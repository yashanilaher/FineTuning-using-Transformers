import torch
import os
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_from_disk, load_dataset # Added load_dataset for fallback
from peft import PeftModel
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.translate.meteor_score import meteor_score # Added METEOR
from sacrebleu.metrics import BLEU, CHRF, TER # Added CHRF
from code_bert_score import score as code_bert_score # Added CodeBERTScore
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings

# --- Setup & Downloads ---
# Suppress specific warnings if necessary
warnings.filterwarnings("ignore", category=UserWarning, message=".*legacy=True.*")

# Download NLTK resources (punkt is needed for word tokenization, wordnet for METEOR)
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' resource found.")
except LookupError: # Correct exception to catch
    print("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
    print("NLTK 'wordnet' resource found.")
except LookupError: # Correct exception to catch
    print("NLTK 'wordnet' resource not found. Downloading...")
    nltk.download('wordnet') # Needed for METEOR


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Dataset ---
dataset_loaded = False
try:
    # Attempt to load the preprocessed dataset first
    test_dataset = load_from_disk("./preprocessed_test")
    print(f"Loaded preprocessed test dataset with {len(test_dataset)} examples")
    dataset_loaded = True
except Exception as e:
    print(f"Warning: Error loading preprocessed test dataset: {e}")
    print("Trying to load tokenized test dataset instead...")
    try:
        test_dataset = load_from_disk("./tokenized_test")
        print(f"Loaded tokenized test dataset with {len(test_dataset)} examples")
        # Ensure required columns exist if loaded from tokenized version
        if "input" not in test_dataset.column_names or "target" not in test_dataset.column_names:
             raise ValueError("Tokenized dataset missing 'input' or 'target' columns.")
        dataset_loaded = True
    except Exception as e:
        print(f"Warning: Error loading tokenized test dataset: {e}")
        print("Falling back to loading and preprocessing original dataset ('code_search_net')...")
        try:
            # Fallback to original dataset
            dataset = load_dataset("code_search_net", "python", split="test", trust_remote_code=True) # Use test split for eval
            print(f"Loaded original 'code_search_net' test split with {len(dataset)} examples")

            # Process dataset
            def preprocess(example):
                # Handle potential None values in documentation
                docstring = example['func_documentation_string'] if example['func_documentation_string'] else "No documentation"
                input_text = f"task: {docstring} -> code:"
                target_text = example["func_code_string"]
                return {"input": input_text, "target": target_text}

            # Filter out examples with empty code strings, as they cause issues with metrics
            dataset = dataset.filter(lambda x: x['func_code_string'] is not None and len(x['func_code_string'].strip()) > 0)
            print(f"Filtered dataset to {len(dataset)} examples with non-empty code.")

            test_dataset = dataset.map(preprocess, remove_columns=dataset.column_names) # Keep only input/target
            print(f"Preprocessed dataset. Using {len(test_dataset)} examples.")
            if len(test_dataset) == 0:
                 raise RuntimeError("No valid data examples after preprocessing.")
            dataset_loaded = True
        except Exception as e_fallback:
            print(f"FATAL: Error loading or processing fallback dataset: {e_fallback}")
            exit() # Exit if no dataset can be loaded

if not dataset_loaded:
    print("FATAL: Could not load any dataset.")
    exit()

# --- Initialize Evaluation Metrics ---
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bleu_scorer = BLEU(effective_order=True) # Using sacrebleu's BLEU with recommended setting
chrf_scorer = CHRF()
nltk_smooth = SmoothingFunction().method1 # NLTK's smoothing for its BLEU

# --- Load Models ---
# Define paths for fine-tuned models
model_paths = {
    # "T5-Base": "t5-base",
    # "SFT": "./sft_t5_model",
    # "LoRA": "./lora_t5_model",
    "Adapter": "./adapter_t5_model"
}

# Function to load models
def load_model(model_name, model_path):
    print(f"Loading {model_name} model from {model_path}")
    try:
        # Always load tokenizer with legacy=False unless specifically needed and tested
        tokenizer = T5Tokenizer.from_pretrained(model_path if os.path.isdir(model_path) else "t5-base", legacy=False)

        if model_name == "T5-Base":
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        elif model_name in ["LoRA", "Adapter"]:
            # For PEFT models, ensure the base model matches what PEFT was trained on
            base_model_peft = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
            # Check if path exists before loading PEFT model
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"PEFT model path not found: {model_path}")
            model = PeftModel.from_pretrained(base_model_peft, model_path).to(device)
            model.eval() # Set PEFT model to evaluation mode
        else: # Standard fine-tuned model (SFT)
            if not os.path.isdir(model_path):
                 raise FileNotFoundError(f"SFT model path not found: {model_path}")
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name} at {model_path}: {e}")
        raise

# --- Code Generation ---
def generate_code(model, tokenizer, input_text, max_length=256): # Increased max_length slightly
    # Handle potential None input
    if input_text is None:
        print("Warning: Received None input_text. Returning empty string.")
        return "", 0.0

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device) # Increased input max_length

    with torch.no_grad():
        start_time = time.time()
        try:
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=5, # Standard beam search setting
                early_stopping=True,
                # pad_token_id=tokenizer.pad_token_id, # Ensure pad token is set if needed
                # eos_token_id=tokenizer.eos_token_id  # Ensure EOS token is set if needed
            )
            generated_ids = outputs[0]
        except Exception as e:
            print(f"Error during model.generate: {e}")
            print(f"Input text causing error: {input_text[:100]}...") # Log problematic input
            return "GENERATION_ERROR", time.time() - start_time # Return specific error string

        end_time = time.time()

    # Decode carefully
    try:
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except Exception as e:
        print(f"Error during tokenizer.decode: {e}")
        generated_code = "DECODING_ERROR"

    generation_time = end_time - start_time

    return generated_code, generation_time

# --- Evaluation Metric Calculations ---

# Simple syntax check
def check_syntax(code):
    if code is None or not isinstance(code, str) or not code.strip():
        return False # Cannot compile None or empty string
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
    except Exception as e: # Catch other potential errors like TypeError, ValueError
        # print(f"Syntax check failed with non-SyntaxError: {e}\nCode: {code[:100]}...") # Optional debug
        return False

# NLTK BLEU (kept for reference/comparison if needed)
def calculate_nltk_bleu(reference, candidate):
    if not isinstance(reference, str) or not isinstance(candidate, str) or not reference or not candidate:
        return 0.0
    reference_tokens = [nltk.word_tokenize(reference)] # NLTK expects list of ref lists
    candidate_tokens = nltk.word_tokenize(candidate)
    if len(candidate_tokens) == 0: return 0.0
    try:
        return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=nltk_smooth)
    except Exception as e:
        # print(f"NLTK BLEU calculation error: {e}")
        return 0.0

# SacreBLEU (preferred)
def calculate_sacrebleu(reference, candidate):
    if not isinstance(reference, str) or not isinstance(candidate, str) or not reference or not candidate:
        return 0.0
    # Sacrebleu expects list of references and list of candidates
    score = bleu_scorer.sentence_score(candidate, [reference])
    return score.score / 100.0 # Sacrebleu score is 0-100, convert to 0-1

# ROUGE scores
def calculate_rouge(reference, candidate):
    if not isinstance(reference, str) or not isinstance(candidate, str) or not reference or not candidate:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    try:
        scores = rouge_scorer_instance.score(reference, candidate)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }
    except Exception as e:
        # print(f"ROUGE calculation error: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

# METEOR score
def calculate_meteor(reference, candidate):
    if not isinstance(reference, str) or not isinstance(candidate, str) or not reference or not candidate:
        return 0.0
    try:
        # NLTK meteor_score expects tokenized strings
        ref_tok = nltk.word_tokenize(reference)
        can_tok = nltk.word_tokenize(candidate)
        # Check for empty candidate after tokenization
        if not can_tok:
            return 0.0
        # Handle potential ZeroDivisionError if reference is empty after tokenization
        if not ref_tok:
            return 0.0
        return meteor_score([ref_tok], can_tok) # Pass reference tokens as list of lists
    except ZeroDivisionError:
        # print(f"METEOR ZeroDivisionError: ref='{reference[:50]}...', cand='{candidate[:50]}...'") # Optional debug
        return 0.0 # Return 0 if division by zero occurs
    except Exception as e:
        # print(f"METEOR calculation error: {e}")
        return 0.0

# chrF score
def calculate_chrf(reference, candidate):
    if not isinstance(reference, str) or not isinstance(candidate, str) or not reference or not candidate:
        return 0.0
    # Sacrebleu expects list of references and list of candidates
    score = chrf_scorer.sentence_score(candidate, [reference])
    return score.score / 100.0 # Sacrebleu score is 0-100, convert to 0-1

# CodeBERTScore
def calculate_codebertscore(reference, candidate):
    if not isinstance(reference, str) or not isinstance(candidate, str) or not reference or not candidate:
        return {"codebertscore_p": 0.0, "codebertscore_r": 0.0, "codebertscore_f1": 0.0}
    try:
        # code_bert_score expects lists of candidates and references
        # Use a model suitable for Python code generation/understanding
        # Common choices: 'microsoft/codebert-base', 'neulab/codebert-python'
        # Using a smaller model if resource constrained: 'microsoft/graphcodebert-base' might also work
        # Let's default to a standard one, ensure you have it downloaded/cached
        # code_bert_score returns P, R, F1, potentially followed by other values (like hash)
        scores = code_bert_score([candidate], [reference], lang='python', model_type='microsoft/codebert-base', device=str(device))

        # Check if scores were returned and have at least 3 elements
        if scores and len(scores) >= 3:
            P, R, F1 = scores[0], scores[1], scores[2] # Unpack the first three
            # The function returns tensors, get the float values
            return {
                "codebertscore_p": P.item() if torch.is_tensor(P) else P,
                "codebertscore_r": R.item() if torch.is_tensor(R) else R,
                "codebertscore_f1": F1.item() if torch.is_tensor(F1) else F1
            }
        else:
             print(f"Warning: code_bert_score did not return expected values. Result: {scores}")
             return {"codebertscore_p": 0.0, "codebertscore_r": 0.0, "codebertscore_f1": 0.0}
    except Exception as e:
        print(f"CodeBERTScore calculation error: {e}") # More likely to have setup/env issues
        return {"codebertscore_p": 0.0, "codebertscore_r": 0.0, "codebertscore_f1": 0.0}

# --- Evaluation Loop ---

# Evaluate a single sample
def evaluate_sample(model, tokenizer, sample):
    # Ensure sample has the necessary keys
    if "input" not in sample or "target" not in sample:
        print("Warning: Sample missing 'input' or 'target' key.")
        return None # Skip this sample or return default error values

    input_text = sample["input"]
    reference = sample["target"]

    # Handle case where reference might be None or empty (should have been filtered, but double-check)
    if reference is None or not reference.strip():
        print(f"Warning: Skipping sample with empty reference. Input: {input_text[:50]}...")
        return None

    generated_code, generation_time = generate_code(model, tokenizer, input_text)

    # Check for generation errors
    if generated_code in ["GENERATION_ERROR", "DECODING_ERROR"]:
         print(f"Warning: Skipping metric calculation due to {generated_code}.")
         # Return partial results or skip
         return {
            "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
            "sacrebleu": 0.0, "meteor": 0.0, "chrf": 0.0,
            "codebertscore_p": 0.0, "codebertscore_r": 0.0, "codebertscore_f1": 0.0,
            "syntax_correct": 0, "generation_time": generation_time,
            "reference": reference, "generated": generated_code # Log the error status
        }


    # Calculate metrics
    rouge_scores = calculate_rouge(reference, generated_code)
    sacrebleu_score = calculate_sacrebleu(reference, generated_code)
    meteor = calculate_meteor(reference, generated_code)
    chrf = calculate_chrf(reference, generated_code)
    cbs_scores = calculate_codebertscore(reference, generated_code)
    syntactically_correct = check_syntax(generated_code)

    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "sacrebleu": sacrebleu_score, # Use SacreBLEU as primary BLEU
        "meteor": meteor,
        "chrf": chrf,
        "codebertscore_p": cbs_scores["codebertscore_p"],
        "codebertscore_r": cbs_scores["codebertscore_r"],
        "codebertscore_f1": cbs_scores["codebertscore_f1"],
        "syntax_correct": int(syntactically_correct),
        "generation_time": generation_time,
        "reference": reference,
        "generated": generated_code
    }

# Main evaluation function for a model
def evaluate_model(model_name, model_path, dataset_to_eval, use_subset=False, subset_size=100):
    try:
        tokenizer, model = load_model(model_name, model_path)
    except Exception as e:
        print(f"Skipping evaluation for {model_name} due to loading error: {e}")
        return None, None # Indicate failure

    # Use full dataset or select a subset
    if use_subset:
        # Ensure subset_size does not exceed dataset length
        actual_subset_size = min(subset_size, len(dataset_to_eval))
        if actual_subset_size < subset_size:
            print(f"Warning: Requested subset size {subset_size} exceeds dataset length {len(dataset_to_eval)}. Using {actual_subset_size} samples.")
        if actual_subset_size == 0:
            print(f"Warning: No samples available to evaluate for {model_name} in subset.")
            return {"model": model_name}, pd.DataFrame() # Return empty results

        evaluation_samples = dataset_to_eval.select(range(actual_subset_size))
        num_samples = actual_subset_size
        print(f"Evaluating {model_name} on a subset of {num_samples} samples...")
    else:
        evaluation_samples = dataset_to_eval
        num_samples = len(evaluation_samples)
        print(f"Evaluating {model_name} on the full dataset with {num_samples} samples...")

    if num_samples == 0:
        print(f"Warning: No samples to evaluate for {model_name}.")
        return {"model": model_name}, pd.DataFrame() # Return empty results


    results = []
    output_dir = f"./evaluation_output/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    for i, sample in enumerate(tqdm(evaluation_samples, desc=f"Evaluating {model_name}")):
        try:
            result = evaluate_sample(model, tokenizer, sample)
            if result is not None: # Only append valid results
                 results.append(result)

            # Save intermediate results periodically (adjust frequency if needed)
            if (i + 1) % 50 == 0 and results: # Save every 50 samples for subsets
                intermediate_df = pd.DataFrame(results)
                intermediate_df["model"] = model_name
                intermediate_df.to_csv(os.path.join(output_dir, f"intermediate_results_{model_name.lower().replace('-', '_')}_subset_{i+1}.csv"), index=False)
                # print(f"Saved intermediate results for {model_name} after {i+1} samples")
        except Exception as e:
            print(f"\nError evaluating sample {i} for model {model_name}: {e}")
            # Optionally log the sample that caused the error
            # print(f"Problematic Sample Input: {sample.get('input', 'N/A')[:100]}...")

    if not results:
        print(f"Warning: No valid results were generated for {model_name}.")
        return {"model": model_name}, pd.DataFrame() # Return empty dataframe


    # Aggregate results
    aggregated_results = {
        "model": model_name,
        "num_samples_evaluated": len(results), # Add count of successfully evaluated samples
        "rouge1": np.mean([r["rouge1"] for r in results]),
        "rouge2": np.mean([r["rouge2"] for r in results]),
        "rougeL": np.mean([r["rougeL"] for r in results]),
        "sacrebleu": np.mean([r["sacrebleu"] for r in results]),
        "meteor": np.mean([r["meteor"] for r in results]),
        "chrf": np.mean([r["chrf"] for r in results]),
        "codebertscore_f1": np.mean([r["codebertscore_f1"] for r in results]), # Focus on F1
        "syntax_correctness": np.mean([r["syntax_correct"] for r in results]) * 100, # As percentage
        "avg_generation_time": np.mean([r["generation_time"] for r in results])
    }

    # Save detailed results
    detailed_results_df = pd.DataFrame(results)
    detailed_results_df["model"] = model_name
    # Adjust filename for subset runs
    filename_suffix = "_subset" if use_subset else ""
    detailed_results_df.to_csv(os.path.join(output_dir, f"detailed_results_{model_name.lower().replace('-', '_')}{filename_suffix}.csv"), index=False)

    print(f"\n--- Results for {model_name} (Evaluated on {len(results)} samples) ---")
    for metric, value in aggregated_results.items():
        if metric not in ["model", "num_samples_evaluated"]:
            print(f"  {metric}: {value:.4f}")
        elif metric == "num_samples_evaluated":
             print(f"  {metric}: {value}")
    print("-" * (len(model_name) + 15))


    return aggregated_results, detailed_results_df # Return the DataFrame

# --- Visualization ---
def create_visualizations(all_aggregated_results):
    if not all_aggregated_results:
        print("No aggregated results to visualize.")
        return

    results_df = pd.DataFrame(all_aggregated_results)
    if results_df.empty:
        print("Aggregated results DataFrame is empty. Cannot visualize.")
        return

    output_dir = "./evaluation_output"
    os.makedirs(output_dir, exist_ok=True)

    # Set up metrics to plot
    metrics = [
        "rouge1", "rouge2", "rougeL",
        "sacrebleu", "meteor", "chrf",
        "codebertscore_f1", "syntax_correctness", "avg_generation_time"
    ]
    # Filter out metrics that might be missing if a model failed completely
    plot_metrics = [m for m in metrics if m in results_df.columns]
    num_metrics = len(plot_metrics)

    if num_metrics == 0:
        print("No valid metrics found in results DataFrame. Cannot visualize.")
        return

    # Determine grid size (e.g., 3x3)
    ncols = 3
    nrows = (num_metrics + ncols - 1) // ncols

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False) # Ensure axes is 2D array
    axes = axes.flatten() # Flatten for easy iteration

    # Plot each metric
    for i, metric in enumerate(plot_metrics):
        ax = axes[i]
        try:
            # Sort by model name for consistent plot order
            plot_df = results_df.sort_values(by="model")
            plot_df.plot(x="model", y=metric, kind="bar", ax=ax, legend=False,
                            title=f"{metric.replace('_', ' ').title()} by Model",
                            rot=45) # Rotate labels for better readability
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_xlabel("") # Remove x-label for cleaner look
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            # Add value labels on top of bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', label_type='edge', fontsize=8, padding=2)
        except Exception as plot_err:
             print(f"Error plotting metric '{metric}': {plot_err}")


    # Hide any unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(pad=2.0) # Add padding
    plot_path = os.path.join(output_dir, "model_comparison_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {plot_path}")
    plt.close(fig) # Close the figure

    # Save aggregate results to CSV
    csv_path = os.path.join(output_dir, "model_evaluation_summary.csv")
    # Sort before saving for consistency
    results_df_sorted = results_df.sort_values(by="model")
    results_df_sorted.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved aggregate results to {csv_path}")

# --- Example Generation ---
def generate_examples(models_dict, dataset_for_examples, num_examples=5):
    if not models_dict:
        print("No models provided for example generation.")
        return []

    example_results = []
    output_dir = "./evaluation_output"
    os.makedirs(output_dir, exist_ok=True)

    # Select examples (ensure index is valid)
    actual_num_examples = min(num_examples, len(dataset_for_examples))
    if actual_num_examples == 0:
        print("No examples in dataset to generate from.")
        return []
    examples = dataset_for_examples.select(range(actual_num_examples))
    print(f"\nGenerating examples for {actual_num_examples} samples...")

    # Pre-load models and tokenizers if memory allows (can speed up significantly)
    # loaded_models = {}
    # print("Pre-loading models for example generation (this might take a while)...")
    # for model_name, model_path in models_dict.items():
    #     try:
    #         tokenizer, model = load_model(model_name, model_path)
    #         loaded_models[model_name] = (tokenizer, model)
    #         print(f"Loaded {model_name}.")
    #     except Exception as e:
    #         print(f"Could not pre-load model {model_name}: {e}")
    # print("Finished pre-loading models.")


    for i, example in enumerate(tqdm(examples, desc="Generating Examples")):
        example_result = {
            "input": example.get("input", "N/A"),
            "reference": example.get("target", "N/A")
        }

        # Generate output from each model
        for model_name, model_path in models_dict.items():
            # if model_name in loaded_models: # Use pre-loaded model if available
            #     tokenizer, model = loaded_models[model_name]
            #     try:
            #         generated, _ = generate_code(model, tokenizer, example_result["input"])
            #         example_result[f"{model_name}_output"] = generated
            #     except Exception as e:
            #         print(f"Error generating example with pre-loaded {model_name} for sample {i}: {e}")
            #         example_result[f"{model_name}_output"] = f"Error: {str(e)}"
            # else: # Load model on the fly if not pre-loaded or pre-loading failed
            try:
                # Load model only if needed (can be slow)
                print(f"\nLoading {model_name} for example {i}...") # Added print statement
                tokenizer, model = load_model(model_name, model_path)
                generated, _ = generate_code(model, tokenizer, example_result["input"])
                example_result[f"{model_name}_output"] = generated
                # Clean up model and tokenizer to free GPU memory if necessary
                del model
                del tokenizer
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                print(f"Unloaded {model_name}.") # Added print statement


            except FileNotFoundError:
                 print(f"Skipping example generation for {model_name}: Model path not found at {model_path}")
                 example_result[f"{model_name}_output"] = "Model Not Found"
            except Exception as e:
                print(f"Error generating example with {model_name} for sample {i}: {e}")
                example_result[f"{model_name}_output"] = f"Error: {str(e)}"
                # Clean up memory on error too
                if 'model' in locals(): del model
                if 'tokenizer' in locals(): del tokenizer
                if torch.cuda.is_available(): torch.cuda.empty_cache()


        example_results.append(example_result)

    # Clean up any pre-loaded models
    # del loaded_models
    # if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save examples to file
    examples_df = pd.DataFrame(example_results)
    examples_csv_path = os.path.join(output_dir, "model_comparison_examples.csv")
    examples_df.to_csv(examples_csv_path, index=False)
    print(f"Saved example outputs to {examples_csv_path}")

    return example_results

# --- Main Execution ---
if __name__ == "__main__":
    # Set random seed for reproducibility (less critical for eval but good practice)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    ############################################################################
    #                   CONFIGURATION: Set subset evaluation                   #
    ############################################################################
    USE_SUBSET_FOR_EVAL = True  # <<< Set to True to run on a subset
    EVAL_SUBSET_SIZE = 10     # <<< Set the desired number of samples here
    ############################################################################


    # --- Run Evaluation ---
    all_aggregated_results = []
    all_detailed_results = {} # Store detailed results DataFrames

    print("\nStarting Model Evaluation...")
    if USE_SUBSET_FOR_EVAL:
        print(f"*** Running evaluation on a SUBSET of {EVAL_SUBSET_SIZE} samples ***")
    else:
        print(f"*** Running evaluation on the FULL dataset ***")

    for model_name, model_path in model_paths.items():
        print(f"\n===== Evaluating: {model_name} =====")
        aggregated_result, detailed_df = evaluate_model(
            model_name, model_path, test_dataset,
            use_subset=USE_SUBSET_FOR_EVAL, subset_size=EVAL_SUBSET_SIZE
        )
        if aggregated_result is not None and 'model' in aggregated_result: # Check if evaluation succeeded
            all_aggregated_results.append(aggregated_result)
        if detailed_df is not None and not detailed_df.empty:
             all_detailed_results[model_name] = detailed_df


    # --- Create Visualizations & Summary ---
    print("\nGenerating Visualizations and Summary...")
    if all_aggregated_results:
        create_visualizations(all_aggregated_results)
    else:
        print("No successful evaluations to visualize.")


    # --- Generate Examples ---
    print("\nGenerating Comparison Examples...")
    # Filter model_paths to only include models that were successfully evaluated
    successful_models = {}
    if all_aggregated_results:
        successful_models = {res['model']: model_paths[res['model']] for res in all_aggregated_results if 'model' in res}

    if successful_models:
         # Generate examples using the *same subset size* or less for consistency/speed
         example_gen_size = min(5, EVAL_SUBSET_SIZE) if USE_SUBSET_FOR_EVAL else 5
         examples = generate_examples(successful_models, test_dataset, num_examples=example_gen_size)
    else:
        print("No models were successfully evaluated, skipping example generation.")

    print("\nEvaluation Script Finished!")