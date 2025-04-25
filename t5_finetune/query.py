import torch
import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
from datasets import load_from_disk
import time
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from rouge_score import rouge_scorer

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model paths
model_paths = {
    "T5-Base": "t5-base",
    "SFT": "./sft_t5_model",
    "LoRA": "./lora_t5_model",
    "Adapter": "./adapter_t5_model"
}

# Function to load models
def load_model(model_name, model_path):
    print(f"Loading {model_name} model from {model_path}")
    try:
        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_path if os.path.isdir(model_path) else "t5-base", legacy=False)

        if model_name == "T5-Base":
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        elif model_name in ["LoRA", "Adapter"]:
            # For PEFT models, load base model first
            base_model_peft = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"PEFT model path not found: {model_path}")
            model = PeftModel.from_pretrained(base_model_peft, model_path).to(device)
            model.eval()
        else:  # Standard fine-tuned model (SFT)
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"SFT model path not found: {model_path}")
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name} at {model_path}: {e}")
        raise

# Function to generate code
def generate_code(model, tokenizer, input_text, max_length=256):
    if input_text is None:
        print("Warning: Received None input_text. Returning empty string.")
        return "", 0.0

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        start_time = time.time()
        try:
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
            generated_ids = outputs[0]
        except Exception as e:
            print(f"Error during model.generate: {e}")
            return "GENERATION_ERROR", time.time() - start_time

        end_time = time.time()

    try:
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except Exception as e:
        print(f"Error during tokenizer.decode: {e}")
        generated_code = "DECODING_ERROR"

    generation_time = end_time - start_time
    return generated_code, generation_time

# Simple Python syntax check
def check_syntax(code):
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

# Set up metrics
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
nltk_smooth = SmoothingFunction().method1

# Calculate BLEU score
def calculate_bleu(reference, candidate):
    if not isinstance(reference, str) or not isinstance(candidate, str) or not reference or not candidate:
        return 0.0
    reference_tokens = [nltk.word_tokenize(reference)]
    candidate_tokens = nltk.word_tokenize(candidate)
    if len(candidate_tokens) == 0: 
        return 0.0
    try:
        return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=nltk_smooth)
    except Exception:
        return 0.0

# Calculate ROUGE scores
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
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

# Function to evaluate on a dataset
def evaluate_on_dataset(dataset_name, dataset_path, num_samples=10):
    print(f"\n--- Evaluating on {dataset_name} dataset ---")
    
    # Try to load the dataset
    try:
        dataset = load_from_disk(dataset_path)
        print(f"Loaded {dataset_name} dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return []
    
    # Randomly sample from the dataset
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        samples = [dataset[int(i)] for i in indices]
    else:
        samples = [dataset[i] for i in range(len(dataset))]
    
    all_results = []
    
    # For each sample, evaluate all models
    for i, sample in enumerate(samples):
        print(f"\n---- Sample {i+1}/{len(samples)} ----")
        
        # Get input and reference
        input_text = sample.get("input", "")
        reference = sample.get("target", "")
        
        if not input_text:
            print("Warning: Empty input text. Skipping sample.")
            continue
        
        print(f"Input: {input_text[:100]}...")
        print(f"Reference: {reference[:100]}...")
        
        sample_results = {
            "dataset": dataset_name,
            "sample_idx": i,
            "input": input_text,
            "reference": reference
        }
        
        # Generate with each model
        for model_name, model_path in model_paths.items():
            try:
                # Load model
                print(f"Running {model_name}...")
                tokenizer, model = load_model(model_name, model_path)
                
                # Generate code
                generated, gen_time = generate_code(model, tokenizer, input_text)
                
                # Check syntax
                is_valid = check_syntax(generated)
                
                # Calculate metrics
                bleu = calculate_bleu(reference, generated)
                rouge_scores = calculate_rouge(reference, generated)
                
                # Save results
                sample_results[f"{model_name}_output"] = generated
                sample_results[f"{model_name}_time"] = gen_time
                sample_results[f"{model_name}_valid_syntax"] = is_valid
                sample_results[f"{model_name}_bleu"] = bleu
                sample_results[f"{model_name}_rouge1"] = rouge_scores["rouge1"]
                sample_results[f"{model_name}_rouge2"] = rouge_scores["rouge2"]
                sample_results[f"{model_name}_rougeL"] = rouge_scores["rougeL"]
                
                # Print metrics
                print(f"  Generated {len(generated)} chars in {gen_time:.2f}s")
                print(f"  BLEU: {bleu:.4f}, ROUGE-L: {rouge_scores['rougeL']:.4f}, Valid Syntax: {is_valid}")
                
                # Clean up
                del model
                del tokenizer
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                sample_results[f"{model_name}_output"] = f"Error: {str(e)}"
                sample_results[f"{model_name}_time"] = 0.0
                sample_results[f"{model_name}_valid_syntax"] = False
                sample_results[f"{model_name}_bleu"] = 0.0
                sample_results[f"{model_name}_rouge1"] = 0.0
                sample_results[f"{model_name}_rouge2"] = 0.0
                sample_results[f"{model_name}_rougeL"] = 0.0
        
        all_results.append(sample_results)
    
    return all_results

# Main function
def main():
    # Create output directory
    os.makedirs("./dataset_comparisons", exist_ok=True)
    
    # Define datasets to evaluate
    datasets = {
        "Train": "./preprocessed_train",
        "Test": "./preprocessed_test"
    }
    
    # Number of examples to evaluate from each dataset
    num_samples = 1  # Adjust as needed
    
    all_results = []
    
    # Evaluate on each dataset
    for dataset_name, dataset_path in datasets.items():
        results = evaluate_on_dataset(dataset_name, dataset_path, num_samples)
        all_results.extend(results)
    
    # Convert to DataFrame and save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = "./dataset_comparisons/model_comparison_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")
        
        # Generate summary stats
        summary_data = []
        for dataset_name in datasets.keys():
            dataset_results = df[df["dataset"] == dataset_name]
            
            for model_name in model_paths.keys():
                avg_bleu = dataset_results[f"{model_name}_bleu"].mean()
                avg_rougeL = dataset_results[f"{model_name}_rougeL"].mean()
                avg_time = dataset_results[f"{model_name}_time"].mean()
                syntax_rate = dataset_results[f"{model_name}_valid_syntax"].mean() * 100
                
                summary_data.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Avg BLEU": avg_bleu,
                    "Avg ROUGE-L": avg_rougeL,
                    "Avg Gen Time (s)": avg_time,
                    "Valid Syntax Rate (%)": syntax_rate
                })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_path = "./dataset_comparisons/summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary statistics to {summary_path}")
        
        # Print summary table
        print("\n----- Summary Statistics -----")
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(summary_df)
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()