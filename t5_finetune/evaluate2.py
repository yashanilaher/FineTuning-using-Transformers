import torch
import os
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import time

# Download NLTK resources
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load test dataset
try:
    test_dataset = load_from_disk("./preprocessed_test")
    print(f"Loaded test dataset with {len(test_dataset)} examples")
except Exception as e:
    print(f"Error loading preprocessed test dataset: {e}")
    print("Trying to load tokenized test dataset instead...")
    test_dataset = load_from_disk("./tokenized_test")
    # Convert back to original format if needed
    if "input" not in test_dataset.column_names:
        print("Converting tokenized dataset back to text format")
        # This is a fallback in case we only have tokenized data

# Initialize evaluation metrics
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1

# Load base model and tokenizer
base_tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=True)
base_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

# Define paths for fine-tuned models
model_paths = {
    "T5-Base": "t5-base",
    "SFT": "./sft_t5_model",
    "LoRA": "./lora_t5_model",
    "Adapter": "./adapter_t5_model"
}

# Function to load models
def load_model(model_name, model_path):
    print(f"Loading {model_name} model from {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=True)
    
    if model_name == "T5-Base":
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    elif model_name in ["LoRA", "Adapter"]:
        # For PEFT models
        base_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
        model = PeftModel.from_pretrained(base_model, model_path).to(device)
    else:
        # Standard fine-tuned model
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    
    return tokenizer, model

# Function to generate code
def generate_code(model, tokenizer, input_text, max_length=128):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        end_time = time.time()
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = end_time - start_time
    
    return generated_code, generation_time

# Function to calculate code correctness (simple syntax check)
def check_syntax(code):
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
    except Exception:
        # Handle other exceptions that might occur with badly formatted code
        return False

# Function to tokenize for BLEU calculation
def tokenize_for_bleu(text):
    return nltk.word_tokenize(text)

# Calculate BLEU score
def calculate_bleu(reference, candidate):
    reference_tokens = tokenize_for_bleu(reference)
    candidate_tokens = tokenize_for_bleu(candidate)
    
    if len(candidate_tokens) == 0:
        return 0.0
    
    try:
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smooth)
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0

# Calculate ROUGE scores
def calculate_rouge(reference, candidate):
    try:
        scores = rouge_scorer_instance.score(reference, candidate)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

# Calculate CodeBLEU (simplified version)
def calculate_codebleu(reference, candidate):
    # This is a simplified version - just using BLEU for now
    # A full CodeBLEU implementation would also consider AST and data flow
    bleu_score = calculate_bleu(reference, candidate)
    return bleu_score

# Sample evaluation function
def evaluate_sample(model, tokenizer, sample):
    input_text = sample["input"]
    reference = sample["target"]
    
    generated_code, generation_time = generate_code(model, tokenizer, input_text)
    
    # Calculate metrics
    rouge_scores = calculate_rouge(reference, generated_code)
    bleu_score = calculate_bleu(reference, generated_code)
    codebleu_score = calculate_codebleu(reference, generated_code)
    syntactically_correct = check_syntax(generated_code)
    
    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu": bleu_score,
        "codebleu": codebleu_score,
        "syntax_correct": int(syntactically_correct),
        "generation_time": generation_time,
        "reference": reference,
        "generated": generated_code
    }

# Main evaluation function
def evaluate_model(model_name, model_path, test_dataset, num_samples=100):
    tokenizer, model = load_model(model_name, model_path)
    
    # Select a subset of test data for evaluation
    if num_samples and num_samples < len(test_dataset):
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        evaluation_samples = [test_dataset[i] for i in indices]
    else:
        evaluation_samples = test_dataset[:num_samples]
        
    print(f"Evaluating {model_name} on {len(evaluation_samples)} samples...")
    
    results = []
    for sample in tqdm(evaluation_samples):
        result = evaluate_sample(model, tokenizer, sample)
        results.append(result)
    
    # Aggregate results
    aggregated_results = {
        "model": model_name,
        "rouge1": np.mean([r["rouge1"] for r in results]),
        "rouge2": np.mean([r["rouge2"] for r in results]),
        "rougeL": np.mean([r["rougeL"] for r in results]),
        "bleu": np.mean([r["bleu"] for r in results]),
        "codebleu": np.mean([r["codebleu"] for r in results]),
        "syntax_correctness": np.mean([r["syntax_correct"] for r in results]) * 100,  # As percentage
        "avg_generation_time": np.mean([r["generation_time"] for r in results])
    }
    
    # Save detailed results
    detailed_results = pd.DataFrame(results)
    detailed_results["model"] = model_name
    detailed_results.to_csv(f"detailed_results_{model_name.lower().replace('-', '_')}.csv", index=False)
    
    print(f"Results for {model_name}:")
    for metric, value in aggregated_results.items():
        if metric != "model":
            print(f"  {metric}: {value:.4f}")
    
    return aggregated_results, results

# Function to create visualizations
def create_visualizations(all_results):
    results_df = pd.DataFrame(all_results)
    
    # Set up metrics to plot
    metrics = ["rouge1", "rouge2", "rougeL", "bleu", "codebleu", "syntax_correctness", "avg_generation_time"]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            results_df.plot(x="model", y=metric, kind="bar", ax=ax, legend=False)
            ax.set_title(f"{metric} by Model")
            ax.set_ylabel(metric)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Hide any unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved visualization to model_comparison.png")
    
    # Save aggregate results to CSV
    results_df.to_csv("model_evaluation_results.csv", index=False)
    print("Saved aggregate results to model_evaluation_results.csv")

# Function to generate example outputs
def generate_examples(models_dict, test_dataset, num_examples=5):
    example_results = []
    
    # Select random examples
    indices = np.random.choice(len(test_dataset), num_examples, replace=False)
    examples = [test_dataset[i] for i in indices]
    
    for example in examples:
        example_result = {
            "input": example["input"],
            "reference": example["target"]
        }
        
        # Generate output from each model
        for model_name, model_path in models_dict.items():
            tokenizer, model = load_model(model_name, model_path)
            generated, _ = generate_code(model, tokenizer, example["input"])
            example_result[f"{model_name}_output"] = generated
        
        example_results.append(example_result)
    
    # Save examples to file
    examples_df = pd.DataFrame(example_results)
    examples_df.to_csv("model_comparison_examples.csv", index=False)
    print("Saved example outputs to model_comparison_examples.csv")
    
    return example_results

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of samples to evaluate (reduce for quicker evaluation)
    NUM_SAMPLES = 100
    
    # Evaluate all models
    all_results = []
    detailed_results = defaultdict(list)
    
    for model_name, model_path in model_paths.items():
        try:
            aggregated_result, detailed_result = evaluate_model(
                model_name, model_path, test_dataset, num_samples=NUM_SAMPLES
            )
            all_results.append(aggregated_result)
            detailed_results[model_name] = detailed_result
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Create visualizations
    if all_results:
        create_visualizations(all_results)
        
        # Generate and save example outputs
        examples = generate_examples(model_paths, test_dataset, num_examples=5)
    
    print("Evaluation complete!")