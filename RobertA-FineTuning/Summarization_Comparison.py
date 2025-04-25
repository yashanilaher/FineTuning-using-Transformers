# Import necessary libraries
import torch
from transformers import RobertaTokenizer, EncoderDecoderModel
from datasets import load_from_disk

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and models
print("Loading models and tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("local_models/roberta_base")

# Load the pre-trained (before fine-tuning) model
pretrained_model = EncoderDecoderModel.from_pretrained("local_models/roberta_encoder_decoder").to(device)

# Load the fine-tuned model 
finetuned_model = EncoderDecoderModel.from_pretrained("./fine_tuned_models/roberta_summarization_model").to(device)

# Load test dataset
print("Loading test dataset...")
dataset = load_from_disk("local_data/CNN_dataset")
test_dataset = dataset["test"]

# Define maximum lengths
max_input_length = 512
max_output_length = 128

# Function to generate summary
def generate_summary(model, text, max_length=128):
    inputs = tokenizer(
        text,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Compare summaries for a few examples
print("\n===== COMPARING SUMMARIES BEFORE AND AFTER FINE-TUNING =====\n")

# Test on more examples for better comparison
for i in range(5):
    article = test_dataset[i]["article"]
    reference_summary = test_dataset[i]["highlights"]
    
    # Generate summary with pre-trained model (before fine-tuning)
    pretrained_summary = generate_summary(pretrained_model, article)
    
    # Generate summary with fine-tuned model
    finetuned_summary = generate_summary(finetuned_model, article)
    
    print(f"\n===== EXAMPLE {i+1} =====")
    print(f"\nORIGINAL ARTICLE (excerpt):\n{article[:300]}...\n")
    print(f"REFERENCE SUMMARY:\n{reference_summary}\n")
    print(f"BEFORE FINE-TUNING:\n{pretrained_summary}\n")
    print(f"AFTER FINE-TUNING:\n{finetuned_summary}\n")
    print("-" * 80)

# Calculate ROUGE scores for both models on a larger sample
from rouge import Rouge
rouge = Rouge()

print("\nCalculating ROUGE scores for 20 examples...")

# Test on a reasonable number of examples
sample_size = 20
pretrained_summaries = []
finetuned_summaries = []
reference_summaries = []

for i in range(sample_size):
    article = test_dataset[i]["article"]
    reference = test_dataset[i]["highlights"]
    
    pretrained_summary = generate_summary(pretrained_model, article)
    finetuned_summary = generate_summary(finetuned_model, article)
    
    pretrained_summaries.append(pretrained_summary)
    finetuned_summaries.append(finetuned_summary)
    reference_summaries.append(reference)

# Calculate ROUGE scores for both models
pretrained_rouge = rouge.get_scores(pretrained_summaries, reference_summaries, avg=True)
finetuned_rouge = rouge.get_scores(finetuned_summaries, reference_summaries, avg=True)

print("\n===== ROUGE SCORES COMPARISON =====")
print("\nBEFORE FINE-TUNING:")
print(f"ROUGE-1: {pretrained_rouge['rouge-1']['f']:.4f}")
print(f"ROUGE-2: {pretrained_rouge['rouge-2']['f']:.4f}")
print(f"ROUGE-L: {pretrained_rouge['rouge-l']['f']:.4f}")

print("\nAFTER FINE-TUNING:")
print(f"ROUGE-1: {finetuned_rouge['rouge-1']['f']:.4f}")
print(f"ROUGE-2: {finetuned_rouge['rouge-2']['f']:.4f}")
print(f"ROUGE-L: {finetuned_rouge['rouge-l']['f']:.4f}")

print("\nComparison complete!")