# Import necessary libraries
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    RobertaTokenizer, 
    EncoderDecoderModel,
    get_linear_schedule_with_warmup
)
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

# Fix for AdamW import
from torch.optim import AdamW

# Download necessary NLTK data
nltk.download('punkt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model from local directory
print("Loading RoBERTa model and tokenizer from local directory...")
tokenizer = RobertaTokenizer.from_pretrained("local_models/roberta_base")
encoder_decoder_model = EncoderDecoderModel.from_pretrained("local_models/roberta_encoder_decoder").to(device)

# Load dataset from local directory
print("Loading CNN/DailyMail dataset from local directory...")
dataset = load_from_disk("local_data/CNN_dataset")
print(f"Dataset loaded with {len(dataset['train'])} training examples")

# Use smaller dataset for faster training/testing
# train_dataset = dataset["train"].select(range(1000))
# val_dataset = dataset["validation"].select(range(1000))
# test_dataset = dataset["test"].select(range(1000))

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Define maximum lengths
max_input_length = 512
max_target_length = 128

# Define preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(
        examples["article"], 
        max_length=max_input_length, 
        padding="max_length", 
        truncation=True
    )
    targets = tokenizer(
        examples["highlights"], 
        max_length=max_target_length, 
        padding="max_length", 
        truncation=True
    )
    
    # Replace padding token id with -100 for loss calculation
    target_ids = targets["input_ids"]
    for i in range(len(target_ids)):
        target_ids[i] = [
            -100 if token == tokenizer.pad_token_id else token 
            for token in target_ids[i]
        ]
    
    inputs["labels"] = target_ids
    return inputs

# Process datasets in batches to avoid memory issues
print("Processing training dataset...")
batch_size = 1000
processed_train = []

for i in tqdm(range(0, len(train_dataset), batch_size)):
    end_idx = min(i + batch_size, len(train_dataset))
    batch = train_dataset.select(range(i, end_idx))
    processed_batch = preprocess_function(batch)
    processed_train.append(processed_batch)

# Function to collate processed batches into one dataset
def collate_processed_batches(processed_batches):
    collated = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    for batch in processed_batches:
        collated["input_ids"].extend(batch["input_ids"])
        collated["attention_mask"].extend(batch["attention_mask"])
        collated["labels"].extend(batch["labels"])
    
    # Convert to PyTorch tensors
    for key in collated:
        collated[key] = torch.tensor(collated[key])
    
    return collated

# Process validation and test sets in one go (they're smaller)
processed_val = preprocess_function(val_dataset)
processed_test = preprocess_function(test_dataset)


train_data = collate_processed_batches(processed_train)


train_tensor_dataset = TensorDataset(
    train_data["input_ids"], 
    train_data["attention_mask"], 
    train_data["labels"]
)

val_tensor_dataset = TensorDataset(
    torch.tensor(processed_val["input_ids"]),
    torch.tensor(processed_val["attention_mask"]),
    torch.tensor(processed_val["labels"])
)

test_tensor_dataset = TensorDataset(
    torch.tensor(processed_test["input_ids"]),
    torch.tensor(processed_test["attention_mask"]),
    torch.tensor(processed_test["labels"])
)

# Create data loaders
train_batch_size = 8
eval_batch_size = 16

train_dataloader = DataLoader(train_tensor_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = DataLoader(val_tensor_dataset, batch_size=eval_batch_size)
test_dataloader = DataLoader(test_tensor_dataset, batch_size=eval_batch_size)

# Prepare model for LoRA by explicitly listing target modules 
# This avoids the need for automatic detection which was causing the error

# First, let's identify modules in our model
print("Model architecture:")
for name, module in encoder_decoder_model.named_modules():
    if "query" in name or "value" in name:
        print(f"- {name}")

# Configure LoRA for efficient fine-tuning with explicit module names
target_modules = []

# For RoBERTa encoder
for i in range(12):  # Assuming 12 layers in RoBERTa base
    target_modules.append(f"encoder.roberta.encoder.layer.{i}.attention.self.query")
    target_modules.append(f"encoder.roberta.encoder.layer.{i}.attention.self.value")

# For decoder (if it's structured similarly)
for i in range(12):  
    target_modules.append(f"decoder.roberta.encoder.layer.{i}.attention.self.query")
    target_modules.append(f"decoder.roberta.encoder.layer.{i}.attention.self.value")

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules  
)

try:
    # Try to apply LoRA to the model
    print("Applying LoRA to RoBERTa encoder-decoder model...")
    roberta_lora_model = get_peft_model(encoder_decoder_model, peft_config)
    print(f"Trainable parameters: {roberta_lora_model.print_trainable_parameters()}")
except Exception as e:
    # Fallback option if LoRA application still fails
    print(f"Error applying LoRA: {e}")
    print("Falling back to standard fine-tuning with gradient checkpointing...")
    roberta_lora_model = encoder_decoder_model
    roberta_lora_model.gradient_checkpointing_enable()

# Training parameters
num_epochs = 4
learning_rate = 5e-5
warmup_ratio = 0.1

# Prepare optimizer 
optimizer = AdamW(roberta_lora_model.parameters(), lr=learning_rate)

# Calculate total steps and warmup steps
total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(total_steps * warmup_ratio)

# Create scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Setup mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Add gradient accumulation
gradient_accumulation_steps = 4

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, accumulation_steps):
    model.train()
    total_loss = 0
    step_count = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        # Use mixed precision
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / accumulation_steps
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        
        total_loss += loss.item() * accumulation_steps
        step_count += 1
        
        # Update parameters after accumulation steps
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item() * accumulation_steps})
        
        # Early printing of loss for large datasets
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item() * accumulation_steps:.4f}")
    
    return total_loss / step_count

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    step_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            step_count += 1
    
    return total_loss / step_count

# Setup checkpointing
output_dir = "./fine_tuned_models"
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize best validation loss for model checkpointing
best_val_loss = float("inf")
best_model_path = os.path.join(output_dir, "best_roberta_lora_model.pt")

# Add early stopping
patience = 2
no_improve_count = 0

# Training loop with validation and early stopping
print(f"\nStarting training for {num_epochs} epochs...")
start_time = time.time()

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # Train
    train_loss = train_epoch(
        roberta_lora_model, 
        train_dataloader, 
        optimizer, 
        scheduler, 
        device, 
        scaler, 
        gradient_accumulation_steps
    )
    print(f"Training Loss: {train_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch+1}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': roberta_lora_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Validate
    val_loss = evaluate(roberta_lora_model, val_dataloader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Save model if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(roberta_lora_model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path}")
        no_improve_count = 0
    else:
        no_improve_count += 1
        print(f"No improvement for {no_improve_count} epochs")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

# Training time measurement
training_time = time.time() - start_time
hours, remainder = divmod(training_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

# Load the best model
roberta_lora_model.load_state_dict(torch.load(best_model_path))

# Save the complete fine-tuned model
final_output_dir = os.path.join(output_dir, "roberta_summarization_model")
os.makedirs(final_output_dir, exist_ok=True)
roberta_lora_model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"Model saved to {final_output_dir}")

# Generate summaries function
def generate_summary(model, tokenizer, text, max_length=128):
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

# Test on a few examples
print("\nGenerating sample summaries...")
test_articles = [test_dataset[i]["article"] for i in range(5)]
reference_summaries = [test_dataset[i]["highlights"] for i in range(5)]

for i, article in enumerate(test_articles):
    print(f"\nExample {i+1}:")
    print(f"Original Article (excerpt): {article[:200]}...")
    print(f"\nReference Summary: {reference_summaries[i]}")
    
    generated_summary = generate_summary(roberta_lora_model, tokenizer, article)
    print(f"\nGenerated Summary: {generated_summary}")
    print("-" * 80)

# Calculate ROUGE scores for the test examples
rouge = Rouge()

print("\nCalculating ROUGE scores for examples...")
generated_summaries = []
for i in tqdm(range(min(100, len(test_dataset)))):  # Test on first 100 examples
    article = test_dataset[i]["article"]
    generated_summary = generate_summary(roberta_lora_model, tokenizer, article)
    generated_summaries.append(generated_summary)

reference_summaries = [test_dataset[i]["highlights"] for i in range(min(100, len(test_dataset)))]
rouge_scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)

print("\nROUGE Scores:")
print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}")
print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")

print("\nTraining and evaluation complete!")