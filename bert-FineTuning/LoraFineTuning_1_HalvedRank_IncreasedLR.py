# Lora FinTuning with Below Changed Parameters
# r=8: Halves the rank
# learning_rate=5e-5: Increase Leraning rate



import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from sklearn.metrics import accuracy_score

# Set environment variables for GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check GPU availability
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA not available, using CPU")

# Define paths
model_output_dir = "./lora_finetuned_bert_1"
local_model_path = "./pretrained_bert_base_cased"  # Pre-downloaded BERT files

# Verify local model path
if not os.path.exists(local_model_path):
    print(f"Error: Local model path '{local_model_path}' not found.")
    print("Ensure 'pretrained_bert_base_cased' is in your directory.")
    exit(1)

# Load IMDb dataset
print("Loading IMDb dataset...", flush=True)
raw_datasets = load_from_disk("imdb_dataset")
print(raw_datasets)

# Load BERT tokenizer from local path
print("Loading BERT tokenizer from local path...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize dataset
print("Tokenizing dataset...", flush=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, batch_size=16)

# Prepare datasets
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# Load base model from local path
print("Loading base model from local path...", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(
    local_model_path,
    num_labels=2,
    torch_dtype=torch.float32
)

# Configure LoRA without quantization
print("Configuring LoRA...", flush=True)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    use_rslora=False,
    use_dora=False
)

# Apply LoRA with error handling
print("Applying LoRA to model...", flush=True)
try:
    model_lora = get_peft_model(model, lora_config)
    print("LoRA applied successfully")
    model_lora.print_trainable_parameters()
except Exception as e:
    print(f"Error applying LoRA: {e}")
    print("Falling back to regular fine-tuning without LoRA")
    model_lora=model

# Define compute_metrics function
def compute_metrics(pred):
    logits, labels = pred
    preds = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    return {"accuracy": accuracy_score(labels, preds)}

# Define training arguments
batch_size = 4
print(f"Using batch size: {batch_size}", flush=True)
training_args = TrainingArguments(
    output_dir="./lora_finetuned_bert_check_1",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=False,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    dataloader_num_workers=1,
    report_to="none"
)

# Create trainer
print("Creating trainer...", flush=True)
trainer = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
print("Starting fine-tuning...", flush=True)
try:
    trainer.train()
    print("Training completed successfully", flush=True)
except Exception as e:
    print(f"Training error encountered: {e}", flush=True)
    print("Try reducing batch size further or using CPU only", flush=True)
    exit(1)

# Save the model (ensure LoRA adapters are saved correctly)
print("Saving fine-tuned model...", flush=True)
model_lora.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

# Evaluate model
print("Evaluating model...", flush=True)
metrics = trainer.evaluate()
print(f"Evaluation Accuracy: {metrics['eval_accuracy']:.4f}", flush=True)

# Load and test the fine-tuned model
print("Loading fine-tuned model for testing...", flush=True)
# Load the full fine-tuned model directly since it was saved as a complete model
model_saved_lora = AutoModelForSequenceClassification.from_pretrained(
    model_output_dir,
    num_labels=2,
    torch_dtype=torch.float32
)
print("Successfully loaded fine-tuned model")

# Load tokenizer from the saved directory
tokenizer = AutoTokenizer.from_pretrained(model_output_dir)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_saved_lora = model_saved_lora.to(device)

# Test the fine-tuned model on a sample
sample_text = "This movie was fantastic and I enjoyed every minute of it!"
encoding = tokenizer(sample_text, return_tensors="pt", truncation=True, padding="max_length")
encoding = {k: v.to(device) for k, v in encoding.items()}

with torch.no_grad():
    outputs = model_saved_lora(**encoding)
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"Sample prediction: {'Positive' if prediction == 1 else 'Negative'}")