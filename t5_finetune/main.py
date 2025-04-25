import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, pipeline
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, AdaLoraConfig
import evaluate

# Set Hugging Face cache
os.environ["HF_HOME"] = "./hf_cache"

# Set device (CUDA for GPUs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=True)
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

# Step 2: Load and save raw CodeSearchNet
dataset = load_dataset("code_search_net", "python", split="train", trust_remote_code=True)
print(f"Raw dataset size: {len(dataset)} examples")
dataset.save_to_disk("./raw_dataset")
print("Saved raw dataset to ./raw_dataset")

# Step 3: Preprocess and split
def preprocess(example):
    input_text = f"task: {example['func_documentation_string']} -> code:"
    target_text = example["func_code_string"]
    return {"input": input_text, "target": target_text}

preprocessed_dataset = dataset.map(preprocess).select_columns(["input", "target"])

# Dynamic split
total_size = len(preprocessed_dataset)
train_size = 350000
valid_size = (total_size - train_size) // 2
test_size = total_size - train_size - valid_size
print(f"Train size: {train_size}, Valid size: {valid_size}, Test size: {test_size}")

train_dataset = preprocessed_dataset.shuffle(seed=42).select(range(train_size))
valid_dataset = preprocessed_dataset.shuffle(seed=42).select(range(train_size, train_size + valid_size))
test_dataset = preprocessed_dataset.shuffle(seed=42).select(range(train_size + valid_size, total_size))

# Save preprocessed datasets
train_dataset.save_to_disk("./preprocessed_train")
valid_dataset.save_to_disk("./preprocessed_valid")
test_dataset.save_to_disk("./preprocessed_test")
print("Saved preprocessed datasets to ./preprocessed_train, ./preprocessed_valid, ./preprocessed_test")

# Step 4: Tokenize
def tokenize(batch):
    inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    targets = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": targets["input_ids"].squeeze()
    }

tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["input", "target"])
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_valid = valid_dataset.map(tokenize, batched=True, remove_columns=["input", "target"])
tokenized_valid.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test = test_dataset.map(tokenize, batched=True, remove_columns=["input", "target"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Save tokenized datasets
tokenized_train.save_to_disk("./tokenized_train")
tokenized_valid.save_to_disk("./tokenized_valid")
tokenized_test.save_to_disk("./tokenized_test")
print("Saved tokenized datasets to ./tokenized_train, ./tokenized_valid, ./tokenized_test")

# Step 5: Fine-Tuning Methods

# Method 1: Standard Fine-Tuning (SFT)
def run_sft():
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    training_args = TrainingArguments(
        output_dir="./sft_t5",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        logging_steps=1000,
        save_steps=2000,
        learning_rate=2e-5,
        eval_strategy="steps",
        eval_steps=2000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid
    )
    print("Starting SFT...")
    trainer.train()
    model.save_pretrained("./sft_t5_model")
    tokenizer.save_pretrained("./sft_t5_model")
    print("SFT done!")

# Method 2: LoRA
def run_lora():
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir="./lora_t5",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        logging_steps=1000,
        save_steps=2000,
        learning_rate=1e-4,
        eval_strategy="steps",
        eval_steps=2000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid
    )
    print("Starting LoRA...")
    trainer.train()
    model.save_pretrained("./lora_t5_model")
    tokenizer.save_pretrained("./lora_t5_model")
    print("LoRA done!")

# Method 3: Adapter Tuning
def run_adapter():
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    adapter_config = AdaLoraConfig(
        init_r=12,
        target_r=8,
        tinit=200,
        tfinal=500,
        deltaT=10,
        total_step=6000,  # Approx 350,000 / (32 * 2) = 5,469
        target_modules=["q", "v"],
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, adapter_config)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir="./adapter_t5",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        logging_steps=1000,
        save_steps=2000,
        learning_rate=3e-4,
        eval_strategy="steps",
        eval_steps=2000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid
    )
    print("Starting Adapter Tuning...")
    trainer.train()
    model.save_pretrained("./adapter_t5_model")
    tokenizer.save_pretrained("./adapter_t5_model")
    print("Adapter Tuning done!")

# Run pipeline
if __name__ == "__main__":
    run_sft()
    run_lora()
    run_adapter()