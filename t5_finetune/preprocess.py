import evaluate
import torch
from datasets import load_dataset
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

# Set device to MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(
    "t5-base", legacy=True
)  # Suppress legacy warning
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

# Step 2: Load and preprocess CodeSearchNet
dataset = load_dataset(
    "code_search_net", "python", split="train[:10%]", trust_remote_code=True
)
print(f"Dataset size: {len(dataset)} examples")


def preprocess(example):
    input_text = f"task: {example['func_documentation_string']} -> code:"
    target_text = example["func_code_string"]
    return {"input": input_text, "target": target_text}


dataset = dataset.map(preprocess).select_columns(["input", "target"])
train_dataset = dataset.shuffle(seed=42).select(range(5000))  # 5k for training
test_dataset = dataset.shuffle(seed=42).select(range(5000, 5500))  # 500 for eval


# Step 3: Tokenize dataset
def tokenize(batch):
    inputs = tokenizer(
        batch["input"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    targets = tokenizer(
        batch["target"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": targets["input_ids"].squeeze(),
    }


tokenized_dataset = train_dataset.map(
    tokenize, batched=True, remove_columns=["input", "target"]
)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test = test_dataset.map(
    tokenize, batched=True, remove_columns=["input", "target"]
)
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Step 4: Fine-Tuning Methods


# Method 1: Standard Fine-Tuning (SFT)
def run_sft():
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    training_args = TrainingArguments(
        output_dir="./sft_t5",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=100,
        save_steps=500,
        learning_rate=2e-5,
        fp16=False,
        use_mps_device=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
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
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir="./lora_t5",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_steps=100,
        save_steps=500,
        learning_rate=1e-4,
        use_mps_device=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
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
        target_modules=["q", "v"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, adapter_config)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir="./adapter_t5",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_steps=100,
        save_steps=500,
        learning_rate=3e-4,
        use_mps_device=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    print("Starting Adapter Tuning...")
    trainer.train()
    model.save_pretrained("./adapter_t5_model")
    tokenizer.save_pretrained("./adapter_t5_model")
    print("Adapter Tuning done!")


# Step 5: Evaluation
def evaluate_models():
    bleu = evaluate.load("bleu")
    for method in ["sft", "lora", "adapter"]:
        model = T5ForConditionalGeneration.from_pretrained(f"./{method}_t5_model").to(
            device
        )
        generator = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer, device=device
        )
        predictions = []
        references = []
        for example in tokenized_test:
            pred = generator(
                tokenizer.decode(example["input_ids"], skip_special_tokens=True),
                max_length=128,
            )[0]["generated_text"]
            ref = tokenizer.decode(example["labels"], skip_special_tokens=True)
            predictions.append(pred)
            references.append(ref)  # BLEU expects single reference per prediction
        bleu_score = bleu.compute(predictions=predictions, references=references)[
            "bleu"
        ]
        print(f"{method.upper()} BLEU Score: {bleu_score:.4f}")


# Run pipeline
if __name__ == "__main__":
    run_sft()
    run_lora()
    run_adapter()
    evaluate_models()
