# Install required libraries (uncomment and run if not installed)
# !pip install transformers torch

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Load the T5-base tokenizer
    print("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
    print("Tokenizer loaded successfully!")

    # Load the T5-base model
    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    print("Model loaded successfully!")

    # Move the model to the appropriate device
    model = model.to(device)
    print(f"Model moved to {device}")

    # Verify the model architecture
    print("Model architecture:\n", model)

    # Simple test inference
    print("Running sample inference...")
    input_text = "translate English to French: Hello world"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50,
        )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Sample output (pre-fine-tuning):", decoded_output)

    # Save the model and tokenizer (optional)
    print("Saving model and tokenizer...")
    model.save_pretrained("./t5_base_initial")
    tokenizer.save_pretrained("./t5_base_initial")
    print("Model and tokenizer saved to './t5_base_initial'")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback

    traceback.print_exc()

# Check memory usage (optional, for debugging)
if torch.cuda.is_available():
    print(
        f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB"
    )
