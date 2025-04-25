# Standard FineTuning


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

from datasets import load_from_disk
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification, DefaultDataCollator
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Add this right after your imports, before creating any models
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Only allow TensorFlow to allocate as much GPU memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load IMDb dataset
raw_datasets = load_from_disk("imdb_dataset")
print("Dataset Loaded:", raw_datasets, flush=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Get full train and test datasets
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

# Load the BERT model
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# Data collator (handles batching)
data_collator = DefaultDataCollator(return_tensors="tf")

# Convert dataset to TensorFlow format
tf_train_dataset = full_train_dataset.to_tf_dataset(
    columns=tokenizer.model_input_names,  # Use only model input columns
    label_cols=["label"],  # Labels
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)

tf_eval_dataset = full_eval_dataset.to_tf_dataset(
    columns=tokenizer.model_input_names,
    label_cols=["label"],
    batch_size=4,
    shuffle=False,
    collate_fn=data_collator
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy()],
)

# Train the model
model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=5)

# Save the trained model
model.save_pretrained("my_imdb_model")

# Convert to PyTorch model
pytorch_model = AutoModelForSequenceClassification.from_pretrained("my_imdb_model", from_tf=True)

# Evaluate the fine-tuned model
eval_results = model.evaluate(tf_eval_dataset)
print(f"Evaluation Loss: {eval_results[0]}, Evaluation Accuracy: {eval_results[1]}", flush=True)

# ------------------- Run Example Predictions -------------------

# Load the original (untrained) BERT model
original_model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# Load the fine-tuned model
fine_tuned_model = TFAutoModelForSequenceClassification.from_pretrained("my_imdb_model")

# Function for making predictions
# def predict_sentiment(model, text):
#     inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
#     outputs = model(inputs)
#     probs = tf.nn.softmax(outputs.logits, axis=-1)
#     sentiment = "Positive" if tf.argmax(probs, axis=-1).numpy()[0] == 1 else "Negative"
#     return sentiment, probs.numpy()

# # Test with a sample review
# test_review = "The movie was fantastic and I loved every moment of it!"
# original_sentiment, original_probs = predict_sentiment(original_model, test_review)
# fine_tuned_sentiment, fine_tuned_probs = predict_sentiment(fine_tuned_model, test_review)

# # Print results
# print("\nOriginal Model Prediction:")
# print(f"Sentiment: {original_sentiment}, Probabilities: {original_probs}")

# print("\nFine-Tuned Model Prediction:")
# print(f"Sentiment: {fine_tuned_sentiment}, Probabilities: {fine_tuned_probs}")

# ------------------- Evaluate Model Accuracy -------------------

# Function to evaluate accuracy
def evaluate_model_accuracy(model, dataset):
    all_preds, all_labels = [], []
    
    for batch in dataset:
        inputs = {key: batch[0][key] for key in tokenizer.model_input_names}  # Extract input features
        labels = batch[1]  # Extract labels

        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        preds = tf.argmax(probs, axis=-1).numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return accuracy_score(all_labels, all_preds)

# Compute accuracy for both models
original_accuracy = evaluate_model_accuracy(original_model, tf_eval_dataset)
fine_tuned_accuracy = evaluate_model_accuracy(fine_tuned_model, tf_eval_dataset)

print(f"\nOriginal Model Accuracy: {original_accuracy:.4f}", flush=True)
print(f"Fine-Tuned Model Accuracy: {fine_tuned_accuracy:.4f}", flush=True)
