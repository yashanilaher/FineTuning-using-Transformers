# Import necessary libraries
import os
import torch
from transformers import RobertaTokenizer, RobertaModel, EncoderDecoderModel
from datasets import load_dataset
import shutil

# Create directories for saving models and datasets
os.makedirs("local_models/roberta_base", exist_ok=True)
os.makedirs("local_data/CNN_dataset", exist_ok=True)

print("Downloading and saving RoBERTa model...")
# Download RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

# Save the model and tokenizer locally
roberta_model.save_pretrained("local_models/roberta_base")
tokenizer.save_pretrained("local_models/roberta_base")
print("RoBERTa model and tokenizer saved to local_models/roberta_base")

# Create and save the encoder-decoder model based on RoBERTa
print("Creating and saving encoder-decoder model from RoBERTa...")
encoder_decoder_model = EncoderDecoderModel.from_encoder_decoder_pretrained('roberta-base', 'roberta-base')

# Configure the decoder
encoder_decoder_model.config.decoder_start_token_id = tokenizer.cls_token_id
encoder_decoder_model.config.eos_token_id = tokenizer.sep_token_id
encoder_decoder_model.config.pad_token_id = tokenizer.pad_token_id
encoder_decoder_model.config.vocab_size = encoder_decoder_model.config.encoder.vocab_size

# Save the encoder-decoder model
encoder_decoder_model.save_pretrained("local_models/roberta_encoder_decoder")
print("RoBERTa encoder-decoder model saved to local_models/roberta_encoder_decoder")

print("Downloading and saving CNN/DailyMail dataset...")
# Download dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Save dataset to disk
dataset.save_to_disk("local_data/CNN_dataset")
print("CNN/DailyMail dataset saved to local_data/CNN_dataset")

print("All resources have been downloaded and saved locally!")

