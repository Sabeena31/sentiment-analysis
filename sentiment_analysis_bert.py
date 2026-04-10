import os
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ------------------- Setup -------------------
print("Script started successfully.")

# Ensure output directories exist
os.makedirs("downloads", exist_ok=True)

# ------------------- Load Dataset -------------------
print("Loading IMDb dataset...")
dataset = load_dataset("imdb", cache_dir="./downloads")
print("Dataset loaded successfully.")

# ------------------- Load Tokenizer and Model -------------------
print("Initializing BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
print("Tokenizer and model initialized.")

# ------------------- Tokenization -------------------
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

print("Tokenizing dataset...")
dataset = dataset.map(tokenize, batched=True, batch_size=16)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
print("Tokenization complete.")

# ------------------- Prepare Training and Testing Sets -------------------
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))
print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

# ------------------- Training Configuration -------------------
training_args = TrainingArguments(
    output_dir="./downloads/results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./downloads/logs",
    logging_steps=10,
)

# ------------------- Trainer Setup -------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# ------------------- Train Model -------------------
print("Starting model training...")
trainer.train()
print("Training completed successfully.")

# ------------------- Save Model -------------------
model_dir = "./downloads/bert_sentiment_model"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"Model and tokenizer saved to {model_dir}.")
