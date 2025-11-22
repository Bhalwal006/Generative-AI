# gpt2_pretrain.py
"""
Continue pretraining / fine-tuning GPT-2 on local .txt files.

Usage:
    python gpt2_pretrain.py

Requirements (install once):
    pip install transformers datasets accelerate safetensors

Notes:
 - If you have an NVIDIA GPU and want much faster runs, install a CUDA-enabled torch:
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
 - For large datasets increase num_train_epochs and adjust batch size.
 - This script uses Trainer from Hugging Face transformers.
"""

import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import torch

# -------------------------
# Configuration (edit as needed)
# -------------------------
MODEL_NAME_OR_PATH = "gpt2"  # base model to continue pretraining
OUTPUT_DIR = "./gpt2_shoolini"  # where to save final model
DATA_FILES = ["Cricket.txt", "Medical.txt", "Education.txt"]  # local text files
BLOCK_SIZE = 512          # context length for grouping tokens
BATCH_SIZE = 2            # per-device batch size (reduce on low-memory machines)
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
GRAD_ACCUM_STEPS = 8      # accumulate gradients to simulate larger batch
SAVE_STEPS = 500
LOGGING_STEPS = 50
FP16 = False              # set True if GPU + mixed precision supported

# -------------------------
# Quick checks
# -------------------------
for f in DATA_FILES:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Data file not found: {f} (put it in the same folder or update DATA_FILES)")

# -------------------------
# Load dataset (text)
# -------------------------
print("Loading text dataset from files:", DATA_FILES)
dataset = load_dataset("text", data_files={"train": DATA_FILES})
print("Number of raw examples:", len(dataset["train"]))

# -------------------------
# Load tokenizer & model
# -------------------------
print("Loading tokenizer and model:", MODEL_NAME_OR_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
# GPT-2 has no pad token by default — set pad token to eos_token for Trainer compatibility
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
# Resize token embeddings if tokenizer was modified (pad token)
model.resize_token_embeddings(len(tokenizer))

# -------------------------
# Tokenize (no truncation here)
# -------------------------
def tokenize_function(examples):
    return tokenizer(examples["text"])

print("Tokenizing dataset (this may take a moment)...")
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# -------------------------
# Group into blocks of BLOCK_SIZE
# -------------------------
def group_texts(examples):
    # concatenate all input_ids
    concatenated = []
    for ids in examples["input_ids"]:
        concatenated.extend(ids)
    total_length = len(concatenated)
    if total_length >= BLOCK_SIZE:
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    else:
        total_length = 0

    result = {"input_ids": [], "attention_mask": []}
    for i in range(0, total_length, BLOCK_SIZE):
        chunk = concatenated[i: i + BLOCK_SIZE]
        result["input_ids"].append(chunk)
        # attention mask of ones
        result["attention_mask"].append([1] * len(chunk))

    # labels are same as input_ids for causal LM
    result["labels"] = result["input_ids"].copy()
    return result

print("Grouping tokenized text into blocks of size", BLOCK_SIZE)
lm_dataset = tokenized.map(group_texts, batched=True, remove_columns=tokenized.column_names["train"])

# If grouping produced empty dataset (too small), fall back to using tokenized examples directly
if len(lm_dataset["train"]) == 0:
    print("Warning: grouping produced zero blocks (dataset too small). Using tokenized dataset with truncation.")
    # Truncate/pad each example to BLOCK_SIZE
    def truncate(ex):
        for k in ex:
            ex[k] = [t[:BLOCK_SIZE] for t in ex[k]]
        ex["labels"] = ex["input_ids"].copy()
        return ex
    lm_dataset = tokenized.map(truncate, batched=True)

print("Number of training blocks:", len(lm_dataset["train"]))

# -------------------------
# Data collator
# -------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------
# Training args (auto GPU/CPU handling)
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    fp16=FP16 and device == "cuda",
    dataloader_pin_memory=(device == "cuda"),
    report_to="none"  # disable wandb/tensorboard unless configured
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    data_collator=data_collator
)

# -------------------------
# Train
# -------------------------
print("Starting training...")
trainer.train()
print("Training finished. Saving model to", OUTPUT_DIR)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved. Done.")
