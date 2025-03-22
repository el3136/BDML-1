import os
import torch
import math
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Load training and testing text file paths
with open("train_txt_files.txt", "r") as f:
    train_txt_files = f.read().splitlines()
with open("test_txt_files.txt", "r") as f:
    test_txt_files = f.read().splitlines()

train_dataset = load_dataset("text", data_files=train_txt_files, streaming=True)
eval_dataset = load_dataset("text", data_files=test_txt_files, streaming=True)

# ================== TRAINING STEP ==================

# Define model path
model_name = "/scratch/BDML25SP/Llama3.2-3B"

# Enable 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load quantized model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure tokenizer padding side for causal LM
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Enable gradient checkpointing for memory optimization
model.gradient_checkpointing_enable()

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)

# Apply LoRA
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# Define training arguments
training_args = TrainingArguments(
    output_dir="/scratch/el3136/your-finetuned-llama",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
)

# Define Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# ================== EVALUATION STEP ==================

def compute_perplexity(model_path, text):
    """Computes perplexity of a given text."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**tokens, labels=tokens['input_ids'])
        loss = outputs.loss.item()
    return math.exp(loss)

# Evaluate on a sample text from the test set
eval_text = next(iter(eval_dataset))["text"]  # Using first test PDF
perplexity = compute_perplexity("/scratch/el3136/your-finetuned-llama", eval_text)
print(f"Perplexity: {perplexity:.2f}")
