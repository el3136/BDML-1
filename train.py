import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, PretrainedConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import random

# Define the directory containing txts
directory = "/scratch/el3136/climate_text_dataset"
txt_files = sorted(
    [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]
)

# Shuffle and split into 90% training and 10% testing
# random.seed(42)  # Ensures consistent splits
random.shuffle(txt_files)
split_index = int(len(txt_files) * 0.9)
train_sets, test_sets = txt_files[:split_index], txt_files[split_index:]

# Save the list of generated text file paths
with open("/scratch/el3136/BDML-1/train_txt_files.txt", "w") as f:
    f.write("\n".join(train_sets))
with open("/scratch/el3136/BDML-1/test_txt_files.txt", "w") as f:
    f.write("\n".join(test_sets))

# Load training and testing text file paths
with open("/scratch/el3136/BDML-1/train_txt_files.txt", "r") as f:
    train_txt_files = f.read().splitlines()
with open("/scratch/el3136/BDML-1/test_txt_files.txt", "r") as f:
    test_txt_files = f.read().splitlines()

# Load training and evaluation datasets
train_dataset = load_dataset("text", data_files=train_txt_files, streaming=True)
eval_dataset = load_dataset("text", data_files=test_txt_files, streaming=True)

# ================== TRAINING STEP ==================

# Define model path
model_name = "/scratch/BDML25SP/Llama3.2-3B/"

# Manually create the configuration object
config = PretrainedConfig(
    model_type="llama",  # Specify the model type
    architectures=["LlamaForCausalLM"],  # Specify the architecture
    dim=3072,
    ffn_dim_multiplier=1.0,
    multiple_of=256,
    n_heads=24,
    n_kv_heads=8,
    n_layers=28,
    norm_eps=1e-05,
    rope_theta=500000.0,
    use_scaled_rope=True,
    vocab_size=128256
)

# Enable 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load quantized model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, quantization_config=quantization_config)
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
    tokenizer=tokenizer,
)

# Start training
trainer.train()
