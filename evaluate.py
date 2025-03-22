import os
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load testing text file paths
with open("test_txt_files.txt", "r") as f:
    test_txt_files = f.read().splitlines()

eval_dataset = load_dataset("text", data_files=test_txt_files, streaming=True)

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
