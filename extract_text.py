import os
import random
import pdfplumber
from datasets import load_dataset

# ================== PREPROCESSING STEP ==================

def extract_text_from_pdf(pdf_path):
    """Extracts and preprocesses text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text.strip()

# Define the directory containing PDFs
directory = "/scratch/el3136/climate_text_dataset"
pdf_files = sorted(
    [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]
)

# Shuffle and split into 90% training and 10% testing
random.shuffle(pdf_files)
split_index = int(len(pdf_files) * 0.9)
train_sets, test_sets = pdf_files[:split_index], pdf_files[split_index:]

def extract_text_to_temp_files(pdf_paths):
    """Extracts text from PDFs and writes them to temporary text files."""
    temp_files = []
    for pdf in pdf_paths:
        text = extract_text_from_pdf(pdf)
        if text:  # Only save non-empty text files
            temp_file = pdf + ".txt"
            with open(temp_file, "w") as f:
                f.write(text)
            temp_files.append(temp_file)
            print(f"Saved: {temp_file}")
        
        # Remove large variables from memory
        del text
    return temp_files

train_txt_files = extract_text_to_temp_files(train_sets)
test_txt_files = extract_text_to_temp_files(test_sets)

# Save the list of generated text file paths
with open("/scratch/el3136/BDML-1/train_txt_files.txt", "w") as f:
    f.write("\n".join(train_txt_files))
with open("/scratch/el3136/BDML-1/test_txt_files.txt", "w") as f:
    f.write("\n".join(test_txt_files))
