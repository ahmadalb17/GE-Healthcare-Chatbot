from transformers import AutoModel, AutoTokenizer
import os

model_name = "sentence-transformers/all-MiniLM-L6-v2"
target_directory = "C:/Users/alkha/Desktop/Bacheloroppgave/models"

# Check if target directory exists
os.makedirs(target_directory, exist_ok=True)

# Download the model
model = AutoModel.from_pretrained(model_name, cache_dir=target_directory)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=target_directory)
