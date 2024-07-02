# Import AutoModelForConditionalGeneration from transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the pre-trained google/flan-t5-base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Specify source and target languages for translation
source_lang = "English"
target_lang = "German"
input_text = f"translate {source_lang} to {target_lang}: How old are you?"

# Use the tokenizer to preprocess input text, with return_tensors="pt" to return PyTorch tensors
inputs = tokenizer(input_text, return_tensors="pt")

# Print the key and values of the preprocessed inputs dictionary
for key, value in inputs.items():
    print(key, value)
