import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

# Use torch.no_grad to prevent gradient accumulation - speeds up inference, reduces memory
with torch.no_grad():
    # Use model.generate method to perform inference
    # Use **inputs to pass the unpacked dictionary as multiple arguments to the generate method
    outputs = model.generate(**inputs)

# The output is a list of length batch_size - number of input texts - which is 1 in this case
# Print the length of outputs list
print(len(outputs))

# Use the decode method of the tokenizer to convert the output to a human readable format
print(tokenizer.decode(outputs[0]))
