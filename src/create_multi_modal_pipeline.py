import os

import huggingface_hub as hf_hub
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import BlipForConditionalGeneration, BlipProcessor

load_dotenv()


# Download only the training split of the dataset
dataset = load_dataset("adirik/fashion_image_caption-100", split="train")

# Import BlipProcessor and BlipForConditionalGeneration from transformers

# Load preprocessor of "Salesforce/blip-image-captioning-base"
preprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load pre-trained "Salesforce/blip-image-captioning-base" model
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Preprocess the first image of the dataset
inputs = preprocessor(dataset[0]["image"], return_tensors="pt")

# With torch.no_grad()
with torch.no_grad():
    # Predict caption using the generate method of the model
    outputs = model.generate(**inputs)

# Decode model output to text
caption = preprocessor.decode(outputs[0], skip_special_tokens=True)

# Print decoded caption
print(caption)


# Create a replace_caption function that takes a data dictionary as input
def replace_caption(data):
    # Preprocess the image value of the data dictionary
    inputs = preprocessor(data["image"], return_tensors="pt")

    # Predict the caption with torch.no_grad and the generate method
    with torch.no_grad():
        output = model.generate(**inputs)

    # Decode model output to text
    caption = preprocessor.decode(output[0], skip_special_tokens=True)

    # Set caption as the new text value of the data dictionary
    data["text"] = caption
    return data


# Use the map function to replace the captions of whole dataset
new_dataset = dataset.map(replace_caption)

# Assign your HUGGINGFACE_TOKEN to a variable named hf_token
hf_token = os.environ["HUGGINGFACE_TOKEN"]

# Login to the HF Hub using your hf_token
hf_hub.login(hf_token)

# Push the new / improved dataset to the hub
your_username = "datttrian"
new_dataset.push_to_hub(f"{your_username}/fashion_image_caption-100")
