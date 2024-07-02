# Import the AutoTokenizer and AutoModel classes from transformers
from transformers import AutoTokenizer, AutoModel

# Load the pre-trained tokenizer of the "cardiffnlp/twitter-roberta-base-emoji" model
tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-emoji")

# Print the tokenizer to see the data preprocessing configuration for this model
print(tokenizer)

# Load the pre-trained "cardiffnlp/twitter-roberta-base-emoji" model
model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-emoji")

# Get the model configuration
config = model.config

# Print model config's architectures attribute to see the model class name
print(model.config.architectures)
