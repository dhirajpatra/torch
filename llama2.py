import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("facebookresearch/Llama2")
tokenizer = AutoTokenizer.from_pretrained("facebookresearch/Llama2")

# Get the input text
text = "I am a large language model."

# Tokenize the text
inputs = tokenizer(text=text, return_tensors="pt")

# Get the predictions
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# Decode the predictions
decoded_predictions = tokenizer.decode(predictions, skip_special_tokens=True)

# Print the predictions
print(decoded_predictions)
