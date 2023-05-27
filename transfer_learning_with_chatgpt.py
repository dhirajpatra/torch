import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model_name = "gpt2"  # or "gpt2-medium", "gpt2-large" for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load and preprocess your custom dataset
dataset_path = "path/to/your/dataset.txt"


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


dataset = load_dataset(dataset_path)
tokenized_dataset = tokenizer.encode(dataset)

# Convert the tokenized dataset into a format suitable for training
train_dataset = TextDataset(tokenized_dataset, tokenizer=tokenizer, block_size=128)

# Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
output_dir = "./fine-tuned-model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
