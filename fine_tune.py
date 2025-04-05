from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the pad_token to the eos_token
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
dataset = load_dataset("text", data_files={"train": "fixed-cowdery.txt"})

def tokenize_function(examples):
    # Tokenize the text
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    # Set labels as a shifted version of input_ids (for causal LM training)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns (e.g., "text") that aren't needed for training
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./cowdery_model",
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_device_train_batch_size=4,
    learning_rate=3e-5,  # Slightly lower learning rate
    weight_decay=0.01,   # Add some regularization
    save_steps=10_000,
    save_total_limit=2,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./cowdery_model")
tokenizer.save_pretrained("./cowdery_model")