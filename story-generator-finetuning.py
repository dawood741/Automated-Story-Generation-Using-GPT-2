import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# Set up the model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare the dataset
def load_and_split_dataset(file_path, test_size=0.1):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().splitlines()
    
    train_texts, val_texts = train_test_split(data, test_size=test_size, random_state=42)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)
    
    return train_dataset, val_dataset

# Load and split the dataset
train_dataset, val_dataset = load_and_split_dataset("path/to/your/train_stories.txt")

# Set up the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
    evaluation_strategy="steps",
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_story_generator")
tokenizer.save_pretrained("./fine_tuned_story_generator")

# Function to generate a story
def generate_story(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

# Example usage
prompt = "Once upon a time in a magical forest"
generated_story = generate_story(prompt)
print(generated_story)
