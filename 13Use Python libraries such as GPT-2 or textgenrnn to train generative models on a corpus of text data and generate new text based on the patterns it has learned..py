#pip -q install transformers datasets accelerate torch
#pip install -U accelerate
import urllib.request
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments
)

print("Loading dataset...")

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

response = urllib.request.urlopen(url)
raw_text = response.read().decode("utf-8")

lines = [
    line.strip()
    for line in raw_text.split("\n")
    if line.strip()
]

dataset = Dataset.from_dict(
    {"text": lines}
)

train_data = dataset.select(
    range(min(1000, len(dataset)))
)

print("Loading tokenizer and model...")

tokenizer = GPT2Tokenizer.from_pretrained(
    "distilgpt2"
)

tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(
    "distilgpt2"
)

def tokenize_function(examples):

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    tokenized["labels"] = (
        tokenized["input_ids"].copy()
    )

    return tokenized

print("Tokenizing dataset...")

tokenized_train = train_data.map(
    tokenize_function,
    batched=True
)

tokenized_train = tokenized_train.remove_columns(
    ["text"]
)

tokenized_train.set_format("torch")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    save_strategy="no",
    logging_steps=20,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train
)

print("Starting training...")

trainer.train()

model.save_pretrained(
    "./fine_tuned_distilgpt2"
)

tokenizer.save_pretrained(
    "./fine_tuned_distilgpt2"
)

print("Model Saved")

prompt = "To be or not to be"

inputs = tokenizer.encode(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    inputs,
    max_length=60,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nGenerated Text:\n")
print(generated_text)
