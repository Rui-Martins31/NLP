# ========================
# Load Data and Libraries
# ========================
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import torch

# Load your labeled dataset
X_train_strings = pd.read_csv('./X_train_string.csv')
X_test_strings = pd.read_csv('./X_test_string.csv')
y_test = pd.read_csv('./y_test.csv')
y_train = pd.read_csv('./y_train.csv')
print(X_train_strings.head())
print(X_test_strings.head())

#-------------------------------------------------------------------------------------------------------

NUM_TO_TRAIN = 25000

# ========================
# (1) DOMAIN ADAPTATION
# ========================
# Save domain corpus file (one review per line)
X_train_strings = pd.read_csv('./X_train_string.csv')

# Load text-only dataset
dataset = load_dataset("text", data_files={"train": "X_train_string.txt"})

# Load model and tokenizer for masked language modeling
base_model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(base_model_name)

# Tokenization function for domain adaptation
def tokenize_mlm(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_mlm, batched=True, remove_columns=["text"])

# Data collator with masking
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# TrainingArguments for domain adaptation
mlm_training_args = TrainingArguments(
    output_dir="./adapted_model",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    save_steps=5000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir="./logs_mlm"
)

# Trainer for domain adaptation
mlm_trainer = Trainer(
    model=mlm_model,
    args=mlm_training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# Run domain adaptation
mlm_trainer.train()

# Save adapted model
mlm_trainer.save_model("./adapted_model")
tokenizer.save_pretrained("./adapted_model")

# ========================
# (2) FINE-TUNING FOR CLASSIFICATION
# ========================
# Use the adapted model for classification
model = AutoModelForSequenceClassification.from_pretrained("./adapted_model", num_labels=3)  # Adjust num_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize data
def tokenize_data(texts):
    return tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")

# Ensure inputs are properly formatted
train_texts = X_train_strings.iloc[:NUM_TO_TRAIN, 0].fillna("")
val_texts = X_test_strings.iloc[:NUM_TO_TRAIN, 0].fillna("")

# Handle labels carefully
if isinstance(y_train, (pd.Series, pd.DataFrame)):
    train_labels = y_train.iloc[:NUM_TO_TRAIN].values.tolist()  # Convert to list
elif isinstance(y_train, list):
    train_labels = y_train[:NUM_TO_TRAIN]
else:
    raise ValueError("y_train must be a list, pandas Series, or DataFrame")

if isinstance(y_test, (pd.Series, pd.DataFrame)):
    val_labels = y_test.iloc[:NUM_TO_TRAIN].values.tolist()
elif isinstance(y_test, list):
    val_labels = y_test[:NUM_TO_TRAIN]
else:
    raise ValueError("y_test must be a list, pandas Series, or DataFrame")

# Verify label lengths
if len(train_labels) != NUM_TO_TRAIN:
    raise ValueError(f"Expected 100 train labels, got {len(train_labels)}")
if len(val_labels) != NUM_TO_TRAIN:
    raise ValueError(f"Expected 100 validation labels, got {len(val_labels)}")

train_encodings = tokenize_data(train_texts)
val_encodings = tokenize_data(val_texts)

# Convert encodings to dataset format for Trainer
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'].squeeze(),
    'attention_mask': train_encodings['attention_mask'].squeeze(),
    'labels': train_labels
})
val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'].squeeze(),
    'attention_mask': val_encodings['attention_mask'].squeeze(),
    'labels': val_labels
})

# Define training arguments
training_args = TrainingArguments(
    output_dir='./fine_tuned_sentiment_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine_tuned_sentiment_model')
tokenizer.save_pretrained('./fine_tuned_sentiment_model')