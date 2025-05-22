# Get files
import pandas as pd

X_train_strings = pd.read_csv('./X_train_string.csv')
X_test_strings = pd.read_csv('./X_test_string.csv')
y_test = pd.read_csv('./y_test.csv')
y_train = pd.read_csv('./y_train.csv')
print(X_train_strings.head())
print(X_test_strings.head())


#-------------------------------------------------------------------------------------------------------

NUM_TO_TRAIN = 25000

# Import libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

# Load tokenizer and model
model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
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