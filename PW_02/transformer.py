import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from tqdm import tqdm
import torch
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

### ---------------------------
### LOAD PREPROCESSED DATASET
### ---------------------------

# Load preprocessed dataset
df = pd.read_csv("preprocessed_b2w.csv")

# Ensure correct data types
df['review_text'] = df['review_text'].fillna("").astype(str)
df['recommend_to_a_friend'] = df['recommend_to_a_friend'].astype(int)

# Debug
print("Dataset shape:", df.shape)
print("Sample data:\n", df.head())
print("Label distribution:\n", df['recommend_to_a_friend'].value_counts())

# Split dataset
X = df['review_text']
y = df['recommend_to_a_friend']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to Hugging Face Dataset
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Debug
print("\nTrain dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))
print("Train label distribution:", train_df['label'].value_counts())
print("Test label distribution:", test_df['label'].value_counts())

### ---------------------------
### TOKENIZATION
### ---------------------------

# Load tokenizer
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

### ---------------------------
### MODEL SETUP
### ---------------------------

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Apply LoRA for parameter-efficient fine-tuning
use_lora = True
if use_lora:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

### ---------------------------
### TRAINING SETUP
### ---------------------------

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bertimbau-finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"
)

# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'macro_f1': report['macro avg']['f1-score']
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

### ---------------------------
### FINE-TUNING
### ---------------------------

print("\nFine-tuning Bertimbau...\n")
trainer.train()

# Save the model
trainer.save_model("./bertimbau-finetuned-final")
tokenizer.save_pretrained("./bertimbau-finetuned-final")

### ---------------------------
### EVALUATION
### ---------------------------

print("\nEvaluating the model...\n")
eval_results = trainer.evaluate()

# Print evaluation metrics
print("\n---BERTIMBAU EVALUATION---")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

# Get predictions for detailed analysis
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(-1)
y_true = predictions.label_ids

# Classification report
print("\nClassification Report (Bertimbau):\n", classification_report(y_true, y_pred, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Bertimbau)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("cm_bertimbau.png")
plt.show()

### ---------------------------
### ERROR ANALYSIS
### ---------------------------

# Identify misclassified examples
misclassified = test_df.iloc[np.where(y_pred != y_true)].copy()
misclassified['predicted_label'] = y_pred[np.where(y_pred != y_true)]

print("\nSample of Misclassified Reviews:")
for idx, row in misclassified.head(5).iterrows():
    print(f"\nReview: {row['text']}")
    print(f"True Label: {row['label']}")
    print(f"Predicted Label: {row['predicted_label']}")

### ---------------------------
### COMPARISON WITH ASSIGNMENT 1
### ---------------------------

# Load Assignment 1 results
assignment1_results = pd.read_csv("assignment1_results.csv").to_dict(orient='index')
assignment1_results = {k: v for k, v in assignment1_results.items()}

print("\n---COMPARISON WITH ASSIGNMENT 1---")
print("Bertimbau Results:")
for key, value in eval_results.items():
    if key.startswith('eval_'):
        print(f"{key.replace('eval_', '')}: {value:.4f}")

print("\nAssignment 1 Results:")
for model_name, metrics in assignment1_results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        if metric != 'Unnamed: 0':
            print(f"{metric}: {value:.4f}")