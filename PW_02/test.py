# TEST --------------------------------------------------------------------------
# Import libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
import pandas as pd


# Get files
X_train_strings = pd.read_csv('./X_train_string.csv')
X_test_strings = pd.read_csv('./X_test_string.csv')
y_test = pd.read_csv('./y_test.csv')
y_train = pd.read_csv('./y_train.csv')


# Predict with fine-tuned model
NUM_TO_TEST = len(X_train_strings)#50000
BATCH_SIZE = 32  # Process texts in smaller batches to avoid memory issues

model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_sentiment_model')
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_sentiment_model')
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

def predict_sentiment(texts, batch_size=BATCH_SIZE):
    predictions = []
    # Ensure texts are strings and handle invalid entries
    texts = [str(text) if pd.notnull(text) else "" for text in texts]  # Convert to string, replace NaN/None with ""

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_predictions = torch.argmax(probabilities, dim=-1).tolist()  # Returns 0 or 1
            predictions.extend(batch_predictions)
        except Exception as e:
            print(f"Error in batch {i // batch_size}: {e}")
            print(f"Problematic batch: {batch_texts}")
            predictions.extend([0] * len(batch_texts))  # Fallback: assign 0 for failed batch
    return predictions

# Test predictions
texts = X_train_strings.iloc[:NUM_TO_TEST, 0].tolist()

# Debug: Check for invalid entries
invalid_texts = [t for t in texts if not isinstance(t, (str, float, int)) or pd.isna(t)]
if invalid_texts:
    print(f"Found {len(invalid_texts)} invalid texts: {invalid_texts[:5]}")
    print("Cleaning texts...")

predictions = predict_sentiment(texts)
for text, pred in zip(texts[:5], predictions[:5]):  # Print first 5 for brevity
    print(f"Text: {text}\nSentiment: {pred}\n")

# Save predictions
df_results = pd.DataFrame({
    'text': texts,
    'mapped_label': predictions
})
df_results.to_csv('./classification_results_mapped.csv', index=False)

# Validation
total = min(NUM_TO_TEST, len(predictions))  # Ensure we don't exceed available data
correct = 0

# Handle y_train based on its type
if isinstance(y_train, pd.DataFrame):
    y_train_values = y_train.iloc[:total, 0].values
elif isinstance(y_train, pd.Series):
    y_train_values = y_train.iloc[:total].values
else:
    y_train_values = y_train[:total]

# Debug: Check y_train
print(f"y_train type: {type(y_train)}, length: {len(y_train_values)}")
print(f"y_train sample: {y_train_values[:5]}")

for i in range(total):
    if predictions[i] == y_train_values[i]:
        correct += 1

accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy:.4f}")