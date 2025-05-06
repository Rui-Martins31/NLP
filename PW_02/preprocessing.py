import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# Set random seed for reproducibility
np.random.seed(42)

### ---------------------------
### INITIALIZATION
### ---------------------------

# Load dataset
df = pd.read_excel("DATASET_B2W_REVIEWS.xlsx", sheet_name="B2W-Reviews01")

# Filter columns
columns_to_delete = [
    'submission_date', 'reviewer_id', 'product_brand', 'site_category_lv1',
    'site_category_lv2', 'reviewer_birth_year', 'reviewer_gender', 'reviewer_state'
]
dataset_filtered = df.drop(columns_to_delete, axis=1).copy()

# Debug
print("Dataset shape:", df.shape)
print("Filtered Dataset:\n", dataset_filtered.head())

### ---------------------------
### CLEAN UP
### ---------------------------

# Download NLTK stopwords
nltk.download('stopwords')
stopwords_pt = set(stopwords.words('portuguese'))

# Initialize stemmer
stemmer = SnowballStemmer("portuguese")

# Load spaCy model for lemmatization
nlp = spacy.load("pt_core_news_lg")

# Copy dataset for preprocessing
dataset_normalized = dataset_filtered.copy()

# Handle missing values and normalize text
dataset_normalized['review_text'] = dataset_normalized['review_text'].fillna("").astype(str)
dataset_normalized['review_text'] = dataset_normalized['review_text'].str.lower().str.strip()
dataset_normalized['recommend_to_a_friend'] = dataset_normalized['recommend_to_a_friend'].fillna("No")

# Preprocessing functions
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords_pt]
    return " ".join(filtered_words)

def remove_symbols_and_numbers(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text.strip()

def apply_lemmatization(text):
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return " ".join(lemmatized_words)

# Apply preprocessing with progress tracking
tqdm.pandas()
dataset_normalized['review_text'] = dataset_normalized['review_text'].progress_apply(remove_symbols_and_numbers)
dataset_normalized['review_text'] = dataset_normalized['review_text'].progress_apply(remove_stopwords)
# Optional: dataset_normalized['review_text'] = dataset_normalized['review_text'].progress_apply(apply_lemmatization)

# Debug
print("\nFiltered review text:\n", dataset_filtered['review_text'].head())
print("\nNormalized review text:\n", dataset_normalized['review_text'].head())

### ---------------------------
### DATA ANALYSIS
### ---------------------------

# Recommend to a Friend Distribution
recommend_counts = dataset_normalized['recommend_to_a_friend'].value_counts()
plt.figure(figsize=(8, 6))
recommend_counts.plot(kind='bar')
plt.title("Distribution of 'Recommend to a Friend'")
plt.xlabel("Label (Yes=1, No=0)")
plt.ylabel("Count of Reviews")
plt.savefig("recommend_distribution.png")
plt.show()
print("\nLabel distribution:\n", recommend_counts)

# Star Rating Distribution
rating_counts = dataset_normalized['overall_rating'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
rating_counts.plot(kind='bar')
plt.title("Star Rating Distribution")
plt.xlabel("Stars")
plt.ylabel("Count of Reviews")
plt.savefig("rating_distribution.png")
plt.show()
print("\nRating distribution:\n", rating_counts)

# Review Text Length Distribution
dataset_normalized['text_length'] = dataset_normalized['review_text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 6))
dataset_normalized['text_length'].hist(bins=50)
plt.title("Distribution of Review Text Length (words)")
plt.xlabel("Word Count")
plt.ylabel("Number of Reviews")
plt.savefig("text_length_distribution.png")
plt.show()
print("\nText length stats:\n", dataset_normalized['text_length'].describe())

# Frequent Words Distribution
all_words = ' '.join(dataset_normalized['review_text']).split()
word_freq = Counter(all_words)
top_words = word_freq.most_common(5)
x, y = zip(*top_words)
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title("Frequent Words Distribution")
plt.xlabel("Words")
plt.ylabel("Count")
plt.savefig("frequent_words.png")
plt.show()
print("\nMost common tokens:", word_freq.most_common(20))

### ---------------------------
### SAVE PREPROCESSED DATA
### ---------------------------

# Save preprocessed dataset for Assignment 2
dataset_to_save = dataset_normalized[['review_text', 'recommend_to_a_friend']].copy()
dataset_to_save['recommend_to_a_friend'] = dataset_to_save['recommend_to_a_friend'].apply(lambda x: 1 if x == 'Yes' else 0)
dataset_to_save.to_csv("preprocessed_b2w.csv", index=False)
print("\nPreprocessed dataset saved to 'preprocessed_b2w.csv'")

### ---------------------------
### PREPARE DATA TO TRAIN
### ---------------------------

print("\nPreparing data to train traditional ML models...\n")
dataset_to_train = dataset_normalized[['review_text', 'recommend_to_a_friend']].copy()
dataset_to_train['recommend_to_a_friend'] = dataset_to_train['recommend_to_a_friend'].apply(lambda x: 1 if x == 'Yes' else 0)

# Debug
print("Checking for NaN in review_text:", dataset_to_train['review_text'].isna().sum())
print("Checking for NaN in recommend_to_a_friend:", dataset_to_train['recommend_to_a_friend'].isna().sum())

### ---------------------------
### TRAIN/TEST
### ---------------------------

print("\nTraining traditional ML models...\n")

X = dataset_to_train['review_text']
cephalometric_analysis = dataset_to_train['recommend_to_a_friend']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Debug
print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Vocabulary size:", len(tfidf.vocabulary_))
print("Train TF-IDF shape:", X_train_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)

# spaCy Embeddings
def spacy_embed_review(review, nlp_model):
    doc = nlp_model(review)
    return doc.vector

X_train_emb = np.array([spacy_embed_review(doc, nlp) for doc in tqdm(X_train, desc="Embedding train set")])
X_test_emb = np.array([spacy_embed_review(doc, nlp) for doc in tqdm(X_test, desc="Embedding test set")])

print("Train embedding shape:", X_train_emb.shape)
print("Test embedding shape:", X_test_emb.shape)

# Dictionary to store results
results = {}

# Naive Bayes
nb_clf = MultinomialNB()
nb_clf.fit(X_train_tfidf, y_train)  # Note: Naive Bayes requires non-negative features, so use TF-IDF
y_pred_nb = nb_clf.predict(X_test_tfidf)
results['Naive Bayes'] = {
    'accuracy': accuracy_score(y_test, y_pred_nb),
    'report': classification_report(y_test, y_pred_nb, output_dict=True, zero_division=0),
    'cm': confusion_matrix(y_test, y_pred_nb)
}

# Logistic Regression
lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train_emb, y_train)
y_pred_lr = lr_clf.predict(X_test_emb)
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'report': classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0),
    'cm': confusion_matrix(y_test, y_pred_lr)
}

# SVM - LinearSVC
svc_clf = LinearSVC(penalty='l1', dual=False, random_state=42)
svc_clf.fit(X_train_emb, y_train)
y_pred_svc = svc_clf.predict(X_test_emb)
results['SVM-SVC'] = {
    'accuracy': accuracy_score(y_test, y_pred_svc),
    'report': classification_report(y_test, y_pred_svc, output_dict=True, zero_division=0),
    'cm': confusion_matrix(y_test, y_pred_svc)
}

# SVM - SGD
sgd_clf = SGDClassifier(penalty='l2', random_state=42)
sgd_clf.fit(X_train_emb, y_train)
y_pred_sgd = sgd_clf.predict(X_test_emb)
results['SVM-SGD'] = {
    'accuracy': accuracy_score(y_test, y_pred_sgd),
    'report': classification_report(y_test, y_pred_sgd, output_dict=True, zero_division=0),
    'cm': confusion_matrix(y_test, y_pred_sgd)
}

# Print results
for model_name, metrics in results.items():
    print(f"\n---{model_name.upper()}---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, metrics['cm'].argmax(axis=1), zero_division=0))
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"cm_{model_name.lower().replace(' ', '_')}.png")
    plt.show()

# Save results for comparison in Assignment 2
results_summary = {
    model: {
        'accuracy': metrics['accuracy'],
        'precision': metrics['report']['weighted avg']['precision'],
        'recall': metrics['report']['weighted avg']['recall'],
        'f1': metrics['report']['weighted avg']['f1-score'],
        'macro_f1': metrics['report']['macro avg']['f1-score']
    } for model, metrics in results.items()
}
pd.DataFrame(results_summary).to_csv("assignment1_results.csv")
print("\nAssignment 1 results saved to 'assignment1_results.csv'")