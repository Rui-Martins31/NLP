"""
classifiers.py

Script to:
1) Load precomputed embeddings and/or preprocessed dataset
2) Train and evaluate various classifiers
3) Print performance metrics for each model
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# -------------------------------
# Helper: Load precomputed embeddings
# -------------------------------
def load_embeddings():
    X_train = np.load("X_train_emb.npy")
    X_test = np.load("X_test_emb.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    print("Loaded embeddings:")
    print("  X_train shape:", X_train.shape)
    print("  X_test shape: ", X_test.shape)
    print("  y_train shape:", y_train.shape)
    print("  y_test shape: ", y_test.shape)
    return X_train, X_test, y_train, y_test

# -------------------------------
# Optionally: Load and vectorize raw text using TF-IDF
# -------------------------------
def load_tfidf_features():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Load the preprocessed dataset (if saved as CSV, for example)
    df = pd.read_excel("DATASET_B2W_REVIEWS.xlsx", sheet_name="B2W-Reviews01")
    # The preprocessed file should have already removed unwanted columns and cleaned text.
    # If not, you can apply your preprocessing steps here.
    # In this example, we assume the text column is "review_text"
    # and the label is "recommend_to_a_friend"
    
    # (Adjust the following if your column names differ.)
    df["review_text"] = df["review_text"].fillna("").astype(str).str.lower().str.strip()
    df["recommend_to_a_friend"] = df["recommend_to_a_friend"].fillna(0)
    X = df["review_text"]
    y = df["recommend_to_a_friend"].apply(lambda x: 1 if x == "Yes" else 0)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        max_df=0.8,       # ignore terms appearing in >80% of documents
        min_df=5,         # ignore terms appearing in <5 documents
        ngram_range=(1,2) # unigrams + bigrams
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print("TF-IDF features:")
    print("  X_train_tfidf shape:", X_train_tfidf.shape)
    print("  X_test_tfidf shape: ", X_test_tfidf.shape)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test

# -------------------------------
# Main function: Train and evaluate classifiers
# -------------------------------
def main():
    # Uncomment the following line if you want to load the embedding features:
    X_train_emb, X_test_emb, y_train, y_test = load_embeddings()

    # Alternatively, if you want to work with TF-IDF features, uncomment:
    # X_train_emb, X_test_emb, y_train, y_test = load_tfidf_features()

    # -------------------------------
    # Example 1: Logistic Regression
    # -------------------------------
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(X_train_emb, y_train)
    y_pred_lr = lr_clf.predict(X_test_emb)
    print("\n--- LOGISTIC REGRESSION ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("Classification Report:\n", classification_report(y_test, y_pred_lr))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

    # -------------------------------
    # Example 2: Linear SVC (with L1 penalty)
    # -------------------------------
    svc_clf = LinearSVC(penalty='l1', dual=False, random_state=42)
    svc_clf.fit(X_train_emb, y_train)
    y_pred_svc = svc_clf.predict(X_test_emb)
    print("\n--- LINEAR SVC ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_svc))
    print("Classification Report:\n", classification_report(y_test, y_pred_svc))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))

    # -------------------------------
    # Example 3: SGD Classifier
    # -------------------------------
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train_emb, y_train)
    y_pred_sgd = sgd_clf.predict(X_test_emb)
    print("\n--- SGD CLASSIFIER ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_sgd))
    print("Classification Report:\n", classification_report(y_test, y_pred_sgd))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_sgd))

    # -------------------------------
    # Example 4: Random Forest (using TF-IDF features)
    # -------------------------------
    # If you want to run Random Forest on TF-IDF features, load them instead:
    # X_train_tfidf, X_test_tfidf, y_train, y_test = load_tfidf_features()
    # rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    # rf_clf.fit(X_train_tfidf, y_train)
    # y_pred_rf = rf_clf.predict(X_test_tfidf)
    # print("\n--- RANDOM FOREST ---")
    # print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    # print("Classification Report:\n", classification_report(y_test, y_pred_rf))
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

    # -------------------------------
    # Example 5: XGBoost (using TF-IDF features)
    # -------------------------------
    # Uncomment and modify if you want to try XGBoost on TF-IDF features:
    # X_train_tfidf, X_test_tfidf, y_train, y_test = load_tfidf_features()
    # counter = Counter(y_train)
    # scale_pos_weight = counter[0] / counter[1]
    # xgb_model = XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.3, random_state=42)
    # xgb_model.fit(X_train_tfidf, y_train)
    # y_pred_xgb = xgb_model.predict(X_test_tfidf)
    # print("\n--- XGBOOST ---")
    # print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
    # print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))


if __name__ == "__main__":
    main()
