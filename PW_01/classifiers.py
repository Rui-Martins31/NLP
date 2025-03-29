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
def load_files():
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
# Main function: Train and evaluate classifiers
# -------------------------------
def main():
    X_train_emb, X_test_emb, y_train, y_test = load_files()

    # -------------------------------
    # Example 1: Logistic Regression
    # -------------------------------
    lr_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
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
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_clf.fit(X_train_emb, y_train)
    y_pred_rf = rf_clf.predict(X_test_emb)
    print("\n--- RANDOM FOREST ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

    # -------------------------------
    # Example 5: XGBoost (using TF-IDF features)
    # -------------------------------
    xgb_clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_clf.fit(X_train_emb, y_train)
    y_pred_xgb = xgb_clf.predict(X_test_emb)
    print("\n--- XGBOOST ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))


if __name__ == "__main__":
    main()
