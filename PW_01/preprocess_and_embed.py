"""
preprocess_and_embed.py

Script to:
1) Load dataset
2) Preprocess text
3) Create embeddings
4) Save embeddings and labels to .npy files
"""

import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# If using enelvo or other libraries, import them here
#import enelvo
#import enelvo.normaliser

# ---------------------------------------------------------
# A) Load and Preprocess
# ---------------------------------------------------------
def load_and_preprocess_data(filename="DATASET_B2W_REVIEWS.xlsx", sheetname="B2W-Reviews01"):
    # IMPORT DATASET
    df = pd.read_excel(io=filename, sheet_name=sheetname)

    # FILTER THE ORIGINAL DATASET
    columnsToDelete = ['submission_date',
                    'reviewer_id',
                    'product_brand',
                    'site_category_lv1',
                    'site_category_lv2',
                    'reviewer_birth_year',
                    'reviewer_gender', 
                    'reviewer_state']
    datasetFiltered = df.drop(columnsToDelete, inplace=False, axis=1).copy()

    # DEBUG
    print("Dataset shape:", df.shape)
    print("Dataset:\n", df.head())
    print("Dataset Filtered:\n", datasetFiltered.head())
    
    # BASIC CLEANUP
    datasetFiltered["review_text"] = datasetFiltered["review_text"].fillna("").astype(str).str.lower().str.strip()
    datasetFiltered["recommend_to_a_friend"] = datasetFiltered["recommend_to_a_friend"].fillna(0)

    # REMOVE STOPWORDS, SYMBOLS, ETC
    nltk.download("stopwords")
    stopwords_pt = set(stopwords.words("portuguese"))
    
    def remove_symbols_and_numbers(text):
        text = re.sub(r"[^a-zA-Zá-úÁ-Ú]", " ", text)  # allow Portuguese characters
        return text.strip()

    def remove_stopwords(text):
        words = text.split()
        filtered = [w for w in words if w not in stopwords_pt]
        return " ".join(filtered)

    tqdm.pandas()
    datasetFiltered["review_text"] = datasetFiltered["review_text"].progress_apply(remove_symbols_and_numbers)
    datasetFiltered["review_text"] = datasetFiltered["review_text"].progress_apply(remove_stopwords)
    
    # Convert "Yes"/"No" or "Liked"/"Not Liked" to numeric if needed
    datasetFiltered["recommend_to_a_friend"] = datasetFiltered["recommend_to_a_friend"].progress_apply(lambda x: 1 if x == "Yes" else 0)
    # If Liked is already 0/1, no conversion needed.

    return datasetFiltered

# ---------------------------------------------------------
# B) Embedding
# ---------------------------------------------------------
def spacy_embed_review(review, nlp_model):
    """
    Given a single review (string) and a spaCy model,
    return the document embedding vector (doc.vector).
    """
    doc = nlp_model(review)
    return doc.vector

def embed_data(df):
    """
    Takes a preprocessed DataFrame with 'review_text' and 'recommend_to_a_friend'
    Splits into train/test, embeds the reviews, returns (X_train_emb, X_test_emb, y_train, y_test).
    """
    # Train/test split
    X = df["review_text"]
    y = df["recommend_to_a_friend"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    # Load spaCy model (Portuguese or English as needed)
    # If Portuguese: python -m spacy download pt_core_news_lg
    # Then nlp = spacy.load("pt_core_news_lg")
    nlp = spacy.load("pt_core_news_lg")  # or "en_core_web_lg" for English

    # Embed train set
    X_train_emb = []
    for doc in tqdm(X_train, desc="Embedding train set"):
        X_train_emb.append(spacy_embed_review(doc, nlp))
    X_train_emb = np.array(X_train_emb)

    # Embed test set
    X_test_emb = []
    for doc in tqdm(X_test, desc="Embedding test set"):
        X_test_emb.append(spacy_embed_review(doc, nlp))
    X_test_emb = np.array(X_test_emb)

    return X_train_emb, X_test_emb, y_train, y_test

# ---------------------------------------------------------
# C) Main
# ---------------------------------------------------------
def main():
    # Load and preprocess
    df = load_and_preprocess_data("DATASET_B2W_REVIEWS.xlsx", "B2W-Reviews01")
    
    # Embed
    X_train_emb, X_test_emb, y_train, y_test = embed_data(df)

    # Save to .npy files for later use
    np.save("X_train_emb.npy", X_train_emb)
    np.save("X_test_emb.npy", X_test_emb)
    np.save("y_train.npy", y_train.to_numpy())
    np.save("y_test.npy", y_test.to_numpy())

    print("Preprocessing and embedding completed. Arrays saved to disk.")

if __name__ == "__main__":
    main()
