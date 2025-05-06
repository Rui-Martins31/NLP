import pandas as pd
import nltk
from nltk.corpus import stopwords
import gensim
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

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

    # REMOVE SYMBOLS AND NUMBERS
    nlp = gensim.models.KeyedVectors.load_word2vec_format("models/cbow_s1000.txt", binary=False)  # Load word2vec model
    nltk.download("stopwords")
    stopwords_pt = set(stopwords.words("portuguese"))
    
    def remove_symbols_and_numbers(text):
        text = re.sub(r"[^a-zA-Zá-úÁ-Ú]", " ", text)  # allow Portuguese characters
        return text.strip()

    # Apply basic preprocessing
    tqdm.pandas()
    datasetFiltered["review_text"] = datasetFiltered["review_text"].progress_apply(remove_symbols_and_numbers)
    
    datasetFiltered["recommend_to_a_friend"] = datasetFiltered["recommend_to_a_friend"].progress_apply(lambda x: 1 if x == "Yes" else 0)  # Convert to binary

    return datasetFiltered

def Word2ec_embed_review_weighted(review, nlp_model, tfidf_vectorizer, vocab):
    words = [word for word in review.split() if word in nlp_model]
    if not words:
        return np.zeros(nlp_model.vectors.shape[1])
    
    tfidf_scores = tfidf_vectorizer.transform([review]).toarray()[0]
    weighted_embedding = np.zeros(nlp_model.vectors.shape[1])
    total_weight = 0
    
    for word in words:
        if word in vocab:
            idx = tfidf_vectorizer.vocabulary_[word]
            weight = tfidf_scores[idx]
            weighted_embedding += nlp_model[word] * weight
            total_weight += weight

    return weighted_embedding / total_weight if total_weight > 0 else weighted_embedding

def embed_data(df):
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

    # Load word2vec model
    nlp = gensim.models.KeyedVectors.load_word2vec_format("models/cbow_s1000.txt", binary=False)

    tfidf = TfidfVectorizer(max_df=0.8, min_df=5)
    tfidf.fit(X_train)  # Fit TF-IDF on training data only
    vocab = tfidf.vocabulary_

    # Embed train set
    X_train_emb = []
    for doc in tqdm(X_train, desc="Embedding train set"):
        X_train_emb.append(Word2ec_embed_review_weighted(doc, nlp, tfidf, vocab))
    X_train_emb = np.array(X_train_emb)

    # Embed test set
    X_test_emb = []
    for doc in tqdm(X_test, desc="Embedding test set"):
        X_test_emb.append(Word2ec_embed_review_weighted(doc, nlp, tfidf, vocab))
    X_test_emb = np.array(X_test_emb)

    # Debug: Check embedding stats
    print("X_train_emb shape:", X_train_emb.shape)
    print("X_train_emb sample:", X_train_emb[:3])
    print("X_train_emb mean:", np.mean(X_train_emb))
    print("X_train_emb std:", np.std(X_train_emb))

    return X_train_emb, X_test_emb, y_train, y_test

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