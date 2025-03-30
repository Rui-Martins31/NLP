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
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter

#import enelvo
#import enelvo.normaliser

# ---------------------------------------------------------
# Analyze Data
# ---------------------------------------------------------
def analyze_data(df):
    """
    Analyzes the given DataFrame by plotting and printing:
    1. Distribution of 'recommend_to_a_friend'
    2. Star rating distribution from 'overall_rating'
    3. Distribution of review text lengths
    4. Frequent words distribution (top 5)
    """
    # ---------------------------
    # Recommend to a Friend Distribution
    # ---------------------------
    recommend_counts = df['recommend_to_a_friend'].value_counts()
    recommend_counts.plot(kind='bar')
    plt.title("Distribution of 'Recommend to a Friend'")
    plt.xlabel("Label (Yes=1, No=0)")
    plt.ylabel("Count of Reviews")
    plt.show()
    print("\nLabel distribution:\n", recommend_counts)
    
    # ---------------------------
    # Star Rating Distribution
    # ---------------------------
    rating_counts = df['overall_rating'].value_counts().sort_index()
    rating_counts.plot(kind='bar')
    plt.title("Star Rating Distribution")
    plt.xlabel("Stars")
    plt.ylabel("Count of Reviews")
    plt.show()
    print("\nRating distribution:\n", rating_counts)
    
    # ---------------------------
    # Review Text Length Distribution
    # ---------------------------
    df['text_length'] = df['review_text'].apply(lambda x: len(x.split()))
    df['text_length'].hist(bins=50)
    plt.title("Distribution of Review Text Length (words)")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Reviews")
    plt.show()
    print("\nText length stats:\n", df['text_length'].describe())
    
    # ---------------------------
    # Frequent Words Distribution
    # ---------------------------
    all_words = ' '.join(df['review_text']).split()
    word_freq = Counter(all_words)
    most_common = word_freq.most_common(5)
    words = [word for word, count in most_common]
    counts = [count for word, count in most_common]
    
    plt.plot(words, counts, marker='o')
    plt.title("Frequent Words Distribution (Top 5)")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.show()
    # Optionally, print the most common words:
    print("\nMost common words:\n", most_common)


# ---------------------------------------------------------
# Load and Preprocess
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
    datasetFiltered["review_title"] = datasetFiltered["review_title"].fillna("").astype(str).str.lower().str.strip()
    datasetFiltered["review_text"] = datasetFiltered["review_text"].fillna("").astype(str).str.lower().str.strip()

    def infer_recommendation(row, threshold=4):
        if pd.isna(row["recommend_to_a_friend"]) or row["recommend_to_a_friend"] == "":
            try:
                rating = float(row["overall_rating"])
            except ValueError:
                return 0
            return 1 if rating >= threshold else 0
        else:
            # convert to binary
            return 1 if row["recommend_to_a_friend"].strip().lower() == "yes" else 0

    # FILL EMPTY FIELDS
    datasetFiltered["recommend_to_a_friend"] = datasetFiltered.apply(infer_recommendation, axis=1)
    #datasetFiltered["recommend_to_a_friend"] = datasetFiltered["recommend_to_a_friend"].fillna(0)
    #datasetFiltered["recommend_to_a_friend"] = datasetFiltered["recommend_to_a_friend"].apply(lambda x: 1 if x == "Yes" else 0) # Conver to binary

    # Combine review title and review text 
    datasetFiltered["review_text"] = (
        datasetFiltered["review_title"] + " " +
        datasetFiltered["review_text"]
    )

    # REMOVE STOPWORDS, SYMBOLS, ETC
    nlp = spacy.load("pt_core_news_lg") # Load Spacy model
    nltk.download("stopwords")
    stopwords_pt = set(stopwords.words("portuguese"))
    
    def remove_symbols_and_numbers(text):
        text = re.sub(r"[^a-zA-Zá-úÁ-Ú]", " ", text)  # allow Portuguese characters
        return text.strip()

    def remove_stopwords(text):
        words = text.split()
        filtered = [w for w in words if w not in stopwords_pt]
        return " ".join(filtered)
    
    def apply_lemmatization(text):
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    tqdm.pandas()
    datasetFiltered["review_text"] = datasetFiltered["review_text"].progress_apply(remove_symbols_and_numbers)
    datasetFiltered["review_text"] = datasetFiltered["review_text"].progress_apply(remove_stopwords)   # The removal of stopwords can influence negatively embeddings performance
    datasetFiltered["review_text"] = datasetFiltered["review_text"].progress_apply(apply_lemmatization)

    return datasetFiltered

# ---------------------------------------------------------
# Embedding
# ---------------------------------------------------------
def spacy_embed_review(review, nlp_model):
    """
    Given a single review (string) and a spaCy model,
    return the document embedding vector (doc.vector).
    """
    doc = nlp_model(review)
    return doc.vector

def spacy_embed_review_weighted(review, nlp_model, tfidf_vectorizer, vocab):
    doc = nlp_model(review)
    
    words = [token.text for token in doc if token.text in vocab]
    if not words:
        return np.zeros(nlp_model.vocab.vectors.shape[1])  # Return zero vector if no valid words
    
    tfidf_scores = tfidf_vectorizer.transform([review]).toarray()[0]
    weighted_embedding = np.zeros(nlp_model.vocab.vectors.shape[1])

    total_weight = 0
    for token in doc:
        if token.text in vocab:
            idx = tfidf_vectorizer.vocabulary_[token.text]
            weight = tfidf_scores[idx]
            weighted_embedding += token.vector * weight
            total_weight += weight

    return weighted_embedding / total_weight if total_weight > 0 else weighted_embedding

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

    # Load spaCy model
    # Portuguese: python -m spacy download pt_core_news_lg
    # Then nlp = spacy.load("pt_core_news_lg")
    nlp = spacy.load("pt_core_news_lg")

    tfidf = TfidfVectorizer(max_df=0.8, min_df=5)
    tfidf.fit(X_train)  # Fit TF-IDF on training data only
    vocab = tfidf.vocabulary_

    # Embed train set
    X_train_emb = []
    for doc in tqdm(X_train, desc="Embedding train set"):
        #X_train_emb.append(spacy_embed_review(doc, nlp))
        X_train_emb.append(spacy_embed_review_weighted(doc, nlp, tfidf, vocab))
    X_train_emb = np.array(X_train_emb)

    # Embed test set
    X_test_emb = []
    for doc in tqdm(X_test, desc="Embedding test set"):
        #X_test_emb.append(spacy_embed_review(doc, nlp))
        X_test_emb.append(spacy_embed_review_weighted(doc, nlp, tfidf, vocab))
    X_test_emb = np.array(X_test_emb)

    # Debug: Check embedding stats
    print("X_train_emb shape:", X_train_emb.shape)
    print("X_train_emb sample:", X_train_emb[:3])
    print("X_train_emb mean:", np.mean(X_train_emb))
    print("X_train_emb std:", np.std(X_train_emb))

    return X_train_emb, X_test_emb, y_train, y_test


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------
def load_tfidf_features(df):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

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

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    # Load and preprocess
    df = load_and_preprocess_data("DATASET_B2W_REVIEWS.xlsx", "B2W-Reviews01")
    analyze_data(df)
    
    # Embed
    X_train_emb, X_test_emb, y_train, y_test = embed_data(df)
    #X_train_emb, X_test_emb, y_train, y_test = load_tfidf_features(df)

    # Save to .npy files for later use
    np.save("X_train_emb.npy", X_train_emb)
    np.save("X_test_emb.npy", X_test_emb)
    np.save("y_train.npy", y_train.to_numpy())
    np.save("y_test.npy", y_test.to_numpy())

    print("Preprocessing and embedding completed. Arrays saved to disk.")

if __name__ == "__main__":
    main()
