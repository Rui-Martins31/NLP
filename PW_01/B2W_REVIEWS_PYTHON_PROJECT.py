import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### ---------------------------
### INITIALIZATION  
### ---------------------------

# IMPORT DATASET
df = pd.read_excel("DATASET_B2W_REVIEWS.xlsx", sheet_name="B2W-Reviews01")

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





### ---------------------------
### CLEAN UP  
### ---------------------------

import enelvo
import enelvo.normaliser
from tqdm import tqdm

normalizer = enelvo.normaliser.Normaliser()

import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')
stopwordsPt = set(stopwords.words('portuguese'))
#print(stopwords.words('portuguese'))

datasetNormalized = datasetFiltered.copy()
#datasetNormalized = datasetNormalized.head(1000) # TEST FOR THE FIRST X ROWS

datasetNormalized['review_text'] = datasetNormalized['review_text'].fillna("").astype(str)
datasetNormalized['review_text'] = datasetNormalized['review_text'].str.lower().str.strip()

datasetNormalized['recommend_to_a_friend'] = datasetNormalized['recommend_to_a_friend'].fillna("No")
    
# METHOD TO REMOVE PT STOPWORDS
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwordsPt]
    return " ".join(filtered_words)

# PROGRESS TRACKING 
tqdm.pandas()
datasetNormalized['review_text'] = datasetNormalized['review_text'].progress_apply(remove_stopwords)
#datasetNormalized['review_text'] = datasetNormalized['review_text'].head(10000).progress_apply(normalizer.normalise)

# DEBUG
print("Dataset filtered - review text:\n", datasetFiltered['review_text'].head())
print("\nDataset normalized - review text:\n", datasetNormalized['review_text'].head())





### ---------------------------
### DATA ANALYSIS  
### ---------------------------

# RECOMMEND TO A FRIEND DISTRIBUTION
recommend_counts = datasetNormalized['recommend_to_a_friend'].value_counts()
recommend_counts.plot(kind='bar')
plt.title("Distribution of 'Recommend to a Friend'")
plt.xlabel("Label (Yes=1, No=0)")
plt.ylabel("Count of Reviews")
plt.show()

print("\nLabel distribution:\n", recommend_counts)


# STAR RATING DISTRIBUTION
rating_counts = datasetNormalized['overall_rating'].value_counts().sort_index()
rating_counts.plot(kind='bar')
plt.title("Star Rating Distribution")
plt.xlabel("Stars")
plt.ylabel("Count of Reviews")
plt.show()

print("\nRating distribution:\n", rating_counts)


# REVIEW TEXT LENGTH DISTRIBUTION
datasetNormalized['text_length'] = datasetNormalized['review_text'].apply(lambda x: len(x.split()))
datasetNormalized['text_length'].hist(bins=50)
plt.title("Distribution of Review Text Length (words)")
plt.xlabel("Word Count")
plt.ylabel("Number of Reviews")
plt.show()

print("\nText length stats:\n", datasetNormalized['text_length'].describe())


# FREQUENT WORDS DISTRIBUTION
from collections import Counter

all_words = ' '.join(datasetNormalized['review_text']).split()
word_freq = Counter(all_words)

y = [count for tag, count in word_freq.most_common(5)]
x = [tag for tag, count in word_freq.most_common(5)]

plt.plot(x, y)
plt.title("Frequent Words Distribution")
plt.xlabel("Words")
plt.ylabel("Count")
plt.show()

#print("Most common tokens:", word_freq.most_common(20))





### ---------------------------
### PREPARE DATA TO TRAIN 
### ---------------------------

print("\n\nPreparing data to train the model.....\n")

columnsToTrain = ['recommend_to_a_friend',
                  'review_text']

datasetToTrain = datasetNormalized[columnsToTrain].copy()
datasetToTrain['recommend_to_a_friend'] = datasetToTrain['recommend_to_a_friend'].apply(lambda x: 1 if x == 'Yes' else 0) # CONVERT TO BINARY

#print(datasetToTrain['recommend_to_a_friend'].value_counts())

# DEBUG
print("Checking for NaN in X (review_text):", datasetToTrain['review_text'].isna().sum())
print("Checking for NaN in y (recommend_to_a_friend):", datasetToTrain['recommend_to_a_friend'].isna().sum())





### ---------------------------
### TRAIN/TEST 
### ---------------------------

print("\n\nTraining the model.....\n")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = datasetToTrain['review_text']
y = datasetToTrain['recommend_to_a_friend']

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# DEBUG
print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

tfidf = TfidfVectorizer(
    max_df=0.8,       # ignore terms appearing in >80% of documents
    min_df=5,         # ignore terms appearing in <5 documents
    ngram_range=(1,2) # unigrams + bigrams
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Vocabulary size:", len(tfidf.vocabulary_))
print("Train TF-IDF shape:", X_train_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)

# NAIVE BAYES
nb_clf = MultinomialNB()
nb_clf.fit(X_train_tfidf, y_train)

y_pred_nb = nb_clf.predict(X_test_tfidf)

print("\n---NAIVE BAYES---")
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report (Naive Bayes):\n", classification_report(y_test, y_pred_nb))
print("\nConfusion Matrix (Naive Bayes):\n", confusion_matrix(y_test, y_pred_nb))

# LOGISTIC REGRESSION
lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train_tfidf, y_train)

y_pred_lr = lr_clf.predict(X_test_tfidf)

print("\n---LOGISTIC REGRESSION---")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report (LR):\n", classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix (LR):\n", confusion_matrix(y_test, y_pred_lr))

# SUPPORT VECTOR MACHINES - SVC
svc_clf = LinearSVC(penalty='l1')   # l1 penalty gives us better results than l2
svc_clf.fit(X_train_tfidf, y_train)

y_pred_svc = svc_clf.predict(X_test_tfidf)

print("\n---SUPPORT VECTOR MACHINES - SVC---")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_svc))
print("\nClassification Report (SVC):\n", classification_report(y_test, y_pred_svc))
print("\nConfusion Matrix (SVC):\n", confusion_matrix(y_test, y_pred_svc))

# SUPPORT VECTOR MACHINES - SGD
sgd_clf = SGDClassifier(penalty='l2')   # l2 penalty gives us better results than l1
sgd_clf.fit(X_train_tfidf, y_train)

y_pred_sgd = sgd_clf.predict(X_test_tfidf)

print("\n---SUPPORT VECTOR MACHINES - SGD---")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_sgd))
print("\nClassification Report (SGD):\n", classification_report(y_test, y_pred_sgd))
print("\nConfusion Matrix (SGD):\n", confusion_matrix(y_test, y_pred_sgd))
