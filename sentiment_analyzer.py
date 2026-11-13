# Name: Korbin Awender
# Course: CSCI 130
# Project: Week 4 Sentiment Analyzer
# File: sentiment_analyzer.py
# Date: 11/13/2026

import nltk
import pandas as pd

import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from nltk.corpus import movie_reviews

# Required datasets
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Build dataset
documents = []

for fileid in movie_reviews.fileids():
    review_text = movie_reviews.raw(fileid)
    label = movie_reviews.categories(fileid)[0]

    documents.append((review_text, label))

# DataFrame
df = pd.DataFrame(
    documents,
    columns=["review", "label"]
)

def preprocess_text(text):

    tokens = word_tokenize(text)

    tokens = [word.lower() for word in tokens]

    tokens = [word for word in tokens if word not in string.punctuation]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

df["processed_review"] = df["review"].apply(preprocess_text)

print("\nPreprocessing Complete")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["processed_review"])

y = df["label"]

print("Vectorization Complete")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train/Test Split Complete")

classifier = MultinomialNB()

classifier.fit(X_train, y_train)

print("Training Complete")

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(accuracy)

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# Show class counts
print("Class Counts")
print(df["label"].value_counts())

print("\nModel Ready")