import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("stopwords")

# Product reviews
def load_reviews(filepath):
    with open(filepath, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


reviews = load_reviews("../reviews/reviews.txt")  # reviews path
# print(reviews)

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("portuguese"))
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


processed_reviews = [preprocess(review) for review in reviews]
print(processed_reviews)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_reviews)

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Displaying results
print("Matriz de Similaridade (Cosseno):\n")
print(
    pd.DataFrame(
        cosine_sim,
        columns=["Review 1", "Review 2", "Review 3"],
        index=["Review 1", "Review 2", "Review 3"],
    )
)

# Simple similarity analysis
pairs = [(i, j, cosine_sim[i][j]) for i in range(3) for j in range(i + 1, 3)]
most_similar = max(pairs, key=lambda x: x[2])
least_similar = min(pairs, key=lambda x: x[2])

print(
    f"\nReviews mais similares: Review {most_similar[0]+1} e Review {most_similar[1]+1} (similaridade = {most_similar[2]:.2f})"
)
print(
    f"Reviews mais distintas: Review {least_similar[0]+1} e Review {least_similar[1]+1} (similaridade = {least_similar[2]:.2f})"
)
