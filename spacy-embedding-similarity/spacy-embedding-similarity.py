0.2
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model with word embeddings vectors
nlp = spacy.load("pt_core_news_md")


# Product Reviews
def load_reviews(filepath):
    with open(filepath, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


reviews = load_reviews("../reviews/reviews.txt")  # ajuste o path conforme necessário

# Pre-processing and generating embeddings for each review
def preprocess_with_spacy(doc):
    doc = nlp(doc.lower())
    tokens = [token for token in doc if not token.is_stop and not token.is_punct]
    if tokens:
        return sum([t.vector for t in tokens]) / len(tokens)
    else:
        return np.zeros(nlp.vocab.vectors_length)


embeddings = np.array([preprocess_with_spacy(text) for text in reviews])

# Cosine similarity between embeddings
cosine_sim = cosine_similarity(embeddings)

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
