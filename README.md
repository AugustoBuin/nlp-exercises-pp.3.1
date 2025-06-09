# Product Review Similarity (TF-IDF vs spaCy Embeddings)

This repository demonstrates two approaches to compute similarity between product reviews written in Portuguese:

1. **TF-IDF + Cosine Similarity** using `scikit-learn`
2. **spaCy Word Embeddings + Cosine Similarity**

---

## 🔍 Objective

To compare how different vectorization methods affect the semantic similarity detection between user reviews from an e-commerce context.

---

## 📂 Project Structure

```
/reviews/
└── reviews.txt                 # Plain text file with one review per line

/tfidf_similarity/
├── requirements.txt
└── tfidf_similarity.py        # TF-IDF based similarity script

/spacy-embedding-similarity/
├── requirements.txt
└── spacy-embedding-similarity.py  # spaCy embeddings based similarity script
```

---

## ⚙️ Setup

### Clone the repo

```bash
git clone https://github.com/your-username/product-review-similarity.git
cd product-review-similarity
```

Place your review texts in the file:

```bash
./reviews/reviews.txt
```

Each line should contain a single product review.

---

## 🧪 Option 1: TF-IDF + Cosine Similarity

### Requirements

```bash
cd tfidf_similarity
pip install -r requirements.txt
```

### Run the script

```bash
python tfidf_similarity.py
```

This script performs:

* Preprocessing: lowercasing, stopword removal, tokenization.
* Loads reviews from `../reviews/reviews.txt`
* Vectorization using `TfidfVectorizer`
* Cosine similarity computation
* Outputs the similarity matrix and identifies the most/least similar reviews.

---

## 🧠 Option 2: spaCy Word Embeddings

### Requirements

```bash
cd spacy-embedding-similarity
pip install -r requirements.txt
python -m spacy download pt_core_news_md
```

### Run the script

```bash
python spacy-embedding-similarity.py
```

This script:

* Loads spaCy’s Portuguese model with word vectors
* Loads reviews from `../reviews/reviews.txt`
* Averages token vectors to represent full reviews
* Computes cosine similarity between dense vectors
* Outputs the similarity matrix and identifies the most/least similar reviews

---

## 🔍 Comparison

| Feature                 | TF-IDF                   | spaCy Embeddings            |
| ----------------------- | ------------------------ | --------------------------- |
| Vector Type             | Sparse, high-dimensional | Dense, low-dimensional      |
| Captures word meaning   | ❌ No                     | ✅ Yes                       |
| Requires language model | ❌ No                     | ✅ Yes (`pt_core_news_md`)   |
| Sensitivity to synonyms | ❌ High                   | ✅ Low                       |
| Interpretability        | ✅ High (frequency-based) | ❌ Low (semantic-based)      |
| Recommended use case    | Keyword-based similarity | Semantic/textual similarity |

---

## 📈 Example Output

```
Most similar reviews: Review 1 and Review 3
Most distinct reviews: Review 2 and Review 3
```

---

## 📜 License

MIT License.
