import re
import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_text


def preprocess_text(text: str) -> str:
    """
    Pra-pemrosesan teks untuk kalimat:
    1. Case folding
    2. Hapus karakter non-alfabet
    3. Stopwords removal (pakai Sastrawi dari preprocess.py)
    4. Strip whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    from src.preprocess import stop_words
    words = [w for w in words if w not in stop_words and len(w) > 1]

    return " ".join(words)


def segment_sentences(reviews: list) -> list:
    """
    Gabung semua ulasan, pecah jadi kalimat pakai nltk.sent_tokenize,
    filter kalimat < 5 kata, hapus duplikat.
    """
    nltk.download('punkt_tab', quiet=True)

    all_text = " ".join([r for r in reviews if isinstance(r, str) and r.strip()])

    if not all_text.strip():
        return []

    raw_sentences = nltk.sent_tokenize(all_text)

    # Filter kalimat dengan < 5 kata
    valid_sentences = [s.strip() for s in raw_sentences if len(s.split()) >= 5]

    # Hapus duplikat (preserve order)
    seen = set()
    unique_sentences = []
    for s in valid_sentences:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique_sentences.append(s)

    return unique_sentences


def build_sentence_vectors(sentences: list) -> tuple:
    """
    TF-IDF Vectorizer per kumpulan kalimat.
    Return (tfidf_matrix, cleaned_sentences)
    """
    if not sentences:
        return np.array([]), []

    cleaned = [preprocess_text(s) for s in sentences]

    if not any(cleaned):
        return np.array([]), []

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(cleaned)

    return matrix, cleaned


def compute_similarity_matrix(tfidf_matrix) -> np.ndarray:
    """
    Hitung cosine similarity matrix, threshold 0.1
    """
    if tfidf_matrix.shape[0] == 0:
        return np.array([])

    sim_matrix = cosine_similarity(tfidf_matrix)
    sim_matrix[sim_matrix < 0.1] = 0
    return sim_matrix


def build_semantic_graph(similarity_matrix: np.ndarray, sentences: list) -> nx.Graph:
    """
    Bangun graf tidak berarah NetworkX:
    - Node = indeks kalimat
    - Edge = similarity > 0, dengan bobot similarity
    """
    G = nx.Graph()

    n = len(sentences)
    for i in range(n):
        G.add_node(i, sentence=sentences[i])

    if similarity_matrix.size == 0:
        return G

    for i in range(n):
        for j in range(i + 1, n):
            weight = similarity_matrix[i][j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    return G
