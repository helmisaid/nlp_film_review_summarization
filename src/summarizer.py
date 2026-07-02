import networkx as nx
from src.graph import (
    segment_sentences,
    build_sentence_vectors,
    compute_similarity_matrix,
    build_semantic_graph,
)


def rank_sentences_textrank(G: nx.Graph) -> dict:
    """
    Hitung PageRank score untuk setiap node kalimat.
    alpha=0.85, weight='weight', max_iter=100.
    Fallback: jika graf kosong, return skor uniform.
    """
    if G.number_of_nodes() == 0:
        return {}

    try:
        scores = nx.pagerank(G, alpha=0.85, weight='weight', max_iter=100)
    except nx.PowerIterationFailedConvergence:
        # Fallback: skor uniform jika PageRank tidak konvergen
        n = G.number_of_nodes()
        scores = {node: 1.0 / n for node in G.nodes()}

    return scores


def extract_top_sentences(scores: dict, sentences: list, n: int = 3) -> list:
    """
    Ambil top-N kalimat berdasarkan skor tertinggi,
    lalu urutkan ascending agar sesuai urutan teks asli.
    """
    if not scores:
        return sentences[:n] if sentences else []

    sorted_nodes = sorted(scores, key=scores.get, reverse=True)
    top_n_indices = sorted_nodes[:n]
    top_n_indices.sort()  # ascending untuk preserve original order

    return [sentences[i] for i in top_n_indices if i < len(sentences)]


def generate_summary(reviews: list, n: int = 3) -> list:
    """
    Orkestrasi pipeline TextRank:
    segment_sentences → build_sentence_vectors → compute_similarity_matrix
    → build_semantic_graph → rank_sentences_textrank → extract_top_sentences

    Edge case: jika < 3 ulasan, return reviews langsung.
    """
    if len(reviews) < 3:
        return reviews

    sentences = segment_sentences(reviews)

    if len(sentences) < n:
        return sentences

    tfidf_matrix, _ = build_sentence_vectors(sentences)

    if tfidf_matrix.shape[0] == 0:
        return sentences[:n]

    similarity = compute_similarity_matrix(tfidf_matrix)

    G = build_semantic_graph(similarity, sentences)

    scores = rank_sentences_textrank(G)

    return extract_top_sentences(scores, sentences, n=n)
