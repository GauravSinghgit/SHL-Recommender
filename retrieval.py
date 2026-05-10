"""
Hybrid retrieval: dense (FAISS) + lexical (BM25) fused with RRF.
Loaded once at startup via load_index(); never called at import time.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RRF_K = 60

# Module-level state populated by load_index()
_embed_model: SentenceTransformer | None = None
_faiss_index: faiss.Index | None = None
_bm25 = None
_records: list[dict] = []


def load_index():
    """Load FAISS index, BM25 corpus, and embedding model into memory."""
    global _embed_model, _faiss_index, _bm25, _records

    catalog_path = DATA_DIR / "catalog.json"
    faiss_path = DATA_DIR / "faiss.index"
    bm25_path = DATA_DIR / "bm25_corpus.pkl"

    import json

    with open(catalog_path, encoding="utf-8") as f:
        _records = json.load(f)

    _faiss_index = faiss.read_index(str(faiss_path))

    with open(bm25_path, "rb") as f:
        bm25_data = pickle.load(f)
    _bm25 = bm25_data["bm25"]

    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    logger.info(
        "Retrieval index loaded: %d records, FAISS dim=%d",
        len(_records),
        _faiss_index.d,
    )


def _dense_scores(query: str, top_k: int) -> list[tuple[int, float]]:
    """Return [(record_idx, l2_distance), ...] for top_k nearest."""
    vec = _embed_model.encode([query], normalize_embeddings=True)
    vec = np.array(vec, dtype="float32")
    distances, indices = _faiss_index.search(vec, top_k)
    return list(zip(indices[0].tolist(), distances[0].tolist()))


def _bm25_scores(query: str) -> list[tuple[int, float]]:
    """Return [(record_idx, bm25_score), ...] for all records, sorted desc."""
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked


def _rrf(dense: list[tuple[int, float]], bm25: list[tuple[int, float]], top_k: int) -> list[int]:
    """Reciprocal Rank Fusion over two ranked lists. Returns ordered record indices."""
    rrf_scores: dict[int, float] = {}

    for rank, (idx, _) in enumerate(dense):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)

    for rank, (idx, _) in enumerate(bm25):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)

    sorted_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
    return sorted_ids[:top_k]


def hybrid_search(query: str, top_k: int = 20) -> list[dict]:
    """
    Search the catalog with hybrid dense + BM25 retrieval.
    Returns up to top_k catalog records sorted by relevance.
    """
    if _faiss_index is None:
        raise RuntimeError("Retrieval index not loaded. Call load_index() first.")

    n = min(top_k * 2, len(_records))
    dense = _dense_scores(query, n)
    bm25 = _bm25_scores(query)[:n]

    top_indices = _rrf(dense, bm25, top_k)
    return [_records[i] for i in top_indices if i < len(_records)]


def get_by_name(name: str) -> dict | None:
    """Exact-match lookup by assessment name."""
    for r in _records:
        if r["name"] == name:
            return r
    return None


def get_by_name_fuzzy(name: str) -> dict | None:
    """
    Fuzzy lookup using difflib when exact match fails.
    Returns the closest match above 0.6 similarity, or None.
    """
    import difflib

    names = [r["name"] for r in _records]
    matches = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    if matches:
        return get_by_name(matches[0])
    return None


def get_all_records() -> list[dict]:
    return _records


def get_catalog_names() -> set[str]:
    return {r["name"] for r in _records}


def get_catalog_urls() -> set[str]:
    return {r["url"] for r in _records}
