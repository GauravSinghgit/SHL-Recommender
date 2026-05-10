import json, pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DATA = Path("data")
with open(DATA / "catalog.json", encoding="utf-8") as f:
    records = json.load(f)

texts = [r.get("embedding_text") or r["name"] for r in records]
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
emb = np.array(emb, dtype="float32")

idx = faiss.IndexFlatL2(emb.shape[1])
idx.add(emb)
faiss.write_index(idx, str(DATA / "faiss.index"))

bm25 = BM25Okapi([t.lower().split() for t in texts])
with open(DATA / "bm25_corpus.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "texts": texts}, f)

print(f"Index built: {idx.ntotal} vectors, dim={emb.shape[1]}")
