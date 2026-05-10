"""
Run once offline: python ingest.py --catalog path/to/shl_product_catalog.json
Produces: data/catalog.json, data/faiss.index, data/bm25_corpus.pkl

Handles raw catalog JSON with embedded newlines in string values.
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# ── JSON cleanup ──────────────────────────────────────────────────────────────

def fix_json_strings(s: str) -> str:
    """Replace bare newlines/tabs inside JSON string values so the parser accepts them."""
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_string and c == "\\":
            result.append(c)
            if i + 1 < len(s):
                result.append(s[i + 1])
                i += 2
            else:
                i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
        elif in_string and c in ("\n", "\r"):
            result.append(" ")
        elif in_string and c == "\t":
            result.append(" ")
        else:
            result.append(c)
        i += 1
    return "".join(result)


# ── Normalisation ─────────────────────────────────────────────────────────────

def parse_duration(raw: str) -> int | None:
    """Extract integer minutes from strings like '20 minutes', '1 hour', etc."""
    if not raw:
        return None
    raw = raw.strip().lower()
    m = re.search(r"(\d+)\s*min", raw)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*hour", raw)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"(\d+)", raw)
    if m:
        return int(m.group(1))
    return None


def normalize_record(raw: dict) -> dict:
    """Map raw catalog fields to our internal flat schema."""
    name = (raw.get("name") or "").strip().replace("  ", " ")
    url = (raw.get("link") or raw.get("url") or "").strip()

    # 'keys' already contains full label strings like "Ability & Aptitude"
    type_labels: list[str] = [k for k in (raw.get("keys") or []) if k]

    languages: list[str] = raw.get("languages") or []
    if isinstance(languages, str):
        languages = [l.strip() for l in languages.split(",") if l.strip()]

    job_levels: list[str] = raw.get("job_levels") or []
    if isinstance(job_levels, str):
        job_levels = [j.strip() for j in job_levels.split(",") if j.strip()]

    duration = parse_duration(raw.get("duration") or raw.get("duration_raw") or "")
    remote = str(raw.get("remote", "")).lower() in ("yes", "true", "1")
    adaptive = str(raw.get("adaptive", "")).lower() in ("yes", "true", "1")

    return {
        "name": name,
        "url": url,
        "test_type": type_labels[0] if type_labels else "",
        "test_type_labels": type_labels,
        "description": (raw.get("description") or "").strip(),
        "job_levels": job_levels,
        "languages": languages,
        "duration_minutes": duration,
        "remote_testing": remote,
        "adaptive_irt": adaptive,
    }


def build_embedding_text(record: dict) -> str:
    name = record["name"]
    types = ", ".join(record["test_type_labels"])
    desc = record["description"][:300]
    levels = ", ".join(record["job_levels"][:6])
    langs = ", ".join(record["languages"][:5])
    return f"{name}. {types}. {desc}. Levels: {levels}. Languages: {langs}"


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest(catalog_path: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(catalog_path, encoding="utf-8") as f:
        raw_text = f.read()

    # Fix bare newlines inside JSON strings
    fixed = fix_json_strings(raw_text)
    raw_catalog = json.loads(fixed)

    if isinstance(raw_catalog, dict):
        # Unwrap if needed
        items = next(
            (v for v in raw_catalog.values() if isinstance(v, list)), []
        )
    else:
        items = raw_catalog

    print(f"Loaded {len(items)} raw items.")

    records = []
    for raw in items:
        rec = normalize_record(raw)
        if not rec["name"] or not rec["url"]:
            continue
        rec["embedding_text"] = build_embedding_text(rec)
        records.append(rec)

    print(f"Normalized {len(records)} records.")

    # Save catalog
    catalog_out = DATA_DIR / "catalog.json"
    with open(catalog_out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved catalog: {catalog_out} ({len(records)} items)")

    # Embed
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [r["embedding_text"] for r in records]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss_path = DATA_DIR / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    print(f"Saved FAISS index: {faiss_path} ({index.ntotal} vectors, dim={dim})")

    # BM25
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_path = DATA_DIR / "bm25_corpus.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts}, f)
    print(f"Saved BM25 corpus: {bm25_path}")

    print("\nIngestion complete.")
    print(f"  Records : {len(records)}")
    print(f"  FAISS   : {faiss_path}")
    print(f"  BM25    : {bm25_path}")
    print(f"  Catalog : {catalog_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="Path to raw SHL catalog JSON.")
    args = parser.parse_args()
    ingest(args.catalog)
