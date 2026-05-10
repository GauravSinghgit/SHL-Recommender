---
title: SHL Assessment Recommender
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# SHL Assessment Recommender

Multi-turn conversational API that recommends SHL assessments to hiring managers.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in API keys
cp .env.example .env
# Edit .env → add GROQ_API_KEY and GEMINI_API_KEY

# 3. Run ingestion (ONE TIME ONLY — needs the catalog JSON)
python ingest.py --catalog path/to/shl_product_catalog.json

# 4. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

### GET /health
```json
{"status": "ok"}
```

### POST /chat
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need an assessment for a senior Java developer"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are recommended assessments...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "Knowledge & Skills"
    }
  ],
  "end_of_conversation": false
}
```

## Test with curl

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need assessments for a graduate management trainee program"}]}'
```

## Deploy to Hugging Face Spaces

1. Create a new Space (Docker SDK)
2. Push this repo
3. Add `GROQ_API_KEY` and `GEMINI_API_KEY` as Space secrets
4. Commit `data/catalog.json`, `data/faiss.index`, `data/bm25_corpus.pkl`

## File structure

```
shl-recommender/
├── main.py          # FastAPI app
├── agent.py         # Router + 5 handlers
├── retrieval.py     # Hybrid search (FAISS + BM25)
├── llm.py           # Groq + Gemini client
├── ingest.py        # One-time data pipeline
├── schemas.py       # Pydantic models
├── requirements.txt
├── Dockerfile
└── data/
    ├── catalog.json
    ├── faiss.index
    └── bm25_corpus.pkl
```
