"""
FastAPI entry point.
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import retrieval
from agent import run_agent
from retrieval import get_catalog_names, get_catalog_urls
from schemas import ChatRequest, ChatResponse, Recommendation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading retrieval index…")
    retrieval.load_index()
    names = get_catalog_names()
    urls = get_catalog_urls()
    logger.info("Index ready: %d assessments loaded.", len(names))
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Validator (hard guarantee before every response) ─────────────────────────

def validate_response(response: ChatResponse) -> ChatResponse:
    catalog_names = get_catalog_names()
    catalog_urls = get_catalog_urls()

    clean: list[Recommendation] = []
    for rec in response.recommendations:
        if rec.name not in catalog_names:
            logger.error("VALIDATOR: dropped hallucinated name '%s'", rec.name)
            continue
        url_str = str(rec.url)
        if url_str not in catalog_urls:
            logger.error("VALIDATOR: dropped hallucinated URL '%s'", url_str)
            continue
        clean.append(rec)

    if len(clean) > 10:
        clean = clean[:10]

    response.recommendations = clean
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = run_agent(request.messages)
        response = validate_response(response)
        return response
    except Exception as exc:
        logger.exception("Unhandled error in /chat: %s", exc)
        return ChatResponse(
            reply="Sorry, something went wrong. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Global exception: %s", exc)
    return JSONResponse(
        status_code=200,
        content={
            "reply": "Sorry, something went wrong. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )
