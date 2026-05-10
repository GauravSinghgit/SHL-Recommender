"""
LLM client: Groq (llama-3.3-70b-versatile) primary, Gemini 2.0 Flash fallback.
Never raises — always returns a string or a safe default.
"""

import json
import logging
import os
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

_groq_client: Groq | None = None
_gemini_model = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def _get_gemini():
    global _gemini_model
    if _gemini_model is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    return _gemini_model


def _messages_to_gemini(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style messages to Gemini contents format."""
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return contents


def call_llm(
    messages: list[dict[str, str]],
    json_mode: bool = False,
    timeout: int = 20,
) -> str:
    """
    Call Groq first; fall back to Gemini on any failure.
    Returns the assistant text string. Never raises.
    """
    json_instruction = (
        "\n\nIMPORTANT: Your response must be valid JSON only. "
        "No markdown, no code fences, no explanation — raw JSON only."
        if json_mode
        else ""
    )

    # Inject JSON instruction into the last user message if needed
    if json_mode and messages:
        messages = list(messages)
        last = messages[-1]
        if last["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": last["content"] + json_instruction,
            }

    # ── Groq ──────────────────────────────────────────────────────────────────
    try:
        client = _get_groq()
        kwargs: dict[str, Any] = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "timeout": timeout,
            "max_tokens": 2048,
            "temperature": 0.2,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        logger.debug("Groq responded (%d chars)", len(text))
        return text
    except Exception as groq_err:
        logger.warning("Groq failed (%s), falling back to Gemini.", groq_err)

    # ── Gemini fallback ───────────────────────────────────────────────────────
    try:
        model = _get_gemini()
        gemini_contents = _messages_to_gemini(messages)
        response = model.generate_content(
            gemini_contents,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
        text = response.text or ""
        # Strip markdown fences if JSON was requested
        if json_mode:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
        logger.debug("Gemini responded (%d chars)", len(text))
        return text
    except Exception as gemini_err:
        logger.error("Gemini also failed (%s). Returning empty string.", gemini_err)
        return ""


def call_llm_json(messages: list[dict[str, str]], timeout: int = 20) -> dict:
    """
    Convenience wrapper: calls call_llm with json_mode=True and parses result.
    Returns parsed dict or {} on failure.
    """
    raw = call_llm(messages, json_mode=True, timeout=timeout)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON substring
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("Could not parse LLM JSON output: %.200s", raw)
        return {}
