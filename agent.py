"""
Deterministic policy engine + hybrid retrieval + LLM for text only.

LLM is called ONLY for: explanation text, clarification questions, comparison summaries.
Everything structural is deterministic:
  - Routing (CLARIFY / RECOMMEND / REFINE / COMPARE / REFUSE)
  - Fact extraction (role, level, skills, test types, constraints)
  - Off-topic detection
  - Compare detection
  - Refinement state detection
  - Turn counting / clarification threshold
  - Schema / recommendation list generation (retrieval results → direct output)
"""

import json
import logging
import re

from llm import call_llm
from retrieval import (
    get_by_name,
    get_by_name_fuzzy,
    hybrid_search,
)
from schemas import ChatResponse, Message, Recommendation

logger = logging.getLogger(__name__)

# ── System prompt (used only when generating natural-language text) ───────────

SYSTEM = """You are an SHL assessment advisor helping hiring managers find the right
SHL Individual Test Solutions. Rules:
- Only reference assessments explicitly provided to you — never invent names or URLs.
- Do not give hiring decisions, legal advice, or general HR consulting.
- Ask at most ONE clarifying question per turn. Keep replies concise."""

# ── Deterministic fact extraction ─────────────────────────────────────────────

_SENIORITY_MAP: dict[str, list[str]] = {
    "graduate": ["graduate", "fresh", "entry level", "entry-level", "new grad", "trainee", "intern"],
    "entry":    ["entry", "junior", "associate", "0-2 years", "0 to 2"],
    "mid":      ["mid", "intermediate", "3-5 years", "3 to 5"],
    "senior":   ["senior", "sr.", "lead", "5+ years", "5 years"],
    "manager":  ["manager", "team lead", "supervisor", "head of"],
    "director": ["director", "vp", "vice president"],
    "CXO":      ["cxo", "ceo", "cto", "cfo", "chief", "c-suite", "c-level"],
}

_TEST_TYPE_MAP: dict[str, list[str]] = {
    "cognitive":    ["cognitive", "aptitude", "reasoning", "numerical", "verbal", "abstract", "logical"],
    "personality":  ["personality", "behavioural", "behavioral", "opr", "occupational"],
    "situational":  ["situational", "sjt", "judgment", "judgement", "scenario"],
    "knowledge":    ["knowledge", "technical", "skill test", "coding", "java", "python", "programming"],
    "simulation":   ["simulation", "work sample", "realistic", "exercise"],
}

_KNOWN_SKILLS: set[str] = {
    "java", "python", "javascript", "sql", "c++", "c#", ".net", "react", "node",
    "sales", "marketing", "finance", "accounting", "leadership", "communication",
    "data analysis", "excel", "project management", "customer service", "coding",
    "devops", "cloud", "aws", "azure", "machine learning", "data science",
}

_JOB_TITLE_WORDS: set[str] = {
    "developer", "engineer", "manager", "analyst", "scientist", "representative",
    "director", "executive", "designer", "consultant", "specialist", "coordinator",
    "architect", "officer", "administrator", "associate", "supervisor", "lead",
    "trainee", "intern", "graduate", "programmer", "accountant", "recruiter",
}

_CONSTRAINT_PATTERNS: list[tuple[str, str]] = [
    (r"\b(\d+)\s*(?:min(?:utes?)?)\b", "max_duration"),
    (r"\bremote\b",                     "remote_testing"),
    (r"\bonline\b",                     "remote_testing"),
    (r"\bshort(?:er)?\b",              "prefer_short"),
    (r"\bquick\b",                      "prefer_short"),
    (r"\buntimed\b",                    "untimed"),
]


def extract_facts(messages: list[dict]) -> dict:
    """Extract structured hiring facts from conversation — no LLM."""
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()

    # Role: grab the job-title word and up to one qualifier before it
    role: str | None = None
    for word in _JOB_TITLE_WORDS:
        if word in user_text:
            match = re.search(rf"(\w+\s+)?{re.escape(word)}", user_text)
            if match:
                role = match.group(0).strip()
                break

    # Seniority
    level: str | None = None
    for lvl, keywords in _SENIORITY_MAP.items():
        if any(kw in user_text for kw in keywords):
            level = lvl
            break

    # Skills
    skills = [s for s in _KNOWN_SKILLS if s in user_text]

    # Test types
    test_types = [tt for tt, kws in _TEST_TYPE_MAP.items() if any(kw in user_text for kw in kws)]

    # Hard constraints
    constraints: list[str] = []
    for pattern, label in _CONSTRAINT_PATTERNS:
        match = re.search(pattern, user_text)
        if match:
            val = match.group(1) if match.lastindex else None
            constraints.append(f"{label}:{val}" if val else label)

    return {
        "role":        role,
        "level":       level,
        "skills":      skills,
        "test_types":  test_types,
        "constraints": constraints,
    }


def facts_to_query(facts: dict, extra: str = "") -> str:
    parts: list[str] = []
    for key in ("role", "level"):
        if facts.get(key):
            parts.append(facts[key])
    parts.extend(facts.get("skills", []))
    parts.extend(facts.get("test_types", []))
    if extra:
        parts.append(extra)
    return " ".join(parts) if parts else "assessment"


# ── Deterministic router ───────────────────────────────────────────────────────

_REFUSE_PATTERNS: list[str] = [
    r"\bweather\b",
    r"\brecipe\b",
    r"\bpolitics?\b",
    r"\bsport\b",
    r"\bignore (?:previous|above|all)\b",
    r"\bforget (?:your|all)\b",
    r"\bact as\b",
    r"\bpretend\b",
    r"\bjailbreak\b",
    r"\blegal advice\b",
    r"\bhiring decision\b",
    r"\bsalary\b",
]

_COMPARE_TRIGGERS: list[str] = [
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference between\b",
    r"\bwhich is better\b",
    r"\bbetter (?:than|between)\b",
]

_REFINE_KEYWORDS: list[str] = [
    "only", "without", "no more than", "shorter", "remove", "instead",
    "exclude", "not include", "prefer", "focus on", "narrow", "filter",
    "change to", "switch to", "replace", "drop", "add more", "fewer",
]


def _is_off_topic(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _REFUSE_PATTERNS)


def _is_compare(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _COMPARE_TRIGGERS)


def _is_refine(text: str, messages: list[dict]) -> bool:
    return bool(_extract_last_recommendations(messages)) and any(
        kw in text.lower() for kw in _REFINE_KEYWORDS
    )


def _has_enough_to_recommend(facts: dict, text: str) -> bool:
    if facts.get("role") or facts.get("skills") or facts.get("test_types"):
        return True
    return any(kw in text.lower() for kw in _JOB_TITLE_WORDS)


def route(messages: list[dict], facts: dict, turns_used: int) -> str:
    """Deterministic policy router — zero LLM calls."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    if _is_off_topic(last_user):
        return "REFUSE"

    if _is_compare(last_user):
        return "COMPARE"

    if _is_refine(last_user, messages):
        return "REFINE"

    if _has_enough_to_recommend(facts, last_user):
        return "RECOMMEND"

    if turns_used >= 6:
        return "RECOMMEND"

    return "CLARIFY"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_constraints(candidates: list[dict], constraints: list[str]) -> list[dict]:
    """Filter / re-sort retrieval results by hard constraints — no LLM."""
    filtered = candidates
    for c in constraints:
        if c.startswith("max_duration:"):
            try:
                cap = int(c.split(":")[1])
                filtered = [r for r in filtered if r.get("duration_minutes") is None or r["duration_minutes"] <= cap]
            except (ValueError, IndexError):
                pass
        elif c == "remote_testing":
            filtered = [r for r in filtered if r.get("remote_testing", False)]
        elif c == "prefer_short":
            filtered = sorted(filtered, key=lambda r: r.get("duration_minutes") or 999)
    return filtered


def _records_to_recommendations(records: list[dict], limit: int = 8) -> list[Recommendation]:
    """Convert catalog records to Recommendation objects — no LLM, no hallucination risk."""
    return [
        Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"])
        for r in records[:limit]
    ]


def _find_catalog_names_in_text(text: str) -> list[dict]:
    """Find catalog assessment records mentioned in free text — fuzzy n-gram matching, no LLM."""
    found: list[dict] = []
    seen: set[str] = set()
    words = text.split()
    for n in range(5, 1, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            record = get_by_name_fuzzy(phrase)
            if record and record["name"] not in seen:
                found.append(record)
                seen.add(record["name"])
    return found


def candidates_to_block(candidates: list[dict]) -> str:
    rows = []
    for i, c in enumerate(candidates):
        rows.append(
            f"{i+1}. Name: {c['name']}\n"
            f"   URL: {c['url']}\n"
            f"   Type: {c['test_type']}\n"
            f"   Description: {c.get('description', '')[:200]}\n"
            f"   Levels: {', '.join(c.get('job_levels', []))}\n"
            f"   Duration: {c.get('duration_minutes', 'unspecified')} min\n"
            f"   Remote: {c.get('remote_testing', False)}\n"
        )
    return "\n".join(rows)


def clean_reply(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*[\s\S]*?```", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{[\s\S]*\"recommendations\"[\s\S]*\}", "", text)
    return text.strip()


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_clarify(facts: dict, messages: list[dict]) -> ChatResponse:
    missing: list[str] = []
    if not facts.get("role") and not facts.get("skills"):
        missing.append("the job role or skills being assessed")
    if not facts.get("level"):
        missing.append("the seniority level")

    question_topic = missing[0] if missing else "what you're looking for"

    clarify_messages = [
        {"role": "system", "content": SYSTEM},
        *messages,
        {
            "role": "user",
            "content": (
                f"Ask the hiring manager ONE short, specific question to clarify: {question_topic}. "
                "Do not ask anything else. Do not recommend yet. Keep it under 2 sentences."
            ),
        },
    ]
    reply = call_llm(clarify_messages, timeout=15) or f"Could you tell me more about {question_topic}?"
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def handle_recommend(facts: dict, messages: list[dict]) -> ChatResponse:
    query = facts_to_query(facts)
    if not query or query == "assessment":
        query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "assessment")

    # Retrieval — deterministic ranking via hybrid FAISS + BM25 + RRF
    candidates = hybrid_search(query, top_k=20)

    # Apply hard constraints deterministically
    if facts.get("constraints"):
        candidates = _apply_constraints(candidates, facts["constraints"])

    # Convert top results directly to recommendations — no LLM selection step
    recommendations = _records_to_recommendations(candidates, limit=8)

    # LLM only for explanation text
    reply_messages = [
        {"role": "system", "content": SYSTEM},
        *messages,
        {
            "role": "user",
            "content": (
                f"These assessments were selected for this hiring need: {[r.name for r in recommendations]}. "
                "Write a short (2-3 sentence) explanation of why this combination fits. "
                "Be specific to the role and level. No lists, no JSON."
            ),
        },
    ]
    reply = clean_reply(
        call_llm(reply_messages, timeout=15) or "Here are the recommended assessments for your hiring need."
    )

    return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=False)


def handle_refine(facts: dict, messages: list[dict]) -> ChatResponse:
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    prior_recs = _extract_last_recommendations(messages)
    prior_names = [r.name for r in prior_recs]

    # Re-run retrieval incorporating the new constraint signal
    query = facts_to_query(facts, extra=last_user)
    candidates = hybrid_search(query, top_k=20)

    if facts.get("constraints"):
        candidates = _apply_constraints(candidates, facts["constraints"])

    recommendations = _records_to_recommendations(candidates, limit=8)

    # LLM only for natural-language acknowledgment
    reply_messages = [
        {"role": "system", "content": SYSTEM},
        *messages,
        {
            "role": "user",
            "content": (
                f"Prior shortlist: {prior_names}. "
                f"Updated shortlist: {[r.name for r in recommendations]}. "
                "Briefly acknowledge the refinement and explain what changed. 1-2 sentences only."
            ),
        },
    ]
    reply = clean_reply(call_llm(reply_messages, timeout=15) or "Here is your updated shortlist.")

    return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=False)


def handle_compare(messages: list[dict]) -> ChatResponse:
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    # Deterministic name extraction via fuzzy n-gram matching — no LLM
    records = _find_catalog_names_in_text(last_user)

    # Fall back to prior recommendations if fewer than 2 found explicitly
    if len(records) < 2:
        seen = {r["name"] for r in records}
        for rec in _extract_last_recommendations(messages):
            r = get_by_name(rec.name)
            if r and r["name"] not in seen:
                records.append(r)
                seen.add(r["name"])
            if len(records) >= 4:
                break

    if len(records) < 2:
        return ChatResponse(
            reply=(
                "I couldn't identify which assessments to compare. "
                "Please name them explicitly, e.g. 'compare Verify G+ vs OPQ32'."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # LLM only for comparison summary text
    compare_messages = [
        {"role": "system", "content": SYSTEM},
        *messages,
        {
            "role": "user",
            "content": (
                f"CATALOG ENTRIES:\n{candidates_to_block(records)}\n\n"
                "Compare these assessments on type, duration, job levels, and use cases. "
                "Be concise and factual. Use only the information in CATALOG ENTRIES."
            ),
        },
    ]
    reply = clean_reply(call_llm(compare_messages, timeout=20) or "")

    if not reply or len(reply) < 60:
        lines = [
            f"{r['name']}: type={r.get('test_type', '')}, "
            f"duration={r.get('duration_minutes', '?')} min, "
            f"levels={', '.join(r.get('job_levels', [])[:3])}."
            for r in records
        ]
        reply = "Comparison:\n" + "\n".join(lines)

    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def handle_refuse() -> ChatResponse:
    return ChatResponse(
        reply=(
            "I can only help with SHL assessment recommendations. "
            "Please ask me about assessments, job roles, or candidate evaluation tools."
        ),
        recommendations=[],
        end_of_conversation=False,
    )


# ── End-of-conversation detector (deterministic keyword set) ─────────────────

_EOC_KEYWORDS: set[str] = {
    "perfect", "thanks", "thank you", "that's all", "thats all",
    "great", "done", "bye", "goodbye", "got it", "that works",
    "exactly what i needed", "looks good", "all set",
}


def _user_is_done(messages: list[dict]) -> bool:
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    lower = last_user.lower().strip()
    return any(kw in lower for kw in _EOC_KEYWORDS)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_agent(messages: list[Message]) -> ChatResponse:
    """
    Pipeline:
      1. Deterministic EOC check
      2. Deterministic fact extraction   (no LLM)
      3. Deterministic routing           (no LLM)
      4. Handler — LLM used only for text generation inside handlers
    """
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    turns_used = sum(1 for m in msg_dicts if m["role"] == "user")

    if _user_is_done(msg_dicts) and turns_used > 1:
        return ChatResponse(
            reply="You're welcome! Good luck with your hiring. Feel free to come back if you need more assessments.",
            recommendations=_extract_last_recommendations(msg_dicts),
            end_of_conversation=True,
        )

    facts = extract_facts(msg_dicts)
    intent = route(msg_dicts, facts, turns_used)

    logger.info(
        "Intent=%s turns=%d role=%s level=%s skills=%s",
        intent, turns_used, facts.get("role"), facts.get("level"), facts.get("skills"),
    )

    if intent == "REFUSE":
        return handle_refuse()
    elif intent == "CLARIFY":
        return handle_clarify(facts, msg_dicts)
    elif intent == "RECOMMEND":
        return handle_recommend(facts, msg_dicts)
    elif intent == "REFINE":
        return handle_refine(facts, msg_dicts)
    elif intent == "COMPARE":
        return handle_compare(msg_dicts)
    else:
        return handle_clarify(facts, msg_dicts)


def _extract_last_recommendations(messages: list[dict]) -> list[Recommendation]:
    """Pull the most recent recommendation list from assistant message history."""
    for msg in reversed(messages):
        if msg["role"] != "assistant":
            continue
        try:
            parsed = json.loads(msg["content"])
            recs = parsed.get("recommendations", [])
            valid: list[Recommendation] = []
            for r in recs:
                record = get_by_name(r.get("name", "")) or get_by_name_fuzzy(r.get("name", ""))
                if record:
                    valid.append(
                        Recommendation(name=record["name"], url=record["url"], test_type=record["test_type"])
                    )
            if valid:
                return valid
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    return []
