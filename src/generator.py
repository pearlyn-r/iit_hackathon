"""
LLM Rationale Generator
========================
Generates brief, grounded rationales for why each retrieved standard
is relevant to the query — using ONLY free, local, open-source methods.
Zero paid APIs. Zero external calls.

CRITICAL DESIGN PRINCIPLE:
- The `retrieved_standards` list (IS numbers) comes ENTIRELY from the retriever.
- The LLM is NEVER asked to suggest, generate, or modify standard IDs.
- The LLM is ONLY asked to write 1-sentence rationales based on provided context.
- This guarantees zero hallucination of standard IDs.

Two modes (both free):
  1. "extract"  — sentence-scoring from chunk text (default, instant, zero deps)
  2. "llm"      — local HuggingFace model via transformers (better quality, needs GPU/RAM)

Fallback: If the local LLM fails for any reason, automatically drops to extract mode.
This ensures the pipeline always produces valid output.
"""

import re
import logging
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_rationales(
    query: str,
    retrieved: list,
    use_llm: bool = False,
    model_path: Optional[str] = None,
    api_key: Optional[str] = None,   # kept for signature compat, ignored
) -> list:
    """
    Generate rationales for retrieved standards.

    Args:
        query       : Original product description query.
        retrieved   : List of dicts from BISRetriever.retrieve().
                      Each must have: standard_id, title, rationale_context,
                      category, score.
        use_llm     : If True, attempt a local HuggingFace model first,
                      then fall back to extract mode on any error.
        model_path  : HuggingFace model ID or local path for LLM mode.
                      Defaults to "TinyLlama/TinyLlama-1.1B-Chat-v1.0".
        api_key     : Ignored — kept only for drop-in compatibility.

    Returns:
        List of dicts with an added 'rationale' field.
        standard_id values are NEVER modified.

    GUARANTEE: This function NEVER modifies the standard_id field.
               All IS codes in output come verbatim from `retrieved`.
    """
    if not retrieved:
        return []

    if use_llm:
        try:
            return _llm_rationales(query, retrieved, model_path)
        except Exception as e:
            logger.warning(f"Local LLM rationale failed: {e}. Falling back to extract mode.")

    return _extract_rationales(query, retrieved)


def generate_rationales_batch(
    queries: list,
    retrieved_batch: list,
    use_llm: bool = False,
    model_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list:
    """
    Batch version of generate_rationales. Processes each query independently.
    In extract mode (default) all calls are near-instant.
    """
    return [
        generate_rationales(q, r, use_llm=use_llm, model_path=model_path)
        for q, r in zip(queries, retrieved_batch)
    ]


# ---------------------------------------------------------------------------
# Mode 1: Sentence-extraction rationale (default — fast, zero deps)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "is", "to", "and", "or", "be",
    "this", "with", "at", "by", "as", "on", "that", "are", "its", "it",
    "was", "from", "has", "have", "not", "shall", "which", "also", "such",
    "may", "all", "any", "been", "their", "they", "more", "used", "when",
    "where", "while", "than", "into", "these", "those", "can", "will",
}


def _tokenize(text: str) -> list:
    return [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in _STOPWORDS]


def _score_sentence(sent: str, query_tokens: set) -> float:
    """Score a sentence by token overlap with query, normalised by length."""
    sent_tokens = set(_tokenize(sent))
    if not sent_tokens:
        return 0.0
    overlap = len(query_tokens & sent_tokens)
    length_bonus = min(len(sent_tokens) / 8, 1.0)
    return (overlap / len(query_tokens | sent_tokens)) * (0.6 + 0.4 * length_bonus)


def _best_sentences(text: str, query_tokens: set, n: int = 2) -> str:
    """Return the top-n most relevant sentences from text, joined."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

    if not sentences:
        words = text.split()[:40]
        return " ".join(words) + ("..." if len(text.split()) > 40 else "")

    scored = sorted(sentences, key=lambda s: _score_sentence(s, query_tokens), reverse=True)
    top = scored[:n]

    order = {s: i for i, s in enumerate(sentences)}
    top.sort(key=lambda s: order.get(s, 999))

    result = " ".join(top)
    return result[:350] + ("..." if len(result) > 350 else "")


def _extract_rationales(query: str, retrieved: list) -> list:
    """
    Extract the most query-relevant sentences from each chunk's text.
    No model needed — fast, deterministic, zero hallucination.
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        query_tokens = set(query.lower().split())

    result = []
    for item in retrieved:
        context = (
            item.get("rationale_context")
            or item.get("text")
            or item.get("chunk_text")
            or ""
        )

        if context.strip():
            rationale = _best_sentences(context, query_tokens, n=2)
        else:
            rationale = _template_rationale_single(query, item)

        result.append({**item, "rationale": rationale})

    return result


# ---------------------------------------------------------------------------
# Mode 2: Local HuggingFace LLM (optional, better quality)
# ---------------------------------------------------------------------------

_llm_pipeline = None  # cached after first load


def _load_llm(model_path: Optional[str]):
    """Load (and cache) a local HuggingFace text-generation pipeline."""
    global _llm_pipeline

    if _llm_pipeline is not None:
        return _llm_pipeline

    from transformers import pipeline as hf_pipeline

    model_id = model_path or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    logger.info(f"Loading local LLM: {model_id} (first call only)")

    _llm_pipeline = hf_pipeline(
        "text-generation",
        model=model_id,
        max_new_tokens=180,
        temperature=0.05,
        do_sample=False,
        device_map="auto",
        return_full_text=False,
    )
    logger.info("Local LLM loaded.")
    return _llm_pipeline


def _build_prompt(query: str, item: dict) -> str:
    """Build a tightly scoped prompt for one standard."""
    sid = item["standard_id"]
    title = item.get("title", "")
    context = (
        item.get("rationale_context")
        or item.get("text")
        or item.get("chunk_text")
        or ""
    )[:400]

    return (
        f"<|system|>You are a BIS Standards compliance expert. "
        f"Answer strictly from the provided excerpt. "
        f"Do NOT mention any standard IDs other than the one given.</s>\n"
        f"<|user|>Product query: {query}\n\n"
        f"Standard: {sid} — {title}\n"
        f"Excerpt: {context}\n\n"
        f"Write ONE sentence (max 50 words) explaining why {sid} is relevant to the query above. "
        f"Start directly with the explanation.</s>\n"
        f"<|assistant|>"
    )


def _llm_rationales(query: str, retrieved: list, model_path: Optional[str]) -> list:
    """
    Generate rationales using a local HuggingFace model.
    Falls back per-item to extract mode if generation fails.
    """
    pipe = _load_llm(model_path)
    query_tokens = set(_tokenize(query))
    result = []

    for item in retrieved:
        try:
            prompt = _build_prompt(query, item)
            out = pipe(prompt)
            raw = out[0]["generated_text"].strip()

            own_id = item["standard_id"]
            raw = _strip_foreign_standard_ids(raw, own_id)
            rationale = raw[:300] if raw else _template_rationale_single(query, item)

        except Exception as e:
            logger.warning(f"LLM generation failed for {item['standard_id']}: {e}")
            rationale = _best_sentences(
                item.get("rationale_context") or item.get("text") or "",
                query_tokens
            ) or _template_rationale_single(query, item)

        result.append({**item, "rationale": rationale})

    return result


def _strip_foreign_standard_ids(text: str, own_id: str) -> str:
    """
    Remove any IS code references from generated text that are NOT own_id.
    Prevents the LLM from sneaking in hallucinated standard IDs.
    """
    own_num = re.search(r'\d+', own_id)
    own_num_str = own_num.group() if own_num else ""

    def _replacer(m):
        if own_num_str and own_num_str in m.group():
            return m.group()
        return "(the standard)"

    return re.sub(r'\bIS[\s:.\-]?\d{2,6}\b', _replacer, text, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Template fallback (used only when chunk text is empty)
# ---------------------------------------------------------------------------

def _template_rationale_single(query: str, item: dict) -> str:
    """Last-resort template rationale built purely from metadata."""
    sid = item["standard_id"]
    title = item.get("title") or "this building material standard"
    category = item.get("category", "Building Materials")

    _templates = {
        "Cement":            f"{sid} specifies requirements for {title[:80]}, directly applicable to your cement query.",
        "Aggregates":        f"{sid} covers {title[:80]}, relevant to your aggregates compliance requirements.",
        "Concrete Products": f"{sid} provides specifications for {title[:80]}, applicable to your concrete product.",
        "Steel":             f"{sid} specifies requirements for {title[:80]}, relevant to your steel product query.",
        "Masonry":           f"{sid} covers {title[:80]}, applicable to your masonry product requirements.",
        "Roofing":           f"{sid} specifies {title[:80]}, directly relevant to your roofing material query.",
        "Pipes":             f"{sid} provides specifications for {title[:80]}, applicable to your pipe query.",
    }
    return _templates.get(
        category,
        f"{sid} — {title[:80]} — is relevant to your product based on material type and application."
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_retrieved = [
        {
            "standard_id": "IS 269",
            "title": "Specification for Ordinary Portland Cement, 33 Grade",
            "category": "Cement",
            "score": 0.95,
            "rationale_context": (
                "This standard covers 33 grade ordinary portland cement. "
                "Chemical requirements include lime saturation factor, silica ratio, "
                "alumina iron ratio. Physical tests cover fineness, setting time, "
                "soundness, and compressive strength at 3 and 7 days."
            ),
        },
        {
            "standard_id": "IS 1489",
            "title": "Portland Pozzolana Cement",
            "category": "Cement",
            "score": 0.88,
            "rationale_context": (
                "This standard covers Portland pozzolana cement made by intergrinding "
                "Portland cement clinker with pozzolanic materials. Suitable for structures "
                "exposed to sulphate attack and marine environments."
            ),
        },
    ]

    print("=== Extract mode (default) ===")
    results = generate_rationales(
        query="33 Grade Ordinary Portland Cement for RCC construction",
        retrieved=test_retrieved,
        use_llm=False,
    )
    for r in results:
        print(f"\n{r['standard_id']}:\n  {r['rationale']}")
