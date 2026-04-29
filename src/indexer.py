"""
Index Builder
=============
Builds FAISS dense index + BM25 sparse index + IDF store from chunks.
Also enriches each chunk with a structured `material_type` field.

New vs previous version:
  - Calls IDFStore.build() and saves idf.pkl alongside the other index files.
  - Calls enrich_material_type() on every chunk before embedding, so the
    retriever can use material_type for metadata filtering at query time.
"""

import json
import logging
import pickle
import re
import math
import numpy as np
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Material-type enrichment ───────────────────────────────────────────────
# Maps material_type label → words that strongly indicate this type.
# Evaluated against chunk title + first 200 chars of text.
# Order matters: more specific entries should come first.
MATERIAL_TYPE_RULES: List[Tuple[str, List[str], List[str]]] = [
    # (label, required_any, excluded_any)
    ("asbestos_cement",        ["asbestos"],                         ["fibre", "fiber"]),
    ("fibre_cement",           ["fibre", "fiber", "grc", "frc"],     ["asbestos"]),
    ("white_cement",           ["white cement", "white portland"],   []),
    ("rapid_hardening_cement", ["rapid hardening", "rhpc"],          []),
    ("portland_cement_opc",    ["ordinary portland", "opc"],         ["slag", "pozzolana", "fly ash", "white", "rapid"]),
    ("blended_cement",         ["pozzolana", "slag cement", "fly ash", "ppc", "psc"], []),
    ("steel_rebar",            ["tmt", "reinforcement bar", "deformed bar", "tor steel"], ["structural", "wire rope"]),
    ("structural_steel",       ["structural steel", "plates", "angles", "channels", "joists"], ["rebar", "reinforcement"]),
    ("wire",                   ["binding wire", "drawn wire", "galvanised wire"], []),
    ("aggregate",              ["coarse aggregate", "fine aggregate", "natural sand", "crushed stone"], []),
    ("brick_masonry",          ["burnt clay brick", "hollow block", "aac block", "autoclaved"],  []),
    ("pipe",                   ["asbestos cement pipe", "pressure pipe", "water pipe", "drain pipe"], []),
    ("roofing_sheet",          ["corrugated sheet", "roofing sheet", "cladding"],  []),
    ("concrete_product",       ["precast", "prestressed", "concrete pipe", "concrete block"], []),
]


def enrich_material_type(chunk: Dict) -> str:
    """
    Assign a material_type label to a chunk based on its title + text.
    Returns the label string (or "general" if nothing matches).
    Stored in chunk["material_type"] in-place and returned.
    """
    probe = (
        chunk.get("title", "") + " " +
        chunk.get("text",  "")[:300]
    ).lower()

    for label, required, excluded in MATERIAL_TYPE_RULES:
        if any(kw in probe for kw in excluded):
            continue
        if any(kw in probe for kw in required):
            chunk["material_type"] = label
            return label

    chunk["material_type"] = "general"
    return "general"


# ── IDF store (mirrors class in retriever.py) ──────────────────────────────

_STOPWORDS = {
    "the","a","an","of","for","in","is","to","and","or","be","this","with",
    "at","by","as","on","that","are","its","it","was","from","has","have",
    "not","shall","which","also","such","may","all","any","been","their",
    "they","more","used","when","where","while","than","into","these","those",
    "can","will","specification","standard","indian","bureau","summary",
    "revision","clause","refer","covers","cover","provides","applicable",
}

def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower())
            if w not in _STOPWORDS]


def build_idf(chunks: List[Dict]) -> Dict[str, float]:
    """Compute IDF scores from corpus. Returns word → idf dict."""
    n_docs = len(chunks)
    df: Counter = Counter()
    for chunk in chunks:
        text  = chunk.get("title", "") + " " + chunk.get("text", "")
        words = set(_tokens(text))
        df.update(words)
    return {
        w: math.log((n_docs + 1) / (count + 1)) + 1.0
        for w, count in df.items()
    }


# ── Embedding text builder ─────────────────────────────────────────────────

def build_embedding_text(chunk: Dict) -> str:
    """Construct the passage text to embed. Includes metadata for semantic coverage."""
    parts = []
    if chunk.get("standard_id"):
        parts.append(f"Standard: {chunk['standard_id']}")
    if chunk.get("title"):
        parts.append(f"Title: {chunk['title']}")
    if chunk.get("category"):
        parts.append(f"Category: {chunk['category']}")
    if chunk.get("material_type") and chunk["material_type"] != "general":
        parts.append(f"Type: {chunk['material_type'].replace('_', ' ')}")
    if chunk.get("keywords"):
        parts.append(f"Keywords: {', '.join(chunk['keywords'][:6])}")
    words = chunk.get("text", "").split()[:400]
    parts.append(" ".join(words))
    return " | ".join(parts)


def build_bm25_text(chunk: Dict) -> str:
    """BM25 text — repeat standard_id for keyword matching boost."""
    parts = []
    if chunk.get("standard_id"):
        parts.extend([chunk["standard_id"]] * 3)
    if chunk.get("all_standard_ids"):
        parts.extend(chunk["all_standard_ids"])
    if chunk.get("title"):
        parts.append(chunk["title"])
    if chunk.get("material_type"):
        parts.append(chunk["material_type"].replace("_", " "))
    if chunk.get("keywords"):
        parts.extend(chunk["keywords"])
    parts.append(chunk.get("text", "")[:2000])
    return " ".join(parts)


# ── Main index builder ─────────────────────────────────────────────────────

def load_chunks(chunks_path: str):
    with open(chunks_path) as f:
        data = json.load(f)
    chunks   = data["chunks"]
    whitelist = set(data.get("standard_id_whitelist", []))
    logger.info(f"Loaded {len(chunks)} chunks, {len(whitelist)} whitelisted standards.")
    return chunks, whitelist


def build_index(
    chunks_path: str,
    index_dir:   str,
    model_name:  str = "BAAI/bge-large-en-v1.5",
):
    """
    Full index build pipeline.
    Saves to index_dir/:
      faiss.index   — FAISS inner-product index
      bm25.pkl      — BM25Okapi object
      idf.pkl       — IDF corpus statistics       ← NEW
      chunks.pkl    — chunk list (with material_type enriched) ← NEW field
      whitelist.pkl — standard ID whitelist
      config.json   — metadata
    """
    import faiss
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    chunks, whitelist = load_chunks(chunks_path)

    # ── Enrich material_type on every chunk ───────────────────────────────
    logger.info("Enriching chunks with material_type metadata...")
    type_counts: Counter = Counter()
    for chunk in chunks:
        mtype = enrich_material_type(chunk)
        type_counts[mtype] += 1
    logger.info(f"Material type distribution: {dict(type_counts.most_common(10))}")

    # ── Build IDF store ───────────────────────────────────────────────────
    logger.info("Building IDF corpus statistics...")
    idf = build_idf(chunks)
    idf_payload = {"idf": idf, "n_docs": len(chunks)}
    with open(f"{index_dir}/idf.pkl", "wb") as f:
        pickle.dump(idf_payload, f)
    logger.info(f"IDF computed for {len(idf)} unique terms.")

    # ── BM25 index ────────────────────────────────────────────────────────
    logger.info("Building BM25 index...")
    bm25_texts = [build_bm25_text(c) for c in chunks]
    tokenized  = [t.lower().split() for t in bm25_texts]
    bm25       = BM25Okapi(tokenized)
    with open(f"{index_dir}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    logger.info("BM25 index built.")

    # ── Dense embedding index ─────────────────────────────────────────────
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.warning(f"Failed to load {model_name}: {e}. Falling back to bge-small.")
        model      = SentenceTransformer("BAAI/bge-small-en-v1.5")
        model_name = "BAAI/bge-small-en-v1.5"

    PASSAGE_PREFIX = "Represent this document for searching relevant passages: "
    embed_texts = [PASSAGE_PREFIX + build_embedding_text(c) for c in chunks]

    logger.info(f"Embedding {len(embed_texts)} chunks (batch_size=32)...")
    batch_size      = 32
    all_embeddings  = []
    for i in range(0, len(embed_texts), batch_size):
        batch = embed_texts[i : i + batch_size]
        embs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs)
        if (i // batch_size + 1) % 20 == 0:
            logger.info(f"  Embedded {min(i+batch_size, len(embed_texts))}/{len(embed_texts)}")

    embeddings = np.vstack(all_embeddings).astype("float32")
    logger.info(f"Embeddings shape: {embeddings.shape}")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, f"{index_dir}/faiss.index")
    logger.info(f"FAISS index built: {index.ntotal} vectors.")

    # ── Save chunks (now enriched with material_type) ─────────────────────
    with open(f"{index_dir}/chunks.pkl",   "wb") as f: pickle.dump(chunks,    f)
    with open(f"{index_dir}/whitelist.pkl","wb") as f: pickle.dump(whitelist, f)

    config = {
        "model_name":      model_name,
        "embedding_dim":   dim,
        "total_chunks":    len(chunks),
        "total_standards": len(whitelist),
        "material_types":  dict(type_counts),
    }
    with open(f"{index_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("All indices saved successfully.")
    logger.info(f"  → {index_dir}/faiss.index")
    logger.info(f"  → {index_dir}/bm25.pkl")
    logger.info(f"  → {index_dir}/idf.pkl   ← new")
    logger.info(f"  → {index_dir}/chunks.pkl (with material_type enrichment)")
    return True


if __name__ == "__main__":
    import sys
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "data/chunks.json"
    index_dir   = sys.argv[2] if len(sys.argv) > 2 else "data/index"
    build_index(chunks_path, index_dir)