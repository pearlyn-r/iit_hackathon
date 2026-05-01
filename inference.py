"""
inference.py -- MANDATORY JUDGE ENTRY POINT

Usage:
    python inference.py --input hidden_private_dataset.json --output team_results.json

Input JSON format:
    [{"id": "q1", "query": "product description here"}, ...]

Output JSON format (STRICT -- judges score on retrieved_standards list):
    [
      {
        "id": "q1",
        "query": "product description here",           <- now included
        "retrieved_standards": ["IS 269", "IS 455", "IS 1489"],
        "latency_seconds": 0.83,
        "details": [                        <- bonus info, not scored
          {
            "standard_id": "IS 269",
            "title": "Ordinary Portland Cement, 33 Grade",
            "confidence": 0.91,
            "rationale": "Covers chemical and physical requirements..."
          }, ...
        ]
      }, ...
    ]
"""

#    path fix: must be at the very top, before any src imports               
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#                                                                            

import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.retriever import BISRetriever
from src.generator import generate_rationales
from src.compliance import build_extractive_rationale, generate_compliance_checklist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

#    Configuration                                                           
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INDEX_DIR = os.path.join(ROOT_DIR, "data", "index")
TMP_INDEX_DIR = "/tmp/index"
INDEX_DIR = (
    os.environ.get("BIS_INDEX_DIR")
    or (DATA_INDEX_DIR if os.path.exists(os.path.join(DATA_INDEX_DIR, "faiss.index")) else TMP_INDEX_DIR)
)
#HF_DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO_ID", "pearlyn/bis_compliance")
#HF_TOKEN = os.environ.get("HF_TOKEN")
INDEX_FILES = [
    "faiss.index",
    "bm25.pkl",
    "chunks.pkl",
    "idf.pkl",
    "whitelist.pkl",
    "config.json",
]
TOP_K_RETRIEVE   = 8      # retrieve more candidates so dedup still gives 5 unique
USE_LLM          = False
LLM_MODEL_PATH   = None
#                                                                           

_retriever = None


def _ensure_index_files(index_dir: str = INDEX_DIR) -> str:
    """Ensure FAISS/BM25 index artifacts are present locally.

    On Hugging Face Spaces the repo should define a secret named HF_TOKEN.
    The artifacts are downloaded from HF_DATASET_REPO_ID as repo_type='dataset'.
    """
    os.makedirs(index_dir, exist_ok=True)
    missing = [name for name in INDEX_FILES if not os.path.exists(os.path.join(index_dir, name))]
    if not missing:
        logger.info("Using BIS index artifacts from %s", index_dir)
        return index_dir

    logger.info(
        "Missing %s index artifact(s) in %s; downloading from dataset %s",
        len(missing),
        index_dir,
        HF_DATASET_REPO_ID,
    )
    for filename in missing:
        downloaded = hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        shutil.copyfile(downloaded, os.path.join(index_dir, filename))
        logger.info("Downloaded %s", filename)
    return index_dir

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = BISRetriever(_ensure_index_files())
    return _retriever


def _bare_standard_id(sid: str) -> str:
    """Normalize a standard ID to bare whitelist form, e.g. 'IS 2185 (Part 2): 1983' -> 'IS 2185'."""
    if not sid:
        return ""
    bare = str(sid).strip().upper()
    bare = re.sub(r'\(\s*PART\s*\d+\s*\)', '', bare, flags=re.IGNORECASE)
    bare = re.sub(r'\bPART\s*\d+\b', '', bare, flags=re.IGNORECASE)
    bare = re.sub(r'\s*:\s*\d{4}\b.*', '', bare)
    bare = re.sub(r'^IS\s*', 'IS ', bare).strip()
    bare = re.sub(r'\s+', ' ', bare).strip()
    return bare


def _dedup_standard_id(sid: str) -> str:
    """
    Normalize a standard ID for deduplication while preserving part numbers.

    Examples:
      - 'IS 2185 (Part 2): 1983' -> 'IS 2185 (PART 2)'
      - 'IS 269: 1989' -> 'IS 269'
    """
    if not sid:
        return ""
    norm = str(sid).strip().upper()
    norm = re.sub(r'\s*:\s*\d{4}\b.*', '', norm)
    norm = re.sub(r'^IS\s*', 'IS ', norm).strip()
    norm = re.sub(r'\s+', ' ', norm).strip()
    return norm


def _deduplicate(items: list) -> list:
    """
    Remove duplicate standard_ids, keeping the first (highest-ranked) occurrence.
    This is the mandatory fix: a standard should never appear twice in results.
    """
    seen = set()
    out  = []
    for item in items:
        sid = item.get("standard_id", "")
        if sid and sid not in seen:
            seen.add(sid)
            out.append(item)
    return out


def _compute_confidence(rank: int, rrf_score: float, n_results: int) -> float:
    """
    Convert retrieval rank + RRF score into a 0-1 confidence value.
    Formula: decay by rank position, clipped to [0.50, 0.99].
    Rank 1 -> ~0.95+, Rank 5 -> ~0.70.
    """
    base  = 0.98 - (rank * 0.055)          # linear decay per rank
    noise = min(rrf_score * 0.15, 0.04)    # small bonus for very high RRF score
    return round(max(0.50, min(0.99, base + noise)), 3)


def process_query(query: str) -> dict:
    retriever = _get_retriever()
    t0 = time.time()

    #    1. Retrieve (fetch extra so dedup still yields 5 unique)           
    chunks = retriever.retrieve(query, top_k=TOP_K_RETRIEVE)

    #    2. Deduplicate chunks by standard_id before generation             
    seen_ids  = set()
    dedup_chunks = []
    for chunk in chunks:
        sid = chunk.get("standard_id")
        # Also consider all_standard_ids so we don't return the same IS twice
        # via different chunk paths
        all_ids = chunk.get("all_standard_ids", [])
        primary = sid or (all_ids[0] if all_ids else None)
        primary_bare = _bare_standard_id(primary)
        primary_key = _dedup_standard_id(primary)
        if primary and primary_key not in seen_ids and primary_bare in retriever._bare_whitelist:
            seen_ids.add(primary_key)
            chunk["standard_id"] = primary   # ensure primary is set
            dedup_chunks.append(chunk)
        if len(dedup_chunks) >= 5:
            break

    #    3. Generate rationales                                             
    results = generate_rationales(query, dedup_chunks, use_llm=USE_LLM, model_path=LLM_MODEL_PATH)
    results = _deduplicate(results)   # second safety pass

    #    4. Build detail records with confidence scores                     
    details = []
    for rank, item in enumerate(results[:5]):
        rrf_score = item.get("score", 0.0)
        confidence = _compute_confidence(rank, rrf_score, len(results))
        rationale_details = build_extractive_rationale(query, item)
        details.append({
            "standard_id": item["standard_id"],
            "title":        item.get("title", "").strip(),
            "category":     item.get("category", ""),
            "year":         item.get("year"),
            "confidence":   confidence,
            "rationale":    rationale_details["summary"] or item.get("rationale", ""),
            "rationale_details": rationale_details,
            "explainability": item.get("_explainability", {}),
            "compliance_checklist": generate_compliance_checklist(item),
        })

    #    5. Plain list of IDs for the mandatory scored field                
    retrieved_standards = [d["standard_id"] for d in details]

    #    6. Pad to 3 minimum if retriever came up short                     
    if len(retrieved_standards) < 3:
        for chunk in chunks:
            for sid in chunk.get("all_standard_ids", []):
                sid_bare = _bare_standard_id(sid)
                retrieved_keys = {_dedup_standard_id(x) for x in retrieved_standards}
                sid_key = _dedup_standard_id(sid)
                if sid_key not in retrieved_keys and sid_bare in retriever._bare_whitelist:
                    fallback_item = {**chunk, "standard_id": sid}
                    rationale_details = build_extractive_rationale(query, fallback_item)
                    retrieved_standards.append(sid)
                    details.append({
                        "standard_id": sid,
                        "title":       chunk.get("title", "").strip(),
                        "category":    chunk.get("category", ""),
                        "year":        chunk.get("year"),
                        "confidence":  0.50,
                        "rationale":   rationale_details["summary"] or "Supplementary match from same document section.",
                        "rationale_details": rationale_details,
                        "explainability": chunk.get("_explainability", {}),
                        "compliance_checklist": generate_compliance_checklist(fallback_item),
                    })
                if len(retrieved_standards) >= 5:
                    break
            if len(retrieved_standards) >= 5:
                break

    return {
        "query":               query,                          # <- ADDED: echo query so API and CLI output both include it
        "retrieved_standards": retrieved_standards[:5],
        "latency_seconds":     round(time.time() - t0, 4),
        "details":             details[:5],
    }


def run_inference(input_path: str, output_path: str):
    logger.info(f"Reading input: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        queries = json.load(f)
    logger.info(f"Total queries: {len(queries)}")

    _get_retriever()   # warm up once

    output        = []
    total_latency = 0.0

    for i, item in enumerate(queries):
        qid   = item.get("id", str(i))
        query = item.get("query", "").strip()

        if not query:
            logger.warning(f"Empty query id={qid}")
            row = {"id": qid, "query": query, "retrieved_standards": [], "latency_seconds": 0.0, "details": []}  # <- ADDED: "query" key in empty-query fallback row
            if "expected_standards" in item:
                row["expected_standards"] = item["expected_standards"]
            output.append(row)
            continue

        try:
            result = process_query(query)  # result now contains "query" automatically
            row = {"id": qid, **result}    # "query" flows through via **result  -  no extra change needed here
            if "expected_standards" in item:
                row["expected_standards"] = item["expected_standards"]
            output.append(row)
            total_latency += result["latency_seconds"]
            logger.info(
                f"[{i+1}/{len(queries)}] {qid} | "
                f"{result['retrieved_standards']} | "
                f"{result['latency_seconds']:.3f}s"
            )
        except Exception as e:
            logger.error(f"Error on id={qid}: {e}", exc_info=True)
            row = {"id": qid, "query": query, "retrieved_standards": [], "latency_seconds": 0.0, "details": []}  # <- ADDED: "query" key in error fallback row
            if "expected_standards" in item:
                row["expected_standards"] = item["expected_standards"]
            output.append(row)

    avg = total_latency / len(queries) if queries else 0
    logger.info(f"Done. Avg latency: {avg:.3f}s")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)
    logger.info(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="BIS Standards RAG Inference")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_inference(args.input, args.output)


if __name__ == "__main__":
    main()
