"""
eval_local.py — Local evaluation script for debugging Hit Rate @3 and MRR @5.

Usage:
    python eval_local.py --results data/results.json --ground-truth data/public_test_set_gt.json

Ground truth format:
    [{"id": "q1", "expected_standards": ["IS 269", "IS 12269"]}, ...]

Results format (from inference.py):
    [{"id": "q1", "retrieved_standards": ["IS 269", "IS 455"], "latency_seconds": 0.83}, ...]
"""

import argparse
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_id(sid: str) -> str:
    """Normalize IS codes for comparison."""
    import re
    m = re.search(r'\d{2,6}', str(sid))
    return f"IS {m.group()}" if m else sid.strip()


def compute_hit_rate_at_k(results: List[Dict], ground_truth: Dict, k: int = 3) -> float:
    """Hit Rate @k: % of queries where at least 1 expected standard in top-k."""
    hits = 0
    total = 0

    for item in results:
        qid = str(item["id"])
        if qid not in ground_truth:
            continue
        total += 1

        expected = set(normalize_id(s) for s in ground_truth[qid])
        retrieved = [normalize_id(s) for s in item["retrieved_standards"][:k]]

        if expected & set(retrieved):
            hits += 1

    rate = (hits / total * 100) if total > 0 else 0
    return rate, hits, total


def compute_mrr_at_k(results: List[Dict], ground_truth: Dict, k: int = 5) -> float:
    """MRR @k: Mean Reciprocal Rank of first correct standard in top-k."""
    rr_sum = 0.0
    total = 0

    for item in results:
        qid = str(item["id"])
        if qid not in ground_truth:
            continue
        total += 1

        expected = set(normalize_id(s) for s in ground_truth[qid])
        retrieved = [normalize_id(s) for s in item["retrieved_standards"][:k]]

        rr = 0.0
        for rank, sid in enumerate(retrieved, 1):
            if sid in expected:
                rr = 1.0 / rank
                break
        rr_sum += rr

    mrr = rr_sum / total if total > 0 else 0
    return mrr, total


def compute_avg_latency(results: List[Dict]) -> float:
    latencies = [r.get("latency_seconds", 0) for r in results]
    return sum(latencies) / len(latencies) if latencies else 0


def analyze_failures(results: List[Dict], ground_truth: Dict, k: int = 3):
    """Print details on queries that missed."""
    print("\n" + "="*60)
    print("FAILURE ANALYSIS (queries that missed Hit@3):")
    print("="*60)
    for item in results:
        qid = str(item["id"])
        if qid not in ground_truth:
            continue
        expected = set(normalize_id(s) for s in ground_truth[qid])
        retrieved = set(normalize_id(s) for s in item["retrieved_standards"][:k])
        if not (expected & retrieved):
            print(f"\nQuery ID: {qid}")
            print(f"  Expected:  {sorted(expected)}")
            print(f"  Retrieved: {sorted(retrieved)}")


def main():
    parser = argparse.ArgumentParser(description="Local evaluation for BIS RAG")
    parser.add_argument("--results", required=True, help="Path to inference output JSON")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--analyze-failures", action="store_true", help="Show failure analysis")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    with open(args.ground_truth) as f:
        gt_raw = json.load(f)

    # Build ground truth dict: {id: [expected_standards]}
    gt = {}
    for item in gt_raw:
        gt[str(item["id"])] = item.get("expected_standards", item.get("standards", []))

    hit_rate, hits, total = compute_hit_rate_at_k(results, gt, k=3)
    mrr, _ = compute_mrr_at_k(results, gt, k=5)
    avg_lat = compute_avg_latency(results)

    print("\n" + "="*60)
    print("BIS RAG EVALUATION RESULTS")
    print("="*60)
    print(f"Total queries evaluated: {total}")
    print(f"\nHit Rate @3:    {hit_rate:.1f}%  (target: >80%) {'✅' if hit_rate > 80 else '❌'}")
    print(f"MRR @5:         {mrr:.4f}   (target: >0.7)  {'✅' if mrr > 0.7 else '❌'}")
    print(f"Avg Latency:    {avg_lat:.3f}s   (target: <5s)   {'✅' if avg_lat < 5 else '❌'}")
    print("="*60)

    if args.analyze_failures:
        analyze_failures(results, gt, k=3)


if __name__ == "__main__":
    main()
