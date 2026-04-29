"""
setup_pipeline.py — Run this ONCE to ingest the PDF and build all indices.

Usage:
    python setup_pipeline.py --pdf data/bis_sp21.pdf

Steps:
    1. Parse PDF -> structured chunks (data/chunks.json)
    2. Build FAISS + BM25 index (data/index/)

After this completes, run inference.py for queries.
"""

# ── path fix: must be at the very top, before any src imports ──────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ───────────────────────────────────────────────────────────────────────────

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="BIS RAG Setup Pipeline")
    parser.add_argument("--pdf", required=True, help="Path to BIS SP 21 PDF")
    parser.add_argument("--chunks-out", default="data/chunks.json", help="Output path for chunks")
    parser.add_argument("--index-dir", default="data/index", help="Output directory for index")
    parser.add_argument(
        "--embed-model",
        default="BAAI/bge-large-en-v1.5",
        help="Sentence transformer model for embeddings. "
             "Use BAAI/bge-small-en-v1.5 if RAM/GPU is limited."
    )
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        logger.error(f"PDF not found: {args.pdf}")
        sys.exit(1)

    # Step 1: Ingest PDF
    logger.info("=" * 60)
    logger.info("STEP 1: PDF Ingestion")
    logger.info("=" * 60)
    from src.ingestion import build_chunks
    chunks = build_chunks(args.pdf, args.chunks_out)
    logger.info(f"Ingestion complete. {len(chunks)} chunks created.")

    # Step 2: Build index
    logger.info("=" * 60)
    logger.info("STEP 2: Building FAISS + BM25 Index")
    logger.info("=" * 60)
    from src.indexer import build_index
    build_index(args.chunks_out, args.index_dir, model_name=args.embed_model)
    logger.info("Index building complete.")

    logger.info("=" * 60)
    logger.info("SETUP COMPLETE. You can now run inference:")
    logger.info("  python inference.py --input data/public_test_set.json --output data/results.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()