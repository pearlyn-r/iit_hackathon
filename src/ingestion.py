"""
BIS SP 21 PDF Ingestion Pipeline
Extracts text + tables, identifies standard blocks, outputs structured chunks.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Regex to detect BIS standard IDs in various formats
STANDARD_ID_PATTERN = re.compile(
    r'\bIS[\s:.\-]?(\d{2,6})(?:[\s:.\-]?(?:Part[\s:.\-]?\d+)?(?:Section[\s:.\-]?\d+)?)?\b',
    re.IGNORECASE
)

# Normalize IS codes → "IS XXXX" canonical form
def normalize_standard_id(raw: str) -> str:
    digits = re.search(r'\d{2,6}', raw)
    if digits:
        return f"IS {digits.group()}"
    return raw.strip()


def extract_all_standard_ids(text: str) -> List[str]:
    """Extract and normalize all standard IDs from a text block."""
    matches = STANDARD_ID_PATTERN.findall(text)
    ids = list(set(f"IS {m.strip()}" for m in matches if m.strip()))
    return ids


def table_to_text(table: List[List]) -> str:
    """Convert pdfplumber table (list of rows) to a readable text block."""
    if not table:
        return ""
    lines = []
    headers = [str(h).strip() if h else "" for h in table[0]]
    for row in table[1:]:
        if not row:
            continue
        parts = []
        for h, cell in zip(headers, row):
            cell_str = str(cell).strip() if cell else ""
            if cell_str:
                parts.append(f"{h}: {cell_str}" if h else cell_str)
        if parts:
            lines.append(" | ".join(parts))
    return "[TABLE] " + " || ".join(lines)


def extract_pages(pdf_path: str) -> List[Dict]:
    """
    Extract all pages from PDF.
    Returns list of {page_num, text, tables_text}.
    """
    logger.info(f"Opening PDF: {pdf_path}")
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        logger.info(f"Total pages: {total}")
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                tables = page.extract_tables()
                tables_text = "\n".join(table_to_text(t) for t in tables if t)
                combined = text
                if tables_text:
                    combined += "\n" + tables_text
                pages.append({
                    "page_num": i + 1,
                    "text": combined.strip()
                })
                if (i + 1) % 100 == 0:
                    logger.info(f"  Processed {i+1}/{total} pages...")
            except Exception as e:
                logger.warning(f"  Page {i+1} error: {e}")
                pages.append({"page_num": i + 1, "text": ""})
    return pages


def split_into_standard_blocks(pages: List[Dict]) -> List[Dict]:
    """
    Heuristically split pages into one chunk per BIS standard.
    Strategy: Detect IS XXXX heading lines as block boundaries.
    """
    # Merge all page text with page markers
    full_text_with_pages = []
    for p in pages:
        if p["text"]:
            full_text_with_pages.append((p["page_num"], p["text"]))

    # Pattern: lines that START with IS followed by a number (section headers)
    # e.g. "IS 269 : 2015 ORDINARY PORTLAND CEMENT"
    BLOCK_START = re.compile(
        r'^(?:IS[\s:.\-]?\d{2,6}[^\n]{0,120})\n',
        re.MULTILINE | re.IGNORECASE
    )

    # Combine all text
    all_text = ""
    page_offsets = []  # (char_offset, page_num)
    for pnum, txt in full_text_with_pages:
        page_offsets.append((len(all_text), pnum))
        all_text += txt + "\n\n"

    # Find all block starts
    block_starts = [(m.start(), m.group()) for m in BLOCK_START.finditer(all_text)]

    if len(block_starts) < 10:
        logger.warning("Few block starts found. Falling back to page-based chunking.")
        return _page_based_chunks(full_text_with_pages)

    logger.info(f"Found {len(block_starts)} standard blocks via regex.")

    blocks = []
    for i, (start, header) in enumerate(block_starts):
        end = block_starts[i + 1][0] if i + 1 < len(block_starts) else len(all_text)
        chunk_text = all_text[start:end].strip()

        # Find which page this starts on
        page_num = 1
        for offset, pnum in page_offsets:
            if offset <= start:
                page_num = pnum

        standard_ids = extract_all_standard_ids(chunk_text[:200])  # Focus on header
        primary_id = standard_ids[0] if standard_ids else None

        blocks.append({
            "chunk_id": f"block_{i}",
            "standard_id": primary_id,
            "all_standard_ids": standard_ids,
            "page_num": page_num,
            "header": header.strip(),
            "text": chunk_text,
            "token_count": len(chunk_text.split())
        })

    return blocks


def _page_based_chunks(pages_text: List) -> List[Dict]:
    """Fallback: chunk by page if block detection fails."""
    chunks = []
    for i, (pnum, txt) in enumerate(pages_text):
        sids = extract_all_standard_ids(txt)
        chunks.append({
            "chunk_id": f"page_{pnum}",
            "standard_id": sids[0] if sids else None,
            "all_standard_ids": sids,
            "page_num": pnum,
            "header": f"Page {pnum}",
            "text": txt,
            "token_count": len(txt.split())
        })
    return chunks


# Lines that are structural noise in BIS SP 21, not titles
_TITLE_SKIP = [
    re.compile(r'^\(.*revision.*\)$',       re.IGNORECASE),
    re.compile(r'^\(.*amendment.*\)$',      re.IGNORECASE),
    re.compile(r'^IS\s*[\d:.\-\s]+$',       re.IGNORECASE),
    re.compile(r'^\d{4}$'),
    re.compile(r'^SP\s*\d+',               re.IGNORECASE),
    re.compile(r'^page\s*\d+',             re.IGNORECASE),
    re.compile(r'^\d+\.\d+$'),
    re.compile(r'^summary\s+of',           re.IGNORECASE),
]

def _is_skip_line(s: str) -> bool:
    return len(s) < 4 or any(p.match(s) for p in _TITLE_SKIP)

def _extract_title(text: str) -> str:
    """
    Extract the human-readable title from a BIS standard chunk.

    BIS SP 21 format:
      IS 269 : 1989 ORDINARY PORTLAND CEMENT          <- IS code + title on one line
      (Fourth Revision)                               <- revision marker (skip)
      1. Scope — ...                                  <- stop here

    Or wrapped:
      IS 459 : 1992 CORRUGATED AND SEMI-CORRUGATED ASBESTOS
      CEMENT SHEETS                                   <- title continuation
      (Third Revision)

    Strategy: find header line, grab text after IS code, then scan
    forward for continuation lines that are ALL-CAPS title fragments.
    Stop at revision markers, scope/clause lines.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    header_idx = 0
    title_parts: List[str] = []

    for i, line in enumerate(lines):
        m = re.match(
            r'^IS[\s:.\-]?\d{2,6}(?:[\s:.\-]+(?:Part[\s:.\-]?\d+)?)?'
            r'(?:\s*:\s*(?:19|20)\d{2})?\s*(.*)',
            line, re.IGNORECASE
        )
        if m:
            header_idx = i
            after = re.sub(r'\s*:?\s*(19|20)\d{2}\s*$', '', m.group(1).strip()).strip()
            if len(after) >= 3 and not _is_skip_line(after):
                title_parts.append(after)
            break

    for line in lines[header_idx + 1 : header_idx + 6]:
        s = line.strip()
        if _is_skip_line(s):
            continue
        if re.match(r'^\d+[\.)]', s) or re.match(r'^(scope|for\s+detail)', s, re.IGNORECASE):
            break
        if len(s) >= 3:
            title_parts.append(s)
            if len(title_parts) >= 2:
                break

    title = ' '.join(title_parts).strip()
    title = re.sub(r'\s+', ' ', title)
    title = re.sub(r'\s*\(.*?revision.*?\)', '', title, flags=re.IGNORECASE).strip()
    # Strip leading bare year ("1989 ORDINARY..." -> "ORDINARY...")
    title = re.sub(r'^\d{4}\s+', '', title).strip()
    return title[:200] if len(title) >= 4 else ""


def extract_metadata(block: Dict) -> Dict:
    """
    Extract rich metadata from a chunk for filtering and display.
    """
    text = block["text"]

    # Extract year
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    year = int(year_match.group()) if year_match else None

    title = _extract_title(text)

    # Category detection keywords
    CATEGORIES = {
        "Cement": ["cement", "cementitious", "portland", "pozzolana", "slag"],
        "Steel": ["steel", "iron", "reinforcement", "bar", "rod", "wire", "structural"],
        "Concrete": ["concrete", "rcc", "pcc", "admixture", "aggregate"],
        "Aggregates": ["aggregate", "sand", "gravel", "crushed", "coarse", "fine"],
        "Bricks": ["brick", "tile", "masonry", "clay", "block"],
        "Timber": ["timber", "wood", "plywood", "particle board"],
        "Glass": ["glass", "glazing"],
        "Paint": ["paint", "varnish", "coating", "primer"],
        "Pipes": ["pipe", "tube", "conduit", "plumbing"],
        "Flooring": ["flooring", "mosaic", "terrazzo", "marble"],
    }
    text_lower = text.lower()
    category = "General"
    for cat, keywords in CATEGORIES.items():
        if any(k in text_lower for k in keywords):
            category = cat
            break

    # Extract keywords (top words from text, skip stopwords)
    STOPWORDS = {"the", "a", "an", "of", "for", "in", "is", "to", "and", "or",
                 "shall", "be", "this", "with", "at", "by", "as", "on", "that"}
    words = re.findall(r'\b[a-z]{4,}\b', text_lower)
    from collections import Counter
    freq = Counter(w for w in words if w not in STOPWORDS)
    keywords = [w for w, _ in freq.most_common(10)]

    block["title"] = title[:200]
    block["year"] = year
    block["category"] = category
    block["keywords"] = keywords
    return block


def build_chunks(pdf_path: str, output_path: str) -> List[Dict]:
    """Full ingestion pipeline: PDF → structured chunks saved to JSON."""
    pages = extract_pages(pdf_path)
    blocks = split_into_standard_blocks(pages)
    
    enriched = []
    for b in blocks:
        b = extract_metadata(b)
        enriched.append(b)

    # Filter empty chunks
    enriched = [b for b in enriched if len(b["text"].split()) > 20]

    logger.info(f"Total chunks after processing: {len(enriched)}")
    
    # Build global standard ID whitelist
    whitelist = set()
    for b in enriched:
        whitelist.update(b["all_standard_ids"])

    output = {
        "chunks": enriched,
        "standard_id_whitelist": sorted(whitelist),
        "total_chunks": len(enriched),
        "total_standards": len(whitelist)
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {len(enriched)} chunks to {output_path}")
    logger.info(f"Whitelist has {len(whitelist)} unique standard IDs")
    return enriched


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.pdf"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/chunks.json"
    build_chunks(pdf, out)