"""
Retrieval Engine  -  Production Architecture
==========================================

Pipeline (in order):
  1. Query expansion               -  abbreviations + synonym normalisation
  2. Dense retrieval               -  FAISS + BGE embeddings
  3. Sparse retrieval              -  BM25
  4. RRF merge                     -  weighted reciprocal rank fusion
  5. Exact IS-code boost           -  hard boost when query contains explicit IS number
  6. Material-type metadata boost  -  IDF-learned corpus classification
  7. Mandatory match gate          -  safety net after material boost
  8. IDF-weighted title F1         -  rare discriminative words weighted higher
  9. Domain keyword boost          -  term-weight overlap between query and chunk
 10. Co-occurrence bonus           -  paired terms (asbestos+cement) as hard discriminator
 11. Negative material penalty     -  suppresses wrong-material near-duplicates
 12. Cross-encoder reranker        -  ms-marco-MiniLM-L-6-v2

Whitelist guarantee: every chunk's standard_id is validated against the
whitelist of IS codes extracted from the actual PDF at index time.
"""

import re
import json
import math
import logging
import pickle
import time
import numpy as np
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

logger = logging.getLogger(__name__)

QUERY_PREFIX        = "Represent this sentence for searching relevant passages: "
STANDARD_ID_PATTERN = re.compile(r'\bIS[\s:.\-]?(\d{2,6})\b', re.IGNORECASE)

_STOPWORDS = {
    "the","a","an","of","for","in","is","to","and","or","be","this","with",
    "at","by","as","on","that","are","its","it","was","from","has","have",
    "not","shall","which","also","such","may","all","any","been","their",
    "they","more","used","when","where","while","than","into","these","those",
    "can","will","specification","standard","indian","bureau","summary",
    "revision","clause","refer","covers","cover","provides","applicable",
}

#    Query expansion map                                                        
QUERY_EXPANSION_MAP: Dict[str, str] = {
    r"\bopc\b":              "ordinary portland cement",
    r"\brcc\b":              "reinforced cement concrete",
    r"\bpcc\b":              "plain cement concrete",
    r"\baac\b":              "autoclaved aerated concrete",
    r"\btmt\b":              "thermo mechanically treated steel",
    r"\bpvc\b":              "polyvinyl chloride",
    r"\bhdpe\b":             "high density polyethylene",
    r"\bpsc\b":              "portland slag cement",
    r"\bppc\b":              "portland pozzolana cement",
    r"\brhpc\b":             "rapid hardening portland cement",
    r"\bfe\s*500\b":         "high strength deformed steel bars fe500",
    r"\bfe\s*415\b":         "high strength deformed steel bars fe415",
    r"\bm\s*25\b":           "concrete mix m25 grade",
    r"\bm\s*30\b":           "concrete mix m30 grade",
}

#    Domain keyword weights                                                     
DOMAIN_KEYWORD_WEIGHTS: Dict[str, float] = {
    "cement":       1.0,  "portland":    1.5,  "clinker":     1.5,
    "slag":         1.5,  "fly ash":     1.5,  "pozzolana":   1.5,
    "asbestos":     2.5,  "corrugated":  1.5,  "roofing":     1.5,
    "sheets":       1.0,  "cladding":    1.2,  "fibre":       2.0,
    "fiber":        2.0,  "reinforced":  1.2,
    "aggregate":    1.5,  "coarse":      1.0,  "fine":        1.0,
    "sand":         1.2,  "gravel":      1.2,  "crushed":     1.0,
    "reinforcement":1.5,  "rebar":       1.5,  "bars":        1.0,
    "structural":   1.0,  "steel":       1.0,  "deformed":    1.5,
    "brick":        1.5,  "masonry":     1.5,  "mortar":      1.2,
    "tile":         1.2,  "block":       1.0,
    "pipe":         1.2,  "plumbing":    1.2,  "drainage":    1.2,
    "water supply": 1.5,  "pressure":    1.0,
}

#    Co-occurrence bonuses                                                      
CO_OCCURRENCE_BONUSES: List[Tuple[Tuple[str, str], float]] = [
    (("asbestos", "cement"),        0.08),
    (("corrugated", "asbestos"),    0.06),
    (("portland", "slag"),          0.05),
    (("fly ash", "cement"),         0.04),
    (("fibre", "cement"),           0.06),
    (("fiber", "cement"),           0.06),
]

#    Negative penalty rules                                                     
NEGATIVE_PENALTY_RULES: List[Dict] = [
    {
        "name": "fibre_not_asbestos",
        "query_signals":       ["asbestos"],
        "chunk_must_have":     ["fibre reinforced", "fiber reinforced", "fibre-reinforced"],
        "chunk_must_not_have": ["asbestos"],
        "penalty": 0.15,
    },
    {
        "name": "asbestos_not_fibre_when_fibre_queried",
        "query_signals":       ["fibre reinforced", "fiber reinforced", "non-asbestos"],
        "chunk_must_have":     ["asbestos cement"],
        "chunk_must_not_have": ["fibre reinforced", "fiber reinforced"],
        "penalty": 0.10,
    },
]

#    Material-type signals                                                      
MATERIAL_TYPE_SIGNALS: Dict[str, Dict[str, Set[str]]] = {
    "asbestos_cement":        {"pos": {"asbestos", "asbesto"},                      "neg": {"fibre", "fiber", "grc"}},
    "fibre_cement":           {"pos": {"fibre", "fiber", "grc", "frc"},             "neg": {"asbestos"}},
    "portland_cement_opc":    {"pos": {"portland", "opc", "ordinary", "33", "43", "53"}, "neg": {"slag", "pozzolana", "fly", "ash", "white", "rapid"}},
    "blended_cement":         {"pos": {"slag", "pozzolana", "ppc", "psc", "fly", "ash"}, "neg": {"ordinary", "opc", "white", "rapid"}},
    "white_cement":           {"pos": {"white"},                                    "neg": {"grey", "ordinary", "opc", "slag"}},
    "rapid_hardening_cement": {"pos": {"rapid", "hardening", "rhpc"},               "neg": {"ordinary", "pozzolana"}},
    "steel_rebar":            {"pos": {"tmt", "rebar", "reinforcement", "bar", "rod", "500", "415"}, "neg": {"structural", "plate", "wire rope", "pipe", "pipes", "concrete"}},
    "structural_steel":       {"pos": {"structural", "plate", "angle", "channel", "beam"}, "neg": {"rebar", "tmt", "reinforcement"}},
    "aggregate":              {"pos": {"aggregate", "gravel", "crushed", "stone", "sand"}, "neg": {"cement", "steel", "brick"}},
    "brick_masonry":          {"pos": {"brick", "masonry", "clay", "burnt", "aac"}, "neg": {"cement", "steel"}},
    "concrete_product":       {"pos": {"precast", "concrete", "block", "blocks", "frame", "frames"}, "neg": {"steel", "aggregate"}},
    "pipe":                   {"pos": {"pipe", "pipes", "tube", "tubes", "pressure", "sewer", "sewers", "culvert", "culverts"}, "neg": {"sheet", "rod", "bar", "brick"}},
}

CONFLICT_PENALTY      = -0.30
MATCH_BOOST           =  0.30
CONFIDENCE_THRESHOLD  =  0.35


#    Utility functions                                                          

def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in _STOPWORDS]

def _token_set(text: str) -> Set[str]:
    return set(_tokens(text))

def _round_float(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)

def _score_weightage_percent(breakdown: Dict[str, Optional[float]]) -> Dict[str, float]:
    raw = {
        "rrf": max(0.0, float(breakdown.get("rrf_score") or 0.0)),
        "is_code": max(0.0, float(breakdown.get("is_code_boost") or 0.0)),
        "material_type": max(0.0, float(breakdown.get("material_type_boost") or 0.0)),
        "idf_title": max(0.0, float(breakdown.get("idf_title_component") or 0.0)),
        "keyword_overlap": max(0.0, float(breakdown.get("keyword_overlap_boost") or 0.0)),
        "cooccurrence": max(0.0, float(breakdown.get("cooccurrence_boost") or 0.0)),
        "cross_encoder": max(0.0, float(breakdown.get("cross_encoder_score") or 0.0)),
    }
    total = sum(raw.values())
    if total <= 0:
        return {key: 0.0 for key in raw}
    return {key: round((value / total) * 100, 1) for key, value in raw.items()}

def normalize_query_standard_ids(query: str) -> List[str]:
    return [f"IS {m}" for m in STANDARD_ID_PATTERN.findall(query)]

def format_standard_id(sid: str, year: Optional[int]) -> str:
    """Return canonical 'IS XXXX : YYYY' or 'IS XXXX' if no year."""
    base = re.sub(r'[\s:.\-]+', ' ', sid).strip().upper()
    if not base.startswith("IS "):
        base = "IS " + re.sub(r'^IS\s*', '', base, flags=re.IGNORECASE).strip()
    return f"{base} : {year}" if year else base


def format_chunk_standard_id(chunk: Dict, bare_sid: str, year: Optional[int]) -> str:
    """
    Build the eval-facing standard ID from chunk metadata.

    This preserves part identifiers for standards like:
      - IS 2185 (Part 2): 1983
      - IS 1489 (Part 2): 1991

    The indexed whitelist still uses bare IDs (e.g. "IS 2185"), so retrieval
    logic should continue to validate against bare IDs only.
    """
    base = re.sub(r'[\s:.\-]+', ' ', bare_sid).strip().upper()
    if not base.startswith("IS "):
        base = "IS " + re.sub(r'^IS\s*', '', base, flags=re.IGNORECASE).strip()

    probe = " ".join([
        str(chunk.get("header", "")),
        str(chunk.get("title", "")),
        str(chunk.get("text", ""))[:300],
    ])
    part_match = re.search(r'\(\s*PART\s*(\d+)\s*\)|\bPART\s*(\d+)\b', probe, re.IGNORECASE)
    if part_match:
        part_num = part_match.group(1) or part_match.group(2)
        base = f"{base} (Part {part_num})"

    return f"{base}: {year}" if year else base


#    Query expansion                                                            

def expand_query(query: str) -> str:
    lower    = query.lower()
    appended = []
    for pattern, expansion in QUERY_EXPANSION_MAP.items():
        if re.search(pattern, lower, re.IGNORECASE) and expansion not in lower:
            appended.append(expansion)
    if appended:
        expanded = query + " " + " ".join(appended)
        logger.debug(f"Query expanded: '{query}' -> '{expanded}'")
        return expanded
    return query


#    IDF store                                                                  

class IDFStore:
    def __init__(self, idf: Dict[str, float], n_docs: int):
        self.idf    = idf
        self.n_docs = n_docs

    def score(self, word: str) -> float:
        return self.idf.get(word, math.log(self.n_docs + 1))

    @classmethod
    def build(cls, chunks: List[Dict]) -> "IDFStore":
        n_docs = len(chunks)
        df: Counter = Counter()
        for chunk in chunks:
            words = set(_tokens(chunk.get("title", "") + " " + chunk.get("text", "")))
            df.update(words)
        idf = {w: math.log((n_docs + 1) / (c + 1)) + 1.0 for w, c in df.items()}
        return cls(idf, n_docs)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"idf": self.idf, "n_docs": self.n_docs}, f)

    @classmethod
    def load(cls, path: str) -> "IDFStore":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["idf"], d["n_docs"])


#    Material-type classifier                                                   

def classify_material_type(query: str, idf_store: IDFStore) -> Tuple[Optional[str], float]:
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return None, 0.0
    net_scores: Dict[str, float] = {}
    for mtype, signals in MATERIAL_TYPE_SIGNALS.items():
        pos = sum(idf_store.score(w) for w in q_tokens & signals["pos"])
        neg = sum(idf_store.score(w) for w in q_tokens & signals["neg"])
        net_scores[mtype] = pos - neg
    best_type  = max(net_scores, key=net_scores.__getitem__)
    best_score = net_scores[best_type]
    if best_score <= 0:
        return None, 0.0
    total      = sum(max(s, 0) for s in net_scores.values())
    confidence = best_score / total if total > 0 else 0.0
    return best_type, round(confidence, 3)


#    IDF-weighted title F1                                                      

def idf_weighted_title_f1(query: str, chunk: Dict, idf_store: IDFStore) -> float:
    q_tokens = _tokens(query)
    title    = chunk.get("title", "") + " " + chunk.get("header", "")
    t_tokens = _tokens(title)
    if not q_tokens or not t_tokens:
        return 0.0
    q_set = set(q_tokens);  t_set = set(t_tokens);  inter = q_set & t_set
    if not inter:
        return 0.0
    idf_inter = sum(idf_store.score(w) for w in inter)
    idf_q     = sum(idf_store.score(w) for w in q_set)
    idf_t     = sum(idf_store.score(w) for w in t_set)
    p = idf_inter / idf_t if idf_t > 0 else 0.0
    r = idf_inter / idf_q if idf_q > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def variant_specificity_score(query: str, chunk: Dict, idf_store: IDFStore) -> float:
    """
    Score how well a chunk matches the query on variant-distinguishing tokens.

    For standards with multiple parts/variants under the same base ID, the
    family name is often shared across chunks while the decisive terms live in
    the variant language: e.g. "lightweight", "calcined clay", "fly ash".

    We therefore compare query tokens against the chunk title/header tokens
    after removing tokens common to every member of that family.
    """
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return 0.0

    title_tokens = set(_tokens(chunk.get("title", "") + " " + chunk.get("header", "")))
    if not title_tokens:
        return 0.0

    family_common = set(chunk.get("_family_common_tokens", []))
    distinctive = title_tokens - family_common
    if not distinctive:
        return 0.0

    overlap = q_tokens & distinctive
    if not overlap:
        return 0.0

    idf_overlap = sum(idf_store.score(tok) for tok in overlap)
    idf_distinctive = sum(idf_store.score(tok) for tok in distinctive)
    idf_query = sum(idf_store.score(tok) for tok in q_tokens)

    p = idf_overlap / idf_distinctive if idf_distinctive > 0 else 0.0
    r = idf_overlap / idf_query if idf_query > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


#    Domain keyword boost                                                       

def compute_keyword_boost(query: str, chunk: Dict) -> float:
    MAX_BOOST   = 0.10
    query_lower = query.lower()
    chunk_text  = (chunk.get("text","") + " " + chunk.get("title","") + " " + chunk.get("category","")).lower()
    boost = 0.0
    for kw, weight in DOMAIN_KEYWORD_WEIGHTS.items():
        if kw in query_lower and kw in chunk_text:
            boost += weight
    return min(boost * 0.01, MAX_BOOST)


def matched_keywords(query: str, chunk: Dict) -> List[str]:
    query_lower = query.lower()
    chunk_text = (
        chunk.get("text", "") + " " +
        chunk.get("title", "") + " " +
        chunk.get("category", "")
    ).lower()
    matches = {
        kw for kw in DOMAIN_KEYWORD_WEIGHTS
        if kw in query_lower and kw in chunk_text
    }
    matches.update(_token_set(query) & _token_set(chunk_text))
    return sorted(matches)


#    Co-occurrence bonus                                                        

def compute_cooccurrence_bonus(query: str, chunk: Dict) -> float:
    query_lower = query.lower()
    chunk_text  = (chunk.get("text","") + " " + chunk.get("title","") + " " + chunk.get("category","")).lower()
    total = 0.0
    for (term_a, term_b), bonus in CO_OCCURRENCE_BONUSES:
        if term_a in query_lower and term_b in query_lower and term_a in chunk_text and term_b in chunk_text:
            total += bonus
            logger.debug(f"Co-occ bonus +{bonus} for ({term_a},{term_b}) on {chunk.get('standard_id','?')}")
    return total


#    Negative penalty                                                           

def compute_negative_penalty(query: str, chunk: Dict) -> float:
    query_lower = query.lower()
    chunk_text  = (chunk.get("text","") + " " + chunk.get("title","") + " " + chunk.get("category","")).lower()
    total = 0.0
    for rule in NEGATIVE_PENALTY_RULES:
        if not any(sig in query_lower for sig in rule["query_signals"]):
            continue
        if not any(t in chunk_text for t in rule["chunk_must_have"]):
            continue
        if any(t in chunk_text for t in rule["chunk_must_not_have"]):
            continue
        total += rule["penalty"]
        logger.debug(f"Penalty -{rule['penalty']} (rule:{rule['name']}) on {chunk.get('standard_id','?')}")
    return total


#    Main retriever                                                             

class BISRetriever:

    def __init__(self, index_dir: str,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.index_dir = index_dir
        self._load_index()
        self._load_reranker(reranker_model)

    #    loading                                                            

    def _load_index(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading index from {self.index_dir}...")
        with open(f"{self.index_dir}/config.json") as f:
            config = json.load(f)

        self.model_name  = config["model_name"]
        try:
            self.embed_model = SentenceTransformer(self.model_name, local_files_only=True)
            logger.info(f"Loaded embedding model from local cache: {self.model_name}")
        except Exception as e:
            logger.warning(f"Local embedding model load failed, retrying default loader: {e}")
            self.embed_model = SentenceTransformer(self.model_name)
        self.faiss_index = faiss.read_index(f"{self.index_dir}/faiss.index")

        with open(f"{self.index_dir}/bm25.pkl",     "rb") as f: self.bm25      = pickle.load(f)
        with open(f"{self.index_dir}/chunks.pkl",   "rb") as f: self.chunks    = pickle.load(f)
        with open(f"{self.index_dir}/whitelist.pkl","rb") as f: self.whitelist = pickle.load(f)

        # Build whitelist lookup in BOTH bare ("IS 269") and year ("IS 269 : 1989") forms
        # so dedup works regardless of how chunks stored the ID
        self._bare_whitelist: Set[str] = set()
        for sid in self.whitelist:
            bare = re.sub(r'\s*:\s*\d{4}\b.*', '', sid).strip().upper()
            bare = re.sub(r'^IS\s*', 'IS ', bare).strip()
            self._bare_whitelist.add(bare)

        # IDF store
        idf_path = f"{self.index_dir}/idf.pkl"
        if Path(idf_path).exists():
            self.idf_store = IDFStore.load(idf_path)
            logger.info(f"IDF store loaded ({self.idf_store.n_docs} docs).")
        else:
            logger.warning("idf.pkl not found  -  building on the fly.")
            self.idf_store = IDFStore.build(self.chunks)
            self.idf_store.save(idf_path)

        logger.info(f"Index loaded: {len(self.chunks)} chunks, "
                    f"{len(self.whitelist)} standards, "
                    f"{len(self._bare_whitelist)} bare IDs.")

    def _load_reranker(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
            try:
                self.reranker = CrossEncoder(model_name, max_length=512, local_files_only=True)
                logger.info(f"Loaded reranker from local cache: {model_name}")
            except Exception as e:
                logger.warning(f"Local reranker load failed, retrying default loader: {e}")
                self.reranker = CrossEncoder(model_name, max_length=512)
            self.has_reranker = True
            logger.info(f"Reranker loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Reranker unavailable: {e}")
            self.reranker     = None
            self.has_reranker = False

    #    embedding                                                          

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_model.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True
        ).astype("float32")

    #    stage 2 & 3: retrieval                                             

    def dense_retrieve(self, query: str, k: int = 30) -> List[Tuple[int, float]]:
        scores, indices = self.faiss_index.search(self.embed_query(query), k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    def sparse_retrieve(self, query: str, k: int = 30) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        top_k  = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_k]

    #    stage 4: RRF                                                       

    def rrf_merge(self, dense, sparse, k=60, dw=0.6, sw=0.4) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense):
            scores[idx] = scores.get(idx, 0.0) + dw / (k + rank + 1)
        for rank, (idx, _) in enumerate(sparse):
            scores[idx] = scores.get(idx, 0.0) + sw / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    #    stage 5: exact IS-code boost                                      

    def boost_exact_matches(self, query: str, rrf) -> List[Tuple[int, float]]:
        query_ids = normalize_query_standard_ids(query)
        if not query_ids:
            return rrf
        out = []
        for idx, score in rrf:
            chunk_ids = set(self.chunks[idx].get("all_standard_ids", []))
            # Also check bare form
            chunk_bare = {re.sub(r'\s*:\s*\d{4}\b.*','',s).strip().upper() for s in chunk_ids}
            boost = 4.0 if any(qid in chunk_ids or qid in chunk_bare for qid in query_ids) else 0.0
            out.append((idx, score + boost))
        return sorted(out, key=lambda x: x[1], reverse=True)

    #    stage 6: material-type boost/penalty                              

    def material_type_rescore(self, query: str, rrf, query_mtype, confidence):
        if query_mtype is None or confidence < CONFIDENCE_THRESHOLD:
            return rrf
        our_pos = MATERIAL_TYPE_SIGNALS[query_mtype]["pos"]
        conflicting = {
            mt for mt, sigs in MATERIAL_TYPE_SIGNALS.items()
            if mt != query_mtype and our_pos & sigs["neg"]
        }
        out = []
        for idx, score in rrf:
            cm = self.chunks[idx].get("material_type", "")
            if cm == query_mtype:
                adj = MATCH_BOOST * confidence
            elif cm in conflicting:
                adj = CONFLICT_PENALTY * confidence
            else:
                adj = 0.0
            out.append((idx, score + adj))
        return sorted(out, key=lambda x: x[1], reverse=True)

    #    stage 7: mandatory match gate                                     

    def mandatory_match_gate(self, rrf, query_mtype, confidence):
        if query_mtype is None or confidence < 0.65:
            return rrf
        top3_mtypes = {self.chunks[idx].get("material_type","") for idx, _ in rrf[:3]}
        if query_mtype in top3_mtypes:
            return rrf
        for i, (idx, score) in enumerate(rrf[3:], start=3):
            if self.chunks[idx].get("material_type","") == query_mtype:
                rrf = list(rrf)
                promoted = rrf.pop(i)
                rrf.insert(2, promoted)
                break
        return rrf

    #    stage 8: IDF-weighted title F1                                    

    def idf_title_rescore(self, query: str, rrf, alpha: float = 0.30):
        if not rrf:
            return rrf
        max_rrf = rrf[0][1];  min_rrf = rrf[-1][1];  rng = max(max_rrf - min_rrf, 1e-9)
        out = []
        for idx, rrf_score in rrf:
            rrf_norm = (rrf_score - min_rrf) / rng
            tf1      = idf_weighted_title_f1(query, self.chunks[idx], self.idf_store)
            out.append((idx, (1 - alpha) * rrf_norm + alpha * tf1))
        return sorted(out, key=lambda x: x[1], reverse=True)

    #    stages 9-11: keyword + co-occurrence + penalty                    

    def apply_signal_adjustments(self, query: str, rrf) -> List[Tuple[int, float]]:
        """Keyword boost + co-occurrence bonus + negative penalty in one pass."""
        out = []
        for idx, score in rrf:
            chunk   = self.chunks[idx]
            kw      = compute_keyword_boost(query, chunk)
            co      = compute_cooccurrence_bonus(query, chunk)
            penalty = compute_negative_penalty(query, chunk)
            out.append((idx, score + kw + co - penalty))
        return sorted(out, key=lambda x: x[1], reverse=True)

    #    stage 12: cross-encoder reranker                                  

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        if not self.has_reranker or len(candidates) <= top_k:
            for c in candidates[:top_k]:
                c["_cross_encoder_score"] = None
            return candidates[:top_k]
        pairs  = [(query, c.get("text", "")[:512]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        out = []
        for c, score in ranked[:top_k]:
            c["_cross_encoder_score"] = float(score)
            out.append(c)
        return out

    def _family_base_id(self, sid: str) -> str:
        base = re.sub(r'\(\s*PART\s*\d+\s*\)', '', str(sid), flags=re.IGNORECASE)
        base = re.sub(r'\bPART\s*\d+\b', '', base, flags=re.IGNORECASE)
        base = re.sub(r'\s*:\s*\d{4}\b.*', '', base)
        base = re.sub(r'[\s:.\-]+', ' ', base).strip().upper()
        base = re.sub(r'^IS\s*', 'IS ', base).strip()
        return base

    def family_variant_rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Reorder chunks inside the same standard family using variant-specific
        title/header evidence.

        This handles families like multi-part standards or closely related
        variants where the base standard is shared but the decisive query words
        identify one specific sibling.
        """
        if len(chunks) < 2:
            return chunks

        families: Dict[str, List[Dict]] = {}
        for chunk in chunks:
            sid = chunk.get("standard_id", "")
            families.setdefault(self._family_base_id(sid), []).append(chunk)

        for family_chunks in families.values():
            if len(family_chunks) < 2:
                continue

            family_token_sets = [
                set(_tokens(c.get("title", "") + " " + c.get("header", "")))
                for c in family_chunks
            ]
            common_tokens = set.intersection(*family_token_sets) if family_token_sets else set()

            for c in family_chunks:
                c["_family_common_tokens"] = sorted(common_tokens)
                c["_variant_specificity_score"] = variant_specificity_score(query, c, self.idf_store)

            family_chunks.sort(
                key=lambda c: (
                    c.get("_variant_specificity_score", 0.0),
                    c.get("_rrf_score", 0.0),
                ),
                reverse=True,
            )

        reordered: List[Dict] = []
        emitted = set()
        for chunk in chunks:
            family = self._family_base_id(chunk.get("standard_id", ""))
            if family in emitted:
                continue
            reordered.extend(families[family])
            emitted.add(family)

        return reordered

    #    public entry point                                                 

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Full production pipeline.
        Returns top_k chunks; standard_id normalised to 'IS XXXX : YYYY'.
        All IDs validated against whitelist  -  no hallucinations possible.
        """
        # 1. Query expansion
        expanded = expand_query(query)

        # 2 & 3. Retrieval
        dense  = self.dense_retrieve(expanded, k=30)
        sparse = self.sparse_retrieve(expanded, k=30)

        # 4. RRF
        merged = self.rrf_merge(dense, sparse)
        rrf_scores = {idx: score for idx, score in merged}

        # 5. Exact IS-code boost (use original query for precision)
        before_exact = {idx: score for idx, score in merged}
        merged = self.boost_exact_matches(query, merged)
        after_exact = {idx: score for idx, score in merged}
        exact_boosts = {
            idx: after_exact.get(idx, before_exact.get(idx, 0.0)) - before_exact.get(idx, 0.0)
            for idx in after_exact
        }

        # Classify material type once  -  used by stages 6 & 7
        query_mtype, confidence = classify_material_type(expanded, self.idf_store)

        # 6. Material-type boost/penalty
        before_material = {idx: score for idx, score in merged}
        merged = self.material_type_rescore(expanded, merged, query_mtype, confidence)
        after_material = {idx: score for idx, score in merged}
        material_adjustments = {
            idx: after_material.get(idx, before_material.get(idx, 0.0)) - before_material.get(idx, 0.0)
            for idx in after_material
        }

        # 7. Mandatory match gate
        merged = self.mandatory_match_gate(merged, query_mtype, confidence)

        # 8. IDF-weighted title F1
        alpha = 0.30
        idf_input = list(merged)
        idf_title_scores = {
            idx: idf_weighted_title_f1(expanded, self.chunks[idx], self.idf_store)
            for idx, _ in idf_input
        }
        if idf_input:
            max_rrf = idf_input[0][1]
            min_rrf = idf_input[-1][1]
            rng = max(max_rrf - min_rrf, 1e-9)
            idf_rrf_norm = {idx: (score - min_rrf) / rng for idx, score in idf_input}
        else:
            idf_rrf_norm = {}
        merged = self.idf_title_rescore(expanded, merged, alpha=alpha)

        # 9-11. Keyword boost + co-occurrence + negative penalty
        signal_details = {}
        for idx, _ in merged:
            chunk = self.chunks[idx]
            kw = compute_keyword_boost(expanded, chunk)
            co = compute_cooccurrence_bonus(expanded, chunk)
            penalty = compute_negative_penalty(expanded, chunk)
            signal_details[idx] = {
                "keyword_overlap_boost": kw,
                "cooccurrence_boost": co,
                "negative_material_penalty": penalty,
                "net_keyword_cooccurrence_adjustment": kw + co - penalty,
            }
        merged = self.apply_signal_adjustments(expanded, merged)

        # Collect top-18 candidates for reranker
        candidates = []
        for idx, score in merged[:18]:
            chunk = dict(self.chunks[idx])       # shallow copy  -  never mutate stored chunk
            chunk["_rrf_score"]   = score
            chunk["score"]        = score
            chunk["_query_mtype"] = query_mtype
            chunk["_mtype_conf"]  = confidence
            chunk["_explainability"] = {
                "matched_keywords": matched_keywords(expanded, chunk),
                "detected_material": {
                    "query_material_type": query_mtype,
                    "query_material_confidence": _round_float(confidence),
                    "standard_material_type": chunk.get("material_type") or None,
                    "material_match": bool(query_mtype and chunk.get("material_type") == query_mtype),
                },
                "score_breakdown": {
                    "rrf_score": _round_float(rrf_scores.get(idx)),
                    "is_code_boost": _round_float(exact_boosts.get(idx, 0.0)),
                    "material_type_boost": _round_float(material_adjustments.get(idx, 0.0)),
                    "idf_title_relevance_score": _round_float(idf_title_scores.get(idx, 0.0)),
                    "idf_rrf_normalized_component": _round_float((1 - alpha) * idf_rrf_norm.get(idx, 0.0)),
                    "idf_title_component": _round_float(alpha * idf_title_scores.get(idx, 0.0)),
                    **{k: _round_float(v) for k, v in signal_details.get(idx, {}).items()},
                    "pre_cross_encoder_score": _round_float(score),
                    "cross_encoder_score": None,
                },
            }
            chunk["_explainability"]["score_weightage_percent"] = _score_weightage_percent(
                chunk["_explainability"]["score_breakdown"]
            )
            candidates.append(chunk)

        # 12. Cross-encoder reranker
        reranked = self.rerank(expanded, candidates, top_k=top_k)
        for chunk in reranked:
            breakdown = chunk.get("_explainability", {}).get("score_breakdown", {})
            breakdown["cross_encoder_score"] = _round_float(chunk.get("_cross_encoder_score"))
            chunk.get("_explainability", {})["score_weightage_percent"] = _score_weightage_percent(breakdown)

        #    Normalise & validate standard IDs                             
        # This is the hallucination-prevention gate: any chunk whose standard_id
        # is not in the whitelist is silently dropped rather than returned.
        validated = []
        for chunk in reranked:
            raw_sid = chunk.get("standard_id", "")
            year    = chunk.get("year")

            # Compute bare form for whitelist lookup
            bare = re.sub(r'\s*:\s*\d{4}\b.*', '', raw_sid).strip().upper()
            bare = re.sub(r'^IS\s*', 'IS ', bare).strip()

            if not bare:
                # Try first of all_standard_ids
                for alt in chunk.get("all_standard_ids", []):
                    alt_bare = re.sub(r'\s*:\s*\d{4}\b.*', '', alt).strip().upper()
                    alt_bare = re.sub(r'^IS\s*', 'IS ', alt_bare).strip()
                    if alt_bare in self._bare_whitelist:
                        bare    = alt_bare
                        raw_sid = alt
                        break

            if bare not in self._bare_whitelist:
                logger.debug(f"Dropping chunk with non-whitelisted ID: '{raw_sid}'")
                continue

            # Preserve part-qualified IDs in the scored output when the chunk header carries them.
            chunk["standard_id"] = format_chunk_standard_id(chunk, bare, year)
            chunk["all_standard_ids"] = [
                format_chunk_standard_id(
                    chunk,
                    re.sub(r'^IS\s*', 'IS ', re.sub(r'\s*:\s*\d{4}\b.*','',s).strip().upper()),
                    year
                )
                for s in chunk.get("all_standard_ids", [])
            ]
            validated.append(chunk)

        validated = self.family_variant_rerank(expanded, validated)
        validated = validated[:top_k]

        for chunk in validated:
            chunk.pop("_family_common_tokens", None)
            chunk.pop("_variant_specificity_score", None)
            chunk.pop("_cross_encoder_score", None)

        return validated


#    smoke test                                                                 

if __name__ == "__main__":
    import sys
    index_dir = sys.argv[1] if len(sys.argv) > 1 else "data/index"
    retriever = BISRetriever(index_dir)

    tests = [
        ("corrugated asbestos cement sheets for factory roofing",   "IS 459"),
        ("fibre reinforced cement long corrugated roofing sheets",   "IS 14871"),
        ("ordinary portland cement 33 grade for RCC",               "IS 269"),
        ("43 grade OPC for general building construction",           "IS 8112"),
        ("fly ash portland pozzolana cement for dam",                "IS 1489"),
        ("TMT Fe500 rebar for earthquake resistant structure",       "IS 1786"),
        ("coarse crushed stone aggregate concrete mix",              "IS 383"),
        ("OPC 53 grade prestressed concrete",                        "IS 12269"),
    ]

    print(f"\n{'Query':<55} {'Expected':<12} {'Got':<25} {'OK'}")
    print("-" * 105)
    for query, expected in tests:
        t0      = time.time()
        results = retriever.retrieve(query, top_k=5)
        top_id  = results[0].get("standard_id", "EMPTY") if results else "EMPTY"
        ok      = " " if expected in top_id else " "
        print(f"{query:<55} {expected:<12} {top_id:<25} {ok}  ({time.time()-t0:.2f}s)")
        if not results:
            print("  WARNING: empty results  -  check whitelist vs chunk standard_ids")
