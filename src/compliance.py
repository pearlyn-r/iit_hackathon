"""
Rule-based compliance checklist generation for StandardsIQ.

The templates are deterministic and curated. They provide MSME-friendly next
steps while avoiding clause-level claims that are not present in the retrieved
summary text.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Dict, List


BASE_CERTIFICATION_STEPS = [
    "Confirm the exact product variety, grade, size, and intended use covered by the selected IS standard.",
    "Prepare manufacturing process flow, plant layout, machinery list, and in-house quality control plan.",
    "Arrange testing through a BIS-recognized or otherwise competent laboratory as applicable to the product.",
    "Submit BIS license application with product details, test reports, manufacturing details, and QC records.",
    "Support factory inspection, sample drawal, and corrective actions requested during BIS assessment.",
    "Maintain routine production testing, calibration records, complaint records, and batch traceability after license grant.",
]

BASE_DOCUMENTS = [
    "BIS application form and applicant identity/business registration details",
    "Manufacturing process description and product specifications",
    "Quality control manual or inspection and test plan",
    "Raw material purchase and acceptance records",
    "In-house and external laboratory test reports",
    "Calibration certificates for measuring and test equipment",
    "Batch, lot, and dispatch records for traceability",
]

TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "cement": {
        "tests": [
            "Chemical composition and loss on ignition checks",
            "Fineness test",
            "Initial and final setting time",
            "Soundness test",
            "Compressive strength at specified ages",
            "Packing, marking, and batch identification checks",
        ],
        "documents": [
            "Clinker, gypsum, fly ash, slag, or pozzolana source records as applicable",
            "Daily cement grinding, blending, and packing QC records",
            "Strength test register with cube preparation and curing details",
        ],
    },
    "steel": {
        "tests": [
            "Chemical composition analysis",
            "Tensile strength, yield stress, and elongation",
            "Bend or rebend test where applicable",
            "Mass, dimension, and tolerance checks",
            "Surface condition and marking verification",
        ],
        "documents": [
            "Heat or batch traceability records",
            "Mill test certificates or in-house metallurgical test records",
            "Rolling, treatment, and marking control records",
        ],
    },
    "aggregate": {
        "tests": [
            "Sieve analysis and grading",
            "Material finer than specified sieve checks",
            "Specific gravity and water absorption",
            "Aggregate crushing or impact value where applicable",
            "Deleterious material and organic impurity checks",
        ],
        "documents": [
            "Quarry or source approval records",
            "Sampling records and gradation test register",
            "Stockpile segregation and moisture control records",
        ],
    },
    "brick_masonry": {
        "tests": [
            "Dimensional tolerance checks",
            "Compressive strength",
            "Water absorption",
            "Efflorescence",
            "Visual defects, warpage, and finish checks",
        ],
        "documents": [
            "Clay, fly ash, lime, cement, or aggregate input records as applicable",
            "Kiln, curing, autoclaving, or drying process records",
            "Lot-wise strength and absorption test reports",
        ],
    },
    "pipe": {
        "tests": [
            "Dimensional and wall thickness checks",
            "Hydraulic pressure or leakage test where applicable",
            "Material identification and grade verification",
            "Impact, flattening, or stiffness checks where applicable",
            "Joint, socket, and marking verification",
        ],
        "documents": [
            "Resin, compound, metal, or cementitious raw material certificates",
            "Extrusion, casting, curing, or forming process records",
            "Pressure test and inspection logs",
        ],
    },
    "roofing": {
        "tests": [
            "Dimension, thickness, and profile checks",
            "Flexural or breaking load test where applicable",
            "Water absorption or permeability checks where applicable",
            "Weathering, finish, and visual defect inspection",
            "Marking and packing verification",
        ],
        "documents": [
            "Fibre, cement, asbestos, resin, or metal input records as applicable",
            "Sheet forming, curing, and finishing QC records",
            "Lot-wise dimensional and strength test records",
        ],
    },
    "default": {
        "tests": [
            "Product dimensions and tolerance checks",
            "Material composition or grade verification",
            "Performance tests specified by the selected IS standard",
            "Workmanship, finish, marking, and packing checks",
        ],
        "documents": [],
    },
}


def _norm(text: str) -> str:
    return (text or "").lower()


def extract_summary_sections(text: str) -> dict:
    clean = re.sub(r"\s+", " ", text or "").strip()
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|(?=\b\d+(?:\.\d+)*\s+[A-Z])", clean)
        if len(s.strip().split()) >= 4
    ]

    buckets = {
        "scope": [],
        "important_requirements": [],
        "test_methods": [],
        "marking_labelling": [],
        "related_standards": sorted(
            set(re.findall(r"\bIS\s*[\-:. ]?\d{2,6}(?:\s*\(Part\s*\d+\))?", clean, re.I))
        ),
    }

    rules = [
        ("scope", re.compile(r"\b(scope|covers|applicable|intended|used for|specifies)\b", re.I)),
        (
            "important_requirements",
            re.compile(
                r"\b(requirement|physical|mechanical|dimension|tolerance|strength|grade|material|composition|shall)\b",
                re.I,
            ),
        ),
        (
            "test_methods",
            re.compile(
                r"\b(test|testing|method|sample|sampling|hydrostatic|bearing|permeability|compressive|tensile|bend|fineness|soundness)\b",
                re.I,
            ),
        ),
        ("marking_labelling", re.compile(r"\b(mark|marked|marking|label|labelling|packing|manufacturer|identification)\b", re.I)),
    ]

    for sent in sentences:
        for key, pattern in rules:
            if pattern.search(sent) and len(buckets[key]) < 4:
                buckets[key].append(sent[:280])

    return buckets


def build_extractive_rationale(query: str, item: dict) -> dict:
    explain = item.get("_explainability", {})
    sections = item.get("summary_sections") or extract_summary_sections(
        item.get("text") or item.get("rationale_context") or ""
    )
    material = explain.get("detected_material", {})
    keywords = explain.get("matched_keywords", [])

    points = []
    if keywords:
        points.append("The query and summary overlap on: " + ", ".join(keywords[:8]) + ".")
    if material.get("query_material_type"):
        points.append(
            "Material signal: query is classified as "
            f"{material.get('query_material_type')} and this standard is tagged "
            f"{material.get('standard_material_type') or 'unknown'}."
        )
    if sections["scope"]:
        points.append("Summary evidence: " + sections["scope"][0])
    elif item.get("title"):
        points.append(
            f"The selected title is '{item.get('title')}', which matched the retrieval signals for the product description."
        )

    return {
        "summary": " ".join(points[:3]),
        "points": points,
        "source_sections": sections,
        "grounding": "Built from retrieved SP 21 summary text, metadata, and retrieval scores only.",
    }


def _template_key(item: dict) -> str:
    material = _norm(item.get("_query_mtype") or item.get("material_type"))
    category = _norm(item.get("category"))
    title = _norm(item.get("title"))
    haystack = " ".join([material, category, title])

    if "cement" in haystack:
        if "sheet" in haystack or "roof" in haystack:
            return "roofing"
        return "cement"
    if "steel" in haystack or "rebar" in haystack or "bar" in haystack:
        return "steel"
    if "aggregate" in haystack or "sand" in haystack or "gravel" in haystack:
        return "aggregate"
    if "brick" in haystack or "masonry" in haystack or "block" in haystack:
        return "brick_masonry"
    if "pipe" in haystack or "tube" in haystack:
        return "pipe"
    if "roof" in haystack or "sheet" in haystack or "cladding" in haystack:
        return "roofing"
    return "default"


def generate_compliance_checklist(item: dict) -> dict:
    key = _template_key(item)
    template = deepcopy(TEMPLATES[key])
    sections = item.get("summary_sections") or extract_summary_sections(
        item.get("text") or item.get("rationale_context") or ""
    )
    docs = BASE_DOCUMENTS + template.get("documents", [])
    tests = list(template["tests"])
    for sent in sections["test_methods"]:
        candidate = sent[:180]
        if candidate not in tests:
            tests.insert(0, candidate)

    return {
        "applicable_standard": {
            "code": item.get("standard_id", ""),
            "title": item.get("title", ""),
        },
        "material_type": item.get("_query_mtype") or item.get("material_type") or key,
        "template_key": key,
        "key_material_dimensional_requirements": sections["important_requirements"],
        "required_tests": tests,
        "marking_labelling": sections["marking_labelling"],
        "related_standards": sections["related_standards"],
        "certification_steps": BASE_CERTIFICATION_STEPS,
        "required_documentation": docs,
        "disclaimer": "Rule-based MSME guidance only; verify exact clauses, sampling frequency, and certification scheme requirements against the current BIS standard and BIS instructions.",
    }
