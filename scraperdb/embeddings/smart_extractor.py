# embeddings/smart_extractor.py
"""
Smart section extraction for company profile documents.
Instead of processing entire 200-page documents, extracts only relevant sections
that define company capabilities, reducing processing time by 5-10x.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Any

# Sections that typically contain important profile information
PROFILE_RELEVANT_SECTIONS = [
    "executive summary",
    "company overview",
    "about us",
    "about the company",
    "who we are",
    "our company",
    "capabilities",
    "core capabilities",
    "our capabilities",
    "services",
    "our services",
    "service offerings",
    "solutions",
    "past performance",
    "project experience",
    "experience",
    "our experience",
    "track record",
    "case studies",
    "client list",
    "our clients",
    "customers",
    "certifications",
    "accreditations",
    "compliance",
    "key personnel",
    "leadership",
    "management team",
    "our team",
    "qualifications",
    "technical qualifications",
    "expertise",
    "domain expertise",
    "specializations",
    "industries",
    "sectors",
    "markets",
    "technologies",
    "technical capabilities",
    "infrastructure",
    "resources",
    "achievements",
    "awards",
    "recognition",
]

# Keywords that indicate important content in tables
TABLE_KEYWORDS = [
    "project",
    "client",
    "contract",
    "value",
    "certification",
    "iso",
    "cmmi",
    "capability",
    "service",
    "technology",
    "experience",
    "year",
    "duration",
]


def _normalize_title(title: str) -> str:
    """Normalize section title for comparison."""
    return re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()


def _section_matches_keywords(title: str) -> bool:
    """Check if section title matches any relevant keywords."""
    normalized = _normalize_title(title)
    return any(keyword in normalized for keyword in PROFILE_RELEVANT_SECTIONS)


def _table_is_relevant(table: Any) -> bool:
    """Check if a table contains relevant information."""
    if isinstance(table, dict):
        table_text = table.get("text", "")
        if not table_text:
            # Try to construct text from other fields
            table_text = " ".join(
                str(v) for v in table.values()
                if isinstance(v, str) and v.strip()
            )
    elif isinstance(table, str):
        table_text = table
    else:
        return False

    table_lower = table_text.lower()
    return any(keyword in table_lower for keyword in TABLE_KEYWORDS)


def _extract_table_text(table: Any) -> str:
    """Extract text content from a table structure."""
    if isinstance(table, dict):
        if "text" in table and isinstance(table["text"], str):
            return table["text"]
        # Combine all string values
        parts = []
        for v in table.values():
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        return " ".join(parts)
    elif isinstance(table, str):
        return table.strip()
    return ""


def extract_relevant_sections(docling_output: Dict[str, Any], max_chars: int = 100000) -> str:
    """
    Extract only the sections that define company capabilities from a Docling output.

    Strategy:
    1. Always include first few pages (usually executive summary/overview)
    2. Find and include sections matching relevant keywords
    3. Include relevant tables (often contain project lists, certifications)
    4. Limit total length to prevent excessive processing

    Args:
        docling_output: Dictionary containing 'text', 'tables', 'sections' from Docling
        max_chars: Maximum characters to extract (default 100k = ~50 pages)

    Returns:
        Combined relevant text
    """
    text = docling_output.get("text", "") or ""
    sections = docling_output.get("sections", []) or []
    tables = docling_output.get("tables", []) or []

    relevant_parts: List[str] = []

    # Strategy 1: Always include first ~15k chars (approximately 5-6 pages)
    # This usually contains executive summary, company overview, etc.
    first_pages_limit = 15000
    if text:
        first_pages = text[:first_pages_limit]
        relevant_parts.append(first_pages)

    # Strategy 2: Find and include specific sections by title matching
    section_texts: List[str] = []
    if sections:
        for section in sections:
            if not isinstance(section, dict):
                continue

            title = section.get("title", "") or section.get("heading", "") or ""
            if not title:
                continue

            if _section_matches_keywords(title):
                section_text = section.get("text", "") or section.get("content", "")
                if section_text and isinstance(section_text, str):
                    # Avoid duplicating content already in first_pages
                    if section_text not in first_pages[:first_pages_limit]:
                        section_texts.append(f"\n### {title}\n{section_text}")

    if section_texts:
        relevant_parts.extend(section_texts)

    # Strategy 3: Include relevant tables (limited to first 15 relevant tables)
    table_texts: List[str] = []
    relevant_table_count = 0
    max_relevant_tables = 15

    for table in tables:
        if relevant_table_count >= max_relevant_tables:
            break

        if _table_is_relevant(table):
            table_text = _extract_table_text(table)
            if table_text and len(table_text) > 20:  # Skip tiny tables
                table_texts.append(f"\n[Table]\n{table_text}")
                relevant_table_count += 1

    if table_texts:
        relevant_parts.append("\n\n--- Tables ---\n")
        relevant_parts.extend(table_texts)

    # Combine all parts
    combined = "\n\n".join(relevant_parts)

    # Limit total length
    if len(combined) > max_chars:
        combined = combined[:max_chars]
        # Try to end at a sentence boundary
        last_period = combined.rfind(".")
        if last_period > max_chars * 0.8:  # Only truncate if we don't lose too much
            combined = combined[:last_period + 1]

    return combined.strip()


def extract_key_info_summary(docling_output: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Extract key structured information from the document for metadata.
    Returns a dictionary with company info that can be used for filtering.
    """
    text = docling_output.get("text", "") or ""
    text_lower = text.lower()

    # Try to identify key information using simple pattern matching
    info: Dict[str, Optional[str]] = {
        "has_iso_certification": None,
        "has_cmmi_certification": None,
        "has_government_experience": None,
        "company_size_indicator": None,
    }

    # Check for ISO certifications
    iso_patterns = ["iso 9001", "iso 27001", "iso 14001", "iso certified", "iso certification"]
    info["has_iso_certification"] = any(pattern in text_lower for pattern in iso_patterns)

    # Check for CMMI certification
    cmmi_patterns = ["cmmi level", "cmmi-dev", "cmmi maturity", "capability maturity"]
    info["has_cmmi_certification"] = any(pattern in text_lower for pattern in cmmi_patterns)

    # Check for government experience
    gov_patterns = [
        "government", "ministry", "department of", "public sector",
        "central government", "state government", "psu", "cpse"
    ]
    info["has_government_experience"] = any(pattern in text_lower for pattern in gov_patterns)

    # Try to determine company size
    if "enterprise" in text_lower or "multinational" in text_lower or "global" in text_lower:
        info["company_size_indicator"] = "large"
    elif "mid-size" in text_lower or "medium" in text_lower:
        info["company_size_indicator"] = "medium"
    elif "startup" in text_lower or "small business" in text_lower:
        info["company_size_indicator"] = "small"

    return info


def get_extraction_stats(
    original_text: str,
    extracted_text: str
) -> Dict[str, Any]:
    """
    Get statistics about the extraction for logging/debugging.
    """
    original_chars = len(original_text)
    extracted_chars = len(extracted_text)

    return {
        "original_chars": original_chars,
        "extracted_chars": extracted_chars,
        "reduction_ratio": round(1 - (extracted_chars / max(original_chars, 1)), 2),
        "original_pages_approx": original_chars // 3000,  # ~3000 chars per page
        "extracted_pages_approx": extracted_chars // 3000,
    }
