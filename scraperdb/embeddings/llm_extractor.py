# embeddings/llm_extractor.py
"""
LLM-powered profile extraction for structured company data.
Uses Gemini to extract key information from company profiles, enabling:
1. Much faster processing (10 seconds vs 5-10 minutes)
2. Structured metadata for filtering
3. Focused embeddings that capture company essence
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

# IMPORTANT: Load dotenv early to ensure env vars are available
# This must happen before reading any env vars at module level
from pathlib import Path
from dotenv import load_dotenv

# Find the .env file relative to this file's location (in scraperdb/)
_this_dir = Path(__file__).resolve().parent  # embeddings/
_project_root = _this_dir.parent  # scraperdb/
_env_file = _project_root / ".env"
_dotenv_loaded = load_dotenv(_env_file)

logger = logging.getLogger(__name__)

# Debug: confirm .env loading
print(f"[LLM_EXTRACTOR] .env file: {_env_file}, exists: {_env_file.exists()}, loaded: {_dotenv_loaded}")

# Configuration - read AFTER load_dotenv() is called
USE_LLM_EXTRACTION = os.getenv("USE_LLM_EXTRACTION", "false").lower() in ("true", "1", "yes")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Debug: print configuration (using print for visibility before logging is configured)
print(f"[LLM_EXTRACTOR] USE_LLM_EXTRACTION={USE_LLM_EXTRACTION}")
print(f"[LLM_EXTRACTOR] GEMINI_API_KEY={'SET (' + GEMINI_API_KEY[:10] + '...)' if GEMINI_API_KEY else 'NOT SET'}")
print(f"[LLM_EXTRACTOR] GEMINI_MODEL={GEMINI_MODEL}")

MAX_INPUT_CHARS = int(os.getenv("LLM_MAX_INPUT_CHARS", "80000"))  # ~20k tokens


PROFILE_EXTRACTION_PROMPT = """Analyze this company document and extract a structured profile.

DOCUMENT:
{document_text}

Extract the following information in JSON format. Be specific and extract actual details from the document. If information is not available, use null.

{{
    "company_name": "string - the company's official name",
    "industries": ["list of industries they work in, e.g., 'Information Technology', 'Defense', 'Healthcare'"],
    "capabilities": ["list of specific technical capabilities, e.g., 'Software Development', 'Cloud Migration', 'Cybersecurity'"],
    "certifications": ["list of certifications like 'ISO 9001', 'ISO 27001', 'CMMI Level 5', etc."],
    "project_types": ["types of projects they handle, e.g., 'Software Development', 'System Integration', 'Consulting'"],
    "past_clients": ["government agencies, companies they've worked with"],
    "geographic_focus": ["regions/countries they operate in, e.g., 'India', 'Pan-India', 'Global'"],
    "company_size": "small/medium/large/enterprise - based on employee count or revenue indicators",
    "key_technologies": ["specific technologies they specialize in, e.g., 'Java', 'Python', 'AWS', 'SAP'"],
    "domain_expertise": ["specific domains like 'Banking & Finance', 'Defense', 'E-Governance', 'Healthcare IT'"],
    "years_of_experience": "number or null - years in business if mentioned",
    "notable_projects": ["brief descriptions of significant projects mentioned"],
    "summary": "2-3 sentence summary of what this company does and their key strengths"
}}

Return ONLY the JSON object, no additional text."""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_llm_json(response_text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_code_fences(response_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return None


def _extract_profile_with_gemini(
    document_text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    logger.info(">>> _extract_profile_with_gemini called with %d chars of text", len(document_text))

    key = api_key or GEMINI_API_KEY
    if not key:
        logger.warning("No Gemini API key configured, skipping LLM extraction")
        return None

    try:
        import google.generativeai as genai
        logger.info("google-generativeai package imported successfully")
    except ImportError:
        logger.warning("google-generativeai package not installed, skipping LLM extraction")
        return None

    model_name = model or GEMINI_MODEL
    logger.info("Using Gemini model: %s", model_name)

    truncated_text = document_text[:MAX_INPUT_CHARS]
    if len(document_text) > MAX_INPUT_CHARS:
        logger.info(f"Truncated document from {len(document_text)} to {MAX_INPUT_CHARS} chars")

    try:
        logger.info("Configuring Gemini API and making request...")
        genai.configure(api_key=key)
        model_client = genai.GenerativeModel(model_name)
        response = model_client.generate_content(
            PROFILE_EXTRACTION_PROMPT.format(document_text=truncated_text),
            generation_config={"temperature": 0.2},
        )
        response_text = (response.text or "").strip()
        logger.info("Gemini API response received, length=%d chars", len(response_text))

        profile_data = _parse_llm_json(response_text)
        if not profile_data:
            logger.warning("Failed to parse Gemini response as JSON")
            return None
        logger.info(f"Successfully extracted profile: {profile_data.get('company_name', 'Unknown')}")
        return profile_data
    except Exception as e:
        logger.exception(f"LLM extraction failed: {e}")
        return None


def extract_profile_with_llm(
    document_text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Use Gemini to extract structured profile data.

    Args:
        document_text: Text content of the company profile document
        api_key: Optional API key override
        model: Optional model override

    Returns:
        Structured profile dictionary or None if extraction fails
    """
    return _extract_profile_with_gemini(document_text, api_key=api_key, model=model)


def create_searchable_profile_text(profile_data: Dict[str, Any]) -> str:
    """
    Convert structured profile data to searchable text for embedding.

    This creates a focused text representation that captures the key
    aspects of the company for semantic matching.

    Args:
        profile_data: Structured profile from extract_profile_with_llm

    Returns:
        Searchable text suitable for embedding
    """
    parts: List[str] = []

    # Company name and summary
    if profile_data.get("company_name"):
        parts.append(f"Company: {profile_data['company_name']}")

    if profile_data.get("summary"):
        parts.append(f"Summary: {profile_data['summary']}")

    # Industries and domains
    if profile_data.get("industries"):
        industries = profile_data["industries"]
        if isinstance(industries, list):
            parts.append(f"Industries: {', '.join(industries)}")

    if profile_data.get("domain_expertise"):
        domains = profile_data["domain_expertise"]
        if isinstance(domains, list):
            parts.append(f"Domain expertise: {', '.join(domains)}")

    # Capabilities and technologies
    if profile_data.get("capabilities"):
        capabilities = profile_data["capabilities"]
        if isinstance(capabilities, list):
            parts.append(f"Capabilities: {', '.join(capabilities)}")

    if profile_data.get("key_technologies"):
        technologies = profile_data["key_technologies"]
        if isinstance(technologies, list):
            parts.append(f"Technologies: {', '.join(technologies)}")

    # Project types
    if profile_data.get("project_types"):
        project_types = profile_data["project_types"]
        if isinstance(project_types, list):
            parts.append(f"Project types: {', '.join(project_types)}")

    # Certifications
    if profile_data.get("certifications"):
        certifications = profile_data["certifications"]
        if isinstance(certifications, list):
            parts.append(f"Certifications: {', '.join(certifications)}")

    # Geographic focus
    if profile_data.get("geographic_focus"):
        geo = profile_data["geographic_focus"]
        if isinstance(geo, list):
            parts.append(f"Geographic focus: {', '.join(geo)}")

    # Past clients (important for government tender matching)
    if profile_data.get("past_clients"):
        clients = profile_data["past_clients"]
        if isinstance(clients, list):
            parts.append(f"Past clients: {', '.join(clients)}")

    # Notable projects
    if profile_data.get("notable_projects"):
        projects = profile_data["notable_projects"]
        if isinstance(projects, list):
            parts.append(f"Notable projects: {'; '.join(projects)}")

    return "\n".join(parts)


def extract_filter_metadata(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata suitable for pre-filtering in search.

    Returns flattened metadata that can be stored in ChromaDB for filtering.

    Args:
        profile_data: Structured profile from extract_profile_with_llm

    Returns:
        Flat dictionary of filterable metadata
    """
    metadata: Dict[str, Any] = {}

    # Company size (useful for tender eligibility)
    if profile_data.get("company_size"):
        metadata["company_size"] = profile_data["company_size"]

    # Years of experience
    if profile_data.get("years_of_experience"):
        try:
            metadata["years_of_experience"] = int(profile_data["years_of_experience"])
        except (ValueError, TypeError):
            pass

    # Has certifications (boolean flags for common ones)
    certifications = profile_data.get("certifications") or []
    if isinstance(certifications, list):
        cert_lower = [c.lower() for c in certifications if isinstance(c, str)]
        metadata["has_iso_9001"] = any("iso 9001" in c or "iso9001" in c for c in cert_lower)
        metadata["has_iso_27001"] = any("iso 27001" in c or "iso27001" in c for c in cert_lower)
        metadata["has_cmmi"] = any("cmmi" in c for c in cert_lower)

    # Industry indicators (for tender category matching)
    industries = profile_data.get("industries") or []
    if isinstance(industries, list):
        ind_lower = [i.lower() for i in industries if isinstance(i, str)]
        metadata["is_it_company"] = any(
            kw in " ".join(ind_lower)
            for kw in ["it", "software", "technology", "digital"]
        )
        metadata["is_defense"] = any("defense" in i or "defence" in i for i in ind_lower)
        metadata["is_healthcare"] = any("health" in i or "medical" in i for i in ind_lower)

    # Geographic flags
    geo = profile_data.get("geographic_focus") or []
    if isinstance(geo, list):
        geo_lower = [g.lower() for g in geo if isinstance(g, str)]
        metadata["operates_pan_india"] = any(
            kw in " ".join(geo_lower)
            for kw in ["pan-india", "pan india", "nationwide", "all india"]
        )
        metadata["operates_global"] = any(
            kw in " ".join(geo_lower)
            for kw in ["global", "international", "worldwide"]
        )

    return metadata


def process_profile_with_llm(
    document_text: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full LLM-based profile processing pipeline.

    Returns both the structured data and searchable text.

    Args:
        document_text: Raw document text
        api_key: Optional Anthropic API key

    Returns:
        Dictionary with:
        - profile_data: Structured profile
        - searchable_text: Text for embedding
        - filter_metadata: Metadata for search filtering
        - success: Whether extraction succeeded
    """
    profile_data = extract_profile_with_llm(document_text, api_key)

    if not profile_data:
        return {
            "profile_data": None,
            "searchable_text": None,
            "filter_metadata": {},
            "success": False,
        }

    searchable_text = create_searchable_profile_text(profile_data)
    filter_metadata = extract_filter_metadata(profile_data)

    return {
        "profile_data": profile_data,
        "searchable_text": searchable_text,
        "filter_metadata": filter_metadata,
        "success": True,
    }


def is_llm_extraction_available() -> bool:
    """Check if LLM extraction is configured and available."""
    if not USE_LLM_EXTRACTION:
        logger.debug("is_llm_extraction_available: USE_LLM_EXTRACTION is False")
        return False

    if not GEMINI_API_KEY:
        logger.debug("is_llm_extraction_available: GEMINI_API_KEY not set")
        return False
    try:
        import google.generativeai
        logger.debug("is_llm_extraction_available: All checks passed, returning True")
        return True
    except ImportError:
        logger.debug("is_llm_extraction_available: google-generativeai not installed")
        return False
