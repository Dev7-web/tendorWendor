# tests/test_optimized_pipeline.py
"""
Test suite for the optimized tender matching pipeline.
Tests all new components: smart extraction, multi-vector, parallel processing, hybrid search.
"""

from __future__ import annotations

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from typing import List


class TestSmartExtractor(unittest.TestCase):
    """Test the smart section extraction module."""

    def test_import(self):
        """Test that smart_extractor can be imported."""
        from embeddings.smart_extractor import (
            extract_relevant_sections,
            get_extraction_stats,
            extract_key_info_summary,
        )
        self.assertIsNotNone(extract_relevant_sections)

    def test_extract_relevant_sections_empty(self):
        """Test extraction with empty input."""
        from embeddings.smart_extractor import extract_relevant_sections

        result = extract_relevant_sections({})
        self.assertEqual(result, "")

    def test_extract_relevant_sections_text_only(self):
        """Test extraction with text only."""
        from embeddings.smart_extractor import extract_relevant_sections

        doc = {"text": "This is a test document about company capabilities."}
        result = extract_relevant_sections(doc)
        self.assertIn("test document", result)

    def test_extract_relevant_sections_with_sections(self):
        """Test extraction with sections."""
        from embeddings.smart_extractor import extract_relevant_sections

        doc = {
            "text": "First page content.",
            "sections": [
                {"title": "Company Overview", "text": "We are a software company."},
                {"title": "Irrelevant Section", "text": "This should not be included."},
                {"title": "Capabilities", "text": "We provide AI solutions."},
            ],
        }
        result = extract_relevant_sections(doc)
        self.assertIn("software company", result)
        self.assertIn("AI solutions", result)

    def test_get_extraction_stats(self):
        """Test extraction statistics."""
        from embeddings.smart_extractor import get_extraction_stats

        stats = get_extraction_stats("a" * 10000, "b" * 1000)
        self.assertEqual(stats["original_chars"], 10000)
        self.assertEqual(stats["extracted_chars"], 1000)
        self.assertAlmostEqual(stats["reduction_ratio"], 0.9, places=2)


class TestProfileEmbeddings(unittest.TestCase):
    """Test the multi-vector profile embeddings module."""

    def test_import(self):
        """Test that profile_embeddings can be imported."""
        from embeddings.profile_embeddings import (
            create_profile_representation,
            select_representative_embeddings,
            compute_multi_vector_similarity,
        )
        self.assertIsNotNone(create_profile_representation)

    def test_select_representative_embeddings_empty(self):
        """Test with empty input."""
        from embeddings.profile_embeddings import select_representative_embeddings

        result = select_representative_embeddings([], [], 5)
        self.assertEqual(result["representative_embeddings"], [])

    def test_mean_vectors(self):
        """Test mean vector computation."""
        from embeddings.profile_embeddings import _mean_vectors

        vectors = [[1.0, 0.0], [0.0, 1.0]]
        result = _mean_vectors(vectors)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_max_pool_vectors(self):
        """Test max pooling."""
        from embeddings.profile_embeddings import _max_pool_vectors

        vectors = [[1.0, 0.5], [0.5, 1.0]]
        result = _max_pool_vectors(vectors)
        self.assertIsNotNone(result)


class TestParallelProcessor(unittest.TestCase):
    """Test the parallel processing module."""

    def test_import(self):
        """Test that parallel_processor can be imported."""
        from embeddings.parallel_processor import (
            process_single_pdf,
            process_pdfs_parallel,
            merge_pdf_outputs,
        )
        self.assertIsNotNone(process_single_pdf)

    def test_merge_empty_outputs(self):
        """Test merging empty outputs."""
        from embeddings.parallel_processor import merge_pdf_outputs

        result = merge_pdf_outputs([])
        self.assertFalse(result["success"])

    def test_merge_outputs(self):
        """Test merging multiple outputs."""
        from embeddings.parallel_processor import merge_pdf_outputs

        outputs = [
            {"success": True, "text": "Document 1", "file_hash": "hash1", "tables": []},
            {"success": True, "text": "Document 2", "file_hash": "hash2", "tables": []},
        ]
        result = merge_pdf_outputs(outputs)
        self.assertTrue(result["success"])
        self.assertEqual(result["document_count"], 2)
        self.assertIn("Document 1", result["text"])
        self.assertIn("Document 2", result["text"])


class TestLLMExtractor(unittest.TestCase):
    """Test the LLM extraction module."""

    def test_import(self):
        """Test that llm_extractor can be imported."""
        from embeddings.llm_extractor import (
            extract_profile_with_llm,
            create_searchable_profile_text,
            extract_filter_metadata,
            is_llm_extraction_available,
        )
        self.assertIsNotNone(create_searchable_profile_text)

    def test_create_searchable_text(self):
        """Test creating searchable text from profile data."""
        from embeddings.llm_extractor import create_searchable_profile_text

        profile_data = {
            "company_name": "Test Corp",
            "industries": ["IT", "Software"],
            "capabilities": ["Cloud", "AI"],
        }
        result = create_searchable_profile_text(profile_data)
        self.assertIn("Test Corp", result)
        self.assertIn("IT", result)
        self.assertIn("Cloud", result)

    def test_extract_filter_metadata(self):
        """Test filter metadata extraction."""
        from embeddings.llm_extractor import extract_filter_metadata

        profile_data = {
            "company_size": "large",
            "certifications": ["ISO 9001", "CMMI Level 5"],
            "industries": ["Information Technology"],
        }
        result = extract_filter_metadata(profile_data)
        self.assertEqual(result["company_size"], "large")
        self.assertTrue(result["has_iso_9001"])
        self.assertTrue(result["has_cmmi"])
        self.assertTrue(result["is_it_company"])


class TestCache(unittest.TestCase):
    """Test the caching module."""

    def test_import(self):
        """Test that cache can be imported."""
        from embeddings.cache import (
            compute_file_hash,
            compute_text_hash,
            FileCache,
            get_file_cache,
        )
        self.assertIsNotNone(compute_file_hash)

    def test_text_hash(self):
        """Test text hashing."""
        from embeddings.cache import compute_text_hash

        hash1 = compute_text_hash("test")
        hash2 = compute_text_hash("test")
        hash3 = compute_text_hash("different")

        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)


class TestHybridSearch(unittest.TestCase):
    """Test the hybrid search module."""

    def test_import(self):
        """Test that hybrid_search can be imported."""
        from api.services.hybrid_search import (
            search_tenders_hybrid,
            search_with_multi_vector,
            fuse_search_results,
            explain_match,
        )
        self.assertIsNotNone(search_tenders_hybrid)

    def test_fuse_search_results(self):
        """Test score fusion."""
        from api.services.hybrid_search import fuse_search_results

        results1 = {"doc1": 0.9, "doc2": 0.8}
        results2 = {"doc2": 0.85, "doc3": 0.7}

        fused = fuse_search_results(results1, results2)
        self.assertEqual(len(fused), 3)
        # doc2 should have max of 0.8 and 0.85
        doc2_score = next(score for id, score in fused if id == "doc2")
        self.assertEqual(doc2_score, 0.85)

    def test_extract_profile_keywords(self):
        """Test keyword extraction from profile."""
        from api.services.hybrid_search import _extract_profile_keywords

        profile_data = {
            "capabilities": ["Software Development", "Cloud Migration"],
            "key_technologies": ["Python", "AWS"],
        }
        keywords = _extract_profile_keywords(profile_data)
        self.assertIn("software", keywords)
        self.assertIn("python", keywords)
        self.assertIn("aws", keywords)


class TestSearchIntegration(unittest.TestCase):
    """Test search module integration."""

    def test_import(self):
        """Test that search can be imported with new functions."""
        from api.services.search import (
            search_tenders_with_embedding,
            search_tenders_for_profile,
            search_tenders_with_profile_representation,
        )
        self.assertIsNotNone(search_tenders_with_profile_representation)


class TestProfileIngestIntegration(unittest.TestCase):
    """Test profile ingest module integration."""

    def test_import(self):
        """Test that profile_ingest can be imported with new functions."""
        from api.services.profile_ingest import (
            build_profile_embedding_from_file,
            build_profile_embedding_from_files,
            process_profile_job,
        )
        self.assertIsNotNone(build_profile_embedding_from_files)


class TestIndexProfilesIntegration(unittest.TestCase):
    """Test index_profiles module integration."""

    def test_import(self):
        """Test that index_profiles can be imported with new functions."""
        from embeddings.index_profiles import (
            index_pending_profiles,
            reindex_all_profiles,
        )
        self.assertIsNotNone(reindex_all_profiles)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSmartExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestProfileEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestCache))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridSearch))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestProfileIngestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestIndexProfilesIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
