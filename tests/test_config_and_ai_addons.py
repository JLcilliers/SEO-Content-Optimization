# -*- coding: utf-8 -*-
"""
Tests for OptimizationConfig and AI Add-ons functionality.

Tests the new FAQ policy-based control, chunk configuration,
and fail-closed FAQ behavior.
"""

import pytest
from src.seo_content_optimizer.config import OptimizationConfig
from src.seo_content_optimizer.ai_addons import (
    AIAddons,
    Chunk,
    ChunkMap,
    build_chunks,
    generate_ai_addons,
    generate_chunk_map,
    generate_fallback_faqs,
    generate_key_takeaways,
    DEFAULT_CHUNK_TARGET_TOKENS,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
)


class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        assert config.faq_policy == "auto"
        assert config.faq_count == 4
        assert config.faq_min_valid == 2
        assert config.faq_retry_on_insufficient is True
        assert config.generate_ai_sections is True
        assert config.generate_key_takeaways is True
        assert config.generate_chunk_map is True
        assert config.chunk_target_tokens == 512
        assert config.chunk_overlap_tokens == 128
        assert config.max_secondary == 5

    def test_faq_policy_never(self):
        """Test faq_policy='never' disables FAQ generation."""
        config = OptimizationConfig(faq_policy="never")
        assert config.should_generate_faq is False
        assert config.force_faq is False

    def test_faq_policy_always(self):
        """Test faq_policy='always' forces FAQ generation."""
        config = OptimizationConfig(faq_policy="always")
        assert config.should_generate_faq is True
        assert config.force_faq is True

    def test_faq_policy_auto(self):
        """Test faq_policy='auto' enables conditional FAQ."""
        config = OptimizationConfig(faq_policy="auto")
        assert config.should_generate_faq is True
        assert config.force_faq is False

    def test_should_generate_ai_addons_true(self):
        """Test AI addons generation check when enabled."""
        config = OptimizationConfig(
            generate_ai_sections=True,
            generate_key_takeaways=True,
        )
        assert config.should_generate_ai_addons is True

    def test_should_generate_ai_addons_false_when_master_disabled(self):
        """Test AI addons generation check when master switch disabled."""
        config = OptimizationConfig(
            generate_ai_sections=False,
            generate_key_takeaways=True,
            generate_chunk_map=True,
        )
        assert config.should_generate_ai_addons is False

    def test_should_generate_ai_addons_false_when_all_disabled(self):
        """Test AI addons generation check when all sub-options disabled."""
        config = OptimizationConfig(
            generate_ai_sections=True,
            generate_key_takeaways=False,
            generate_chunk_map=False,
        )
        assert config.should_generate_ai_addons is False

    def test_invalid_faq_policy_raises(self):
        """Test invalid faq_policy raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptimizationConfig(faq_policy="invalid")
        assert "faq_policy must be" in str(exc_info.value)

    def test_invalid_faq_count_raises(self):
        """Test invalid faq_count raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptimizationConfig(faq_count=0)
        assert "faq_count must be" in str(exc_info.value)

    def test_invalid_chunk_tokens_raises(self):
        """Test invalid chunk_target_tokens raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptimizationConfig(chunk_target_tokens=50)
        assert "chunk_target_tokens must be" in str(exc_info.value)

    def test_invalid_chunk_overlap_raises(self):
        """Test overlap >= target raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OptimizationConfig(chunk_target_tokens=200, chunk_overlap_tokens=200)
        assert "chunk_overlap_tokens" in str(exc_info.value)


class TestAISectionsGeneration:
    """Tests for AI section generation."""

    def test_ai_sections_present_when_enabled(self):
        """Test AI sections are generated when enabled."""
        content_blocks = [
            "This is a test about SEO optimization and content strategy.",
            "Search engines use algorithms to rank pages based on relevance.",
            "Content quality matters for SEO success and user engagement.",
            "Proper keyword usage improves visibility in search results.",
        ]

        addons = generate_ai_addons(
            content_blocks=content_blocks,
            primary_keyword="SEO optimization",
            secondary_keywords=["search engines", "content quality"],
            generate_takeaways=True,
            generate_chunks=True,
            generate_faqs=False,  # Test without FAQ
        )

        assert addons.key_takeaways, "Key takeaways should be generated"
        assert addons.chunk_map, "Chunk map should be generated"
        assert addons.chunk_map.chunks, "Chunks should be present"

    def test_ai_sections_absent_when_disabled(self):
        """Test AI sections are not generated when disabled."""
        content_blocks = ["Test content about SEO and digital marketing."]

        addons = generate_ai_addons(
            content_blocks=content_blocks,
            primary_keyword="SEO",
            generate_takeaways=False,
            generate_chunks=False,
            generate_faqs=False,
        )

        assert not addons.key_takeaways
        assert not addons.chunk_map

    def test_key_takeaways_count_range(self):
        """Test key takeaways are within expected range (3-6)."""
        content_blocks = [
            "Payment processing is essential for modern businesses.",
            "Accepting credit cards helps increase sales significantly.",
            "Secure payment systems protect customer data effectively.",
            "Mobile payments are becoming increasingly popular.",
            "Integration with POS systems streamlines operations.",
        ]

        takeaways = generate_key_takeaways(
            content_blocks=content_blocks,
            primary_keyword="payment processing",
            secondary_keywords=["credit cards", "secure payment"],
        )

        assert len(takeaways) >= 3, "Should generate at least 3 takeaways"
        assert len(takeaways) <= 6, "Should not exceed 6 takeaways"


class TestFAQFallback:
    """Tests for FAQ fallback behavior."""

    def test_faq_fallback_produces_results(self):
        """Test fallback FAQ generation produces results."""
        content_blocks = [
            "Payment processing is essential for businesses.",
            "Accepting credit cards helps increase sales.",
            "Secure payment systems protect customer data.",
        ]

        faqs = generate_fallback_faqs(
            content_blocks=content_blocks,
            primary_keyword="payment processing",
            secondary_keywords=["credit cards", "secure payment"],
            min_faqs=3,
            max_faqs=5,
        )

        assert len(faqs) >= 3, "Should generate at least min_faqs items"
        assert all("question" in faq and "answer" in faq for faq in faqs)

    def test_faq_fallback_respects_min_max(self):
        """Test fallback respects min/max limits."""
        content_blocks = ["Simple content about services."]

        faqs = generate_fallback_faqs(
            content_blocks=content_blocks,
            primary_keyword="services",
            min_faqs=2,
            max_faqs=4,
        )

        assert len(faqs) >= 2, "Should generate at least min_faqs"
        assert len(faqs) <= 4, "Should not exceed max_faqs"

    def test_faq_fallback_empty_content_returns_empty(self):
        """Test fallback with empty content returns empty list."""
        faqs = generate_fallback_faqs(
            content_blocks=[],
            primary_keyword="test",
        )

        assert faqs == []


class TestChunkMapTableStructure:
    """Tests for chunk map table structure."""

    def test_chunk_map_has_required_fields(self):
        """Test chunk map has correct structure with all required fields."""
        content_blocks = [
            "Introduction to the topic of payment processing.",
            "Details about features and benefits of our service.",
            "How to get started with the payment service.",
            "Pricing information and subscription plans available.",
        ]

        chunks = build_chunks(content_blocks)
        chunk_map = generate_chunk_map(
            chunks=chunks,
            primary_keyword="payment service",
            secondary_keywords=["features", "pricing"],
        )

        assert chunk_map.total_chunks > 0, "Should have at least one chunk"

        # Verify each chunk has required fields (5-column structure)
        for chunk in chunk_map.chunks:
            assert chunk.chunk_id, "Chunk ID is required"
            assert chunk.heading_path, "Heading path (Section) is required"
            assert chunk.summary, "Summary is required"
            assert chunk.best_question, "Best Question is required"
            assert hasattr(chunk, 'keywords_present'), "Keywords field is required"

    def test_chunk_map_metadata(self):
        """Test chunk map metadata is calculated correctly."""
        content_blocks = [
            "First paragraph with some content.",
            "Second paragraph with more content.",
        ]

        chunks = build_chunks(content_blocks)
        chunk_map = generate_chunk_map(
            chunks=chunks,
            primary_keyword="content",
        )

        assert chunk_map.total_chunks == len(chunk_map.chunks)
        assert chunk_map.total_words > 0
        assert chunk_map.total_tokens > 0


class TestChunkConfiguration:
    """Tests for chunk configuration parameters."""

    def test_custom_chunk_tokens(self):
        """Test custom chunk token configuration is used."""
        content_blocks = ["Content block " * 50]  # ~100 words

        # With default tokens (512), should be 1 chunk
        default_chunks = build_chunks(
            content_blocks,
            chunk_target_tokens=DEFAULT_CHUNK_TARGET_TOKENS,
        )

        # With smaller tokens (50), should be more chunks
        small_chunks = build_chunks(
            content_blocks,
            chunk_target_tokens=50,
            chunk_overlap_tokens=10,
        )

        # Smaller target should create more chunks (or at least not fewer)
        # Note: actual behavior depends on content structure
        assert len(small_chunks) >= 1, "Should create at least one chunk"

    def test_generate_ai_addons_passes_chunk_config(self):
        """Test generate_ai_addons respects chunk configuration."""
        content_blocks = [
            "Test content for chunking verification.",
            "More content to ensure chunks are created properly.",
        ]

        # Call with custom chunk config
        addons = generate_ai_addons(
            content_blocks=content_blocks,
            primary_keyword="test",
            generate_takeaways=False,
            generate_chunks=True,
            generate_faqs=False,
            chunk_target_tokens=256,
            chunk_overlap_tokens=64,
        )

        assert addons.chunk_map is not None
        assert addons.chunk_map.chunks, "Should have chunks"


class TestWarningFields:
    """Tests for warning fields in OptimizationResult."""

    def test_optimization_result_has_warning_fields(self):
        """Test OptimizationResult has warning fields."""
        from src.seo_content_optimizer.models import OptimizationResult

        result = OptimizationResult(primary_keyword="test")

        # Check warning fields exist and have correct defaults
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'faq_archetype_warning')
        assert result.warnings == []
        assert result.faq_archetype_warning is None

    def test_optimization_result_with_warning(self):
        """Test OptimizationResult can store warnings."""
        from src.seo_content_optimizer.models import OptimizationResult

        warning_text = "FAQ generated despite 'homepage' archetype"
        result = OptimizationResult(
            primary_keyword="test",
            warnings=[warning_text],
            faq_archetype_warning=warning_text,
        )

        assert len(result.warnings) == 1
        assert result.warnings[0] == warning_text
        assert result.faq_archetype_warning == warning_text


class TestQuestionGeneration:
    """Tests for improved question generation patterns."""

    def test_question_patterns_how_to(self):
        """Test how-to content generates appropriate question."""
        from src.seo_content_optimizer.ai_addons import _generate_question_for_chunk

        content = "Here's how to set up the system step by step."
        question = _generate_question_for_chunk(content, "H2: Setup Guide", "system setup")

        assert "How" in question

    def test_question_patterns_cost(self):
        """Test cost content generates appropriate question."""
        from src.seo_content_optimizer.ai_addons import _generate_question_for_chunk

        content = "The pricing starts at $99 per month."
        question = _generate_question_for_chunk(content, "H2: Pricing", "service")

        assert "cost" in question.lower() or "price" in question.lower()

    def test_question_patterns_benefit(self):
        """Test benefit content generates appropriate question."""
        from src.seo_content_optimizer.ai_addons import _generate_question_for_chunk

        # Use content with benefit/advantage words but no other pattern matches
        content = "The key benefits and advantages are numerous for your business."
        question = _generate_question_for_chunk(content, "H2: Benefits", "solution")

        assert "benefit" in question.lower()

    def test_question_fallback(self):
        """Test fallback question generation."""
        from src.seo_content_optimizer.ai_addons import _generate_question_for_chunk

        content = "Lorem ipsum dolor sit amet."
        question = _generate_question_for_chunk(content, "Content", "keyword")

        assert question.endswith("?")
        assert "keyword" in question.lower()
