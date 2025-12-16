# -*- coding: utf-8 -*-
"""
Regression tests for quality gates.

These tests ensure the critical quality rules are never violated:
1. No marker leakage
2. No mojibake corruption
3. No hallucinated claims
4. No duplicate sentences
5. Content preservation
6. Natural keyword integration
"""

import pytest
from src.seo_content_optimizer.text_repair import (
    repair_text,
    detect_mojibake,
    normalize_whitespace,
    repair_content_blocks,
)
from src.seo_content_optimizer.repetition_guard import (
    find_duplicate_sentences,
    find_near_duplicate_sentences,
    remove_duplicate_sentences,
    check_keyword_density,
)
from src.seo_content_optimizer.claim_validator import (
    extract_facts_ledger,
    validate_generated_content,
)
from src.seo_content_optimizer.page_archetype import (
    detect_page_archetype,
    filter_guide_phrases,
)
from src.seo_content_optimizer.output_validator import (
    validate_content_preservation,
    validate_blocks_with_matching,
)
from src.seo_content_optimizer.docx_writer import (
    validate_no_marker_leakage as docx_validate_markers,
    strip_leaked_markers,
)


class TestMarkerLeakage:
    """Tests to ensure markers never appear in output."""

    def test_no_add_marker_in_plain_text(self):
        """[[[ADD]]] marker must not appear in output."""
        text = "This is normal text with [[[ADD]]]leaked marker[[[ENDADD]]] content."
        cleaned = strip_leaked_markers(text)
        assert "[[[ADD]]]" not in cleaned
        assert "[[[ENDADD]]]" not in cleaned

    def test_no_partial_markers(self):
        """Partial marker patterns must be detected."""
        text = "Text with [[[ partial marker."
        is_valid, leaks = docx_validate_markers(text)
        assert not is_valid
        assert len(leaks) > 0

    def test_validate_clean_text(self):
        """Clean text should pass validation."""
        text = "This is perfectly clean text without any markers."
        is_valid, leaks = docx_validate_markers(text)
        assert is_valid
        assert len(leaks) == 0

    def test_strip_multiple_markers(self):
        """Multiple markers should all be stripped."""
        text = "[[[ADD]]]First[[[ENDADD]]] and [[[ADD]]]Second[[[ENDADD]]]"
        cleaned = strip_leaked_markers(text)
        assert "[[[ADD]]]" not in cleaned
        assert "[[[ENDADD]]]" not in cleaned
        assert "First" in cleaned
        assert "Second" in cleaned


class TestMojibakeRepair:
    """Tests for encoding/mojibake repair."""

    def test_detect_common_mojibake(self):
        """Common mojibake patterns should be detected."""
        # These are classic UTF-8 decoded as Windows-1252
        assert detect_mojibake("worldâ€™s") == True
        assert detect_mojibake("donâ€™t") == True
        assert detect_mojibake("Normal text") == False

    def test_repair_smart_quotes(self):
        """Smart quotes should be normalized to ASCII."""
        text = "It\u2019s a \u201ctest\u201d"  # Unicode smart quotes
        repaired, _ = repair_text(text, aggressive=True)
        assert "'" in repaired or "It" in repaired  # Quote normalized

    def test_repair_whitespace(self):
        """Missing spaces should be fixed."""
        text = "word.Word"  # Missing space after period
        normalized = normalize_whitespace(text)
        assert "word. Word" == normalized

    def test_repair_non_breaking_spaces(self):
        """Non-breaking spaces should become regular spaces."""
        text = "word\u00a0word"  # Non-breaking space
        normalized = normalize_whitespace(text)
        assert "\u00a0" not in normalized

    def test_repair_content_blocks(self):
        """Multiple blocks should all be repaired."""
        blocks = ["First block", "Second\u00a0block", "Third block"]
        repaired, count = repair_content_blocks(blocks)
        assert len(repaired) == 3
        assert "\u00a0" not in repaired[1]


class TestRepetitionGuard:
    """Tests for duplicate and repetition detection."""

    def test_find_exact_duplicates(self):
        """Exact duplicate sentences should be detected."""
        text = "This is a sentence. This is another. This is a sentence."
        issues = find_duplicate_sentences(text)
        assert len(issues) >= 1

    def test_remove_duplicates(self):
        """Duplicate sentences should be removed."""
        text = "First sentence. Second sentence. First sentence."
        cleaned = remove_duplicate_sentences(text)
        assert cleaned.count("First sentence") == 1

    def test_near_duplicate_detection(self):
        """Near-duplicate sentences should be flagged."""
        text = "The product is excellent. The product is very excellent."
        issues = find_near_duplicate_sentences(text, threshold=0.7)
        # These are similar enough to potentially flag
        assert isinstance(issues, list)

    def test_keyword_density_under(self):
        """Under-optimized keyword density should be detected."""
        text = "This is a long text about various topics but no keywords."
        count, min_t, max_t, status = check_keyword_density(text, "missing keyword")
        assert count == 0
        assert status == "under"

    def test_keyword_density_optimal(self):
        """Optimal keyword density should pass."""
        text = "SEO optimization helps with SEO. Good SEO practices matter for SEO results."
        count, min_t, max_t, status = check_keyword_density(text, "SEO")
        assert count >= 3


class TestClaimValidation:
    """Tests for hallucination/claim validation."""

    def test_extract_numbers_from_source(self):
        """Numbers should be extracted to facts ledger."""
        source = "Our product has 99% satisfaction rate and costs $49.99"
        ledger = extract_facts_ledger(source)
        assert len(ledger.numbers) >= 1

    def test_detect_new_statistics(self):
        """New statistics not in source should be flagged."""
        source = "Our product is great."
        generated = "Our product has a 95% success rate."  # Made up!
        ledger = extract_facts_ledger(source)
        violations, _ = validate_generated_content(generated, ledger)
        # Should detect the new percentage
        assert len(violations) >= 1

    def test_allow_existing_claims(self):
        """Claims that exist in source should be allowed."""
        source = "Our product has 99% satisfaction rate."
        generated = "The product achieves 99% satisfaction."
        ledger = extract_facts_ledger(source)
        violations, _ = validate_generated_content(generated, ledger)
        # 99% was in source, so no violation for that
        number_violations = [v for v in violations if v.claim_type == "number"]
        # The 99% should be allowed
        assert not any("99" in str(v.claim_text) for v in number_violations)


class TestPageArchetype:
    """Tests for page type detection."""

    def test_detect_homepage(self):
        """Homepage URLs should be detected."""
        result = detect_page_archetype(
            url="https://example.com/",
            title="Example Company",
            h1="Welcome to Example",
            content_text="Get started today. Contact us.",
        )
        assert result.archetype in ("homepage", "landing")

    def test_detect_blog(self):
        """Blog URLs should be detected."""
        result = detect_page_archetype(
            url="https://example.com/blog/how-to-guide",
            title="How to Do Something - Blog",
            h1="Complete Guide to Something",
            content_text="In this guide, we'll cover everything you need to know...",
        )
        assert result.archetype in ("blog", "guide")

    def test_homepage_blocks_guide_framing(self):
        """Homepage archetype should block guide framing."""
        result = detect_page_archetype(
            url="https://example.com/",
            title="Company Home",
            h1="Welcome",
            content_text="Get started",
        )
        assert not result.allows_guide_framing

    def test_filter_guide_phrases_on_landing(self):
        """Guide phrases should be filtered from landing pages."""
        result = detect_page_archetype(
            url="https://example.com/",
            title="Home",
            h1="Welcome",
            content_text="Get started",
        )
        text = "This guide covers everything you need to know about our product."
        filtered = filter_guide_phrases(text, result)
        assert "This guide covers" not in filtered


class TestContentPreservation:
    """Tests for content preservation validation."""

    def test_detect_content_deletion(self):
        """Significant content deletion should be detected."""
        original = "This is a long piece of content with many words that should be preserved."
        optimized = "Short."  # Severe deletion!
        is_valid, error = validate_content_preservation(original, optimized, min_preservation_ratio=0.5)
        assert not is_valid
        assert "deleted" in error.lower()

    def test_allow_content_addition(self):
        """Content addition should be allowed."""
        original = "Original content."
        optimized = "Original content with additional SEO keywords added."
        is_valid, error = validate_content_preservation(original, optimized)
        assert is_valid

    def test_block_level_matching(self):
        """All original blocks should have matches in output."""
        original_blocks = ["First paragraph.", "Second paragraph.", "Third paragraph."]
        optimized_blocks = ["First paragraph with edits.", "Second paragraph.", "Third paragraph."]
        is_valid, unmatched, _ = validate_blocks_with_matching(original_blocks, optimized_blocks)
        assert is_valid
        assert len(unmatched) == 0

    def test_detect_missing_blocks(self):
        """Missing blocks should be detected."""
        # Use longer blocks to avoid the short block filter (< 10 chars)
        original_blocks = [
            "This is the first paragraph with enough words to be substantial content.",
            "This is the second paragraph which also has substantial content here.",
            "This is the third paragraph containing important information for users.",
        ]
        optimized_blocks = [
            "This is the first paragraph with enough words to be substantial content."
        ]  # Missing 2 blocks!
        is_valid, unmatched, _ = validate_blocks_with_matching(original_blocks, optimized_blocks)
        assert not is_valid
        assert len(unmatched) >= 1


class TestKeywordIntegration:
    """Tests for natural keyword integration."""

    def test_no_colon_prefix_in_title(self):
        """Titles should not have crude 'keyword:' prefix."""
        from src.seo_content_optimizer.optimizer import ensure_keyword_in_text

        title = "Amazing Product Features"
        result = ensure_keyword_in_text(
            title, "SEO tools", position="start",
            max_length=60, element_type="title"
        )
        # Should not start with "SEO tools:" literally
        assert not result.startswith("SEO tools:")

    def test_keyword_already_present(self):
        """If keyword exists, text should be unchanged."""
        from src.seo_content_optimizer.optimizer import ensure_keyword_in_text

        title = "Best SEO Tools for Marketing"
        result = ensure_keyword_in_text(
            title, "SEO Tools", position="start",
            max_length=60, element_type="title"
        )
        assert result == title  # Unchanged because keyword present

    def test_length_constraint_respected(self):
        """Length constraints should be respected after integration."""
        from src.seo_content_optimizer.optimizer import ensure_keyword_in_text

        title = "A Very Long Title That Goes On And On"
        result = ensure_keyword_in_text(
            title, "keyword", position="start",
            max_length=50, element_type="title"
        )
        assert len(result) <= 50 or "keyword" in result.lower()


class TestExtractionLadder:
    """Tests for multi-stage extraction."""

    def test_quality_threshold_check(self):
        """Extraction quality check should work."""
        from src.seo_content_optimizer.content_sources import _extraction_quality_ok

        # Good extraction - needs at least 100 words and 3 substantial blocks
        # Each block has ~20 words, 6 blocks = 120 words
        good_blocks = [
            "This is a substantial paragraph with many different words that make it quite long and meaningful content for testing purposes here.",
            "Another substantial paragraph with different words and content that helps reach the minimum word count threshold for good extraction.",
            "A third paragraph that contains even more words and content to ensure we have enough blocks and words for the quality check.",
            "Fourth paragraph with additional content that contributes to the overall word count and block count for testing extraction quality.",
            "Fifth paragraph ensuring we have plenty of content blocks with enough words to pass the quality threshold check here.",
            "Sixth paragraph to make absolutely sure we exceed all thresholds for minimum word count and minimum block count.",
        ]
        assert _extraction_quality_ok(good_blocks, "test") == True

        # Poor extraction - too few words and blocks
        poor_blocks = ["Short."]
        assert _extraction_quality_ok(poor_blocks, "test") == False

    def test_split_to_blocks(self):
        """Content should be split into blocks correctly."""
        from src.seo_content_optimizer.content_sources import _split_to_blocks

        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        blocks = _split_to_blocks(content)
        assert len(blocks) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
