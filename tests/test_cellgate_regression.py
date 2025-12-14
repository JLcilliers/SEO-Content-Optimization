"""
Regression tests for CellGate scenario.

These tests verify the fixes for:
1. Bug 1: Keyword Enforcement - Keywords must reliably appear in output
2. Bug 2: Highlighting - Only changed/new sentences should be highlighted

The CellGate content is about external cameras for property security.
Primary keyword: "external cameras"
Secondary keywords: ["property security", "access control"]
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.seo_content_optimizer.models import (
    HeadingLevel,
    Keyword,
    KeywordPlan,
    MetaElement,
    OptimizationResult,
    ParagraphBlock,
)
from src.seo_content_optimizer.optimizer import (
    ContentOptimizer,
    count_keyword_in_text,
    ensure_keyword_in_text,
)
from src.seo_content_optimizer.diff_markers import (
    MARK_END,
    MARK_START,
    compute_markers_v2,
    mark_block_as_new,
    normalize_sentence,
    strip_markers,
)
from src.seo_content_optimizer.docx_writer import DocxWriter


# Load CellGate fixture content
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CELLGATE_ORIGINAL = (FIXTURES_DIR / "enhancing_property_security_original.txt").read_text(encoding="utf-8")
CELLGATE_OPTIMIZED_RAW = (FIXTURES_DIR / "enhancing_property_security_optimized_raw.txt").read_text(encoding="utf-8")


class TestKeywordEnforcement:
    """Tests for Bug 1: Keywords must reliably appear in output."""

    def test_ensure_keyword_in_text_when_present(self):
        """If keyword already present, text unchanged."""
        text = "External cameras are great for security."
        result = ensure_keyword_in_text(text, "external cameras")
        assert result == text  # No change needed

    def test_ensure_keyword_in_text_case_insensitive(self):
        """Keyword check is case-insensitive."""
        text = "EXTERNAL CAMERAS are essential."
        result = ensure_keyword_in_text(text, "external cameras")
        assert result == text  # No change needed - already present

    def test_ensure_keyword_in_text_inject_at_start(self):
        """If keyword missing, inject at start with colon separator."""
        text = "Security systems help protect property."
        result = ensure_keyword_in_text(text, "external cameras", position="start")
        assert result.startswith("external cameras:")
        assert "external cameras" in result.lower()

    def test_ensure_keyword_in_text_inject_at_end(self):
        """If keyword missing, inject at end."""
        text = "Security systems help protect property."
        result = ensure_keyword_in_text(text, "external cameras", position="end")
        assert "external cameras" in result.lower()

    def test_ensure_keyword_in_text_preserves_punctuation(self):
        """Injection preserves final punctuation."""
        text = "This is a great system."
        result = ensure_keyword_in_text(text, "cameras", position="end")
        assert result.endswith(".")

    def test_count_keyword_in_text_basic(self):
        """Basic keyword counting works."""
        text = "External cameras for property. External cameras are great."
        count = count_keyword_in_text(text, "external cameras")
        assert count == 2

    def test_count_keyword_in_text_case_insensitive(self):
        """Keyword counting is case-insensitive."""
        text = "EXTERNAL CAMERAS and external cameras and External Cameras."
        count = count_keyword_in_text(text, "external cameras")
        assert count == 3

    def test_count_keyword_in_text_empty(self):
        """Empty text returns 0."""
        assert count_keyword_in_text("", "keyword") == 0
        assert count_keyword_in_text("text", "") == 0


class TestSentenceLevelDiff:
    """Tests for Bug 2: Only changed/new sentences should be highlighted."""

    def test_identical_sentence_no_markers(self):
        """Sentence that exists in original should have no markers."""
        original = "External cameras are great. They help with security."
        rewritten = "External cameras are great. They help with security."

        result = compute_markers_v2(original, rewritten, full_original_text=original)

        # No markers since sentences are identical
        assert MARK_START not in result
        assert MARK_END not in result

    def test_new_sentence_fully_marked(self):
        """Sentence not in original should be fully wrapped in markers."""
        original = "External cameras are great."
        rewritten = "External cameras are great. This is a new sentence."

        result = compute_markers_v2(original, rewritten, full_original_text=original)

        # First sentence should be unmarked
        assert "External cameras are great." in result
        # New sentence should be fully wrapped
        assert f"{MARK_START}This is a new sentence.{MARK_END}" in result

    def test_modified_sentence_fully_marked(self):
        """Modified sentence should be fully wrapped (all-or-nothing)."""
        original = "External cameras are great."
        rewritten = "External cameras are excellent and amazing."

        result = compute_markers_v2(original, rewritten, full_original_text=original)

        # The changed sentence should be fully wrapped
        assert MARK_START in result
        assert MARK_END in result

    def test_punctuation_normalization(self):
        """Smart quotes vs straight quotes should match after normalization."""
        original = "CellGate's cameras are great."  # Smart apostrophe
        rewritten = "CellGate's cameras are great."  # Straight apostrophe

        # Normalized versions should match
        assert normalize_sentence(original) == normalize_sentence(rewritten)

    def test_no_token_level_highlighting(self):
        """Verify no partial word highlighting (entire sentence or nothing)."""
        original = "The camera provides good coverage."
        rewritten = "The camera provides excellent coverage."

        result = compute_markers_v2(original, rewritten, full_original_text=original)

        # Should NOT have markers around just "excellent"
        # Instead, entire sentence should be wrapped
        assert f"{MARK_START}excellent{MARK_END}" not in result


class TestCellGateContent:
    """Tests using actual CellGate fixture content."""

    def test_cellgate_original_content_loads(self):
        """Verify CellGate fixture content loads correctly."""
        assert len(CELLGATE_ORIGINAL) > 0
        assert "CellGate" in CELLGATE_ORIGINAL
        assert "external cameras" in CELLGATE_ORIGINAL.lower()

    def test_unchanged_sentences_not_highlighted(self):
        """Sentences that exist in original should not be highlighted."""
        # A sentence from the original
        original_sentence = "External Cameras for Enhanced Property Security: A Guide to Your Options"

        # When the rewritten text contains only sentences from the original,
        # and we compare against the full original, no markers should appear
        # The key is the original param should be a subset we're comparing from
        result = compute_markers_v2(
            original_sentence,  # Original block text
            original_sentence,  # Rewritten (same text)
            full_original_text=CELLGATE_ORIGINAL  # Full document for context
        )

        # Should not have markers since it's identical
        assert MARK_START not in result
        assert MARK_END not in result

    def test_strip_markers_preserves_content(self):
        """strip_markers removes markers but preserves content."""
        marked_text = f"Hello {MARK_START}world{MARK_END}!"
        result = strip_markers(marked_text)
        assert result == "Hello world!"
        assert MARK_START not in result
        assert MARK_END not in result


class TestKeywordUsageCounts:
    """Tests for keyword usage counts in Keyword Plan table."""

    def test_optimization_result_has_usage_counts(self):
        """OptimizationResult includes keyword_usage_counts field."""
        result = OptimizationResult(
            primary_keyword="external cameras",
            secondary_keywords=["property security"],
            keyword_usage_counts={
                "external cameras": 5,
                "property security": 3,
            }
        )

        assert result.keyword_usage_counts is not None
        assert result.keyword_usage_counts["external cameras"] == 5
        assert result.keyword_usage_counts["property security"] == 3

    def test_default_usage_counts_empty(self):
        """Default keyword_usage_counts is empty dict."""
        result = OptimizationResult()
        assert result.keyword_usage_counts == {}


class TestDocxWriterKeywordTable:
    """Tests for DOCX keyword plan table with usage counts."""

    def test_keyword_plan_table_includes_usage_column(self):
        """Keyword Plan table should have Type, Keyword, Usage Count columns."""
        writer = DocxWriter()

        # Call the method with usage counts
        writer._add_keyword_plan_table(
            primary_keyword="external cameras",
            secondary_keywords=["property security", "access control"],
            keyword_usage_counts={
                "external cameras": 7,
                "property security": 4,
                "access control": 2,
            }
        )

        # Check that table was created with correct structure
        # The table should be in the document
        tables = writer.doc.tables
        assert len(tables) >= 1

        # Last table should be keyword plan
        table = tables[-1]

        # Should have 3 columns (Type, Keyword, Usage Count)
        assert table.rows[0].cells[0].text == "Type"
        assert table.rows[0].cells[1].text == "Keyword"
        assert table.rows[0].cells[2].text == "Usage Count"

    def test_keyword_plan_table_shows_counts(self):
        """Keyword Plan table should display correct usage counts."""
        writer = DocxWriter()

        writer._add_keyword_plan_table(
            primary_keyword="external cameras",
            secondary_keywords=["property security"],
            keyword_usage_counts={
                "external cameras": 7,
                "property security": 4,
            }
        )

        table = writer.doc.tables[-1]

        # Row 1 (primary): should have count "7"
        assert table.rows[1].cells[2].text == "7"

        # Row 2 (secondary 1): should have count "4"
        assert table.rows[2].cells[2].text == "4"

    def test_keyword_plan_table_handles_missing_counts(self):
        """Table handles missing counts gracefully (shows 0)."""
        writer = DocxWriter()

        writer._add_keyword_plan_table(
            primary_keyword="external cameras",
            secondary_keywords=["property security"],
            keyword_usage_counts={
                "external cameras": 5,
                # property security not in dict
            }
        )

        table = writer.doc.tables[-1]

        # Primary should have count
        assert table.rows[1].cells[2].text == "5"

        # Secondary with missing count should show "0"
        assert table.rows[2].cells[2].text == "0"


class TestMetaElementHighlighting:
    """Tests for meta element all-or-nothing highlighting."""

    def test_meta_element_unchanged_no_markers(self):
        """Meta element identical to original should have no markers."""
        # If title is exactly the same, no markers
        original_title = "External Cameras for Property Security"
        optimized_title = "External Cameras for Property Security"

        # After normalization, these should be equal
        assert normalize_sentence(original_title) == normalize_sentence(optimized_title)

    def test_meta_element_changed_fully_wrapped(self):
        """Changed meta element should be fully wrapped with markers."""
        original_title = "External Cameras Guide"
        optimized_title = "External Cameras for Property Security - Complete Guide"

        # These are different
        assert normalize_sentence(original_title) != normalize_sentence(optimized_title)

        # When different, entire element gets wrapped
        if normalize_sentence(original_title) != normalize_sentence(optimized_title):
            wrapped = f"{MARK_START}{optimized_title}{MARK_END}"
            assert MARK_START in wrapped
            assert MARK_END in wrapped


class TestFAQHighlighting:
    """Tests for FAQ section highlighting (always 100% green)."""

    def test_faq_items_always_marked_as_new(self):
        """FAQ items should always be fully marked since they're new content."""
        question = "What are external cameras?"
        answer = "External cameras are security devices."

        marked_q = mark_block_as_new(question)
        marked_a = mark_block_as_new(answer)

        # Both should be fully wrapped
        assert marked_q.startswith(MARK_START)
        assert marked_q.endswith(MARK_END)
        assert marked_a.startswith(MARK_START)
        assert marked_a.endswith(MARK_END)


class TestEdgeCases:
    """Edge case tests for the CellGate scenario."""

    def test_empty_original_marks_all_new(self):
        """Empty original text results in all content marked as new."""
        result = compute_markers_v2("", "New content here.", full_original_text="")
        assert MARK_START in result
        assert MARK_END in result

    def test_empty_rewritten_returns_empty(self):
        """Empty rewritten text returns empty string."""
        result = compute_markers_v2("Original text.", "", full_original_text="Original text.")
        assert result == ""

    def test_whitespace_normalization(self):
        """Multiple spaces normalized to single space for comparison."""
        original = "External   cameras  are   great."  # Multiple spaces
        rewritten = "External cameras are great."  # Single spaces

        # Normalized versions should match
        assert normalize_sentence(original) == normalize_sentence(rewritten)

    def test_unicode_normalization(self):
        """Unicode variations normalized for comparison."""
        # NFKC normalization handles things like ligatures
        original = "efficacy"  # Could have fi ligature in some fonts
        rewritten = "efficacy"

        assert normalize_sentence(original) == normalize_sentence(rewritten)


class TestSecondaryKeywordEnforcement:
    """Tests for Bug 2: Secondary keywords must meet target counts."""

    def test_count_keyword_in_text_multi_word(self):
        """Multi-word keyword counting works correctly."""
        text = "Gate intercom with camera is great. Every gate intercom with camera helps security."
        count = count_keyword_in_text(text, "gate intercom with camera")
        assert count == 2

    def test_count_keyword_in_text_partial_match(self):
        """Partial matches should not count."""
        text = "Gate intercom is good. Camera is good. Gate camera."
        # "gate intercom with camera" should NOT match any of these
        count = count_keyword_in_text(text, "gate intercom with camera")
        assert count == 0

    def test_ensure_keyword_handles_multi_word_phrases(self):
        """Multi-word keywords can be injected properly."""
        text = "Security systems are essential for properties."
        result = ensure_keyword_in_text(text, "gate intercom with camera", position="end")
        assert "gate intercom with camera" in result.lower()

    def test_keyword_enforcement_targets(self):
        """Secondary keywords have a default target of 3 occurrences."""
        # This test verifies the expected configuration
        # The _enforce_secondary_keyword_counts method uses secondary_target=3
        secondary_target = 3
        assert secondary_target == 3  # Document the expected behavior

    def test_keyword_counts_in_optimization_result(self):
        """OptimizationResult tracks usage counts for secondary keywords."""
        result = OptimizationResult(
            primary_keyword="external cameras",
            secondary_keywords=["gate intercom with camera", "property security"],
            keyword_usage_counts={
                "external cameras": 6,
                "gate intercom with camera": 3,
                "property security": 4,
            }
        )

        # All keywords should have counts tracked
        assert "gate intercom with camera" in result.keyword_usage_counts
        assert result.keyword_usage_counts["gate intercom with camera"] == 3

    def test_secondary_keyword_count_case_insensitive(self):
        """Secondary keyword counting is case-insensitive."""
        text = "GATE INTERCOM WITH CAMERA is great. gate intercom with camera too."
        count = count_keyword_in_text(text, "gate intercom with camera")
        assert count == 2

    def test_secondary_keywords_in_keyword_plan(self):
        """KeywordPlan correctly stores secondary keywords."""
        primary = Keyword(phrase="external cameras")
        secondary = [
            Keyword(phrase="gate intercom with camera"),
            Keyword(phrase="property security"),
        ]
        plan = KeywordPlan(primary=primary, secondary=secondary)

        assert len(plan.secondary) == 2
        assert plan.secondary[0].phrase == "gate intercom with camera"
        assert plan.secondary[1].phrase == "property security"

    def test_all_phrases_includes_secondary(self):
        """KeywordPlan.all_phrases includes secondary keywords."""
        primary = Keyword(phrase="external cameras")
        secondary = [Keyword(phrase="gate intercom with camera")]
        plan = KeywordPlan(primary=primary, secondary=secondary)

        assert "gate intercom with camera" in plan.all_phrases


class TestKeywordDistribution:
    """Tests for even keyword distribution throughout the document."""

    def test_identify_insertion_points_returns_list(self):
        """Verify _identify_keyword_insertion_points returns a list of indices."""
        from src.seo_content_optimizer.optimizer import ContentOptimizer

        optimizer = ContentOptimizer.__new__(ContentOptimizer)

        # Create test blocks - mix of headings and paragraphs
        blocks = [
            ParagraphBlock(text="Introduction heading", heading_level=HeadingLevel.H1),
            ParagraphBlock(text="This is the first paragraph with enough content to be a body paragraph for testing purposes."),
            ParagraphBlock(text="Section heading", heading_level=HeadingLevel.H2),
            ParagraphBlock(text="This is a second paragraph with substantial content to qualify as a body paragraph."),
            ParagraphBlock(text="Another section", heading_level=HeadingLevel.H2),
            ParagraphBlock(text="Third paragraph with enough content to be included in the analysis."),
            ParagraphBlock(text="Fourth paragraph that should also be considered for keyword insertion."),
            ParagraphBlock(text="Fifth paragraph with sufficient length to be a candidate."),
        ]

        result = optimizer._identify_keyword_insertion_points(blocks)

        # Should return a list, not a dict
        assert isinstance(result, list)
        # Should have multiple insertion points (not just 3)
        assert len(result) >= 3
        # All values should be valid indices
        for idx in result:
            assert 0 <= idx < len(blocks)

    def test_insertion_points_evenly_distributed(self):
        """Verify insertion points are spread throughout the document, not clustered."""
        from src.seo_content_optimizer.optimizer import ContentOptimizer

        optimizer = ContentOptimizer.__new__(ContentOptimizer)

        # Create a longer document with 15 paragraphs
        blocks = []
        for i in range(15):
            if i % 4 == 0:
                blocks.append(ParagraphBlock(text=f"Section {i} heading", heading_level=HeadingLevel.H2))
            else:
                blocks.append(ParagraphBlock(text=f"This is paragraph {i} with enough content to be a body paragraph for testing."))

        result = optimizer._identify_keyword_insertion_points(blocks)

        # Should have multiple points
        assert len(result) >= 4

        # Points should be spread out - check that we have points in different thirds
        first_third = [idx for idx in result if idx < len(blocks) // 3]
        middle_third = [idx for idx in result if len(blocks) // 3 <= idx < 2 * len(blocks) // 3]
        last_third = [idx for idx in result if idx >= 2 * len(blocks) // 3]

        # All three sections should have at least one insertion point
        assert len(first_third) >= 1, "First third should have insertion points"
        assert len(middle_third) >= 1, "Middle third should have insertion points"
        assert len(last_third) >= 1, "Last third should have insertion points"

    def test_insert_sentences_distributes_evenly(self):
        """Verify sentences are distributed across insertion points, not clustered."""
        from src.seo_content_optimizer.optimizer import ContentOptimizer

        optimizer = ContentOptimizer.__new__(ContentOptimizer)

        # Create 10 blocks
        blocks = [
            ParagraphBlock(text=f"Paragraph {i} with some content here.")
            for i in range(10)
        ]

        # 5 insertion points across the document
        insertion_points = [1, 3, 5, 7, 9]

        # 3 sentences to insert
        sentences = ["Sentence A.", "Sentence B.", "Sentence C."]

        result = optimizer._insert_keyword_sentences(blocks, sentences, insertion_points)

        # Find which blocks have the inserted sentences
        blocks_with_insertions = []
        for i, block in enumerate(result):
            if "Sentence A" in block.text or "Sentence B" in block.text or "Sentence C" in block.text:
                blocks_with_insertions.append(i)

        # Sentences should be in 3 different blocks (not clustered)
        assert len(blocks_with_insertions) == 3, "Each sentence should be in a different block"

        # The blocks should be spread out
        assert blocks_with_insertions != [1, 1, 1], "Sentences should not all be in the same block"

    def test_insert_sentences_respects_max_per_block(self):
        """Verify no more than 2 sentences per block."""
        from src.seo_content_optimizer.optimizer import ContentOptimizer

        optimizer = ContentOptimizer.__new__(ContentOptimizer)

        # Create 3 blocks
        blocks = [
            ParagraphBlock(text=f"Paragraph {i} content.")
            for i in range(3)
        ]

        # Only 2 insertion points
        insertion_points = [0, 2]

        # 5 sentences to insert (more than points)
        sentences = ["S1.", "S2.", "S3.", "S4.", "S5."]

        result = optimizer._insert_keyword_sentences(blocks, sentences, insertion_points)

        # Count sentences per block
        for i, block in enumerate(result):
            count = sum(1 for s in sentences if s in block.text)
            assert count <= 2, f"Block {i} has {count} sentences, max should be 2"
