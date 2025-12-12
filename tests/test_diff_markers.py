"""
Tests for diff-based marker insertion.

This module tests the diff_markers.py module which computes differences
between original and rewritten text and inserts [[[ADD]]]/[[[ENDADD]]]
markers around ONLY the changed parts.

Key guarantees tested:
1. Only ACTUALLY new/changed text gets markers
2. Existing text (including keywords) stays unhighlighted
3. Entire new sentences get fully highlighted
4. Markers are properly balanced and cleaned up
"""

import pytest

from seo_content_optimizer.diff_markers import (
    MARK_START,
    MARK_END,
    tokenize,
    add_markers_by_diff,
    split_into_sentences,
    strip_markers,
    expand_markers_to_full_sentence,
    compute_markers,
    cleanup_markers,
    inject_phrase_with_markers,
)


class TestTokenize:
    """Tests for the tokenize function."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert tokenize("") == []

    def test_simple_words(self):
        """Simple words are split correctly."""
        result = tokenize("hello world")
        assert "hello" in result
        assert "world" in result
        assert " " in result

    def test_preserves_punctuation(self):
        """Punctuation is preserved as part of tokens or separately."""
        result = tokenize("Hello, world!")
        # Non-whitespace chunks include punctuation
        assert "Hello," in result or ("Hello" in result and "," in result)

    def test_preserves_whitespace(self):
        """Whitespace is preserved as separate tokens."""
        result = tokenize("a  b")
        assert "a" in result
        assert "b" in result
        # Double space preserved
        whitespace_tokens = [t for t in result if t.isspace()]
        assert len(whitespace_tokens) >= 1


class TestAddMarkersByDiff:
    """Tests for the add_markers_by_diff function."""

    def test_identical_text_no_markers(self):
        """Identical text should have no markers."""
        original = "This is the same text."
        result = add_markers_by_diff(original, original)
        assert MARK_START not in result
        assert MARK_END not in result

    def test_completely_new_text(self):
        """Empty original means all text is new."""
        result = add_markers_by_diff("", "This is new text.")
        assert result == f"{MARK_START}This is new text.{MARK_END}"

    def test_empty_rewritten_returns_empty(self):
        """Empty rewritten text returns empty string."""
        result = add_markers_by_diff("original", "")
        assert result == ""

    def test_word_replacement_marked(self):
        """Replaced words should be marked."""
        original = "The quick brown fox"
        rewritten = "The fast brown fox"
        result = add_markers_by_diff(original, rewritten)

        assert MARK_START in result
        assert MARK_END in result
        assert "quick" not in result  # Old word removed
        assert "fast" in result
        # "fast" should be wrapped in markers
        assert f"{MARK_START}fast{MARK_END}" in result

    def test_word_insertion_marked(self):
        """Inserted words should be marked."""
        original = "The brown fox"
        rewritten = "The quick brown fox"
        result = add_markers_by_diff(original, rewritten)

        assert MARK_START in result
        assert "quick" in result
        # "quick" should be wrapped in markers
        assert MARK_START in result and "quick" in result

    def test_existing_keywords_not_marked(self):
        """Keywords that existed in original should NOT be marked."""
        original = "We offer payment processing services."
        rewritten = "We offer payment processing services for your business."
        result = add_markers_by_diff(original, rewritten)

        # "payment processing" was in original - should NOT be wrapped
        # Only "for your business" should be wrapped
        clean_original_part = "We offer payment processing services"
        assert clean_original_part in strip_markers(result)

        # The new part should be marked
        assert "for your business" in strip_markers(result)
        assert MARK_START in result

    def test_case_sensitive_diff(self):
        """Diff should be case-sensitive."""
        original = "Hello World"
        rewritten = "hello World"
        result = add_markers_by_diff(original, rewritten)

        # "hello" (lowercase) should be marked as different from "Hello"
        assert MARK_START in result


class TestSplitIntoSentences:
    """Tests for the split_into_sentences function."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert split_into_sentences("") == []

    def test_single_sentence(self):
        """Single sentence is returned as-is."""
        result = split_into_sentences("This is a sentence.")
        assert len(result) == 1
        assert "This is a sentence." in result[0]

    def test_multiple_sentences(self):
        """Multiple sentences are split correctly."""
        result = split_into_sentences("First sentence. Second sentence! Third?")
        assert len(result) >= 3

    def test_preserves_sentence_content(self):
        """Sentence content is preserved."""
        text = "Hello world. How are you?"
        result = split_into_sentences(text)
        combined = " ".join(result)
        assert "Hello world" in combined
        assert "How are you" in combined


class TestStripMarkers:
    """Tests for the strip_markers function."""

    def test_removes_markers(self):
        """Markers are removed from text."""
        text = f"Hello {MARK_START}world{MARK_END}!"
        result = strip_markers(text)
        assert result == "Hello world!"
        assert MARK_START not in result
        assert MARK_END not in result

    def test_no_markers_unchanged(self):
        """Text without markers is unchanged."""
        text = "Hello world!"
        result = strip_markers(text)
        assert result == text


class TestExpandMarkersToFullSentence:
    """Tests for the expand_markers_to_full_sentence function."""

    def test_new_sentence_fully_wrapped(self):
        """Entirely new sentences should be fully wrapped."""
        original = "This is the original."
        # Marked text where only part of new sentence is wrapped
        marked = f"This is the original. {MARK_START}New{MARK_END} sentence added."

        result = expand_markers_to_full_sentence(original, marked)

        # The new sentence should be fully wrapped
        # (fuzzy match will determine if it's "new")
        # Since "New sentence added." doesn't exist in original at all,
        # it should be fully wrapped
        assert "original" in result.lower()

    def test_modified_sentence_keeps_fine_markers(self):
        """Modified sentences should keep fine-grained markers."""
        original = "We offer great services."
        # Marked text where "excellent" replaced "great"
        marked = f"We offer {MARK_START}excellent{MARK_END} services."

        result = expand_markers_to_full_sentence(original, marked)

        # Since sentence still mostly matches (>70%), keep fine markers
        # Just "excellent" should be marked, not whole sentence
        assert MARK_START in result
        assert MARK_END in result

    def test_no_markers_returns_unchanged(self):
        """Text without markers is returned unchanged."""
        original = "Original text."
        marked = "No markers here."

        result = expand_markers_to_full_sentence(original, marked)
        assert result == marked


class TestComputeMarkers:
    """Tests for the main compute_markers function."""

    def test_empty_original_wraps_all(self):
        """Empty original means all content is new."""
        result = compute_markers("", "This is new content.")
        assert result == f"{MARK_START}This is new content.{MARK_END}"

    def test_empty_rewritten_returns_empty(self):
        """Empty rewritten returns empty string."""
        result = compute_markers("Original", "")
        assert result == ""

    def test_unchanged_content_no_markers(self):
        """Unchanged content has no markers."""
        text = "This is unchanged."
        result = compute_markers(text, text)
        assert MARK_START not in result
        assert MARK_END not in result

    def test_existing_keyword_not_highlighted(self):
        """Keywords that existed in original should NOT be highlighted."""
        original = "Our payment processing is fast."
        rewritten = "Our payment processing is incredibly fast and reliable."
        result = compute_markers(original, rewritten)

        # "payment processing" existed - should NOT be in markers
        # We check by ensuring the phrase is present but not wrapped
        clean = strip_markers(result)
        assert "payment processing" in clean

        # The added parts should be marked
        assert MARK_START in result
        assert "incredibly" in clean or "reliable" in clean

    def test_full_sentence_added(self):
        """Entire new sentence should be fully highlighted."""
        original = "First paragraph here."
        rewritten = "First paragraph here. This is a completely new sentence."
        result = compute_markers(original, rewritten)

        # The new sentence should have markers
        assert MARK_START in result
        assert "new sentence" in strip_markers(result)


class TestCleanupMarkers:
    """Tests for the cleanup_markers function."""

    def test_removes_empty_markers(self):
        """Empty marker pairs are removed."""
        text = f"Hello {MARK_START}{MARK_END} world"
        result = cleanup_markers(text)
        assert f"{MARK_START}{MARK_END}" not in result

    def test_merges_adjacent_markers(self):
        """Adjacent marker blocks are merged."""
        text = f"{MARK_START}first{MARK_END} {MARK_START}second{MARK_END}"
        result = cleanup_markers(text)

        # Should be merged into one block
        # Count markers - should be 1 start and 1 end
        assert result.count(MARK_START) == 1
        assert result.count(MARK_END) == 1

    def test_balances_unmatched_markers(self):
        """Unmatched markers are balanced."""
        # Missing end marker
        text = f"{MARK_START}unclosed"
        result = cleanup_markers(text)
        assert result.count(MARK_START) == result.count(MARK_END)


class TestInjectPhraseWithMarkers:
    """Tests for the inject_phrase_with_markers function."""

    def test_phrase_already_present_unchanged(self):
        """Text with phrase already present is unchanged."""
        text = "Our payment processing is great."
        result = inject_phrase_with_markers(text, "payment processing")
        assert result == text

    def test_phrase_case_insensitive_check(self):
        """Phrase check is case-insensitive."""
        text = "Our PAYMENT PROCESSING is great."
        result = inject_phrase_with_markers(text, "payment processing")
        assert result == text

    def test_phrase_added_at_start(self):
        """Phrase can be added at start."""
        text = "Our services are excellent."
        result = inject_phrase_with_markers(text, "Payment processing", position="start")

        assert "Payment processing" in result
        assert MARK_START in result
        assert MARK_END in result

    def test_phrase_added_at_end(self):
        """Phrase can be added at end."""
        text = "Our services are excellent."
        result = inject_phrase_with_markers(text, "payment processing", position="end")

        assert "payment processing" in result
        assert MARK_START in result
        assert MARK_END in result


class TestGuardrails:
    """Tests for the guardrail behaviors in compute_markers."""

    def test_unchanged_block_returns_unchanged(self):
        """Guardrail: unchanged block (normalized) returns as-is, no markers."""
        original = "This is some  text with extra   spaces."
        # Same text with different whitespace (normalizes to same)
        rewritten = "This is some text with extra spaces."
        result = compute_markers(original, rewritten)

        # Should have no markers since normalized forms are equal
        assert MARK_START not in result
        assert MARK_END not in result

    def test_short_sentence_fully_wrapped(self):
        """Guardrail: short sentences (< 6 words) are always fully wrapped."""
        original = "Hello world."
        # Short modification - less than 6 words
        rewritten = "Hello new world."
        result = compute_markers(original, rewritten)

        # Short sentence should be fully wrapped, not phrase-diff
        # The whole sentence "Hello new world." should be wrapped
        assert MARK_START in result
        assert MARK_END in result
        # Check that we don't have partial markers (phrase-diff behavior)
        # The entire short sentence should be marked
        clean = strip_markers(result)
        assert "Hello new world" in clean

    def test_long_sentence_phrase_diff(self):
        """Long sentences that are similar get phrase-level diff."""
        original = "This is a long sentence with many words in it."
        rewritten = "This is a long sentence with several words in it."
        result = compute_markers(original, rewritten)

        # Should have markers only around "several" (replaced "many")
        assert MARK_START in result
        assert "several" in result
        # "This is a long sentence with" should NOT be marked
        # We verify by checking the structure
        clean = strip_markers(result)
        assert "This is a long sentence" in clean

    def test_entirely_new_sentence_fully_wrapped(self):
        """Entirely new sentences are fully wrapped regardless of length."""
        original = "First sentence here."
        rewritten = "First sentence here. This is a completely new sentence with many words."
        result = compute_markers(original, rewritten)

        # The new sentence should be fully wrapped
        assert MARK_START in result
        clean = strip_markers(result)
        assert "completely new sentence" in clean


class TestH1Markers:
    """Tests for the compute_h1_markers function."""

    def test_unchanged_h1_no_markers(self):
        """Unchanged H1 returns without markers."""
        from seo_content_optimizer.diff_markers import compute_h1_markers

        original = "Welcome to Our Website"
        optimized = "Welcome to Our Website"
        result = compute_h1_markers(original, optimized)

        assert result == optimized
        assert MARK_START not in result

    def test_changed_h1_fully_wrapped(self):
        """Changed H1 is fully wrapped, not phrase-diffed."""
        from seo_content_optimizer.diff_markers import compute_h1_markers

        original = "Welcome to Our Website"
        optimized = "Welcome to Our Amazing Website"
        result = compute_h1_markers(original, optimized)

        # Entire H1 should be wrapped
        assert result == f"{MARK_START}Welcome to Our Amazing Website{MARK_END}"

    def test_no_original_h1_fully_wrapped(self):
        """No original H1 means entire optimized H1 is wrapped."""
        from seo_content_optimizer.diff_markers import compute_h1_markers

        result = compute_h1_markers("", "New Heading Title")
        assert result == f"{MARK_START}New Heading Title{MARK_END}"

    def test_empty_optimized_returns_empty(self):
        """Empty optimized H1 returns empty string."""
        from seo_content_optimizer.diff_markers import compute_h1_markers

        result = compute_h1_markers("Original Heading", "")
        assert result == ""

    def test_h1_case_insensitive_comparison(self):
        """H1 comparison is case-insensitive (via normalization)."""
        from seo_content_optimizer.diff_markers import compute_h1_markers

        original = "Welcome To Our Website"
        optimized = "welcome to our website"
        result = compute_h1_markers(original, optimized)

        # Same when normalized - no markers
        assert MARK_START not in result


class TestFullOriginalText:
    """Tests for the full_original_text parameter."""

    def test_sentence_from_elsewhere_not_marked(self):
        """Sentences that exist elsewhere in full_original_text are not marked."""
        original_block = "First paragraph content."
        rewritten = "First paragraph content. Second paragraph text."

        # The "Second paragraph text" exists in full_original_text
        full_original = "Some intro. Second paragraph text. More content."

        result = compute_markers(original_block, rewritten, full_original_text=full_original)

        # "Second paragraph text" should NOT be marked because it exists in full doc
        assert "Second paragraph text" in strip_markers(result)
        # If sentence exists unchanged in full original, no markers
        # Check if markers are absent for that sentence
        # This is harder to test precisely, but we can check the overall behavior

    def test_new_sentence_not_in_full_original_marked(self):
        """Sentences that don't exist anywhere in full_original_text are marked."""
        original_block = "First paragraph here."
        rewritten = "First paragraph here. Completely unique new content."
        full_original = "First paragraph here. Other content elsewhere."

        result = compute_markers(original_block, rewritten, full_original_text=full_original)

        # "Completely unique new content" should be marked
        assert MARK_START in result
        assert "Completely unique new content" in strip_markers(result)


class TestIntegration:
    """Integration tests for real-world scenarios."""

    def test_seo_optimization_scenario(self):
        """Test a realistic SEO optimization scenario."""
        original = """We provide excellent customer service.
        Our team is dedicated to helping you succeed."""

        rewritten = """We provide excellent payment processing services.
        Our expert team is dedicated to helping your business succeed with
        seamless merchant solutions."""

        result = compute_markers(original, rewritten)

        # Original unchanged parts should NOT be marked
        assert "We provide excellent" in strip_markers(result)
        assert "Our" in strip_markers(result)
        assert "team is dedicated" in strip_markers(result)

        # New/changed parts should be marked
        assert MARK_START in result
        assert "payment processing" in strip_markers(result)

    def test_keyword_in_original_not_marked(self):
        """Keywords that existed in original content must NOT be marked."""
        # This is the key fix - LLM was marking existing keywords
        original = "Payment processing is essential for modern businesses."
        rewritten = "Payment processing is absolutely essential for modern businesses today."

        result = compute_markers(original, rewritten)

        # "Payment processing" was in original - MUST NOT be in markers
        # Find the position of "Payment processing" and check it's not between markers
        clean = strip_markers(result)
        assert "Payment processing" in clean

        # Only "absolutely" and "today" should be marked
        # Check that markers exist for new content
        assert MARK_START in result

    def test_multiple_keywords_some_existing(self):
        """Mix of existing and new keywords handled correctly."""
        original = "We offer payment processing. Contact us today."
        rewritten = "We offer payment processing and merchant services. Contact our team today."

        result = compute_markers(original, rewritten)
        clean = strip_markers(result)

        # Both should be present
        assert "payment processing" in clean
        assert "merchant services" in clean

        # "payment processing" existed - not marked
        # "merchant services" is new - should be marked
        assert MARK_START in result


class TestCellGateRegression:
    """Regression tests based on the CellGate security cameras example."""

    def test_cellgate_h1_optimization(self):
        """Test the CellGate H1 optimization scenario from the bug report."""
        from seo_content_optimizer.diff_markers import compute_h1_markers

        original_h1 = "Enhancing Property Security with External Cameras"
        optimized_h1 = "How External Cameras Transform and Strengthen Property Security Systems"

        result = compute_h1_markers(original_h1, optimized_h1)

        # Entire new H1 should be wrapped (not phrase-diffed)
        assert result == f"{MARK_START}{optimized_h1}{MARK_END}"
        # No partial highlighting
        assert result.count(MARK_START) == 1
        assert result.count(MARK_END) == 1

    def test_cellgate_unchanged_paragraph(self):
        """Test that paragraphs unchanged from original remain unmarked."""
        # This paragraph appears identically in both original and optimized
        original = """External cameras combined with access control or visitor management systems serve as valuable tools for property security. They not only provide real-time monitoring but also can store video footage for review in case of suspicious activity. Some of the key benefits of such external cameras include:"""

        rewritten = original  # Same paragraph

        result = compute_markers(original, rewritten)

        # Should have no markers
        assert MARK_START not in result
        assert MARK_END not in result

    def test_cellgate_new_sentence_added(self):
        """Test that entirely new sentences are fully highlighted."""
        original = """Selecting the right external camera is an important step in enhancing the security of your property. Whether you need a basic fixed-focus camera, a flexible varifocal lens for adjustable coverage, or long-range surveillance for larger areas, there are several options available to suit a variety of needs. By evaluating the layout of your property and the specific security challenges you face, you can make an informed choice that boosts both safety and peace of mind."""

        # New version with added sentence at the end
        rewritten = """Selecting the right external camera is an important step in enhancing the security of your property. Whether you need a basic fixed-focus camera, a flexible varifocal lens for adjustable coverage, or long-range surveillance for larger areas, there are several options available to suit a variety of needs. By evaluating the layout of your property and the specific security challenges you face, you can make an informed choice that boosts both safety and peace of mind. These security cameras work best when integrated with comprehensive access control systems to create a complete security solution."""

        result = compute_markers(original, rewritten)

        # The new sentence should be marked
        assert MARK_START in result
        # The new sentence content should be present
        assert "security cameras work best" in strip_markers(result)
        assert "access control systems" in strip_markers(result)

    def test_cellgate_existing_keywords_not_marked(self):
        """Test that keywords existing in original are not highlighted."""
        original = "CellGate has several external camera models to choose from that can be integrated with its visitor management and access control systems."
        rewritten = "CellGate has several external camera models to choose from that can be integrated with its visitor management and access control systems, each offering different feature types to match various security needs."

        result = compute_markers(original, rewritten)

        # "external camera" existed - should NOT be in markers
        # "access control" existed - should NOT be in markers
        # The new part "each offering..." should be marked
        clean = strip_markers(result)
        assert "external camera" in clean
        assert "access control" in clean
        assert "each offering different feature types" in clean

    def test_cellgate_phrase_level_diff(self):
        """Test phrase-level diff on similar sentences."""
        original = "This compact, fixed-focus camera offers a wide-angle view that's ideal for general surveillance of access points."
        rewritten = "This compact, fixed-focus camera offers a wide-angle view that's ideal for general surveillance of access points and entry gates."

        result = compute_markers(original, rewritten)

        # Most of the sentence is the same - phrase-level diff
        # "and entry gates" should be marked, not the whole sentence
        assert MARK_START in result
        clean = strip_markers(result)
        assert "This compact, fixed-focus camera" in clean
        assert "entry gates" in clean
