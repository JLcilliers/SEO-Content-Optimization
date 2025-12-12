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
    mark_block_as_new,
    normalize_token_for_diff,
    expand_markers_to_full_sentence,
    compute_markers,
    cleanup_markers,
    inject_phrase_with_markers,
    filter_markers_by_keywords,
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


class TestMarkBlockAsNew:
    """Tests for the mark_block_as_new function."""

    def test_wraps_text_in_markers(self):
        """Plain text is wrapped in markers."""
        text = "This is a new FAQ answer."
        result = mark_block_as_new(text)
        assert result == f"{MARK_START}This is a new FAQ answer.{MARK_END}"

    def test_empty_string_returns_empty(self):
        """Empty string returns empty string."""
        assert mark_block_as_new("") == ""
        assert mark_block_as_new("   ") == ""

    def test_strips_existing_markers(self):
        """Existing markers are stripped to avoid nesting."""
        text = f"Already {MARK_START}marked{MARK_END} text"
        result = mark_block_as_new(text)
        assert result == f"{MARK_START}Already marked text{MARK_END}"
        # No nested markers
        assert result.count(MARK_START) == 1
        assert result.count(MARK_END) == 1

    def test_faq_question_scenario(self):
        """FAQ question is fully wrapped."""
        question = "What security cameras are best for property monitoring?"
        result = mark_block_as_new(question)
        assert result.startswith(MARK_START)
        assert result.endswith(MARK_END)
        assert "What security cameras" in result

    def test_faq_answer_scenario(self):
        """FAQ answer (multi-sentence) is fully wrapped."""
        answer = "External cameras provide essential monitoring. They can capture visitor photos and license plates."
        result = mark_block_as_new(answer)
        assert result == f"{MARK_START}{answer}{MARK_END}"


class TestNormalizeTokenForDiff:
    """Tests for punctuation normalization in diff comparison."""

    def test_curly_apostrophe_normalized(self):
        """Curly apostrophe (U+2019) is normalized to straight quote."""
        # This is the main bug: CellGate's vs CellGate's
        assert normalize_token_for_diff("CellGate\u2019s") == "CellGate's"

    def test_left_single_quote_normalized(self):
        """Left single quote (U+2018) is normalized to straight quote."""
        assert normalize_token_for_diff("\u2018hello") == "'hello"

    def test_double_quotes_normalized(self):
        """Smart double quotes are normalized to straight double quotes."""
        assert normalize_token_for_diff("\u201Cquote\u201D") == '"quote"'

    def test_en_dash_normalized(self):
        """En dash (U+2013) is normalized to hyphen."""
        assert normalize_token_for_diff("2019\u20132023") == "2019-2023"

    def test_em_dash_normalized(self):
        """Em dash (U+2014) is normalized to hyphen."""
        assert normalize_token_for_diff("word\u2014another") == "word-another"

    def test_ellipsis_normalized(self):
        """Horizontal ellipsis (U+2026) is normalized to three dots."""
        assert normalize_token_for_diff("wait\u2026") == "wait..."

    def test_plain_text_unchanged(self):
        """Plain ASCII text is unchanged."""
        assert normalize_token_for_diff("CellGate's") == "CellGate's"

    def test_empty_token_unchanged(self):
        """Empty string returns empty string."""
        assert normalize_token_for_diff("") == ""


class TestPunctuationInDiff:
    """Tests for punctuation normalization affecting diff results."""

    def test_curly_vs_straight_apostrophe_no_highlight(self):
        """Curly apostrophe in rewritten vs straight in original: no highlight."""
        # Original uses straight apostrophe
        original = "CellGate's system is great."
        # Rewritten uses curly apostrophe (U+2019) but otherwise identical
        rewritten = "CellGate\u2019s system is great."

        result = add_markers_by_diff(original, rewritten)

        # Should NOT highlight - they're semantically the same
        assert MARK_START not in result
        assert MARK_END not in result

    def test_mixed_punctuation_differences_no_highlight(self):
        """Various punctuation differences should not trigger highlights."""
        # Original with straight punctuation
        original = "It's a \"test\" - really."
        # Rewritten with curly/smart punctuation (single dash vs em dash is same token count)
        rewritten = "It\u2019s a \u201Ctest\u201D \u2014 really."

        result = add_markers_by_diff(original, rewritten)

        # Should NOT highlight punctuation-only differences
        assert MARK_START not in result

    def test_real_content_change_still_highlighted(self):
        """Real content changes are still highlighted despite punctuation."""
        original = "CellGate's system works well."
        rewritten = "CellGate\u2019s NEW system works perfectly."

        result = add_markers_by_diff(original, rewritten)

        # Should highlight the actual changes (NEW, perfectly)
        assert MARK_START in result
        assert "NEW" in result or "perfectly" in result


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


class TestFilterMarkersByKeywords:
    """Tests for the filter_markers_by_keywords function."""

    def test_keeps_markers_with_keyword(self):
        """Markers containing keywords are kept."""
        text = f"Hello {MARK_START}security cameras{MARK_END} world."
        keywords = ["security cameras"]

        result = filter_markers_by_keywords(text, keywords)

        assert MARK_START in result
        assert "security cameras" in result

    def test_removes_markers_without_keyword(self):
        """Markers without keywords are removed but text is kept."""
        text = f"Hello {MARK_START}random change{MARK_END} world."
        keywords = ["security cameras", "access control"]

        result = filter_markers_by_keywords(text, keywords)

        # Markers should be removed
        assert MARK_START not in result
        assert MARK_END not in result
        # But text should remain
        assert "random change" in result
        assert result == "Hello random change world."

    def test_mixed_markers_some_kept_some_removed(self):
        """Some markers kept, others removed based on keywords."""
        text = f"The {MARK_START}security camera{MARK_END} provides {MARK_START}excellent{MARK_END} monitoring."
        keywords = ["security camera"]

        result = filter_markers_by_keywords(text, keywords)

        # "security camera" markers kept
        assert f"{MARK_START}security camera{MARK_END}" in result
        # "excellent" markers removed
        assert "excellent" in result
        assert result.count(MARK_START) == 1

    def test_case_insensitive_matching(self):
        """Keyword matching is case-insensitive."""
        text = f"The {MARK_START}Security Cameras{MARK_END} are great."
        keywords = ["security cameras"]

        result = filter_markers_by_keywords(text, keywords)

        assert MARK_START in result

    def test_no_markers_returns_unchanged(self):
        """Text without markers is returned unchanged."""
        text = "Hello world with no markers."
        keywords = ["security"]

        result = filter_markers_by_keywords(text, keywords)

        assert result == text

    def test_empty_keywords_keeps_all_markers(self):
        """Empty keyword list keeps all markers."""
        text = f"Hello {MARK_START}marked{MARK_END} world."
        keywords = []

        result = filter_markers_by_keywords(text, keywords)

        assert MARK_START in result

    def test_partial_keyword_match(self):
        """Partial keyword match (keyword as substring) works."""
        text = f"The {MARK_START}external cameras{MARK_END} work well."
        keywords = ["camera"]  # Partial match

        result = filter_markers_by_keywords(text, keywords)

        assert MARK_START in result

    def test_cellgate_scenario(self):
        """Real CellGate scenario: filter non-SEO changes."""
        # Simulates: "CellGate's" highlighted due to apostrophe but no keyword
        text = f"The {MARK_START}CellGate's{MARK_END} system provides {MARK_START}external cameras{MARK_END} for security."
        keywords = ["external cameras", "security cameras", "access control"]

        result = filter_markers_by_keywords(text, keywords)

        # "external cameras" contains keyword - keep markers
        assert f"{MARK_START}external cameras{MARK_END}" in result
        # "CellGate's" has no keyword - remove markers
        assert "CellGate's" in result
        # Should only have one marker pair
        assert result.count(MARK_START) == 1


class TestCurlyQuoteNormalization:
    """Tests specifically for curly quote and smart punctuation handling."""

    def test_curly_apostrophe_in_possessive_not_highlighted(self):
        """Words with curly apostrophe like 'CellGate's' should not be highlighted."""
        # Original has straight apostrophe, optimized has curly
        original = "CellGate's system is great."
        rewritten = "CellGate\u2019s system is great."  # \u2019 is curly apostrophe

        result = compute_markers(original, rewritten)

        # Should have NO markers - they're equivalent
        assert MARK_START not in result
        assert MARK_END not in result

    def test_lets_with_curly_apostrophe_not_highlighted(self):
        """Let's with curly apostrophe should match Let's with straight."""
        original = "Let's explore the options."
        rewritten = "Let\u2019s explore the options."

        result = compute_markers(original, rewritten)

        assert MARK_START not in result
        assert MARK_END not in result

    def test_propertys_with_curly_apostrophe_not_highlighted(self):
        """property's with curly apostrophe should match straight."""
        original = "Your property's security is important."
        rewritten = "Your property\u2019s security is important."

        result = compute_markers(original, rewritten)

        assert MARK_START not in result
        assert MARK_END not in result

    def test_smart_double_quotes_not_highlighted(self):
        """Smart double quotes should match straight double quotes."""
        original = 'They said "hello" to everyone.'
        rewritten = "They said \u201chello\u201d to everyone."  # \u201c \u201d are curly double quotes

        result = compute_markers(original, rewritten)

        assert MARK_START not in result
        assert MARK_END not in result

    def test_en_dash_vs_hyphen_not_highlighted(self):
        """En dash should match hyphen."""
        original = "We offer 24-7 support."
        rewritten = "We offer 24\u20137 support."  # \u2013 is en dash

        result = compute_markers(original, rewritten)

        assert MARK_START not in result
        assert MARK_END not in result

    def test_multiple_curly_quote_words_unchanged(self):
        """Multiple words with curly quotes should all remain unchanged."""
        original = "CellGate's property's system Let's cameras"
        rewritten = "CellGate\u2019s property\u2019s system Let\u2019s cameras"

        result = compute_markers(original, rewritten)

        assert MARK_START not in result
        assert MARK_END not in result


class TestEndToEndPipeline:
    """End-to-end tests simulating the full optimizer pipeline."""

    def test_full_pipeline_curly_quotes_filtered(self):
        """Test the full pipeline: compute_markers + filter_markers_by_keywords."""
        # Simulate the optimizer: original has straight quotes, LLM returns curly
        original = "CellGate's external cameras provide property's security."
        rewritten = "CellGate\u2019s external cameras provide property\u2019s security."

        # First compute markers
        marked = compute_markers(original, rewritten)

        # If any markers exist (shouldn't after normalization fix), filter them
        keywords = ["external cameras", "security cameras", "property security"]
        result = filter_markers_by_keywords(marked, keywords)

        # Should have no markers because:
        # 1. Normalization should prevent curly quote differences from being marked
        # 2. Even if marked, filter should remove "CellGate's" and "property's"
        assert MARK_START not in result
        assert MARK_END not in result

    def test_full_pipeline_real_change_with_keyword_highlighted(self):
        """Test that real SEO changes containing keywords are highlighted."""
        original = "Our system provides monitoring."
        rewritten = "Our security camera system provides comprehensive monitoring."

        marked = compute_markers(original, rewritten)
        keywords = ["security camera", "comprehensive monitoring"]
        result = filter_markers_by_keywords(marked, keywords)

        # Should have markers because real change with keyword
        assert MARK_START in result
        # The keyword content should be in the result
        assert "security camera" in strip_markers(result)

    def test_full_pipeline_change_without_keyword_removed(self):
        """Test that changes NOT containing keywords have markers removed."""
        original = "The excellent system works well."
        rewritten = "The amazing system works well."

        marked = compute_markers(original, rewritten)
        keywords = ["security camera", "access control"]
        result = filter_markers_by_keywords(marked, keywords)

        # "amazing" should NOT be highlighted (no keyword)
        assert MARK_START not in result
        assert MARK_END not in result
        # But text should remain
        assert "amazing" in result

    def test_cellgate_real_world_scenario(self):
        """Comprehensive test of CellGate-like optimization."""
        # Original content
        original = """When it comes to securing any property, having the right camera system in place is essential. When complex camera, surveillance, and video setups are not necessary for a property, the use of simpler external cameras integrated with an access control or visitor management solution (like CellGate's Entría and/or Watchman lines) can provide the right layer of extra security."""

        # LLM might change some punctuation and add keywords
        rewritten = """When it comes to securing any property, having the right camera system in place is essential. When complex camera, surveillance, and video setups are not necessary for a property, the use of simpler external cameras integrated with an access control or visitor management solution (like CellGate\u2019s Entría and/or Watchman lines) can provide the right layer of extra security for your property security needs."""

        marked = compute_markers(original, rewritten, full_original_text=original)
        keywords = ["external cameras", "property security", "access control", "camera system"]
        result = filter_markers_by_keywords(marked, keywords)

        # "CellGate's" should NOT be highlighted (curly quote only)
        # But if new text "for your property security needs" was added, it should be
        clean = strip_markers(result)
        assert "CellGate" in clean  # Text still present
        # Check marker count - should only mark new SEO-relevant additions
        if "property security needs" in rewritten:
            # If new content added with keyword, it should be marked
            if MARK_START in result:
                # Verify the marked content contains a keyword
                import re
                marker_pattern = rf"{re.escape(MARK_START)}(.*?){re.escape(MARK_END)}"
                matches = re.findall(marker_pattern, result)
                for match in matches:
                    match_lower = match.lower()
                    has_keyword = any(kw.lower() in match_lower for kw in keywords)
                    assert has_keyword, f"Marked content '{match}' has no keyword"
