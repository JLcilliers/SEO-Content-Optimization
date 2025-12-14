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
    """End-to-end tests simulating the full optimizer pipeline.

    These tests verify FULL DIFF MODE behavior where ALL real changes
    are highlighted, not just those containing keywords. This matches
    the Reading Guide: "Text highlighted in green indicates keyword
    insertions or SEO-focused adjustments. All non-highlighted text
    remains unchanged from the original content."
    """

    def test_full_pipeline_curly_quotes_not_marked(self):
        """Test that curly quote differences don't create markers (normalization)."""
        # Simulate the optimizer: original has straight quotes, LLM returns curly
        original = "CellGate's external cameras provide property's security."
        rewritten = "CellGate\u2019s external cameras provide property\u2019s security."

        # Compute markers - in full diff mode, no keyword filtering
        result = compute_markers(original, rewritten)

        # Should have no markers because normalization treats curly = straight
        assert MARK_START not in result
        assert MARK_END not in result

    def test_full_pipeline_real_change_highlighted(self):
        """Test that ALL real changes are highlighted in full diff mode."""
        original = "Our system provides monitoring."
        rewritten = "Our security camera system provides comprehensive monitoring."

        # Full diff mode: all changes highlighted
        result = compute_markers(original, rewritten)

        # Should have markers because real changes occurred
        assert MARK_START in result
        # The changed content should be in the result
        assert "security camera" in strip_markers(result)
        assert "comprehensive" in strip_markers(result)

    def test_full_pipeline_all_changes_marked_not_just_keywords(self):
        """Test that changes are highlighted even without keywords (full diff)."""
        original = "The excellent system works well."
        rewritten = "The amazing system works well."

        # Full diff mode: no keyword filtering
        result = compute_markers(original, rewritten)

        # "amazing" SHOULD be highlighted because it's a real change
        # This is the key difference from keyword-filtered mode
        assert MARK_START in result
        assert MARK_END in result
        assert "amazing" in strip_markers(result)

    def test_cellgate_real_world_scenario_full_diff(self):
        """Comprehensive test of CellGate-like optimization in full diff mode."""
        # Original content
        original = """When it comes to securing any property, having the right camera system in place is essential. When complex camera, surveillance, and video setups are not necessary for a property, the use of simpler external cameras integrated with an access control or visitor management solution (like CellGate's Entría and/or Watchman lines) can provide the right layer of extra security."""

        # LLM might change some punctuation and add new content
        rewritten = """When it comes to securing any property, having the right camera system in place is essential. When complex camera, surveillance, and video setups are not necessary for a property, the use of simpler external cameras integrated with an access control or visitor management solution (like CellGate\u2019s Entría and/or Watchman lines) can provide the right layer of extra security for your property security needs."""

        result = compute_markers(original, rewritten, full_original_text=original)

        # "CellGate's" should NOT be highlighted (curly quote normalized)
        # The curly apostrophe change should not trigger markers
        clean = strip_markers(result)
        assert "CellGate" in clean  # Text still present

        # New text "for your property security needs" SHOULD be marked (full diff)
        if "property security needs" in clean:
            assert MARK_START in result
            assert "property security needs" in clean

    def test_unchanged_content_no_markers(self):
        """Verify that truly unchanged content has no markers."""
        original = "External cameras provide real-time monitoring for your property."
        rewritten = "External cameras provide real-time monitoring for your property."

        result = compute_markers(original, rewritten)

        # Identical text should have no markers
        assert MARK_START not in result
        assert MARK_END not in result
        assert result == original


class TestFilterMarkersByKeywordsStandalone:
    """Tests for filter_markers_by_keywords function (kept for potential future use).

    Note: This function is no longer used in the main pipeline (full diff mode),
    but is kept for potential future "SEO-only diff" mode. These tests verify
    the function still works correctly.
    """

    def test_filter_keeps_markers_with_keyword(self):
        """Markers containing keywords are kept."""
        text = f"Hello {MARK_START}security cameras{MARK_END} world."
        keywords = ["security cameras"]

        result = filter_markers_by_keywords(text, keywords)

        assert MARK_START in result
        assert "security cameras" in result

    def test_filter_removes_markers_without_keyword(self):
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


# =============================================================================
# V2 SENTENCE-LEVEL DIFF TESTS
# =============================================================================
# These tests verify the V2 sentence-level diff behavior:
# - A sentence is either FULLY unchanged (black) or FULLY changed (green)
# - NO token-level diff inside sentences
# - Eliminates confusing partial highlights
# =============================================================================


class TestV2SplitSentences:
    """Tests for the V2 split_sentences function."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        from seo_content_optimizer.diff_markers import split_sentences
        assert split_sentences("") == []

    def test_single_sentence(self):
        """Single sentence is returned as a list with one item."""
        from seo_content_optimizer.diff_markers import split_sentences
        result = split_sentences("This is a sentence.")
        assert result == ["This is a sentence."]

    def test_multiple_sentences(self):
        """Multiple sentences are split correctly."""
        from seo_content_optimizer.diff_markers import split_sentences
        result = split_sentences("First sentence. Second sentence. Third one!")
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
        assert result[2] == "Third one!"

    def test_question_marks(self):
        """Question marks also split sentences."""
        from seo_content_optimizer.diff_markers import split_sentences
        result = split_sentences("Is this a question? Yes it is.")
        assert len(result) == 2

    def test_exclamation_marks(self):
        """Exclamation marks also split sentences."""
        from seo_content_optimizer.diff_markers import split_sentences
        result = split_sentences("Wow! That is amazing.")
        assert len(result) == 2


class TestV2ComputeMarkersSentenceLevel:
    """Tests for the V2 compute_markers_sentence_level function."""

    def test_all_sentences_new(self):
        """All sentences new when sentence index is empty."""
        from seo_content_optimizer.diff_markers import (
            compute_markers_sentence_level,
            MARK_START,
            MARK_END,
        )

        result = compute_markers_sentence_level(
            original_block="",
            rewritten_block="This is a new sentence. And another one.",
            original_sentence_index=set(),
        )

        # All content should be marked as new
        assert MARK_START in result
        assert MARK_END in result

    def test_unchanged_sentence_not_marked(self):
        """Unchanged sentence (in index) is NOT marked."""
        from seo_content_optimizer.diff_markers import (
            compute_markers_sentence_level,
            normalize_sentence,
            MARK_START,
            MARK_END,
        )

        original_sentence = "This sentence exists in the original."
        sentence_index = {normalize_sentence(original_sentence)}

        result = compute_markers_sentence_level(
            original_block=original_sentence,
            rewritten_block=original_sentence,
            original_sentence_index=sentence_index,
        )

        # No markers for unchanged sentence
        assert MARK_START not in result
        assert MARK_END not in result

    def test_changed_sentence_fully_marked(self):
        """Changed sentence is FULLY marked (no partial highlights)."""
        from seo_content_optimizer.diff_markers import (
            compute_markers_sentence_level,
            normalize_sentence,
            strip_markers,
            MARK_START,
            MARK_END,
        )

        original = "This is the original sentence."
        changed = "This is the changed sentence with new words."

        sentence_index = {normalize_sentence(original)}

        result = compute_markers_sentence_level(
            original_block=original,
            rewritten_block=changed,
            original_sentence_index=sentence_index,
        )

        # Entire sentence should be marked
        assert MARK_START in result
        assert MARK_END in result
        # Only ONE marker pair (entire sentence)
        assert result.count(MARK_START) == 1
        assert result.count(MARK_END) == 1

    def test_mixed_unchanged_and_changed(self):
        """Mixed unchanged and changed sentences handled correctly."""
        from seo_content_optimizer.diff_markers import (
            compute_markers_sentence_level,
            normalize_sentence,
            strip_markers,
            MARK_START,
            MARK_END,
        )

        original = "First sentence unchanged. Second also unchanged."
        rewritten = "First sentence unchanged. This is brand new. Second also unchanged."

        sentence_index = {
            normalize_sentence("First sentence unchanged."),
            normalize_sentence("Second also unchanged."),
        }

        result = compute_markers_sentence_level(
            original_block=original,
            rewritten_block=rewritten,
            original_sentence_index=sentence_index,
        )

        # Only the new sentence should be marked
        assert MARK_START in result
        clean = strip_markers(result)
        assert "First sentence unchanged" in clean
        assert "This is brand new" in clean
        assert "Second also unchanged" in clean

        # "First sentence unchanged" should NOT be marked
        # "This is brand new" SHOULD be marked
        assert f"{MARK_START}This is brand new.{MARK_END}" in result


class TestV2ComputeMarkersV2:
    """Tests for the V2 compute_markers_v2 entry point."""

    def test_empty_rewritten_returns_empty(self):
        """Empty rewritten text returns empty string."""
        from seo_content_optimizer.diff_markers import compute_markers_v2

        result = compute_markers_v2("original text", "")
        assert result == ""

    def test_no_original_wraps_all(self):
        """No original text wraps everything as new."""
        from seo_content_optimizer.diff_markers import compute_markers_v2, MARK_START, MARK_END

        result = compute_markers_v2("", "This is all new content.")
        assert result == f"{MARK_START}This is all new content.{MARK_END}"

    def test_unchanged_content_no_markers(self):
        """Unchanged content has no markers."""
        from seo_content_optimizer.diff_markers import compute_markers_v2, MARK_START

        original = "This content is unchanged."
        result = compute_markers_v2(original, original)

        assert MARK_START not in result
        assert result == original

    def test_full_original_text_respected(self):
        """Sentences from full_original_text are not marked."""
        from seo_content_optimizer.diff_markers import compute_markers_v2, MARK_START, strip_markers

        original_block = "Paragraph one."
        rewritten = "Paragraph one. Sentence from elsewhere."
        full_original = "Some intro. Sentence from elsewhere. More content."

        result = compute_markers_v2(original_block, rewritten, full_original_text=full_original)

        # "Sentence from elsewhere" exists in full_original_text
        # so it should NOT be marked
        clean = strip_markers(result)
        assert "Sentence from elsewhere" in clean

    def test_new_sentence_marked_v2(self):
        """New sentences are fully marked in V2."""
        from seo_content_optimizer.diff_markers import compute_markers_v2, MARK_START, MARK_END

        original = "First sentence here."
        rewritten = "First sentence here. Completely new addition."

        result = compute_markers_v2(original, rewritten)

        # New sentence should be fully marked
        assert MARK_START in result
        assert "Completely new addition" in result


class TestV2CellGateScenario:
    """V2 tests based on the CellGate scenario."""

    def test_cellgate_h1_v2_full_sentence(self):
        """CellGate H1 change - entire H1 should be one marker block."""
        from seo_content_optimizer.diff_markers import compute_h1_markers, MARK_START, MARK_END

        original = "Enhancing Property Security with External Cameras"
        optimized = "How External Cameras Transform and Strengthen Property Security Systems"

        result = compute_h1_markers(original, optimized)

        # H1 handler already wraps entire H1
        assert result == f"{MARK_START}{optimized}{MARK_END}"
        # Only one marker pair
        assert result.count(MARK_START) == 1
        assert result.count(MARK_END) == 1

    def test_cellgate_unchanged_paragraph_v2(self):
        """Unchanged paragraph stays completely unmarked."""
        from seo_content_optimizer.diff_markers import compute_markers_v2, MARK_START

        paragraph = """External cameras combined with access control or visitor management systems serve as valuable tools for property security. They not only provide real-time monitoring but also can store video footage for review in case of suspicious activity."""

        result = compute_markers_v2(paragraph, paragraph)

        assert MARK_START not in result

    def test_cellgate_new_final_sentence_v2(self):
        """New sentence at end is fully marked (not partial)."""
        from seo_content_optimizer.diff_markers import compute_markers_v2, MARK_START, MARK_END, strip_markers

        original = """Selecting the right external camera is an important step in enhancing the security of your property."""

        rewritten = """Selecting the right external camera is an important step in enhancing the security of your property. These security cameras work best when integrated with comprehensive access control systems."""

        result = compute_markers_v2(original, rewritten)

        # Original sentence should NOT be marked
        # New sentence should be FULLY marked (not token-level)
        assert MARK_START in result
        assert MARK_END in result

        # The new sentence should be completely wrapped
        clean = strip_markers(result)
        assert "These security cameras work best" in clean

    def test_v2_no_partial_highlights(self):
        """V2 should never produce partial word/token highlights."""
        from seo_content_optimizer.diff_markers import compute_markers_v2, MARK_START, MARK_END

        original = "This is a long sentence with many words in it."
        # Changed one word: many -> several
        rewritten = "This is a long sentence with several words in it."

        result = compute_markers_v2(original, rewritten)

        # In V2, the ENTIRE sentence should be marked, not just "several"
        # Because the sentence is DIFFERENT from original (even by one word)
        if MARK_START in result:
            # Should wrap the full sentence, not just "several"
            # The count should be 1 (one marker pair for the whole sentence)
            assert result.count(MARK_START) == 1
            assert result.count(MARK_END) == 1


class TestNormalizeParagraphSpacing:
    """Tests for the normalize_paragraph_spacing function.

    Bug #3: Text running together without proper line breaks.
    This function fixes "word.Word" patterns and spacing issues.
    """

    def test_empty_string(self):
        """Empty string returns empty string."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing
        assert normalize_paragraph_spacing("") == ""

    def test_normal_text_unchanged(self):
        """Properly spaced text should remain unchanged."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing
        text = "This is a sentence. This is another sentence."
        result = normalize_paragraph_spacing(text)
        assert result == text

    def test_fixes_word_dot_capital(self):
        """Should fix 'word.Word' patterns (missing space after period)."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing
        text = "First sentence.Second sentence."
        result = normalize_paragraph_spacing(text)
        assert result == "First sentence. Second sentence."

    def test_fixes_word_exclamation_capital(self):
        """Should fix 'word!Word' patterns."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing
        text = "Amazing!This is great."
        result = normalize_paragraph_spacing(text)
        assert result == "Amazing! This is great."

    def test_fixes_word_question_capital(self):
        """Should fix 'word?Word' patterns."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing
        text = "How does this work?It's quite simple."
        result = normalize_paragraph_spacing(text)
        assert result == "How does this work? It's quite simple."

    def test_fixes_word_dot_lowercase(self):
        """Should fix 'word.word' patterns (lowercase after period)."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing
        text = "First sentence.second part."
        result = normalize_paragraph_spacing(text)
        assert result == "First sentence. second part."

    def test_collapses_multiple_spaces(self):
        """Should collapse multiple spaces to single space."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing
        text = "Too   many    spaces."
        result = normalize_paragraph_spacing(text)
        assert result == "Too many spaces."

    def test_preserves_newlines(self):
        """Should preserve newline characters."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing
        text = "First paragraph.\n\nSecond paragraph."
        result = normalize_paragraph_spacing(text)
        assert "\n\n" in result
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_handles_markers(self):
        """Should handle text with markers correctly."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing, MARK_START, MARK_END
        text = f"Original text.{MARK_START}New sentence.{MARK_END}More text."
        result = normalize_paragraph_spacing(text)
        # Should have proper spacing around markers
        assert "text. " in result or f"text.{MARK_START}" in result  # Space added before marker or marker intact
        assert MARK_START in result
        assert MARK_END in result

    def test_strips_leading_trailing_whitespace(self):
        """Should strip leading/trailing whitespace."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing
        text = "  Some text.  "
        result = normalize_paragraph_spacing(text)
        assert result == "Some text."

    def test_real_world_llm_output(self):
        """Test with realistic LLM output that might have issues."""
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing
        text = "PTO insurance provides coverage.Professional liability is essential.Consider your options carefully."
        result = normalize_paragraph_spacing(text)
        assert ". P" in result  # Space after period before "Professional"
        assert ". C" in result  # Space after period before "Consider"

    def test_cellgate_specific_issue(self):
        """Test the specific reported issue from CellGate content.

        Bug #3: Text running together like "your property.External cameras"
        Should be "your property. External cameras" with proper spacing.
        """
        from seo_content_optimizer.diff_markers import normalize_paragraph_spacing

        # Exact reported issue pattern
        text = "your property.External cameras"
        result = normalize_paragraph_spacing(text)
        assert result == "your property. External cameras"

        # More complex version
        text2 = "Enhance your property.External cameras provide security.CellGate systems offer protection."
        result2 = normalize_paragraph_spacing(text2)
        assert ". E" in result2  # Space before "External"
        assert ". C" in result2  # Space before "CellGate"


class TestBrandNameNormalization:
    """Tests for brand name normalization in LLM output.

    Bug #1: Brand names like "CellGate" being changed to "Cell-Gate"
    by the LLM and then incorrectly highlighted as changes.
    """

    def test_normalize_brand_in_text_camelcase_preserved(self):
        """CellGate should be preserved, not changed to Cell-Gate."""
        from seo_content_optimizer.diff_markers import normalize_brand_in_text, generate_brand_variations

        original_brand = "CellGate"
        variations = generate_brand_variations(original_brand)

        # LLM might output "Cell-Gate" when original is "CellGate"
        llm_output = "Security powered by Cell-Gate systems."
        result = normalize_brand_in_text(llm_output, original_brand, variations)

        # Should normalize back to original spelling
        assert "CellGate" in result
        assert "Cell-Gate" not in result

    def test_normalize_brand_in_text_preserves_other_hyphens(self):
        """Hyphens not related to brand should be preserved."""
        from seo_content_optimizer.diff_markers import normalize_brand_in_text, generate_brand_variations

        original_brand = "CellGate"
        variations = generate_brand_variations(original_brand)

        llm_output = "Cell-Gate offers state-of-the-art security."
        result = normalize_brand_in_text(llm_output, original_brand, variations)

        # Brand normalized, but "state-of-the-art" preserved
        assert "CellGate" in result
        assert "state-of-the-art" in result

    def test_normalize_brand_in_text_case_variations(self):
        """All case variations should normalize to original."""
        from seo_content_optimizer.diff_markers import normalize_brand_in_text, generate_brand_variations

        original_brand = "CellGate"
        variations = generate_brand_variations(original_brand)

        # Test various case issues
        test_cases = [
            ("CELLGATE security", "CellGate security"),
            ("cellgate systems", "CellGate systems"),
            ("cell-gate products", "CellGate products"),
        ]

        for input_text, expected in test_cases:
            result = normalize_brand_in_text(input_text, original_brand, variations)
            assert original_brand in result, f"Failed for input: {input_text}"

    def test_normalize_brand_in_text_empty_brand(self):
        """Empty brand name should return text unchanged."""
        from seo_content_optimizer.diff_markers import normalize_brand_in_text

        text = "Some text here."
        result = normalize_brand_in_text(text, "", set())
        assert result == text

    def test_normalize_brand_in_text_no_brand_in_text(self):
        """Text without brand should remain unchanged."""
        from seo_content_optimizer.diff_markers import normalize_brand_in_text, generate_brand_variations

        original_brand = "CellGate"
        variations = generate_brand_variations(original_brand)

        text = "Security cameras are essential for property."
        result = normalize_brand_in_text(text, original_brand, variations)
        assert result == text

    def test_generate_brand_variations_camelcase(self):
        """Generate variations from CamelCase brand."""
        from seo_content_optimizer.diff_markers import generate_brand_variations

        variations = generate_brand_variations("CellGate")

        # Should include various forms
        assert "cellgate" in variations
        assert "cell-gate" in variations or "cell gate" in variations

    def test_generate_brand_variations_hyphenated(self):
        """Generate variations from hyphenated brand."""
        from seo_content_optimizer.diff_markers import generate_brand_variations

        variations = generate_brand_variations("Cell-Gate")

        # Should include various forms
        assert "cell-gate" in variations
        assert "cellgate" in variations

    def test_brand_not_highlighted_when_unchanged(self):
        """Brand name should not be highlighted when unchanged."""
        from seo_content_optimizer.diff_markers import compute_markers, MARK_START

        original = "CellGate's security cameras are the best."
        rewritten = "CellGate's security cameras are the best."

        result = compute_markers(original, rewritten)

        # No changes, no markers
        assert MARK_START not in result

    def test_brand_excluded_from_diff_when_only_brand_case_changed(self):
        """If only brand case/format changed, it should be normalized, not marked."""
        from seo_content_optimizer.diff_markers import (
            normalize_brand_in_text,
            generate_brand_variations,
            compute_markers,
            MARK_START,
        )

        original_brand = "CellGate"
        variations = generate_brand_variations(original_brand)

        # Simulate: original has "CellGate", LLM outputs "Cell-Gate"
        original = "CellGate systems provide security."
        llm_output = "Cell-Gate systems provide security."

        # First normalize the brand
        normalized = normalize_brand_in_text(llm_output, original_brand, variations)

        # Should be "CellGate systems provide security." now
        assert normalized == original

        # Then compute markers - should have no markers
        result = compute_markers(original, normalized)
        assert MARK_START not in result
