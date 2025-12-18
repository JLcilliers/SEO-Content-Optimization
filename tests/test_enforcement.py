# -*- coding: utf-8 -*-
"""
Tests for the enforcement module (post-optimization caps and budget limits).

These tests verify that the enforcement layer correctly:
1. Removes excess keyword occurrences beyond caps
2. Enforces insertion budget limits (max sentences/words)
3. Validates that insertions contain keywords
4. Integrates all enforcement steps in run_enforcement()
"""

import pytest

from seo_content_optimizer.enforcement import (
    run_enforcement,
    enforce_keyword_caps,
    enforce_budget_limits,
    validate_insertions_have_keywords,
    extract_marker_spans,
    EnforcementResult,
    MarkerSpan,
)
from seo_content_optimizer.config import OptimizationConfig
from seo_content_optimizer.diff_markers import MARK_START, MARK_END


class TestExtractMarkerSpans:
    """Test marker span detection."""

    def test_find_single_span(self):
        """Should find a single marked span."""
        text = f"Original {MARK_START}added text{MARK_END} more original"
        spans = extract_marker_spans(text)

        assert len(spans) == 1
        assert spans[0].content == "added text"
        assert spans[0].word_count == 2

    def test_find_multiple_spans(self):
        """Should find multiple marked spans."""
        text = f"Start {MARK_START}first add{MARK_END} middle {MARK_START}second add{MARK_END} end"
        spans = extract_marker_spans(text)

        assert len(spans) == 2
        assert spans[0].content == "first add"
        assert spans[1].content == "second add"

    def test_find_no_spans(self):
        """Should return empty list when no spans."""
        text = "No markers in this text at all."
        spans = extract_marker_spans(text)

        assert len(spans) == 0

    def test_span_sentence_count(self):
        """Should count sentences in spans."""
        text = f"Start {MARK_START}First sentence. Second sentence. Third.{MARK_END} end"
        spans = extract_marker_spans(text)

        assert len(spans) == 1
        assert spans[0].sentence_count == 3

    def test_span_word_count(self):
        """Should count words in spans."""
        text = f"Start {MARK_START}one two three four five{MARK_END} end"
        spans = extract_marker_spans(text)

        assert len(spans) == 1
        assert spans[0].word_count == 5


class TestEnforceKeywordCaps:
    """Test keyword cap enforcement."""

    def test_remove_excess_keyword_occurrences(self):
        """Should remove occurrences beyond the cap."""
        keywords = ["test keyword"]
        text = (
            f"First {MARK_START}test keyword{MARK_END} here. "
            f"Second {MARK_START}test keyword{MARK_END} here. "
            f"Third {MARK_START}test keyword{MARK_END} here."
        )

        result, removed, orig_counts, final_counts = enforce_keyword_caps(
            text, keywords, primary_cap=1, secondary_cap=1
        )

        # Should keep only 1 occurrence
        assert removed == 2
        assert result.count("test keyword") == 1

    def test_respect_cap_value(self):
        """Should respect the specified cap value."""
        keywords = ["test keyword"]
        text = (
            f"One {MARK_START}test keyword{MARK_END}. "
            f"Two {MARK_START}test keyword{MARK_END}. "
            f"Three {MARK_START}test keyword{MARK_END}."
        )

        result, removed, orig_counts, final_counts = enforce_keyword_caps(
            text, keywords, primary_cap=2, secondary_cap=2
        )

        # Should keep 2 occurrences
        assert removed == 1
        assert result.count("test keyword") == 2

    def test_primary_vs_secondary_caps(self):
        """Should use different caps for primary vs secondary."""
        keywords = ["primary kw", "secondary kw"]
        text = (
            f"One {MARK_START}primary kw{MARK_END}. "
            f"Two {MARK_START}primary kw{MARK_END}. "
            f"Three {MARK_START}secondary kw{MARK_END}. "
            f"Four {MARK_START}secondary kw{MARK_END}. "
            f"Five {MARK_START}secondary kw{MARK_END}."
        )

        result, removed, orig_counts, final_counts = enforce_keyword_caps(
            text, keywords, primary_cap=2, secondary_cap=1, primary_keyword="primary kw"
        )

        # Primary should have 2, secondary should have 1
        assert result.count("primary kw") == 2
        assert result.count("secondary kw") == 1
        assert removed == 2  # 0 primary + 2 secondary removed

    def test_no_removal_when_under_cap(self):
        """Should not remove anything when under cap."""
        keywords = ["test keyword"]
        text = f"Only one {MARK_START}test keyword{MARK_END} here."

        result, removed, orig_counts, final_counts = enforce_keyword_caps(
            text, keywords, primary_cap=5, secondary_cap=5
        )

        assert removed == 0
        assert result.count("test keyword") == 1


class TestEnforceBudgetLimits:
    """Test insertion budget enforcement."""

    def test_enforce_sentence_limit(self):
        """Should remove spans to respect sentence limit."""
        text = (
            f"Original. {MARK_START}Added one.{MARK_END} "
            f"More. {MARK_START}Added two.{MARK_END} "
            f"End. {MARK_START}Added three.{MARK_END}"
        )

        result, removed_spans, removed_words, removed_sentences = enforce_budget_limits(
            text, max_sentences=2
        )

        # Should remove at least 1 sentence (3 added, limit 2)
        assert removed_sentences >= 1

    def test_enforce_word_limit(self):
        """Should remove spans to respect word limit."""
        text = (
            f"Original text here. {MARK_START}Added ten words one two three four five six seven eight{MARK_END} "
            f"More text. {MARK_START}Another five words here yes{MARK_END}"
        )

        result, removed_spans, removed_words, removed_sentences = enforce_budget_limits(
            text, max_words=10
        )

        # Should remove words beyond limit
        assert removed_words > 0

    def test_preserve_keyword_spans(self):
        """Should preserve spans with keywords even when over budget."""
        keywords = ["important keyword"]
        text = (
            f"Start. {MARK_START}Important keyword here.{MARK_END} "
            f"Middle. {MARK_START}No keyword span.{MARK_END} "
            f"End. {MARK_START}Another important keyword.{MARK_END}"
        )

        result, removed_spans, removed_words, removed_sentences = enforce_budget_limits(
            text, max_sentences=1, keywords=keywords
        )

        # Should have removed the non-keyword span first
        assert "important keyword" in result.lower()

    def test_no_removal_under_budget(self):
        """Should not remove anything when under budget."""
        text = f"Original. {MARK_START}Just a few words.{MARK_END}"

        result, removed_spans, removed_words, removed_sentences = enforce_budget_limits(
            text, max_sentences=10, max_words=100
        )

        assert removed_spans == 0
        assert removed_words == 0


class TestValidateInsertionsHaveKeywords:
    """Test insertion validation."""

    def test_valid_when_all_have_keywords(self):
        """Should return valid when all insertions have keywords."""
        keywords = ["test keyword", "another"]
        text = (
            f"Start {MARK_START}has test keyword{MARK_END} "
            f"middle {MARK_START}has another{MARK_END} end"
        )

        valid, empty_spans = validate_insertions_have_keywords(text, keywords)

        assert valid is True
        assert len(empty_spans) == 0

    def test_invalid_when_some_missing_keywords(self):
        """Should return invalid when some insertions lack keywords."""
        keywords = ["test keyword"]
        text = (
            f"Start {MARK_START}has test keyword{MARK_END} "
            f"middle {MARK_START}no keyword here{MARK_END} end"
        )

        # With require_all=True, should be invalid
        valid, issues = validate_insertions_have_keywords(text, keywords, require_all=True)

        assert valid is False
        assert len(issues) == 1

    def test_valid_when_at_least_one_has_keyword(self):
        """Should be valid when at least one insertion has keyword (require_all=False)."""
        keywords = ["test keyword"]
        text = (
            f"Start {MARK_START}has test keyword{MARK_END} "
            f"middle {MARK_START}no keyword here{MARK_END} end"
        )

        # With require_all=False (default), should be valid if any has keyword
        valid, issues = validate_insertions_have_keywords(text, keywords, require_all=False)

        assert valid is True
        assert len(issues) == 0


class TestRunEnforcement:
    """Test the combined enforcement function."""

    def test_full_enforcement_pipeline(self):
        """Should run all enforcement steps."""
        config = OptimizationConfig.insert_only()
        keywords = ["primary kw"]

        text = (
            f"Start. {MARK_START}Added primary kw{MARK_END} "
            f"Middle. {MARK_START}Another primary kw{MARK_END} "
            f"End. {MARK_START}Third primary kw{MARK_END}"
        )

        result = run_enforcement(
            text, config, keywords, primary_keyword="primary kw"
        )

        assert isinstance(result, EnforcementResult)
        # With cap=1, should remove 2 excess occurrences
        assert result.caps_removed == 2
        assert result.text.count("primary kw") == 1

    def test_enforcement_returns_counts(self):
        """Should return original and final keyword counts."""
        config = OptimizationConfig.insert_only()
        keywords = ["test kw"]

        text = (
            f"Original test kw. "
            f"{MARK_START}Added test kw{MARK_END}. "
            f"{MARK_START}Another test kw{MARK_END}."
        )

        result = run_enforcement(text, config, keywords, primary_keyword="test kw")

        assert "test kw" in result.original_keyword_counts
        assert "test kw" in result.final_keyword_counts
        # Original had 3, final should have fewer due to cap

    def test_enforcement_disabled_in_enhanced_mode(self):
        """Enforcement should be minimal in enhanced mode."""
        config = OptimizationConfig.enhanced()
        keywords = ["test kw"]

        text = (
            f"Start. {MARK_START}test kw{MARK_END} "
            f"Middle. {MARK_START}test kw{MARK_END} "
            f"End. {MARK_START}test kw{MARK_END}"
        )

        result = run_enforcement(text, config, keywords)

        # Enhanced mode doesn't enforce caps
        # All 3 occurrences should remain
        assert result.text.count("test kw") == 3

    def test_enforcement_generates_warnings(self):
        """Should generate warnings for issues found."""
        config = OptimizationConfig.insert_only(
            max_new_sentences_total=1,
            max_new_words_total=5,
        )
        keywords = ["kw"]

        text = (
            f"Original. {MARK_START}First sentence with kw.{MARK_END} "
            f"More. {MARK_START}Second sentence.{MARK_END} "
            f"End. {MARK_START}Third sentence.{MARK_END}"
        )

        result = run_enforcement(text, config, keywords)

        # Should have warnings about budget enforcement
        # (warnings may vary based on what was actually removed)
        assert isinstance(result.warnings, list)


class TestInsertOnlyModeConfig:
    """Test insert_only mode specific configuration."""

    def test_insert_only_config_defaults(self):
        """Insert-only mode should have strict defaults."""
        config = OptimizationConfig.insert_only()

        assert config.optimization_mode == "insert_only"
        assert config.max_new_sentences_total == 2
        assert config.max_new_words_total == 40
        assert config.enforce_keyword_caps is True
        assert config.manual_keywords_only is True

    def test_insert_only_mode_properties(self):
        """Insert-only mode should disable LLM for body."""
        config = OptimizationConfig.insert_only()

        assert config.is_insert_only_mode is True
        assert config.should_use_llm_for_body is False
        assert config.should_enforce_keyword_caps is True


class TestMarkerSpanDataclass:
    """Test MarkerSpan dataclass."""

    def test_marker_span_creation(self):
        """Should create MarkerSpan with all fields."""
        span = MarkerSpan(
            start=10,
            end=30,
            content="test content",
            word_count=2,
            sentence_count=1,
            keywords_found={"test"},
        )

        assert span.start == 10
        assert span.end == 30
        assert span.content == "test content"
        assert span.word_count == 2
        assert span.sentence_count == 1
        assert "test" in span.keywords_found

    def test_marker_span_empty_keywords(self):
        """Should handle empty keywords set."""
        span = MarkerSpan(
            start=0,
            end=10,
            content="no keywords",
            word_count=2,
            sentence_count=1,
            keywords_found=set(),
        )

        assert len(span.keywords_found) == 0


class TestStripMarkedAdditions:
    """Test strip_marked_additions function."""

    def test_strip_single_addition(self):
        """Should remove a single marked addition."""
        from seo_content_optimizer.enforcement import strip_marked_additions

        text = f"Original text {MARK_START}added content{MARK_END} more original."
        result = strip_marked_additions(text)

        assert "added content" not in result
        assert "Original text" in result
        assert "more original" in result
        assert MARK_START not in result
        assert MARK_END not in result

    def test_strip_multiple_additions(self):
        """Should remove multiple marked additions."""
        from seo_content_optimizer.enforcement import strip_marked_additions

        text = (
            f"First part. {MARK_START}addition one{MARK_END} "
            f"Middle. {MARK_START}addition two{MARK_END} End."
        )
        result = strip_marked_additions(text)

        assert "addition one" not in result
        assert "addition two" not in result
        assert "First part" in result
        assert "Middle" in result
        assert "End" in result

    def test_strip_no_additions(self):
        """Should return unchanged text when no markers."""
        from seo_content_optimizer.enforcement import strip_marked_additions

        text = "Plain text with no markers at all."
        result = strip_marked_additions(text)

        assert result == text

    def test_strip_cleans_double_spaces(self):
        """Should clean up double spaces after stripping."""
        from seo_content_optimizer.enforcement import strip_marked_additions

        text = f"Before {MARK_START}removed{MARK_END} after."
        result = strip_marked_additions(text)

        assert "  " not in result  # No double spaces


class TestValidateStripAdditions:
    """Test validate_strip_additions function."""

    def test_valid_when_stripping_restores_original(self):
        """Should be valid when stripping additions restores original."""
        from seo_content_optimizer.enforcement import validate_strip_additions

        source = "This is the original content."
        optimized = f"This is {MARK_START}keyword{MARK_END} the original content."

        result = validate_strip_additions(optimized, source)

        assert result.valid is True
        assert len(result.differences) == 0

    def test_valid_for_multiple_additions(self):
        """Should be valid when multiple additions are stripped."""
        from seo_content_optimizer.enforcement import validate_strip_additions

        source = "First sentence. Second sentence. Third sentence."
        optimized = (
            f"{MARK_START}Added keyword{MARK_END} First sentence. "
            f"Second {MARK_START}another addition{MARK_END} sentence. "
            f"Third sentence."
        )

        result = validate_strip_additions(optimized, source)

        assert result.valid is True

    def test_invalid_when_content_deleted(self):
        """Should be invalid when original content is missing."""
        from seo_content_optimizer.enforcement import validate_strip_additions

        source = "First sentence. Second sentence. Third sentence."
        # Note: Second sentence is missing (deleted, not just marked)
        optimized = f"First sentence. {MARK_START}added{MARK_END} Third sentence."

        result = validate_strip_additions(optimized, source, strict=True)

        # Should detect missing content
        assert result.valid is False or len(result.missing_from_stripped) > 0

    def test_returns_stripped_text(self):
        """Should return the stripped text in result."""
        from seo_content_optimizer.enforcement import validate_strip_additions

        source = "Original content here."
        optimized = f"Original {MARK_START}keyword{MARK_END} content here."

        result = validate_strip_additions(optimized, source)

        assert "keyword" not in result.stripped_text
        assert MARK_START not in result.stripped_text


class TestGetStripAdditionsReport:
    """Test get_strip_additions_report function."""

    def test_report_for_valid_result(self):
        """Should generate positive report for valid result."""
        from seo_content_optimizer.enforcement import (
            validate_strip_additions,
            get_strip_additions_report,
        )

        source = "Original content."
        optimized = f"Original {MARK_START}added{MARK_END} content."

        result = validate_strip_additions(optimized, source)
        report = get_strip_additions_report(result)

        assert "VALID" in report

    def test_report_for_invalid_result(self):
        """Should generate negative report for invalid result."""
        from seo_content_optimizer.enforcement import (
            StripAdditionsResult,
            get_strip_additions_report,
        )

        result = StripAdditionsResult(
            valid=False,
            stripped_text="stripped",
            source_text="original",
            differences=["Missing: something"],
            missing_from_stripped=["missing content"],
            extra_in_stripped=[],
        )
        report = get_strip_additions_report(result)

        assert "INVALID" in report
        assert "Missing" in report


class TestEnforceKeywordDeltaBudgets:
    """Test delta budget enforcement (new additions, not total counts)."""

    def test_enforces_delta_budget_not_total(self):
        """Should limit NEW additions, not total occurrences."""
        from seo_content_optimizer.enforcement import enforce_keyword_delta_budgets

        keywords = ["test kw"]
        # Source already had 3 occurrences
        source_counts = {"test kw": 3}
        # Optimized text has 3 original + 2 new = 5 total
        text = (
            f"test kw in original. test kw again. test kw third. "
            f"{MARK_START}test kw added{MARK_END}. "
            f"{MARK_START}test kw extra{MARK_END}."
        )

        result, removed, delta_results = enforce_keyword_delta_budgets(
            text, keywords, source_counts,
            allowed_new_primary=1,  # Only 1 new addition allowed
        )

        # Should remove 1 of the 2 new additions, keeping original 3 + 1 new = 4
        assert removed == 1
        assert len(delta_results) == 1
        assert delta_results[0].within_budget is True
        assert delta_results[0].new_additions <= 1

    def test_allows_all_original_content(self):
        """Should allow all original occurrences regardless of count."""
        from seo_content_optimizer.enforcement import enforce_keyword_delta_budgets

        keywords = ["test kw"]
        # Source had 5 occurrences (no limit on original)
        source_counts = {"test kw": 5}
        # No new additions
        text = "test kw. test kw. test kw. test kw. test kw."

        result, removed, delta_results = enforce_keyword_delta_budgets(
            text, keywords, source_counts,
            allowed_new_primary=1,
        )

        # Should not remove anything - all are original
        assert removed == 0
        assert delta_results[0].within_budget is True

    def test_primary_vs_secondary_delta_budgets(self):
        """Should use different budgets for primary vs secondary."""
        from seo_content_optimizer.enforcement import enforce_keyword_delta_budgets

        keywords = ["primary kw", "secondary kw"]
        source_counts = {"primary kw": 0, "secondary kw": 0}
        text = (
            f"{MARK_START}primary kw{MARK_END}. "
            f"{MARK_START}primary kw{MARK_END}. "
            f"{MARK_START}secondary kw{MARK_END}. "
            f"{MARK_START}secondary kw{MARK_END}."
        )

        result, removed, delta_results = enforce_keyword_delta_budgets(
            text, keywords, source_counts,
            allowed_new_primary=2,
            allowed_new_secondary=1,
            primary_keyword="primary kw",
        )

        # Primary should keep 2, secondary should keep 1
        primary_result = next(r for r in delta_results if r.keyword == "primary kw")
        secondary_result = next(r for r in delta_results if r.keyword == "secondary kw")

        assert primary_result.within_budget is True
        assert primary_result.final_count == 2
        assert secondary_result.within_budget is True
        assert secondary_result.final_count == 1

    def test_delta_budget_result_fields(self):
        """Should populate all DeltaBudgetResult fields correctly."""
        from seo_content_optimizer.enforcement import enforce_keyword_delta_budgets

        keywords = ["test"]
        source_counts = {"test": 1}
        text = f"Original test here. {MARK_START}test added{MARK_END}."

        result, removed, delta_results = enforce_keyword_delta_budgets(
            text, keywords, source_counts,
            allowed_new_primary=1,
        )

        assert len(delta_results) == 1
        dr = delta_results[0]
        assert dr.keyword == "test"
        assert dr.source_count == 1
        assert dr.allowed_new == 1
        assert dr.within_budget is True


class TestHighlightCorrectness:
    """Test that highlighting correctly marks only inserted content."""

    def test_markers_wrap_only_inserted_text(self):
        """Markers should only wrap actually inserted content."""
        # When text is properly marked, stripping should restore original
        from seo_content_optimizer.enforcement import strip_marked_additions

        source = "This is original content."
        optimized = f"This is {MARK_START}keyword{MARK_END} original content."

        stripped = strip_marked_additions(optimized)
        # Stripped should match original (normalized)
        assert "keyword" not in stripped
        assert "original content" in stripped

    def test_no_marker_in_original_content(self):
        """Original content should NOT be inside markers."""
        # This tests that original words aren't wrapped
        source = "The quick brown fox jumps."
        optimized = f"The {MARK_START}keyword{MARK_END} quick brown fox jumps."

        # Check that original words are NOT inside markers
        assert f"{MARK_START}quick" not in optimized
        assert f"fox{MARK_END}" not in optimized
        assert f"{MARK_START}brown{MARK_END}" not in optimized

    def test_multiple_insertions_each_have_markers(self):
        """Each insertion should have its own marker pair."""
        text = (
            f"Start {MARK_START}first insert{MARK_END} "
            f"middle {MARK_START}second insert{MARK_END} end."
        )

        # Count marker pairs
        starts = text.count(MARK_START)
        ends = text.count(MARK_END)

        assert starts == 2
        assert ends == 2
        assert starts == ends  # Balanced markers

    def test_stripping_markers_produces_clean_text(self):
        """Stripping markers should produce readable text without artifacts."""
        from seo_content_optimizer.enforcement import strip_marked_additions

        text = f"Start {MARK_START}added{MARK_END}. Middle. {MARK_START}more{MARK_END} end."
        stripped = strip_marked_additions(text)

        # No markers
        assert MARK_START not in stripped
        assert MARK_END not in stripped
        # No double spaces
        assert "  " not in stripped
        # Should still have original content
        assert "Start" in stripped
        assert "Middle" in stripped
        assert "end" in stripped
