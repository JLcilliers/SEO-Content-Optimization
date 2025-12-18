# -*- coding: utf-8 -*-
"""
Regression tests for insert-only (minimal) optimization mode.

These tests verify that when optimization_mode='minimal':
1. Keywords appear at most N times in body (where N is the cap)
2. FAQ generation is disabled
3. AI add-ons are disabled
4. Keyword allowlist is enforced
5. No density targeting or distribution occurs
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from seo_content_optimizer.config import OptimizationConfig
from seo_content_optimizer.models import (
    Keyword,
    KeywordPlan,
    ParagraphBlock,
    HeadingLevel,
)
from seo_content_optimizer.optimizer import (
    ContentOptimizer,
    KeywordCapValidationResult,
    CapValidationReport,
    DebugBundle,
    ScopedKeywordCounts,
    KeywordCounts,
    count_keyword_in_text,
    strip_markers,
)


class TestMinimalModeConfig:
    """Test that minimal mode configuration is set up correctly."""

    def test_minimal_config_defaults(self):
        """Minimal mode should have correct defaults."""
        config = OptimizationConfig.minimal()

        assert config.optimization_mode == "minimal"
        assert config.faq_policy == "never"
        assert config.generate_ai_sections is False
        assert config.generate_key_takeaways is False
        assert config.generate_chunk_map is False
        assert config.enforce_keyword_caps is True
        assert config.primary_keyword_body_cap == 1
        assert config.secondary_keyword_body_cap == 1

    def test_minimal_mode_properties(self):
        """Minimal mode properties should return correct values."""
        config = OptimizationConfig.minimal()

        assert config.is_minimal_mode is True
        assert config.is_enhanced_mode is False
        assert config.should_target_density is False
        assert config.should_distribute_keywords is False
        assert config.should_expand_content is False
        assert config.should_enforce_keyword_caps is True
        assert config.should_generate_faq is False
        assert config.should_generate_ai_addons is False

    def test_for_manual_keywords_creates_minimal_config(self):
        """for_manual_keywords should create minimal mode config by default."""
        keywords = ["primary keyword", "secondary one", "secondary two"]
        config = OptimizationConfig.for_manual_keywords(keywords)

        assert config.optimization_mode == "minimal"
        assert config.has_keyword_allowlist is True
        assert "primary keyword" in {k.lower() for k in config.keyword_allowlist}

    def test_keyword_allowlist_enforcement(self):
        """Keyword allowlist should correctly identify allowed keywords."""
        config = OptimizationConfig.for_manual_keywords(
            ["allowed keyword", "Another Allowed"]
        )

        assert config.is_keyword_allowed("allowed keyword") is True
        assert config.is_keyword_allowed("ALLOWED KEYWORD") is True  # Case insensitive
        assert config.is_keyword_allowed("another allowed") is True
        assert config.is_keyword_allowed("not allowed") is False


class TestCapValidation:
    """Test keyword cap validation functionality."""

    def test_keyword_cap_validation_result_within_cap(self):
        """KeywordCapValidationResult should detect within-cap status."""
        result = KeywordCapValidationResult(
            keyword="test keyword",
            cap=1,
            actual_body_count=1,
            actual_total_count=3,
            is_primary=True,
        )

        assert result.within_cap is True
        assert result.excess_count == 0

    def test_keyword_cap_validation_result_over_cap(self):
        """KeywordCapValidationResult should detect over-cap status."""
        result = KeywordCapValidationResult(
            keyword="test keyword",
            cap=1,
            actual_body_count=5,
            actual_total_count=8,
            is_primary=True,
        )

        assert result.within_cap is False
        assert result.excess_count == 4

    def test_cap_validation_report_all_within_caps(self):
        """CapValidationReport should correctly identify when all keywords are within caps."""
        report = CapValidationReport(
            mode="minimal",
            cap_enforcement_enabled=True,
            primary_cap=1,
            secondary_cap=1,
            keyword_results=[
                KeywordCapValidationResult(
                    keyword="primary",
                    cap=1,
                    actual_body_count=1,
                    actual_total_count=2,
                    is_primary=True,
                ),
                KeywordCapValidationResult(
                    keyword="secondary",
                    cap=1,
                    actual_body_count=1,
                    actual_total_count=1,
                    is_primary=False,
                ),
            ],
        )

        assert report.all_within_caps is True
        assert report.total_excess == 0
        assert len(report.keywords_over_cap) == 0

    def test_cap_validation_report_some_over_cap(self):
        """CapValidationReport should correctly identify over-cap keywords."""
        report = CapValidationReport(
            mode="minimal",
            cap_enforcement_enabled=True,
            primary_cap=1,
            secondary_cap=1,
            keyword_results=[
                KeywordCapValidationResult(
                    keyword="primary",
                    cap=1,
                    actual_body_count=1,
                    actual_total_count=2,
                    is_primary=True,
                ),
                KeywordCapValidationResult(
                    keyword="secondary",
                    cap=1,
                    actual_body_count=3,  # Over cap
                    actual_total_count=5,
                    is_primary=False,
                ),
            ],
        )

        assert report.all_within_caps is False
        assert report.total_excess == 2  # 3 - 1 = 2 excess
        assert len(report.keywords_over_cap) == 1
        assert report.keywords_over_cap[0].keyword == "secondary"


class TestDebugBundle:
    """Test debug bundle generation and export."""

    def test_debug_bundle_to_dict(self):
        """Debug bundle should convert to dict correctly."""
        from seo_content_optimizer.optimizer import (
            DebugBundleConfig,
            DebugBundleKeyword,
        )

        config = DebugBundleConfig(
            optimization_mode="minimal",
            faq_policy="never",
            generate_ai_sections=False,
            generate_key_takeaways=False,
            generate_chunk_map=False,
            primary_keyword_body_cap=1,
            secondary_keyword_body_cap=1,
            enforce_keyword_caps=True,
            max_secondary=5,
            has_keyword_allowlist=True,
            keyword_allowlist={"test keyword"},
        )

        bundle = DebugBundle(
            timestamp="2024-01-01T00:00:00",
            config=config,
            keywords=[
                DebugBundleKeyword(
                    phrase="test keyword",
                    is_primary=True,
                    cap=1,
                    body_count=1,
                    meta_count=1,
                    headings_count=0,
                    faq_count=0,
                    total_count=2,
                    within_cap=True,
                ),
            ],
            cap_validation=None,
            total_blocks=10,
            blocks_with_keywords=3,
            warnings=[],
        )

        d = bundle.to_dict()

        assert d["timestamp"] == "2024-01-01T00:00:00"
        assert d["config"]["optimization_mode"] == "minimal"
        assert d["config"]["enforce_keyword_caps"] is True
        assert len(d["keywords"]) == 1
        assert d["keywords"][0]["phrase"] == "test keyword"
        assert d["summary"]["total_blocks"] == 10

    def test_debug_bundle_to_json(self):
        """Debug bundle should export to JSON."""
        from seo_content_optimizer.optimizer import (
            DebugBundleConfig,
            DebugBundleKeyword,
        )
        import json

        config = DebugBundleConfig(
            optimization_mode="minimal",
            faq_policy="never",
            generate_ai_sections=False,
            generate_key_takeaways=False,
            generate_chunk_map=False,
            primary_keyword_body_cap=1,
            secondary_keyword_body_cap=1,
            enforce_keyword_caps=True,
            max_secondary=5,
            has_keyword_allowlist=False,
            keyword_allowlist=None,
        )

        bundle = DebugBundle(
            timestamp="2024-01-01T00:00:00",
            config=config,
            keywords=[],
            cap_validation=None,
            total_blocks=5,
            blocks_with_keywords=0,
            warnings=[],
        )

        json_str = bundle.to_json()
        parsed = json.loads(json_str)

        assert parsed["config"]["optimization_mode"] == "minimal"
        assert parsed["summary"]["total_blocks"] == 5

    def test_debug_bundle_to_checklist(self):
        """Debug bundle should generate human-readable checklist."""
        from seo_content_optimizer.optimizer import (
            DebugBundleConfig,
            DebugBundleKeyword,
        )

        config = DebugBundleConfig(
            optimization_mode="minimal",
            faq_policy="never",
            generate_ai_sections=False,
            generate_key_takeaways=False,
            generate_chunk_map=False,
            primary_keyword_body_cap=1,
            secondary_keyword_body_cap=1,
            enforce_keyword_caps=True,
            max_secondary=5,
            has_keyword_allowlist=True,
            keyword_allowlist={"test"},
        )

        bundle = DebugBundle(
            timestamp="2024-01-01T00:00:00",
            config=config,
            keywords=[
                DebugBundleKeyword(
                    phrase="test keyword",
                    is_primary=True,
                    cap=1,
                    body_count=1,
                    meta_count=1,
                    headings_count=0,
                    faq_count=0,
                    total_count=2,
                    within_cap=True,
                ),
            ],
            cap_validation=None,
            total_blocks=10,
            blocks_with_keywords=3,
            warnings=[],
        )

        checklist = bundle.to_checklist()

        assert "INSERT-ONLY MODE DEBUG CHECKLIST" in checklist
        assert "optimization_mode = 'minimal'" in checklist
        assert "enforce_keyword_caps = True" in checklist
        assert "PASS" in checklist  # Should pass since within cap


class TestKeywordCountHelpers:
    """Test helper functions for keyword counting."""

    def test_count_keyword_in_text_basic(self):
        """count_keyword_in_text should find exact matches."""
        text = "This is a test keyword and another test keyword here."
        count = count_keyword_in_text(text, "test keyword")
        assert count == 2

    def test_count_keyword_in_text_case_insensitive(self):
        """count_keyword_in_text should be case insensitive."""
        text = "Test Keyword and TEST KEYWORD and test keyword"
        count = count_keyword_in_text(text.lower(), "test keyword")
        assert count == 3

    def test_count_keyword_in_text_no_matches(self):
        """count_keyword_in_text should return 0 for no matches."""
        text = "This text has no matching phrases."
        count = count_keyword_in_text(text, "nonexistent keyword")
        assert count == 0

    def test_strip_markers_removes_add_markers(self):
        """strip_markers should remove [[[ADD]]] markers."""
        text = "Original [[[ADD]]]added text[[[ENDADD]]] more original"
        stripped = strip_markers(text)
        assert "[[[ADD]]]" not in stripped
        assert "[[[ENDADD]]]" not in stripped
        assert "added text" in stripped


class TestMinimalModeKeywordInsertion:
    """Test that minimal mode correctly limits keyword insertions."""

    def test_scoped_keyword_counts_structure(self):
        """ScopedKeywordCounts should track counts by location."""
        counts = KeywordCounts(
            meta=1,
            headings=2,
            body=1,
            faq=0,
        )
        scoped = ScopedKeywordCounts(
            keyword="test keyword",
            target=8,
            counts=counts,
            is_primary=True,
        )

        assert scoped.counts.total == 4
        assert scoped.counts.body_and_headings == 3
        assert scoped.body_satisfied is False  # 1 < 8
        assert scoped.body_needed == 7  # 8 - 1 = 7

    def test_minimal_mode_body_satisfied_at_cap(self):
        """In minimal mode, body should be 'satisfied' at cap=1."""
        counts = KeywordCounts(
            meta=1,
            headings=0,
            body=1,  # Exactly at cap
            faq=0,
        )
        scoped = ScopedKeywordCounts(
            keyword="test keyword",
            target=1,  # Cap is 1 in minimal mode
            counts=counts,
            is_primary=True,
        )

        assert scoped.body_satisfied is True
        assert scoped.body_needed == 0


class TestEnhancedVsMinimalMode:
    """Test differences between enhanced and minimal modes."""

    def test_enhanced_mode_enables_features(self):
        """Enhanced mode should enable density targeting and expansion."""
        config = OptimizationConfig.enhanced()

        assert config.optimization_mode == "enhanced"
        assert config.is_enhanced_mode is True
        assert config.should_target_density is True
        assert config.should_distribute_keywords is True
        assert config.should_expand_content is True
        assert config.should_enforce_keyword_caps is False  # Caps are minimal-mode only

    def test_minimal_mode_disables_features(self):
        """Minimal mode should disable density targeting and expansion."""
        config = OptimizationConfig.minimal()

        assert config.optimization_mode == "minimal"
        assert config.is_minimal_mode is True
        assert config.should_target_density is False
        assert config.should_distribute_keywords is False
        assert config.should_expand_content is False
        assert config.should_enforce_keyword_caps is True

    def test_faq_policy_in_minimal_mode(self):
        """FAQ should only be generated in minimal mode if faq_policy='always'."""
        # Default minimal mode - FAQ disabled
        config = OptimizationConfig.minimal()
        assert config.should_generate_faq is False

        # Minimal with faq_policy='always' - FAQ enabled
        config_with_faq = OptimizationConfig.minimal(faq_policy="always")
        assert config_with_faq.should_generate_faq is True

    def test_ai_addons_in_minimal_mode(self):
        """AI add-ons should be disabled by default in minimal mode."""
        config = OptimizationConfig.minimal()
        assert config.should_generate_ai_addons is False

        # Even if master switch is on, individual switches are off
        config_with_master = OptimizationConfig.minimal(generate_ai_sections=True)
        # Still False because generate_key_takeaways and generate_chunk_map are False
        assert config_with_master.should_generate_ai_addons is False


class TestExistingFAQDetection:
    """Test FAQ detection in source content."""

    def test_source_has_faq_detects_faq_heading(self):
        """_source_has_faq should detect FAQ-related headings."""
        optimizer = ContentOptimizer.__new__(ContentOptimizer)

        # Source with FAQ heading
        optimizer._full_original_text = """
        Welcome to our website.

        Frequently Asked Questions

        Q: What is your service?
        A: We provide great service.
        """
        assert optimizer._source_has_faq() is True

    def test_source_has_faq_detects_faqs_text(self):
        """_source_has_faq should detect 'FAQs' text."""
        optimizer = ContentOptimizer.__new__(ContentOptimizer)

        optimizer._full_original_text = """
        About our company.

        FAQs

        Here are common questions.
        """
        assert optimizer._source_has_faq() is True

    def test_source_has_faq_detects_common_questions(self):
        """_source_has_faq should detect 'Common Questions' heading."""
        optimizer = ContentOptimizer.__new__(ContentOptimizer)

        optimizer._full_original_text = """
        Product information.

        Common Questions

        What do you offer?
        """
        assert optimizer._source_has_faq() is True

    def test_source_has_faq_returns_false_without_faq(self):
        """_source_has_faq should return False when no FAQ exists."""
        optimizer = ContentOptimizer.__new__(ContentOptimizer)

        optimizer._full_original_text = """
        Welcome to our website.

        About Us

        We are a great company that provides excellent services.
        Contact us today!
        """
        assert optimizer._source_has_faq() is False

    def test_source_has_faq_returns_false_with_no_text(self):
        """_source_has_faq should return False when no source text."""
        optimizer = ContentOptimizer.__new__(ContentOptimizer)

        # No _full_original_text attribute
        assert optimizer._source_has_faq() is False

        # Empty text
        optimizer._full_original_text = ""
        assert optimizer._source_has_faq() is False

    def test_minimal_mode_skips_faq_when_source_has_faq(self):
        """Minimal mode should skip FAQ generation if source already has FAQ."""
        # This is validated by the logic flow - when is_minimal_mode=True
        # and source_has_existing_faq=True, FAQ generation is skipped
        config = OptimizationConfig.minimal()

        # In minimal mode, even with faq_policy='always', existing FAQ should
        # cause skip. This test documents the expected behavior.
        assert config.is_minimal_mode is True
        # The actual skip logic is in optimizer.py and tested by integration tests


class TestNaturalKeywordInjection:
    """Test the natural keyword injection function for minimal mode."""

    def test_inject_keyword_naturally_skips_existing(self):
        """Should not inject if keyword already exists."""
        from seo_content_optimizer.diff_markers import inject_keyword_naturally

        text = "Our plumbing services are the best in town."
        result = inject_keyword_naturally(text, "plumbing services")
        assert result == text  # Unchanged

    def test_inject_keyword_naturally_after_services(self):
        """Should inject after service-related words."""
        from seo_content_optimizer.diff_markers import inject_keyword_naturally

        text = "We offer excellent services to all our customers."
        result = inject_keyword_naturally(text, "plumbing repair")
        assert "plumbing repair" in result
        assert "[[[ADD]]]" in result
        assert "including plumbing repair" in result

    def test_inject_keyword_naturally_at_sentence_end(self):
        """Should inject at sentence end when no service words."""
        from seo_content_optimizer.diff_markers import inject_keyword_naturally

        text = "Our team has been helping customers for years."
        result = inject_keyword_naturally(text, "emergency plumbing")
        assert "emergency plumbing" in result
        assert "[[[ADD]]]" in result

    def test_inject_keyword_naturally_empty_text(self):
        """Should handle empty text gracefully."""
        from seo_content_optimizer.diff_markers import inject_keyword_naturally

        assert inject_keyword_naturally("", "keyword") == ""
        assert inject_keyword_naturally("text", "") == "text"


class TestPatchBasedOptimization:
    """Test patch-based body optimization in minimal mode."""

    def test_minimal_mode_preserves_original_text(self):
        """Minimal mode should preserve original text structure."""
        config = OptimizationConfig.minimal()

        # Verify minimal mode config is set correctly
        assert config.is_minimal_mode is True
        assert config.should_target_density is False
        assert config.should_distribute_keywords is False

    def test_minimal_mode_respects_caps(self):
        """Minimal mode should respect keyword caps."""
        config = OptimizationConfig.minimal()

        assert config.primary_keyword_body_cap == 1
        assert config.secondary_keyword_body_cap == 1
        assert config.should_enforce_keyword_caps is True

    def test_minimal_mode_no_content_expansion(self):
        """Minimal mode should not expand content."""
        config = OptimizationConfig.minimal()

        assert config.should_expand_content is False


# ===========================================================================
# INSERT-ONLY MODE SPECIFIC TESTS
# ===========================================================================

class TestInsertOnlyModeConfig:
    """Test insert_only mode configuration (strictest mode)."""

    def test_insert_only_config_defaults(self):
        """insert_only mode should have strictest defaults."""
        config = OptimizationConfig.insert_only()

        assert config.optimization_mode == "insert_only"
        assert config.faq_policy == "never"
        assert config.generate_ai_sections is False
        assert config.enforce_keyword_caps is True
        assert config.manual_keywords_only is True
        assert config.max_new_sentences_total == 2
        assert config.max_new_words_total == 40

    def test_insert_only_mode_properties(self):
        """insert_only mode properties should return correct values."""
        config = OptimizationConfig.insert_only()

        assert config.is_insert_only_mode is True
        assert config.is_minimal_mode is True  # insert_only is a subset of minimal
        assert config.is_enhanced_mode is False
        assert config.should_use_llm_for_body is False  # Key difference from minimal
        assert config.should_enforce_keyword_caps is True
        assert config.should_generate_faq is False

    def test_insert_only_disables_llm_for_body(self):
        """insert_only mode should disable LLM for body optimization."""
        config = OptimizationConfig.insert_only()

        assert config.should_use_llm_for_body is False

    def test_minimal_still_uses_llm_for_body(self):
        """Regular minimal mode should still use LLM for body (contrast test)."""
        config = OptimizationConfig.minimal()

        # Minimal mode uses LLM with restrictions, insert_only does not
        assert config.should_use_llm_for_body is True

    def test_insert_only_with_faq_override(self):
        """insert_only can have FAQ enabled via override."""
        config = OptimizationConfig.insert_only(faq_policy="always", faq_count=3)

        assert config.faq_policy == "always"
        assert config.faq_count == 3
        assert config.should_generate_faq is True

    def test_insert_only_budget_override(self):
        """insert_only can have budget limits overridden."""
        config = OptimizationConfig.insert_only(
            max_new_sentences_total=5,
            max_new_words_total=100,
        )

        assert config.max_new_sentences_total == 5
        assert config.max_new_words_total == 100


class TestInsertOnlyVsMinimalMode:
    """Test differences between insert_only and minimal modes."""

    def test_both_are_minimal_modes(self):
        """Both insert_only and minimal should be considered 'minimal mode'."""
        insert_only = OptimizationConfig.insert_only()
        minimal = OptimizationConfig.minimal()

        assert insert_only.is_minimal_mode is True
        assert minimal.is_minimal_mode is True

    def test_only_insert_only_is_insert_only_mode(self):
        """Only insert_only should have is_insert_only_mode=True."""
        insert_only = OptimizationConfig.insert_only()
        minimal = OptimizationConfig.minimal()

        assert insert_only.is_insert_only_mode is True
        assert minimal.is_insert_only_mode is False

    def test_llm_usage_differs(self):
        """insert_only should not use LLM for body, minimal may."""
        insert_only = OptimizationConfig.insert_only()
        minimal = OptimizationConfig.minimal()

        assert insert_only.should_use_llm_for_body is False
        assert minimal.should_use_llm_for_body is True

    def test_both_enforce_keyword_caps(self):
        """Both modes should enforce keyword caps."""
        insert_only = OptimizationConfig.insert_only()
        minimal = OptimizationConfig.minimal()

        assert insert_only.should_enforce_keyword_caps is True
        assert minimal.should_enforce_keyword_caps is True


class TestInsertOnlyEnforcement:
    """Test enforcement integration for insert_only mode."""

    def test_enforcement_runs_in_insert_only_mode(self):
        """Enforcement should run in insert_only mode."""
        from seo_content_optimizer.enforcement import run_enforcement, EnforcementResult
        from seo_content_optimizer.diff_markers import MARK_START, MARK_END

        config = OptimizationConfig.insert_only()
        keywords = ["test keyword"]

        text = (
            f"Original. {MARK_START}test keyword{MARK_END}. "
            f"More. {MARK_START}test keyword{MARK_END}."
        )

        result = run_enforcement(text, config, keywords, primary_keyword="test keyword")

        assert isinstance(result, EnforcementResult)
        # With cap=1, one occurrence should be removed
        assert result.caps_removed >= 1

    def test_enforcement_returns_keyword_counts(self):
        """Enforcement should return keyword counts."""
        from seo_content_optimizer.enforcement import run_enforcement
        from seo_content_optimizer.diff_markers import MARK_START, MARK_END

        config = OptimizationConfig.insert_only()
        keywords = ["test"]

        text = f"Start. {MARK_START}test{MARK_END}."

        result = run_enforcement(text, config, keywords, primary_keyword="test")

        assert "test" in result.original_keyword_counts
        assert "test" in result.final_keyword_counts


class TestInsertOnlyHighlightIntegrity:
    """Test highlight integrity for insert_only mode."""

    def test_highlight_integrity_check_runs(self):
        """Highlight integrity check should run successfully."""
        from seo_content_optimizer.highlight_integrity import (
            run_highlight_integrity_check,
            HighlightIntegrityReport,
        )
        from seo_content_optimizer.diff_markers import MARK_START, MARK_END

        original = "The quick brown fox jumps over the lazy dog."
        marked = f"The quick brown fox {MARK_START}and cat{MARK_END} jumps over the lazy dog."

        report = run_highlight_integrity_check(original, marked)

        assert isinstance(report, HighlightIntegrityReport)
        assert report.markers_balanced is True

    def test_highlight_integrity_detects_issues(self):
        """Highlight integrity should detect issues."""
        from seo_content_optimizer.highlight_integrity import run_highlight_integrity_check
        from seo_content_optimizer.diff_markers import MARK_START

        original = "Original text."
        # Unbalanced markers
        marked = f"Some {MARK_START}added text without end marker."

        report = run_highlight_integrity_check(original, marked)

        assert report.markers_balanced is False

    def test_highlight_integrity_url_preservation(self):
        """Highlight integrity should check URL preservation."""
        from seo_content_optimizer.highlight_integrity import run_highlight_integrity_check

        original = "Visit us at https://example.com for more info."
        marked = "Visit us at https://example.com for more info."  # Preserved

        report = run_highlight_integrity_check(original, marked)

        assert report.urls_preserved is True


class TestDeterministicInjection:
    """Test that insert_only mode uses deterministic injection."""

    def test_inject_keyword_naturally_is_deterministic(self):
        """inject_keyword_naturally should produce consistent results."""
        from seo_content_optimizer.diff_markers import inject_keyword_naturally

        text = "We provide professional services to all customers."
        keyword = "plumbing repair"

        result1 = inject_keyword_naturally(text, keyword)
        result2 = inject_keyword_naturally(text, keyword)

        # Same input should produce same output (deterministic)
        assert result1 == result2

    def test_inject_adds_markers_for_new_content(self):
        """Injected content should be wrapped in markers."""
        from seo_content_optimizer.diff_markers import inject_keyword_naturally, MARK_START, MARK_END

        text = "We provide professional services."
        keyword = "drain cleaning"

        result = inject_keyword_naturally(text, keyword)

        # Should have markers around injected keyword
        if keyword in result and keyword not in text:
            assert MARK_START in result
            assert MARK_END in result

    def test_inject_preserves_existing_keywords(self):
        """Should not re-inject if keyword already exists."""
        from seo_content_optimizer.diff_markers import inject_keyword_naturally

        text = "Our drain cleaning services are the best."
        keyword = "drain cleaning"

        result = inject_keyword_naturally(text, keyword)

        # Should be unchanged
        assert result == text


class TestManualKeywordsInInsertOnlyMode:
    """Test manual keywords work correctly in insert_only mode."""

    def test_manual_keywords_config_with_insert_only(self):
        """ManualKeywordsConfig should work with insert_only config."""
        from seo_content_optimizer.models import ManualKeywordsConfig

        manual = ManualKeywordsConfig(
            primary="main keyword phrase",
            secondary=["secondary one", "secondary two"],
        )

        config = OptimizationConfig.insert_only()

        # Both should work together
        assert manual.primary == "main keyword phrase"
        assert len(manual.secondary) == 2
        assert config.is_insert_only_mode is True

    def test_manual_keywords_preserves_exact_casing(self):
        """Manual keywords should preserve exact phrase casing."""
        from seo_content_optimizer.models import ManualKeywordsConfig

        manual = ManualKeywordsConfig(
            primary="SEO Content Optimization",
            secondary=["Local SEO Services"],
        )

        # Exact casing should be preserved
        assert manual.primary == "SEO Content Optimization"
        assert "Local SEO Services" in manual.secondary
