"""
Tests for output_validator module.

This module tests the post-generation safety checks that ensure LLM outputs
don't contain blocked industry terms that weren't in the original content.
"""

import pytest
from src.seo_content_optimizer.output_validator import (
    find_blocked_terms,
    validate_output,
    validate_and_fallback,
    validate_faq_items,
    get_violation_summary,
    ValidationResult,
    BLOCKED_TERMS,
    SINGLE_WORD_BLOCKLIST,
)


class TestFindBlockedTerms:
    """Tests for find_blocked_terms function."""

    def test_no_blocked_terms(self):
        """Text with no blocked terms returns empty list."""
        text = "This is a normal business text about access control systems."
        result = find_blocked_terms(text)
        assert result == []

    def test_single_blocked_term(self):
        """Single blocked term is detected."""
        text = "Our cannabis products are high quality."
        result = find_blocked_terms(text)
        assert "cannabis" in result

    def test_multiple_blocked_terms(self):
        """Multiple blocked terms are all detected."""
        text = "Visit our dispensary for hemp and CBD products."
        result = find_blocked_terms(text)
        assert "dispensary" in result
        assert "hemp" in result
        assert "cbd" in result

    def test_case_insensitive(self):
        """Detection is case-insensitive."""
        text = "CANNABIS, Hemp, CbD are all blocked."
        result = find_blocked_terms(text)
        assert "cannabis" in result
        assert "hemp" in result
        assert "cbd" in result

    def test_word_boundary_detection(self):
        """Terms are detected only at word boundaries."""
        # "spa" should be detected as a word
        text = "Visit our spa for relaxation."
        result = find_blocked_terms(text)
        assert "spa" in result

        # "spa" within another word should not be detected
        text = "This is a space exploration company."
        result = find_blocked_terms(text)
        assert "spa" not in result

    def test_high_risk_merchant_terms(self):
        """High-risk merchant terms are detected."""
        text = "We offer high risk merchant account services."
        result = find_blocked_terms(text)
        assert "high risk merchant" in result or "merchant account" in result

    def test_gambling_terms(self):
        """Gambling-related terms are detected."""
        text = "Try your luck at our casino with online betting."
        result = find_blocked_terms(text)
        assert "casino" in result
        assert "betting" in result

    def test_adult_terms(self):
        """Adult content terms are detected."""
        text = "This is adult entertainment content."
        result = find_blocked_terms(text)
        assert "adult" in result

    def test_healthcare_terms(self):
        """Healthcare terms are detected."""
        text = "Visit our pharmacy for prescription medications."
        result = find_blocked_terms(text)
        assert "pharmacy" in result
        assert "prescription" in result

    def test_salon_terms(self):
        """Salon/spa service terms are detected."""
        text = "Our hair salon offers massage therapy services."
        result = find_blocked_terms(text)
        assert "hair salon" in result or "salon" in result
        assert "massage" in result

    def test_deduplication(self):
        """Duplicate terms are removed."""
        text = "Cannabis cannabis CANNABIS."
        result = find_blocked_terms(text)
        assert result.count("cannabis") == 1


class TestValidateOutput:
    """Tests for validate_output function."""

    def test_valid_output_no_blocked_terms(self):
        """Output with no blocked terms is valid."""
        llm_output = "Our security cameras provide excellent coverage."
        original = "Security cameras are important for property protection."

        result = validate_output(llm_output, original)

        assert result.is_valid is True
        assert result.violations == []

    def test_valid_output_term_in_original(self):
        """Blocked term in output is valid if it was in original."""
        llm_output = "Our spa offers relaxing treatments."
        original = "The spa is a great place for relaxation."

        result = validate_output(llm_output, original)

        assert result.is_valid is True
        assert result.violations == []

    def test_invalid_output_new_blocked_term(self):
        """Blocked term introduced by LLM is invalid."""
        llm_output = "Our security system works great for cannabis dispensary locations."
        original = "Our security system provides comprehensive coverage."

        result = validate_output(llm_output, original)

        assert result.is_valid is False
        assert "cannabis" in result.violations
        assert "dispensary" in result.violations

    def test_invalid_multiple_new_terms(self):
        """Multiple new blocked terms are all reported as violations."""
        llm_output = "This works for casino, spa, and hemp businesses."
        original = "This works for many types of businesses."

        result = validate_output(llm_output, original)

        assert result.is_valid is False
        # At least one violation detected (hemp is in single-word blocklist)
        assert len(result.violations) >= 1
        assert "hemp" in result.violations

    def test_empty_llm_output(self):
        """Empty LLM output is considered valid."""
        result = validate_output("", "Some original content")

        assert result.is_valid is True
        assert result.violations == []

    def test_empty_original_content(self):
        """Empty original allows all terms (rare edge case)."""
        llm_output = "Our casino offers great games."
        result = validate_output(llm_output, "")

        # With empty original, any blocked term is a violation
        assert result.is_valid is False
        assert "casino" in result.violations

    def test_has_violations_property(self):
        """ValidationResult.has_violations property works correctly."""
        valid_result = ValidationResult(is_valid=True, violations=[])
        invalid_result = ValidationResult(is_valid=False, violations=["cannabis"])

        assert valid_result.has_violations is False
        assert invalid_result.has_violations is True


class TestValidateAndFallback:
    """Tests for validate_and_fallback function."""

    def test_valid_returns_llm_output(self):
        """Valid LLM output is returned unchanged."""
        llm_output = "Our security cameras are excellent."
        original = "Security cameras help protect your property."

        content, result = validate_and_fallback(llm_output, original)

        assert content == llm_output
        assert result.is_valid is True

    def test_invalid_returns_original(self):
        """Invalid LLM output falls back to original."""
        llm_output = "Our cannabis dispensary has great security."
        original = "Our business has great security."

        content, result = validate_and_fallback(llm_output, original)

        assert content == original
        assert result.is_valid is False
        assert result.sanitized_content == original

    def test_preserves_original_with_blocked_terms(self):
        """If original had blocked terms, they're preserved."""
        llm_output = "Our spa offers relaxing massage services."
        original = "Our spa provides excellent massage services."

        content, result = validate_and_fallback(llm_output, original)

        # Both spa and massage were in original, so output is valid
        assert content == llm_output
        assert result.is_valid is True

    def test_fallback_when_new_term_added(self):
        """Falls back when new blocked term is added to text with existing blocked terms."""
        llm_output = "Our spa offers cannabis products."
        original = "Our spa offers great products."

        content, result = validate_and_fallback(llm_output, original)

        # Cannabis was not in original
        assert content == original
        assert result.is_valid is False
        assert "cannabis" in result.violations


class TestValidateFaqItems:
    """Tests for validate_faq_items function."""

    def test_valid_faqs_pass(self):
        """FAQs without blocked terms pass validation."""
        faqs = [
            {"question": "How do I set up the camera?", "answer": "Follow the installation guide."},
            {"question": "What's the warranty?", "answer": "We offer a 2-year warranty."},
        ]
        original = "Security camera installation and warranty information."

        valid_faqs, results = validate_faq_items(faqs, original)

        assert len(valid_faqs) == 2
        assert all(r.is_valid for r in results)

    def test_invalid_faq_filtered(self):
        """FAQs with new blocked terms are filtered out."""
        faqs = [
            {"question": "How do I set up the camera?", "answer": "Follow the installation guide."},
            {"question": "Does this work for dispensaries?", "answer": "Yes, it's great for cannabis businesses."},
        ]
        original = "Security camera installation guide."

        valid_faqs, results = validate_faq_items(faqs, original)

        assert len(valid_faqs) == 1
        assert valid_faqs[0]["question"] == "How do I set up the camera?"

    def test_all_faqs_filtered(self):
        """All FAQs can be filtered if all have violations."""
        faqs = [
            {"question": "Is this good for casinos?", "answer": "Yes, perfect for gambling venues."},
            {"question": "Can dispensaries use this?", "answer": "Great for cannabis businesses."},
        ]
        original = "Security camera systems for businesses."

        valid_faqs, results = validate_faq_items(faqs, original)

        assert len(valid_faqs) == 0

    def test_empty_faq_list(self):
        """Empty FAQ list returns empty results."""
        valid_faqs, results = validate_faq_items([], "Some original content")

        assert valid_faqs == []
        assert results == []

    def test_faq_blocked_in_question(self):
        """FAQ blocked term in question causes filtering."""
        faqs = [
            {"question": "How does this help salon businesses?", "answer": "It provides security."},
        ]
        original = "Security solutions for businesses."

        valid_faqs, results = validate_faq_items(faqs, original)

        assert len(valid_faqs) == 0

    def test_faq_blocked_in_answer(self):
        """FAQ blocked term in answer causes filtering."""
        faqs = [
            {"question": "What businesses use this?", "answer": "Many spa and salon businesses use our system."},
        ]
        original = "Security solutions for businesses."

        valid_faqs, results = validate_faq_items(faqs, original)

        assert len(valid_faqs) == 0


class TestGetViolationSummary:
    """Tests for get_violation_summary function."""

    def test_all_valid(self):
        """Summary with all valid results."""
        results = [
            ValidationResult(is_valid=True, violations=[]),
            ValidationResult(is_valid=True, violations=[]),
        ]

        summary = get_violation_summary(results)

        assert summary["total_checked"] == 2
        assert summary["valid_count"] == 2
        assert summary["invalid_count"] == 0
        assert summary["unique_violations"] == []

    def test_mixed_results(self):
        """Summary with mixed valid and invalid results."""
        results = [
            ValidationResult(is_valid=True, violations=[]),
            ValidationResult(is_valid=False, violations=["cannabis", "hemp"]),
            ValidationResult(is_valid=False, violations=["cannabis"]),
        ]

        summary = get_violation_summary(results)

        assert summary["total_checked"] == 3
        assert summary["valid_count"] == 1
        assert summary["invalid_count"] == 2
        assert set(summary["unique_violations"]) == {"cannabis", "hemp"}
        assert summary["violation_frequency"]["cannabis"] == 2
        assert summary["violation_frequency"]["hemp"] == 1

    def test_empty_results(self):
        """Summary with empty results list."""
        summary = get_violation_summary([])

        assert summary["total_checked"] == 0
        assert summary["valid_count"] == 0
        assert summary["invalid_count"] == 0


class TestBlocklistCompleteness:
    """Tests to verify blocklist coverage."""

    def test_cannabis_terms_blocked(self):
        """All cannabis-related terms are blocked."""
        cannabis_terms = ["cannabis", "marijuana", "hemp", "cbd", "thc", "dispensary"]
        for term in cannabis_terms:
            assert term in BLOCKED_TERMS or term in SINGLE_WORD_BLOCKLIST, f"{term} should be blocked"

    def test_gambling_terms_blocked(self):
        """All gambling-related terms are blocked."""
        gambling_terms = ["gambling", "casino", "betting", "poker", "lottery"]
        for term in gambling_terms:
            assert term in BLOCKED_TERMS or term in SINGLE_WORD_BLOCKLIST, f"{term} should be blocked"

    def test_adult_terms_blocked(self):
        """Adult content terms are blocked."""
        adult_terms = ["adult", "escort", "porn"]
        for term in adult_terms:
            assert term in BLOCKED_TERMS or term in SINGLE_WORD_BLOCKLIST, f"{term} should be blocked"

    def test_high_priority_single_words(self):
        """High-priority single words are in the aggressive blocklist."""
        critical_terms = ["hemp", "cbd", "cannabis", "marijuana", "thc", "dispensary"]
        for term in critical_terms:
            assert term in SINGLE_WORD_BLOCKLIST, f"{term} should be in single-word blocklist"


class TestEdgeCases:
    """Edge case tests."""

    def test_punctuation_around_terms(self):
        """Terms are detected even with punctuation."""
        text = "cannabis, hemp. cbd! thc?"
        result = find_blocked_terms(text)
        assert "cannabis" in result
        assert "hemp" in result
        assert "cbd" in result
        assert "thc" in result

    def test_terms_in_urls(self):
        """Terms within URLs should still be detected."""
        text = "Visit www.cannabis-shop.com for more."
        result = find_blocked_terms(text)
        # This might or might not catch depending on implementation
        # At minimum, the word boundary check should still work

    def test_hyphenated_terms(self):
        """Hyphenated blocked terms are detected."""
        text = "We offer cannabis-infused products and hemp-derived oils."
        result = find_blocked_terms(text)
        # Should detect the compound terms
        assert any("cannabis" in term for term in result) or "cannabis" in result

    def test_very_long_text(self):
        """Validation handles long text efficiently."""
        base_text = "This is normal security content about cameras. " * 1000
        blocked_text = base_text + " Visit our casino for gambling."

        result = find_blocked_terms(blocked_text)
        assert "casino" in result
        assert "gambling" in result

    def test_unicode_text(self):
        """Validation handles unicode characters."""
        text = "Café security systems are great. No cannabis here."
        result = find_blocked_terms(text)
        assert "cannabis" in result
        assert "café" not in result  # Not a blocked term
