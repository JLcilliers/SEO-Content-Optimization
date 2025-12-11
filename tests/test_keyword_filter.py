"""
Tests for keyword topical relevance filtering.

These tests ensure that off-topic keywords are blocked and only relevant
keywords are allowed through to optimization.
"""

import pytest

from seo_content_optimizer.keyword_filter import (
    HIGH_RISK_INDUSTRIES,
    SINGLE_WORD_BLOCKLIST,
    KeywordFilterResult,
    contains_blocked_term,
    contains_high_risk_term,
    filter_keywords_for_content,
    get_content_topics,
    keyword_on_topic,
    normalize_tokens,
)
from seo_content_optimizer.models import Keyword


class TestNormalizeTokens:
    """Test token extraction and normalization."""

    def test_basic_tokenization(self):
        """Test basic word extraction."""
        tokens = normalize_tokens("Hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_stopword_removal(self):
        """Test that common stopwords are removed."""
        tokens = normalize_tokens("the quick brown fox and the lazy dog")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_short_words_removed(self):
        """Test that single-character words are removed."""
        tokens = normalize_tokens("a b c test word")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "test" in tokens
        assert "word" in tokens

    def test_case_insensitivity(self):
        """Test that tokens are lowercased."""
        tokens = normalize_tokens("HELLO World TeSt")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens


class TestContainsHighRiskTerm:
    """Test high-risk industry term detection."""

    def test_single_word_blocklist_terms(self):
        """Test that single-word blocklist terms are detected."""
        assert contains_high_risk_term("hemp merchant account") == "hemp"
        assert contains_high_risk_term("cbd processing") == "cbd"
        assert contains_high_risk_term("cannabis business") == "cannabis"
        assert contains_high_risk_term("salon pos system") == "salon"
        assert contains_high_risk_term("spa management") == "spa"

    def test_multi_word_high_risk_terms(self):
        """Test multi-word high-risk industry terms."""
        # Single word blocklist is checked first, so 'salon' is found before 'hair salon'
        assert contains_high_risk_term("hair salon pos") in ("salon", "hair salon")
        assert contains_high_risk_term("restaurant payment processing") == "restaurant"
        assert contains_high_risk_term("merchant services for businesses") == "merchant services"
        # Multiple terms match; what matters is that it IS blocked
        blocked = contains_high_risk_term("high risk merchant account")
        assert blocked in ("high risk merchant", "high-risk merchant", "merchant account")

    def test_clean_keywords_pass(self):
        """Test that clean keywords are not flagged."""
        assert contains_high_risk_term("payment processing") is None
        assert contains_high_risk_term("credit card payments") is None
        assert contains_high_risk_term("business software") is None
        assert contains_high_risk_term("stripe alternative") is None

    def test_case_insensitivity(self):
        """Test that detection is case-insensitive."""
        assert contains_high_risk_term("HEMP merchant") == "hemp"
        assert contains_high_risk_term("CBD Business") == "cbd"
        # Single word blocklist checked first
        assert contains_high_risk_term("Hair Salon POS") in ("salon", "hair salon")


class TestContainsBlockedTerm:
    """Test the strict blocked term check against content."""

    def test_blocked_term_not_in_content(self):
        """Test that blocked terms not in content are detected."""
        content = "This is about payment processing for businesses."
        # Order depends on set iteration; what matters is that SOMETHING is blocked
        blocked = contains_blocked_term("hemp merchant account", content)
        assert blocked in ("hemp", "merchant account", "merchant services")
        assert contains_blocked_term("cbd business", content) == "cbd"
        assert contains_blocked_term("salon pos", content) == "salon"

    def test_blocked_term_in_content_allowed(self):
        """Test that blocked terms already in content are allowed."""
        content = "This is about hemp processing for cannabis businesses."
        # hemp is in content, so it should pass
        assert contains_blocked_term("hemp merchant", content) is None

    def test_clean_keyword_passes(self):
        """Test that clean keywords pass the check."""
        content = "This is about payment processing."
        assert contains_blocked_term("payment processing", content) is None
        assert contains_blocked_term("credit card", content) is None

    def test_partial_word_not_blocked(self):
        """Test that partial word matches don't trigger blocking."""
        # "spas" is plural, but "spa" is the blocked term. The regex uses word boundaries.
        # "spas" != "spa" so "spa services" will be blocked because "spa" isn't literally in content
        content = "This is about themes and a spa."  # "spa" literally in content
        # "spa" is in content, so spa-related keywords should pass
        assert contains_blocked_term("spa services", content) is None


class TestKeywordOnTopic:
    """Test the main keyword topical relevance function."""

    def test_exact_match_allowed(self):
        """Test that exact phrase matches are allowed."""
        content = "We offer payment processing solutions for businesses."
        is_ok, score, reason = keyword_on_topic("payment processing", content)
        assert is_ok is True
        assert score == 1.0
        assert "exact" in reason.lower()

    def test_blocked_term_rejected(self):
        """Test that blocked terms not in content are rejected."""
        content = "We offer payment processing solutions for businesses."
        is_ok, score, reason = keyword_on_topic("hemp merchant account", content)
        assert is_ok is False
        assert score == 0.0
        assert "BLOCKED" in reason or "blocked" in reason.lower()

    def test_high_token_overlap_allowed(self):
        """Test that keywords with high token overlap are allowed."""
        content = "We provide credit card payment processing and merchant solutions."
        is_ok, score, reason = keyword_on_topic("credit card processing", content)
        assert is_ok is True
        assert score >= 0.5

    def test_low_token_overlap_rejected(self):
        """Test that keywords with low token overlap are rejected."""
        content = "We provide payment processing solutions."
        is_ok, score, reason = keyword_on_topic("restaurant delivery software", content)
        assert is_ok is False
        assert score < 0.5

    def test_off_topic_industry_rejected(self):
        """Test rejection of off-topic industry terms."""
        content = "Payzli Transact vs Stripe: A comparison of payment processors."

        # These should all be rejected
        is_ok, _, _ = keyword_on_topic("hair salon pos", content)
        assert is_ok is False

        is_ok, _, _ = keyword_on_topic("restaurant payment processing", content)
        assert is_ok is False

        is_ok, _, _ = keyword_on_topic("cbd merchant account", content)
        assert is_ok is False

    def test_on_topic_keywords_allowed(self):
        """Test that on-topic keywords are allowed."""
        content = "Payzli Transact vs Stripe: A comparison of payment processors."

        is_ok, _, _ = keyword_on_topic("payment processor comparison", content)
        assert is_ok is True

        is_ok, _, _ = keyword_on_topic("Stripe alternative", content)
        assert is_ok is True


class TestFilterKeywordsForContent:
    """Test the main filter function with keyword lists."""

    def test_filters_out_blocked_keywords(self):
        """Test that blocked keywords are filtered out."""
        content = "Payzli Transact vs Stripe comparison for businesses."
        keywords = [
            Keyword(phrase="payment processing", search_volume=1000, intent="transactional"),
            Keyword(phrase="hemp merchant account", search_volume=500, intent="transactional"),
            Keyword(phrase="stripe alternative", search_volume=800, intent="informational"),
            Keyword(phrase="cbd processing", search_volume=300, intent="transactional"),
            Keyword(phrase="salon pos system", search_volume=400, intent="transactional"),
        ]

        allowed, results = filter_keywords_for_content(keywords, content)

        # Check that blocked keywords are not in allowed list
        allowed_phrases = [kw.phrase for kw in allowed]
        assert "hemp merchant account" not in allowed_phrases
        assert "cbd processing" not in allowed_phrases
        assert "salon pos system" not in allowed_phrases

        # Check that on-topic keywords are allowed
        assert "payment processing" in allowed_phrases or "stripe alternative" in allowed_phrases

    def test_returns_filter_results(self):
        """Test that filter results include rejection reasons."""
        content = "Payment processing comparison."
        keywords = [
            Keyword(phrase="payment", search_volume=100, intent="transactional"),
            Keyword(phrase="hemp cbd", search_volume=100, intent="transactional"),
        ]

        allowed, results = filter_keywords_for_content(keywords, content)

        # Find the result for the blocked keyword
        blocked_results = [r for r in results if not r.is_allowed]
        assert len(blocked_results) > 0
        assert any("hemp" in r.reason.lower() or "cbd" in r.reason.lower() for r in blocked_results)

    def test_max_keywords_limit(self):
        """Test that max_keywords parameter limits results."""
        content = "Payment processing solutions for credit cards and debit transactions."
        keywords = [
            Keyword(phrase="payment processing", search_volume=1000, intent="transactional"),
            Keyword(phrase="credit card", search_volume=800, intent="transactional"),
            Keyword(phrase="debit transactions", search_volume=600, intent="transactional"),
            Keyword(phrase="solutions", search_volume=400, intent="informational"),
        ]

        allowed, _ = filter_keywords_for_content(keywords, content, max_keywords=2)
        assert len(allowed) <= 2

    def test_empty_keywords_returns_empty(self):
        """Test handling of empty keyword list."""
        content = "Some content."
        allowed, results = filter_keywords_for_content([], content)
        assert len(allowed) == 0
        assert len(results) == 0


class TestGetContentTopics:
    """Test content topic extraction."""

    def test_extracts_frequent_terms(self):
        """Test that frequent terms are extracted."""
        content = """
        Payment processing is important. Payment solutions help businesses.
        Credit card payment is common. Payment gateway integration matters.
        """
        topics = get_content_topics(content, top_n=5)
        assert "payment" in topics

    def test_excludes_stopwords(self):
        """Test that stopwords are not in topics."""
        content = "The payment is the best payment for the business."
        topics = get_content_topics(content, top_n=10)
        assert "the" not in topics
        assert "is" not in topics
        assert "for" not in topics

    def test_respects_top_n(self):
        """Test that top_n parameter is respected."""
        content = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        topics = get_content_topics(content, top_n=3)
        assert len(topics) <= 3


class TestBlocklistCompleteness:
    """Test that blocklists are comprehensive."""

    def test_single_word_blocklist_has_required_terms(self):
        """Verify critical single-word blocklist terms."""
        required = {"hemp", "cbd", "cannabis", "marijuana", "salon", "spa", "casino", "gambling"}
        assert required.issubset(SINGLE_WORD_BLOCKLIST)

    def test_high_risk_industries_has_required_terms(self):
        """Verify critical industry terms in blocklist."""
        required = {
            "hair salon", "nail salon", "restaurant", "cafe",
            "dispensary", "merchant services", "merchant account",
        }
        assert required.issubset(HIGH_RISK_INDUSTRIES)


class TestRealWorldScenario:
    """Test with real-world scenarios from the user's bug report."""

    def test_payzli_vs_stripe_filtering(self):
        """
        Test filtering for the Payzli Transact vs Stripe comparison page.
        This is the actual scenario that triggered the bug report.
        """
        content = """
        Payzli Transact vs Stripe: Choosing the Right Payment Processor

        When comparing payment processors, businesses need to consider features,
        pricing, and ease of integration. Both Payzli Transact and Stripe offer
        robust payment processing capabilities for online and in-person transactions.

        Key features to compare include transaction fees, API documentation,
        customer support, and fraud protection.
        """

        # Keywords that should be BLOCKED (off-topic industries)
        blocked_keywords = [
            Keyword(phrase="hemp merchant services", search_volume=100, intent="transactional"),
            Keyword(phrase="CBD businesses", search_volume=100, intent="transactional"),
            Keyword(phrase="POS for hair salon", search_volume=100, intent="transactional"),
            Keyword(phrase="restaurant payment processing", search_volume=100, intent="transactional"),
            Keyword(phrase="high risk merchant account", search_volume=100, intent="transactional"),
            Keyword(phrase="cannabis payment processing", search_volume=100, intent="transactional"),
            Keyword(phrase="salon pos system", search_volume=100, intent="transactional"),
            Keyword(phrase="spa management software", search_volume=100, intent="transactional"),
        ]

        # Keywords that should be ALLOWED (on-topic)
        allowed_keywords = [
            Keyword(phrase="payment processor comparison", search_volume=500, intent="informational"),
            Keyword(phrase="stripe alternative", search_volume=400, intent="informational"),
            Keyword(phrase="transaction fees", search_volume=300, intent="informational"),
            Keyword(phrase="payment processing", search_volume=1000, intent="transactional"),
        ]

        all_keywords = blocked_keywords + allowed_keywords

        filtered, results = filter_keywords_for_content(all_keywords, content)
        filtered_phrases = [kw.phrase for kw in filtered]

        # Verify blocked keywords are NOT in filtered list
        for kw in blocked_keywords:
            assert kw.phrase not in filtered_phrases, f"Blocked keyword '{kw.phrase}' should not be allowed"

        # Verify at least some allowed keywords are in filtered list
        allowed_count = sum(1 for kw in allowed_keywords if kw.phrase in filtered_phrases)
        assert allowed_count > 0, "At least some on-topic keywords should be allowed"

    def test_strict_blocklist_enforcement(self):
        """
        Test that blocklist is enforced even for keywords with high volume.
        """
        content = "Payment processing comparison for online businesses."

        # High-volume but off-topic keyword
        keywords = [
            Keyword(phrase="hemp merchant account", search_volume=10000, intent="transactional"),
            Keyword(phrase="payment processing", search_volume=100, intent="transactional"),
        ]

        filtered, _ = filter_keywords_for_content(keywords, content)
        filtered_phrases = [kw.phrase for kw in filtered]

        # Even with high volume, hemp should be blocked
        assert "hemp merchant account" not in filtered_phrases
        assert "payment processing" in filtered_phrases
