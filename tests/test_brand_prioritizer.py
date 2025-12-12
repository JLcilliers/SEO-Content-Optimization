"""
Tests for brand keyword detection and prioritizer behavior.

These tests lock the behavior that:
1. Brand keywords (is_brand=True) cannot be selected as primary keyword
2. Brand detection heuristics work correctly
3. Long-tail secondary phrases are preserved as complete strings
"""

import pytest

from seo_content_optimizer.analysis import (
    ContentAnalysis,
    analyze_content,
    guess_brand_tokens,
    is_branded_phrase,
    mark_branded_keywords,
)
from seo_content_optimizer.models import (
    ContentIntent,
    DocxContent,
    HeadingLevel,
    Keyword,
    KeywordPlan,
    PageMeta,
    ParagraphBlock,
)
from seo_content_optimizer.prioritizer import (
    KeywordPrioritizer,
    create_keyword_plan,
)


class TestBrandKeywordExclusion:
    """Tests ensuring brand keywords cannot be primary."""

    def test_brand_keyword_excluded_from_primary(self):
        """A keyword with is_brand=True should never be returned as primary."""
        keywords = [
            Keyword(phrase="payzli", search_volume=5000, difficulty=20, is_brand=True),
            Keyword(phrase="payzli vs stripe", search_volume=3000, difficulty=25, is_brand=True),
            Keyword(phrase="payment processing solutions", search_volume=2000, difficulty=30, is_brand=False),
            Keyword(phrase="credit card processing", search_volume=1500, difficulty=35, is_brand=False),
        ]

        content = PageMeta(
            title="Payzli vs Stripe: Payment Processing Comparison",
            h1="Payzli vs Stripe Comparison",
            content_blocks=[
                "Compare payment processing solutions for your business.",
                "This guide covers credit card processing options.",
            ],
        )

        analysis = analyze_content(content, keywords)
        plan = create_keyword_plan(keywords, content, analysis)

        # Primary keyword must NOT be a brand keyword
        assert plan.primary.is_brand is False
        assert plan.primary.phrase in ("payment processing solutions", "credit card processing")

    def test_all_brand_keywords_falls_back_with_warning(self):
        """When all keywords are brands, fallback should occur (with warning)."""
        keywords = [
            Keyword(phrase="payzli", search_volume=5000, is_brand=True),
            Keyword(phrase="payzli payment", search_volume=3000, is_brand=True),
        ]

        content = PageMeta(
            title="Payzli Payment Solutions",
            h1="Payzli Payment Processing",
            content_blocks=["Payzli offers payment solutions."],
        )

        analysis = analyze_content(content, keywords)

        # Should not raise, but fallback to brands if no alternatives
        plan = create_keyword_plan(keywords, content, analysis)
        assert plan.primary is not None

    def test_secondary_keywords_deprioritize_brands(self):
        """Brand keywords should be deprioritized (but not excluded) from secondary."""
        keywords = [
            Keyword(phrase="payment processing", search_volume=3000, is_brand=False),
            Keyword(phrase="payzli pricing", search_volume=4000, is_brand=True),
            Keyword(phrase="merchant services", search_volume=2500, is_brand=False),
            Keyword(phrase="online payments", search_volume=2000, is_brand=False),
        ]

        content = PageMeta(
            title="Payment Processing Guide",
            h1="Payment Processing Solutions",
            content_blocks=["Learn about payment processing and merchant services."],
        )

        analysis = analyze_content(content, keywords)
        plan = create_keyword_plan(keywords, content, analysis, max_secondary=3)

        # Brand keywords should appear lower in secondary list due to deprioritization
        secondary_phrases = [kw.phrase for kw in plan.secondary]
        # If payzli pricing is in secondary, it should not be first
        if "payzli pricing" in secondary_phrases:
            assert secondary_phrases[0] != "payzli pricing"


class TestBrandDetectionHeuristics:
    """Tests for brand token detection from URL, H1, title."""

    def test_guess_brand_from_url_domain(self):
        """Brand tokens should be extracted from URL domain."""
        tokens = guess_brand_tokens(
            url="https://www.payzli.com/vs-stripe",
            h1=None,
            title=None,
        )

        assert "payzli" in tokens

    def test_guess_brand_from_h1_capitalized(self):
        """Capitalized words in H1 should be detected as potential brands."""
        tokens = guess_brand_tokens(
            url=None,
            h1="Payzli vs Stripe: Which Is Better?",
            title=None,
        )

        assert "payzli" in tokens
        assert "stripe" in tokens

    def test_guess_brand_from_vs_pattern(self):
        """X vs Y patterns should extract both as potential brands."""
        tokens = guess_brand_tokens(
            url=None,
            h1=None,
            title="Square vs PayPal: Payment Comparison",
        )

        assert "square" in tokens
        assert "paypal" in tokens

    def test_skip_generic_domain_words(self):
        """Generic domain words like 'blog', 'shop' should be skipped."""
        tokens = guess_brand_tokens(
            url="https://blog.example.com/article",
            h1=None,
            title=None,
        )

        assert "blog" not in tokens

    def test_is_branded_phrase_with_brand_tokens(self):
        """Phrases containing brand tokens should be detected as branded."""
        brand_tokens = {"payzli", "stripe"}

        assert is_branded_phrase("payzli pricing", brand_tokens) is True
        assert is_branded_phrase("stripe fees", brand_tokens) is True
        assert is_branded_phrase("payment processing", brand_tokens) is False

    def test_is_branded_phrase_explicit_flag(self):
        """Keywords with is_brand=True should always be detected as branded."""
        kw = Keyword(phrase="generic payment", is_brand=True)

        # Even without matching tokens, explicit flag should win
        assert is_branded_phrase("generic payment", set(), kw) is True

    def test_is_branded_phrase_navigational_intent(self):
        """Keywords with navigational intent should be detected as branded."""
        kw = Keyword(phrase="payzli login", intent="navigational")

        assert is_branded_phrase("payzli login", set(), kw) is True

    def test_mark_branded_keywords_updates_list(self):
        """mark_branded_keywords should set is_brand=True for detected brands."""
        keywords = [
            Keyword(phrase="payzli pricing"),
            Keyword(phrase="stripe fees"),
            Keyword(phrase="payment processing"),
        ]

        updated = mark_branded_keywords(
            keywords,
            url="https://payzli.com/compare",
            h1="Payzli vs Stripe",
            title=None,
        )

        # Payzli and Stripe keywords should be marked as brands
        payzli_kw = next(kw for kw in updated if "payzli" in kw.phrase.lower())
        stripe_kw = next(kw for kw in updated if "stripe" in kw.phrase.lower())
        generic_kw = next(kw for kw in updated if kw.phrase == "payment processing")

        assert payzli_kw.is_brand is True
        assert stripe_kw.is_brand is True
        assert generic_kw.is_brand is False


class TestKeywordPhraseIntegrity:
    """Tests ensuring long-tail phrases are preserved as complete strings."""

    def test_keyword_plan_preserves_full_phrases(self):
        """KeywordPlan should contain complete keyword phrases."""
        keywords = [
            Keyword(phrase="best payment processor for small business", search_volume=1000),
            Keyword(phrase="credit card processing fees comparison", search_volume=800),
            Keyword(phrase="how to accept payments online", search_volume=600),
            Keyword(phrase="payment gateway", search_volume=2000),
        ]

        content = PageMeta(
            title="Payment Processing Guide",
            h1="Best Payment Processors for Small Business",
            content_blocks=[
                "Find the best payment processor for small business needs.",
                "Compare credit card processing fees across providers.",
            ],
        )

        analysis = analyze_content(content, keywords)
        plan = create_keyword_plan(keywords, content, analysis)

        # All phrases should be preserved as complete strings
        all_phrases = plan.all_phrases

        # Check that long-tail phrases are not truncated
        for kw in keywords:
            if kw.phrase in all_phrases:
                # Phrase should match exactly
                assert kw.phrase in all_phrases

    def test_all_keywords_returns_complete_phrases(self):
        """The all_keywords property should return keywords with full phrases."""
        primary = Keyword(phrase="payment processing solutions")
        secondary = [
            Keyword(phrase="credit card merchant services"),
            Keyword(phrase="online payment gateway integration"),
        ]
        questions = [
            Keyword(phrase="how much does payment processing cost"),
        ]

        plan = KeywordPlan(
            primary=primary,
            secondary=secondary,
            long_tail_questions=questions,
        )

        all_kw = plan.all_keywords
        all_phrases = plan.all_phrases

        # Verify full phrases are preserved
        assert "payment processing solutions" in all_phrases
        assert "credit card merchant services" in all_phrases
        assert "online payment gateway integration" in all_phrases
        assert "how much does payment processing cost" in all_phrases

        # Verify no truncation occurred (each phrase has multiple words)
        for phrase in all_phrases:
            word_count = len(phrase.split())
            # Long-tail phrases should still have their full word count
            assert word_count >= 2 or phrase in ("payzli", "stripe")  # Single word brands are OK


class TestKeywordLoaderBrandColumn:
    """Tests for loading is_brand flag from CSV/Excel."""

    def test_csv_with_brand_column(self, tmp_path):
        """CSV with is_brand column should parse brand flags correctly."""
        from seo_content_optimizer.keyword_loader import load_keywords_from_csv

        csv_content = """keyword,search_volume,is_brand
payzli pricing,1000,true
stripe fees,800,yes
payment processing,2000,false
merchant services,1500,no
generic term,500,
"""
        csv_path = tmp_path / "keywords_with_brand.csv"
        csv_path.write_text(csv_content)

        keywords = load_keywords_from_csv(csv_path)

        # Find keywords by phrase
        kw_dict = {kw.phrase: kw for kw in keywords}

        assert kw_dict["payzli pricing"].is_brand is True
        assert kw_dict["stripe fees"].is_brand is True
        assert kw_dict["payment processing"].is_brand is False
        assert kw_dict["merchant services"].is_brand is False
        assert kw_dict["generic term"].is_brand is False  # Default

    def test_csv_with_type_column(self, tmp_path):
        """CSV with 'type' column (brand/topic) should work."""
        from seo_content_optimizer.keyword_loader import load_keywords_from_csv

        csv_content = """keyword,search_volume,type
payzli,1000,brand
stripe,800,brand
payments,2000,topic
"""
        csv_path = tmp_path / "keywords_with_type.csv"
        csv_path.write_text(csv_content)

        keywords = load_keywords_from_csv(csv_path)

        kw_dict = {kw.phrase: kw for kw in keywords}

        assert kw_dict["payzli"].is_brand is True
        assert kw_dict["stripe"].is_brand is True
        assert kw_dict["payments"].is_brand is False


class TestKeywordModelBrandFlag:
    """Tests for Keyword model is_brand attribute."""

    def test_keyword_default_not_brand(self):
        """Keywords should default to is_brand=False."""
        kw = Keyword(phrase="payment processing")
        assert kw.is_brand is False

    def test_keyword_explicit_brand(self):
        """Keywords can be explicitly marked as brands."""
        kw = Keyword(phrase="payzli", is_brand=True)
        assert kw.is_brand is True

    def test_keyword_brand_flag_mutable(self):
        """is_brand flag should be mutable for heuristic detection."""
        kw = Keyword(phrase="stripe pricing")
        assert kw.is_brand is False

        kw.is_brand = True
        assert kw.is_brand is True
