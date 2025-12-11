"""Tests for content analysis functionality."""

import pytest

from seo_content_optimizer.analysis import (
    ContentAnalysis,
    analyze_content,
    calculate_keyword_density,
    extract_key_terms,
    find_missing_keywords,
    find_underused_keywords,
    get_keyword_usage_stats,
)
from seo_content_optimizer.models import (
    ContentIntent,
    DocxContent,
    HeadingLevel,
    Keyword,
    PageMeta,
    ParagraphBlock,
)


class TestAnalyzeContent:
    """Tests for the analyze_content function."""

    def test_analyze_page_meta(self):
        """Test analyzing PageMeta content."""
        content = PageMeta(
            title="Professional Liability Insurance Guide",
            meta_description="Learn about professional liability insurance.",
            h1="Understanding Professional Liability Insurance",
            content_blocks=[
                "Professional liability insurance protects your business.",
                "This coverage is essential for consultants and contractors.",
            ],
        )

        analysis = analyze_content(content)

        assert isinstance(analysis, ContentAnalysis)
        assert analysis.word_count > 0
        assert "liability" in analysis.topic.lower() or "insurance" in analysis.topic.lower()

    def test_analyze_docx_content(self):
        """Test analyzing DocxContent."""
        content = DocxContent(
            paragraphs=[
                ParagraphBlock(text="How to Choose Insurance", heading_level=HeadingLevel.H1),
                ParagraphBlock(text="This guide explains insurance options.", heading_level=HeadingLevel.BODY),
                ParagraphBlock(text="Types of Coverage", heading_level=HeadingLevel.H2),
            ]
        )

        analysis = analyze_content(content)

        assert analysis.word_count > 0
        assert analysis.heading_count >= 1
        assert analysis.paragraph_count == 3

    def test_analyze_detects_informational_intent(self):
        """Test detection of informational content intent."""
        content = PageMeta(
            title="How to Choose Professional Liability Insurance",
            h1="A Complete Guide to Professional Liability Coverage",
            content_blocks=[
                "This guide will help you understand the basics of insurance.",
                "Learn what coverage options are available.",
                "Follow these steps to find the right policy.",
            ],
        )

        analysis = analyze_content(content)

        assert analysis.intent == ContentIntent.INFORMATIONAL

    def test_analyze_detects_transactional_intent(self):
        """Test detection of transactional content intent."""
        content = PageMeta(
            title="Buy Professional Liability Insurance Now",
            h1="Get Your Quote Today",
            content_blocks=[
                "Purchase our premium insurance products.",
                "Contact us for a free quote.",
                "Sign up now and save 20%.",
            ],
        )

        analysis = analyze_content(content)

        assert analysis.intent == ContentIntent.TRANSACTIONAL

    def test_analyze_with_keywords(self):
        """Test analysis with keyword tracking."""
        content = PageMeta(
            title="PTO Insurance Coverage",
            h1="Professional Liability Insurance",
            content_blocks=[
                "PTO insurance provides essential coverage for your business.",
                "Liability protection is crucial for consultants.",
            ],
        )

        keywords = [
            Keyword(phrase="PTO insurance"),
            Keyword(phrase="liability protection"),
            Keyword(phrase="missing keyword"),
        ]

        analysis = analyze_content(content, keywords)

        assert "PTO insurance" in analysis.existing_keywords
        assert analysis.existing_keywords["PTO insurance"]["count_in_body"] >= 1


class TestGetKeywordUsageStats:
    """Tests for keyword usage statistics."""

    def test_count_in_body(self):
        """Test counting keyword occurrences in body."""
        stats = get_keyword_usage_stats(
            phrase="insurance",
            full_text="Insurance is important. Get insurance today. Insurance coverage.",
        )

        assert stats.count_in_body == 3

    def test_in_title(self):
        """Test detection in title."""
        stats = get_keyword_usage_stats(
            phrase="insurance",
            full_text="Get coverage today.",
            title="Professional Insurance Guide",
        )

        assert stats.in_title is True
        assert stats.in_meta_description is False

    def test_in_meta_description(self):
        """Test detection in meta description."""
        stats = get_keyword_usage_stats(
            phrase="coverage",
            full_text="Some content here.",
            meta_description="Learn about insurance coverage options.",
        )

        assert stats.in_meta_description is True

    def test_in_h1(self):
        """Test detection in H1."""
        stats = get_keyword_usage_stats(
            phrase="liability insurance",
            full_text="Body content here.",
            h1="Professional Liability Insurance Guide",
        )

        assert stats.in_h1 is True

    def test_in_first_100_words(self):
        """Test detection in first 100 words."""
        # Create text with keyword in first 100 words
        first_part = "Insurance coverage is important. " * 5
        second_part = "Other content. " * 50

        stats = get_keyword_usage_stats(
            phrase="insurance coverage",
            full_text=first_part + second_part,
        )

        assert stats.in_first_100_words is True

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        stats = get_keyword_usage_stats(
            phrase="PTO Insurance",
            full_text="We offer pto insurance for your needs.",
        )

        assert stats.count_in_body == 1


class TestExtractKeyTerms:
    """Tests for key term extraction."""

    def test_extract_removes_stopwords(self):
        """Test that stopwords are removed."""
        text = "The insurance is very important and the coverage is good."
        terms = extract_key_terms(text)

        # Common words like 'the', 'is', 'and' should not appear
        term_words = [t[0] for t in terms]
        assert "the" not in term_words
        assert "and" not in term_words

    def test_extract_returns_top_n(self):
        """Test that only top N terms are returned."""
        text = "Insurance coverage protection policy premium claims benefits " * 10
        terms = extract_key_terms(text, top_n=5)

        assert len(terms) <= 5

    def test_extract_counts_frequency(self):
        """Test that frequencies are counted correctly."""
        text = "insurance insurance insurance coverage coverage policy"
        terms = extract_key_terms(text, top_n=10)

        # Insurance should have highest count
        term_dict = dict(terms)
        assert term_dict.get("insurance", 0) == 3
        assert term_dict.get("coverage", 0) == 2


class TestCalculateKeywordDensity:
    """Tests for keyword density calculation."""

    def test_density_calculation(self):
        """Test basic density calculation."""
        # 10 total words, 2 occurrences of 1-word phrase = 20%
        text = "insurance coverage insurance protection liability claims benefits policy premium quote"
        density = calculate_keyword_density("insurance", text)

        assert density == 20.0

    def test_density_multi_word_phrase(self):
        """Test density with multi-word phrase."""
        text = "PTO insurance is great. Get PTO insurance today."
        density = calculate_keyword_density("PTO insurance", text)

        # 2 occurrences * 2 words / 10 total words = 40%
        assert density > 0

    def test_density_empty_text(self):
        """Test density with empty text."""
        density = calculate_keyword_density("keyword", "")
        assert density == 0.0


class TestFindMissingKeywords:
    """Tests for finding missing keywords."""

    def test_find_missing(self):
        """Test finding keywords not in content."""
        content = PageMeta(
            title="Insurance Guide",
            content_blocks=["This content discusses insurance coverage."],
        )

        keywords = [
            Keyword(phrase="insurance"),  # Present
            Keyword(phrase="liability"),  # Missing
            Keyword(phrase="coverage"),  # Present
        ]

        missing = find_missing_keywords(content, keywords)

        assert len(missing) == 1
        assert missing[0].phrase == "liability"

    def test_all_present(self):
        """Test when all keywords are present."""
        content = PageMeta(
            title="Insurance and Liability",
            content_blocks=["Coverage options available."],
        )

        keywords = [
            Keyword(phrase="insurance"),
            Keyword(phrase="liability"),
            Keyword(phrase="coverage"),
        ]

        missing = find_missing_keywords(content, keywords)

        assert len(missing) == 0


class TestFindUnderusedKeywords:
    """Tests for finding underused keywords."""

    def test_find_underused(self):
        """Test finding keywords used fewer than threshold times."""
        content = PageMeta(
            title="Insurance",
            content_blocks=[
                "Insurance coverage insurance policy insurance. Coverage once."
            ],
        )

        keywords = [
            Keyword(phrase="insurance"),  # Used 4 times
            Keyword(phrase="coverage"),  # Used 2 times
        ]

        underused = find_underused_keywords(content, keywords, min_occurrences=3)

        assert len(underused) == 1
        assert underused[0].phrase == "coverage"
