"""Tests for keyword loading functionality."""

import pytest
from pathlib import Path

from seo_content_optimizer.keyword_loader import (
    KeywordLoadError,
    deduplicate_keywords,
    filter_keywords_by_intent,
    load_keywords,
    load_keywords_from_csv,
    load_keywords_from_excel,
    sort_keywords_by_priority,
)
from seo_content_optimizer.models import Keyword


class TestLoadKeywordsFromCSV:
    """Tests for CSV keyword loading."""

    def test_load_valid_csv(self, sample_keywords_csv: Path):
        """Test loading a valid CSV file."""
        keywords = load_keywords_from_csv(sample_keywords_csv)

        assert len(keywords) > 0
        assert all(isinstance(kw, Keyword) for kw in keywords)
        assert keywords[0].phrase == "PTO insurance"
        assert keywords[0].search_volume == 1200
        assert keywords[0].difficulty == 45.0

    def test_load_csv_with_intent(self, sample_keywords_csv: Path):
        """Test that intent is correctly parsed."""
        keywords = load_keywords_from_csv(sample_keywords_csv)

        # Find keyword with informational intent
        info_kw = next((kw for kw in keywords if kw.intent == "informational"), None)
        assert info_kw is not None

        # Find keyword with transactional intent
        trans_kw = next((kw for kw in keywords if kw.intent == "transactional"), None)
        assert trans_kw is not None

    def test_load_nonexistent_csv(self, tmp_path: Path):
        """Test loading a non-existent file raises error."""
        with pytest.raises(KeywordLoadError, match="File not found"):
            load_keywords_from_csv(tmp_path / "nonexistent.csv")

    def test_load_csv_without_keyword_column(self, tmp_path: Path):
        """Test loading CSV without required keyword column."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("volume,difficulty\n100,50\n200,60")

        with pytest.raises(KeywordLoadError, match="No keyword column found"):
            load_keywords_from_csv(csv_path)

    def test_load_empty_csv(self, tmp_path: Path):
        """Test loading empty CSV raises error."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("keyword\n")

        with pytest.raises(KeywordLoadError, match="Keyword file is empty"):
            load_keywords_from_csv(csv_path)


class TestLoadKeywordsFromExcel:
    """Tests for Excel keyword loading."""

    def test_load_valid_excel(self, sample_keywords_excel: Path):
        """Test loading a valid Excel file."""
        keywords = load_keywords_from_excel(sample_keywords_excel)

        assert len(keywords) == 3
        assert keywords[0].phrase == "PTO insurance"
        assert keywords[0].search_volume == 1000

    def test_load_nonexistent_excel(self, tmp_path: Path):
        """Test loading non-existent Excel file."""
        with pytest.raises(KeywordLoadError, match="File not found"):
            load_keywords_from_excel(tmp_path / "nonexistent.xlsx")


class TestLoadKeywords:
    """Tests for the generic load_keywords function."""

    def test_auto_detect_csv(self, sample_keywords_csv: Path):
        """Test auto-detection of CSV format."""
        keywords = load_keywords(sample_keywords_csv)
        assert len(keywords) > 0

    def test_auto_detect_excel(self, sample_keywords_excel: Path):
        """Test auto-detection of Excel format."""
        keywords = load_keywords(sample_keywords_excel)
        assert len(keywords) > 0

    def test_unsupported_format(self, tmp_path: Path):
        """Test loading unsupported format raises error."""
        txt_path = tmp_path / "keywords.txt"
        txt_path.write_text("keyword1\nkeyword2")

        with pytest.raises(KeywordLoadError, match="Unsupported file format"):
            load_keywords(txt_path)


class TestKeywordUtilities:
    """Tests for keyword utility functions."""

    def test_deduplicate_keywords(self):
        """Test keyword deduplication."""
        keywords = [
            Keyword(phrase="PTO insurance"),
            Keyword(phrase="pto insurance"),  # Duplicate (case-insensitive)
            Keyword(phrase="liability coverage"),
        ]

        unique = deduplicate_keywords(keywords)

        assert len(unique) == 2
        assert unique[0].phrase == "PTO insurance"

    def test_filter_keywords_by_intent(self):
        """Test filtering by intent."""
        keywords = [
            Keyword(phrase="kw1", intent="informational"),
            Keyword(phrase="kw2", intent="transactional"),
            Keyword(phrase="kw3", intent="informational"),
            Keyword(phrase="kw4", intent=None),
        ]

        info_only = filter_keywords_by_intent(keywords, "informational", include_none=False)
        assert len(info_only) == 2

        info_with_none = filter_keywords_by_intent(keywords, "informational", include_none=True)
        assert len(info_with_none) == 3

    def test_sort_keywords_by_priority(self):
        """Test sorting by volume and difficulty."""
        keywords = [
            Keyword(phrase="low", search_volume=100, difficulty=80),
            Keyword(phrase="high", search_volume=1000, difficulty=30),
            Keyword(phrase="medium", search_volume=500, difficulty=50),
        ]

        sorted_kw = sort_keywords_by_priority(keywords)

        # Highest volume should be first
        assert sorted_kw[0].phrase == "high"
        assert sorted_kw[1].phrase == "medium"
        assert sorted_kw[2].phrase == "low"


class TestKeywordModel:
    """Tests for the Keyword model."""

    def test_keyword_normalization(self):
        """Test that keyword phrases are normalized."""
        kw = Keyword(phrase="  PTO insurance  ")
        assert kw.phrase == "PTO insurance"

    def test_is_question(self):
        """Test question detection."""
        q1 = Keyword(phrase="how much does PTO insurance cost?")
        assert q1.is_question

        q2 = Keyword(phrase="what is professional liability")
        assert q2.is_question

        q3 = Keyword(phrase="PTO insurance coverage")
        assert not q3.is_question
