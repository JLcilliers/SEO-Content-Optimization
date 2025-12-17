# -*- coding: utf-8 -*-
"""
Tests for bulk optimization endpoint.

Tests the Excel parsing, bulk processing logic, and API endpoint
for optimizing multiple URLs at once.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import zipfile
import io
import sys

# Add api directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from index import (
    parse_bulk_optimization_excel,
    BulkExcelParseError,
    BulkOptimizationRow,
    BulkOptimizationStatus,
    BulkItemResult,
    _find_column_variant,
    _ensure_unique_filename,
    _create_manifest,
)


class TestColumnVariantFinding:
    """Tests for _find_column_variant helper."""

    def test_find_exact_match(self):
        """Test finding exact column name match."""
        df = pd.DataFrame({"url": ["test"], "keyword": ["test"]})
        result = _find_column_variant(df, ["url", "source_url"])
        assert result == "url"

    def test_find_case_insensitive_match(self):
        """Test case-insensitive column matching."""
        df = pd.DataFrame({"URL": ["test"], "KEYWORD": ["test"]})
        result = _find_column_variant(df, ["url", "source_url"])
        assert result == "URL"

    def test_find_variant_match(self):
        """Test matching column variants."""
        df = pd.DataFrame({"source_url": ["test"], "main_keyword": ["test"]})
        result = _find_column_variant(df, ["url", "source_url", "page_url"])
        assert result == "source_url"

    def test_find_with_spaces_and_underscores(self):
        """Test matching with spaces converted to underscores."""
        df = pd.DataFrame({"Primary Keyword": ["test"]})
        result = _find_column_variant(df, ["primary_keyword", "primary"])
        assert result == "Primary Keyword"

    def test_find_no_match_returns_none(self):
        """Test that no match returns None."""
        df = pd.DataFrame({"unrelated": ["test"]})
        result = _find_column_variant(df, ["url", "source_url"])
        assert result is None


class TestBulkExcelParsing:
    """Tests for Excel parsing logic."""

    def test_parse_valid_excel(self, tmp_path: Path):
        """Test parsing a valid bulk optimization Excel file."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "URL": ["https://example.com/page1", "https://example.com/page2"],
            "Primary Keyword": ["keyword one", "keyword two"],
            "Secondary Keywords": ["sec1, sec2", "sec3"],
        })
        df.to_excel(excel_path, index=False)

        rows = parse_bulk_optimization_excel(excel_path)

        assert len(rows) == 2
        assert rows[0].url == "https://example.com/page1"
        assert rows[0].primary_keyword == "keyword one"
        assert rows[0].secondary_keywords == ["sec1", "sec2"]
        assert rows[1].secondary_keywords == ["sec3"]

    def test_parse_excel_case_insensitive_columns(self, tmp_path: Path):
        """Test that column names are matched case-insensitively."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "url": ["https://example.com/page1"],
            "PRIMARY_KEYWORD": ["keyword"],
            "secondary keywords": ["sec1"],
        })
        df.to_excel(excel_path, index=False)

        rows = parse_bulk_optimization_excel(excel_path)

        assert len(rows) == 1
        assert rows[0].primary_keyword == "keyword"

    def test_parse_excel_missing_url_column(self, tmp_path: Path):
        """Test error when URL column is missing."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "Primary Keyword": ["keyword"],
        })
        df.to_excel(excel_path, index=False)

        with pytest.raises(BulkExcelParseError, match="Missing required column: URL"):
            parse_bulk_optimization_excel(excel_path)

    def test_parse_excel_missing_primary_keyword_column(self, tmp_path: Path):
        """Test error when Primary Keyword column is missing."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "URL": ["https://example.com/page1"],
        })
        df.to_excel(excel_path, index=False)

        with pytest.raises(BulkExcelParseError, match="Missing required column: Primary Keyword"):
            parse_bulk_optimization_excel(excel_path)

    def test_parse_excel_exceeds_batch_limit(self, tmp_path: Path):
        """Test error when Excel has more than 10 rows."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "URL": [f"https://example.com/page{i}" for i in range(11)],
            "Primary Keyword": [f"keyword{i}" for i in range(11)],
        })
        df.to_excel(excel_path, index=False)

        with pytest.raises(BulkExcelParseError, match="Batch limit exceeded: 11 URLs"):
            parse_bulk_optimization_excel(excel_path)

    def test_parse_excel_invalid_url_format(self, tmp_path: Path):
        """Test error when URL doesn't start with http/https."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "URL": ["example.com/page1"],
            "Primary Keyword": ["keyword"],
        })
        df.to_excel(excel_path, index=False)

        with pytest.raises(BulkExcelParseError, match="Invalid URL format"):
            parse_bulk_optimization_excel(excel_path)

    def test_parse_excel_empty_primary_keyword(self, tmp_path: Path):
        """Test error when primary keyword is empty."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "URL": ["https://example.com/page1"],
            "Primary Keyword": [""],
        })
        df.to_excel(excel_path, index=False)

        with pytest.raises(BulkExcelParseError, match="Primary keyword is required"):
            parse_bulk_optimization_excel(excel_path)

    def test_parse_excel_empty_file(self, tmp_path: Path):
        """Test error when Excel file is empty."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "URL": pd.Series([], dtype=str),
            "Primary Keyword": pd.Series([], dtype=str),
        })
        df.to_excel(excel_path, index=False)

        with pytest.raises(BulkExcelParseError, match="contains no data rows"):
            parse_bulk_optimization_excel(excel_path)

    def test_parse_excel_max_rows_allowed(self, tmp_path: Path):
        """Test that exactly 10 rows is allowed."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "URL": [f"https://example.com/page{i}" for i in range(10)],
            "Primary Keyword": [f"keyword{i}" for i in range(10)],
        })
        df.to_excel(excel_path, index=False)

        rows = parse_bulk_optimization_excel(excel_path)
        assert len(rows) == 10

    def test_parse_excel_optional_secondary_keywords(self, tmp_path: Path):
        """Test parsing without secondary keywords column."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "URL": ["https://example.com/page1"],
            "Primary Keyword": ["keyword"],
        })
        df.to_excel(excel_path, index=False)

        rows = parse_bulk_optimization_excel(excel_path)
        assert rows[0].secondary_keywords == []

    def test_parse_excel_secondary_keywords_limit(self, tmp_path: Path):
        """Test that secondary keywords are limited to 5."""
        excel_path = tmp_path / "bulk.xlsx"
        df = pd.DataFrame({
            "URL": ["https://example.com/page1"],
            "Primary Keyword": ["keyword"],
            "Secondary Keywords": ["kw1, kw2, kw3, kw4, kw5, kw6, kw7"],
        })
        df.to_excel(excel_path, index=False)

        rows = parse_bulk_optimization_excel(excel_path)
        assert len(rows[0].secondary_keywords) == 5


class TestUniqueFilename:
    """Tests for _ensure_unique_filename helper."""

    def test_unique_filename_no_collision(self):
        """Test filename is returned unchanged when no collision."""
        result = _ensure_unique_filename("test.docx", [])
        assert result == "test.docx"

    def test_unique_filename_with_collision(self):
        """Test filename gets suffix on collision."""
        result = _ensure_unique_filename("test.docx", ["test.docx"])
        assert result == "test-1.docx"

    def test_unique_filename_multiple_collisions(self):
        """Test filename handles multiple collisions."""
        result = _ensure_unique_filename(
            "test.docx",
            ["test.docx", "test-1.docx", "test-2.docx"]
        )
        assert result == "test-3.docx"


class TestManifestCreation:
    """Tests for _create_manifest helper."""

    def test_manifest_success_items(self):
        """Test manifest with successful items."""
        results = [
            BulkItemResult(
                row_number=2,
                url="https://example.com/page1",
                primary_keyword="keyword1",
                status=BulkOptimizationStatus.SUCCESS,
                filename="page1-optimized-content.docx",
                secondary_keywords_used=["sec1", "sec2"],
            )
        ]

        manifest = _create_manifest(results)

        assert "Bulk Optimization Results" in manifest
        assert "Total URLs processed: 1" in manifest
        assert "Successful: 1" in manifest
        assert "Failed: 0" in manifest
        assert "[OK]" in manifest
        assert "https://example.com/page1" in manifest
        assert "page1-optimized-content.docx" in manifest

    def test_manifest_failed_items(self):
        """Test manifest with failed items."""
        results = [
            BulkItemResult(
                row_number=2,
                url="https://example.com/page1",
                primary_keyword="keyword1",
                status=BulkOptimizationStatus.FAILED,
                error_message="Network error",
            )
        ]

        manifest = _create_manifest(results)

        assert "Failed: 1" in manifest
        assert "[FAIL]" in manifest
        assert "Network error" in manifest

    def test_manifest_mixed_results(self):
        """Test manifest with mixed success/failure."""
        results = [
            BulkItemResult(
                row_number=2,
                url="https://example.com/page1",
                primary_keyword="keyword1",
                status=BulkOptimizationStatus.SUCCESS,
                filename="page1.docx",
            ),
            BulkItemResult(
                row_number=3,
                url="https://example.com/page2",
                primary_keyword="keyword2",
                status=BulkOptimizationStatus.FAILED,
                error_message="Timeout",
            ),
        ]

        manifest = _create_manifest(results)

        assert "Total URLs processed: 2" in manifest
        assert "Successful: 1" in manifest
        assert "Failed: 1" in manifest


class TestBulkOptimizationRow:
    """Tests for BulkOptimizationRow NamedTuple."""

    def test_row_creation(self):
        """Test creating a bulk optimization row."""
        row = BulkOptimizationRow(
            row_number=2,
            url="https://example.com/test",
            primary_keyword="test keyword",
            secondary_keywords=["sec1", "sec2"],
        )

        assert row.row_number == 2
        assert row.url == "https://example.com/test"
        assert row.primary_keyword == "test keyword"
        assert row.secondary_keywords == ["sec1", "sec2"]

    def test_row_immutable(self):
        """Test that row is immutable (NamedTuple)."""
        row = BulkOptimizationRow(
            row_number=2,
            url="https://example.com/test",
            primary_keyword="test",
            secondary_keywords=[],
        )

        with pytest.raises(AttributeError):
            row.url = "https://other.com"


class TestBulkItemResult:
    """Tests for BulkItemResult Pydantic model."""

    def test_success_result(self):
        """Test creating a success result."""
        result = BulkItemResult(
            row_number=2,
            url="https://example.com/test",
            primary_keyword="test",
            status=BulkOptimizationStatus.SUCCESS,
            filename="test.docx",
            secondary_keywords_used=["sec1"],
        )

        assert result.status == BulkOptimizationStatus.SUCCESS
        assert result.error_message is None

    def test_failed_result(self):
        """Test creating a failed result."""
        result = BulkItemResult(
            row_number=2,
            url="https://example.com/test",
            primary_keyword="test",
            status=BulkOptimizationStatus.FAILED,
            error_message="Connection timeout",
        )

        assert result.status == BulkOptimizationStatus.FAILED
        assert result.filename is None
        assert "timeout" in result.error_message.lower()

    def test_default_secondary_keywords(self):
        """Test default empty list for secondary keywords."""
        result = BulkItemResult(
            row_number=2,
            url="https://example.com/test",
            primary_keyword="test",
            status=BulkOptimizationStatus.PENDING,
        )

        assert result.secondary_keywords_used == []
