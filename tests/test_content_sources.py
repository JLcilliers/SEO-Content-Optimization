"""Tests for content source loading functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from seo_content_optimizer.content_sources import (
    ContentExtractionError,
    convert_page_meta_to_blocks,
    fetch_url_content,
    load_content,
    load_docx_content,
)
from seo_content_optimizer.models import DocxContent, HeadingLevel, PageMeta


class TestLoadDocxContent:
    """Tests for Word document loading."""

    def test_load_valid_docx(self, sample_docx: Path):
        """Test loading a valid Word document."""
        content = load_docx_content(sample_docx)

        assert isinstance(content, DocxContent)
        assert len(content.paragraphs) > 0
        assert content.source_path == str(sample_docx)

    def test_docx_heading_detection(self, sample_docx: Path):
        """Test that headings are detected correctly."""
        content = load_docx_content(sample_docx)

        # Find H1 heading
        h1_blocks = [p for p in content.paragraphs if p.heading_level == HeadingLevel.H1]
        assert len(h1_blocks) >= 1

        # Find H2 headings
        h2_blocks = [p for p in content.paragraphs if p.heading_level == HeadingLevel.H2]
        assert len(h2_blocks) >= 1

    def test_docx_h1_property(self, sample_docx: Path):
        """Test the h1 property."""
        content = load_docx_content(sample_docx)
        assert content.h1 is not None
        assert "Insurance" in content.h1 or "Liability" in content.h1

    def test_docx_full_text(self, sample_docx: Path):
        """Test full_text property."""
        content = load_docx_content(sample_docx)
        full_text = content.full_text

        assert len(full_text) > 0
        assert "professional" in full_text.lower() or "liability" in full_text.lower()

    def test_load_nonexistent_docx(self, tmp_path: Path):
        """Test loading non-existent file."""
        with pytest.raises(ContentExtractionError, match="File not found"):
            load_docx_content(tmp_path / "nonexistent.docx")

    def test_load_non_docx_file(self, tmp_path: Path):
        """Test loading non-docx file."""
        txt_path = tmp_path / "file.txt"
        txt_path.write_text("Not a docx")

        with pytest.raises(ContentExtractionError, match="must be a .docx file"):
            load_docx_content(txt_path)


class TestFetchUrlContent:
    """Tests for URL content fetching."""

    def test_invalid_url(self):
        """Test that invalid URLs raise errors."""
        with pytest.raises(ContentExtractionError, match="Invalid URL"):
            fetch_url_content("not-a-url")

    @patch("seo_content_optimizer.content_sources.requests.get")
    def test_fetch_url_extracts_meta(self, mock_get, sample_html_content: str):
        """Test that meta elements are extracted from HTML."""
        mock_response = Mock()
        mock_response.text = sample_html_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        content = fetch_url_content("https://example.com/page")

        assert isinstance(content, PageMeta)
        assert content.title == "Professional Liability Insurance | Expert Guide"
        assert "professional liability insurance" in content.meta_description.lower()
        assert "Understanding Professional Liability Insurance" in content.h1

    @patch("seo_content_optimizer.content_sources.requests.get")
    def test_fetch_url_extracts_content(self, mock_get, sample_html_content: str):
        """Test that body content is extracted."""
        mock_response = Mock()
        mock_response.text = sample_html_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        content = fetch_url_content("https://example.com/page")

        assert len(content.content_blocks) > 0
        full_text = content.full_text.lower()
        assert "professional" in full_text or "insurance" in full_text


class TestLoadContent:
    """Tests for the generic load_content function."""

    def test_load_content_docx(self, sample_docx: Path):
        """Test loading a docx file via load_content."""
        content = load_content(str(sample_docx))
        assert isinstance(content, DocxContent)

    @patch("seo_content_optimizer.content_sources.requests.get")
    def test_load_content_url(self, mock_get, sample_html_content: str):
        """Test loading a URL via load_content."""
        mock_response = Mock()
        mock_response.text = sample_html_content
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        content = load_content("https://example.com/page")
        assert isinstance(content, PageMeta)

    def test_load_content_invalid_source(self, tmp_path: Path):
        """Test loading an invalid source."""
        txt_path = tmp_path / "file.txt"
        txt_path.write_text("text content")

        with pytest.raises(ContentExtractionError, match="Invalid source"):
            load_content(str(txt_path))


class TestConvertPageMetaToBlocks:
    """Tests for converting PageMeta to blocks."""

    def test_convert_with_h1(self):
        """Test conversion includes H1."""
        meta = PageMeta(
            title="Test Title",
            h1="Main Heading",
            content_blocks=["First paragraph.", "Second paragraph."],
        )

        blocks = convert_page_meta_to_blocks(meta)

        assert len(blocks) >= 3
        h1_block = blocks[0]
        assert h1_block.heading_level == HeadingLevel.H1
        assert h1_block.text == "Main Heading"

    def test_convert_empty_meta(self):
        """Test conversion of empty meta."""
        meta = PageMeta()
        blocks = convert_page_meta_to_blocks(meta)
        assert len(blocks) == 0
