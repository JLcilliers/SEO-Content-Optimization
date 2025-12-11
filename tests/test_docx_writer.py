"""Tests for Word document writing with green highlights."""

import pytest
from pathlib import Path

from docx import Document

from seo_content_optimizer.docx_writer import (
    DocxWriter,
    create_simple_docx_with_highlights,
    write_optimization_result,
)
from seo_content_optimizer.models import (
    FAQItem,
    HeadingLevel,
    MetaElement,
    OptimizationResult,
    ParagraphBlock,
)


class TestDocxWriter:
    """Tests for the DocxWriter class."""

    def test_parse_marker_segments_no_markers(self):
        """Test parsing text without markers."""
        writer = DocxWriter()
        segments = writer._parse_marker_segments("Plain text without markers.")

        assert len(segments) == 1
        assert segments[0] == ("Plain text without markers.", False)

    def test_parse_marker_segments_with_markers(self):
        """Test parsing text with ADD markers."""
        writer = DocxWriter()
        text = "Original [[[ADD]]]new content[[[ENDADD]]] more original."
        segments = writer._parse_marker_segments(text)

        assert len(segments) == 3
        assert segments[0] == ("Original ", False)
        assert segments[1] == ("new content", True)
        assert segments[2] == (" more original.", False)

    def test_parse_marker_segments_multiple_markers(self):
        """Test parsing text with multiple marker sections."""
        writer = DocxWriter()
        text = "[[[ADD]]]First[[[ENDADD]]] middle [[[ADD]]]second[[[ENDADD]]] end"
        segments = writer._parse_marker_segments(text)

        assert len(segments) == 4
        assert segments[0] == ("First", True)
        assert segments[1] == (" middle ", False)
        assert segments[2] == ("second", True)
        assert segments[3] == (" end", False)

    def test_parse_marker_segments_only_markers(self):
        """Test parsing text that is entirely marked."""
        writer = DocxWriter()
        text = "[[[ADD]]]All new content here[[[ENDADD]]]"
        segments = writer._parse_marker_segments(text)

        assert len(segments) == 1
        assert segments[0] == ("All new content here", True)

    def test_write_creates_file(self, tmp_path: Path):
        """Test that write creates a file."""
        result = OptimizationResult(
            meta_elements=[
                MetaElement(
                    element_name="Title Tag",
                    current="Old Title",
                    optimized="[[[ADD]]]New[[[ENDADD]]] Title",
                    why_changed="Added keyword",
                ),
            ],
            optimized_blocks=[
                ParagraphBlock(
                    text="This is [[[ADD]]]optimized[[[ENDADD]]] content.",
                    heading_level=HeadingLevel.BODY,
                ),
            ],
            primary_keyword="test keyword",
        )

        output_path = tmp_path / "output.docx"
        writer = DocxWriter()
        created_path = writer.write(result, output_path)

        assert created_path.exists()
        assert created_path.suffix == ".docx"

    def test_write_adds_docx_extension(self, tmp_path: Path):
        """Test that .docx extension is added if missing."""
        result = OptimizationResult(primary_keyword="test")
        output_path = tmp_path / "output"

        writer = DocxWriter()
        created_path = writer.write(result, output_path)

        assert created_path.suffix == ".docx"

    def test_write_includes_reading_guide(self, tmp_path: Path):
        """Test that the reading guide is included."""
        result = OptimizationResult(primary_keyword="test")
        output_path = tmp_path / "output.docx"

        writer = DocxWriter()
        writer.write(result, output_path)

        # Read the document and check for reading guide
        doc = Document(str(output_path))
        text = "\n".join(p.text for p in doc.paragraphs)

        assert "Reading Guide" in text or any(
            "Reading Guide" in p.style.name for p in doc.paragraphs if p.style
        )

    def test_write_includes_meta_table(self, tmp_path: Path):
        """Test that meta elements table is included."""
        result = OptimizationResult(
            meta_elements=[
                MetaElement(
                    element_name="Title Tag",
                    current="Old",
                    optimized="New",
                    why_changed="Reason",
                ),
            ],
            primary_keyword="test",
        )
        output_path = tmp_path / "output.docx"

        writer = DocxWriter()
        writer.write(result, output_path)

        doc = Document(str(output_path))
        assert len(doc.tables) >= 1

    def test_write_includes_faq(self, tmp_path: Path):
        """Test that FAQ section is included when present."""
        result = OptimizationResult(
            faq_items=[
                FAQItem(
                    question="[[[ADD]]]What is SEO?[[[ENDADD]]]",
                    answer="[[[ADD]]]SEO is search engine optimization.[[[ENDADD]]]",
                ),
            ],
            primary_keyword="test",
        )
        output_path = tmp_path / "output.docx"

        writer = DocxWriter()
        writer.write(result, output_path)

        doc = Document(str(output_path))
        text = "\n".join(p.text for p in doc.paragraphs)

        assert "Frequently Asked Questions" in text or "FAQ" in text


class TestCreateSimpleDocx:
    """Tests for the simple docx creation helper."""

    def test_create_simple_docx(self, tmp_path: Path):
        """Test creating a simple docx with highlights."""
        text = "Normal text [[[ADD]]]highlighted text[[[ENDADD]]] more normal."
        output_path = tmp_path / "simple.docx"

        created_path = create_simple_docx_with_highlights(text, output_path)

        assert created_path.exists()
        doc = Document(str(created_path))
        assert len(doc.paragraphs) > 0


class TestWriteOptimizationResult:
    """Tests for the convenience function."""

    def test_write_optimization_result(self, tmp_path: Path):
        """Test the convenience function."""
        result = OptimizationResult(
            meta_elements=[],
            optimized_blocks=[
                ParagraphBlock(text="Test content", heading_level=HeadingLevel.BODY),
            ],
            primary_keyword="test",
        )
        output_path = tmp_path / "result.docx"

        created_path = write_optimization_result(result, output_path)

        assert created_path.exists()
