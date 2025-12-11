"""
Word document writer with green highlight support.

This module generates the final optimized .docx output with:
- Reading guide at the top
- Current vs Optimized Meta Elements table
- Optimized content with green-highlighted changes
"""

import re
from pathlib import Path
from typing import Optional, Union

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.table import Table

from .llm_client import ADD_END, ADD_START, strip_markers
from .models import FAQItem, HeadingLevel, MetaElement, OptimizationResult, ParagraphBlock


# Green highlight color (bright green for visibility)
HIGHLIGHT_GREEN = "brightGreen"  # Word's built-in bright green


def set_cell_shading(cell, color: str) -> None:
    """Set background color/shading for a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def add_highlight_to_run(run, highlight_color: str = HIGHLIGHT_GREEN) -> None:
    """Add highlight color to a run."""
    run.font.highlight_color = getattr(
        __import__("docx.enum.text", fromlist=["WD_COLOR_INDEX"]).WD_COLOR_INDEX,
        highlight_color.upper(),
        None,
    )
    # Fallback: use manual XML if enum doesn't exist
    if run.font.highlight_color is None:
        # Set via XML
        rPr = run._r.get_or_add_rPr()
        highlight = OxmlElement("w:highlight")
        highlight.set(qn("w:val"), highlight_color)
        rPr.append(highlight)


class DocxWriter:
    """
    Writes optimization results to a Word document.

    Creates a professional document with:
    - Reading guide explaining the green highlight convention
    - Table comparing current vs optimized meta elements
    - Full optimized content with changes highlighted
    """

    def __init__(self):
        """Initialize the document writer."""
        self.doc = Document()
        self._setup_styles()

    def _setup_styles(self) -> None:
        """Configure document styles."""
        # Set default font
        style = self.doc.styles["Normal"]
        style.font.name = "Calibri"
        style.font.size = Pt(11)

    def write(
        self,
        result: OptimizationResult,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Write the optimization result to a Word document.

        Args:
            result: OptimizationResult containing all optimized content.
            output_path: Path for the output .docx file.

        Returns:
            Path to the created document.
        """
        output_path = Path(output_path)

        # Ensure .docx extension
        if output_path.suffix.lower() != ".docx":
            output_path = output_path.with_suffix(".docx")

        # Add reading guide
        self._add_reading_guide()

        # Add meta elements table
        self._add_meta_table(result.meta_elements)

        # Add optimized content header
        self._add_section_header("OPTIMIZED CONTENT")

        # Add optimized blocks
        self._add_optimized_blocks(result.optimized_blocks)

        # Add FAQ section if present
        if result.faq_items:
            self._add_faq_section(result.faq_items)

        # Save document
        self.doc.save(str(output_path))

        return output_path

    def _add_reading_guide(self) -> None:
        """Add the reading guide section at the top."""
        # Add title
        title = self.doc.add_heading("Reading Guide", level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.LEFT

        # Add guide text
        guide_para = self.doc.add_paragraph()

        # Add the guide text with a highlighted sample
        guide_para.add_run(
            "This document contains SEO-optimized content. Text highlighted in "
        )

        # Add highlighted sample
        sample_run = guide_para.add_run("green like this")
        add_highlight_to_run(sample_run)

        guide_para.add_run(
            " indicates keyword insertions or SEO-focused adjustments. "
            "All non-highlighted text remains unchanged from the original content.\n\n"
            "Review the table below to see a summary of meta element changes, "
            "then scroll down to see the full optimized content."
        )

        # Add spacing
        self.doc.add_paragraph()

    def _add_meta_table(self, meta_elements: list[MetaElement]) -> None:
        """Add the Current vs Optimized Meta Elements table."""
        # Add section header
        self.doc.add_heading("Current vs Optimized Meta Elements", level=2)

        # Create table with 4 columns
        table = self.doc.add_table(rows=1, cols=4)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.LEFT

        # Set column widths
        widths = [Inches(1.2), Inches(2.0), Inches(2.5), Inches(2.3)]
        for i, width in enumerate(widths):
            table.columns[i].width = width

        # Add header row
        header_cells = table.rows[0].cells
        headers = ["Element", "Current", "Optimized", "Why Changed"]
        for i, header in enumerate(headers):
            header_cells[i].text = header
            # Bold header text
            for paragraph in header_cells[i].paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
            # Gray background for header
            set_cell_shading(header_cells[i], "D9D9D9")

        # Add data rows
        for meta in meta_elements:
            row = table.add_row()
            cells = row.cells

            # Element name
            cells[0].text = meta.element_name

            # Current value
            cells[1].text = meta.current or "(None)"

            # Optimized value - with highlights
            self._add_text_with_highlights(cells[2], meta.optimized)

            # Why changed
            cells[3].text = meta.why_changed

        # Add spacing after table
        self.doc.add_paragraph()

    def _add_section_header(self, title: str) -> None:
        """Add a section header."""
        self.doc.add_heading(title, level=1)

    def _add_optimized_blocks(self, blocks: list[ParagraphBlock]) -> None:
        """Add optimized content blocks with highlights."""
        for block in blocks:
            if block.is_heading:
                # Add as heading
                level = min(block.heading_level.value, 6) if block.heading_level.value > 0 else 2
                heading = self.doc.add_heading(level=level)
                self._add_text_with_highlights_to_paragraph(heading, block.text)
            else:
                # Add as paragraph
                para = self.doc.add_paragraph()
                self._add_text_with_highlights_to_paragraph(para, block.text)

    def _add_faq_section(self, faq_items: list[FAQItem]) -> None:
        """Add the FAQ section."""
        # Add FAQ header
        self.doc.add_heading("Frequently Asked Questions", level=2)

        for item in faq_items:
            # Question as H3
            q_heading = self.doc.add_heading(level=3)
            self._add_text_with_highlights_to_paragraph(q_heading, item.question)

            # Answer as paragraph
            a_para = self.doc.add_paragraph()
            self._add_text_with_highlights_to_paragraph(a_para, item.answer)

    def _add_text_with_highlights(self, cell, text: str) -> None:
        """Add text to a table cell with highlight markers processed."""
        # Clear existing content
        cell.text = ""
        para = cell.paragraphs[0]
        self._add_text_with_highlights_to_paragraph(para, text)

    def _add_text_with_highlights_to_paragraph(self, paragraph, text: str) -> None:
        """
        Add text to a paragraph, applying green highlights to marked sections.

        Parses [[[ADD]]]...[[[ENDADD]]] markers and highlights those sections.
        """
        # Split text by markers
        segments = self._parse_marker_segments(text)

        for segment_text, is_highlighted in segments:
            if not segment_text:
                continue

            run = paragraph.add_run(segment_text)

            if is_highlighted:
                add_highlight_to_run(run)

    def _parse_marker_segments(self, text: str) -> list[tuple[str, bool]]:
        """
        Parse text with markers into segments.

        Returns list of (text, is_highlighted) tuples.
        """
        segments: list[tuple[str, bool]] = []

        # Pattern to match [[[ADD]]]...[[[ENDADD]]]
        pattern = re.compile(
            rf"{re.escape(ADD_START)}(.*?){re.escape(ADD_END)}",
            re.DOTALL,
        )

        last_end = 0

        for match in pattern.finditer(text):
            # Add text before this match (not highlighted)
            if match.start() > last_end:
                before_text = text[last_end : match.start()]
                if before_text:
                    segments.append((before_text, False))

            # Add matched content (highlighted)
            highlighted_text = match.group(1)
            if highlighted_text:
                segments.append((highlighted_text, True))

            last_end = match.end()

        # Add any remaining text after last match
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining:
                segments.append((remaining, False))

        # If no markers found, return original text unhighlighted
        if not segments:
            segments.append((text, False))

        return segments


def write_optimization_result(
    result: OptimizationResult,
    output_path: Union[str, Path],
) -> Path:
    """
    Convenience function to write optimization result to docx.

    Args:
        result: OptimizationResult to write.
        output_path: Output file path.

    Returns:
        Path to created document.
    """
    writer = DocxWriter()
    return writer.write(result, output_path)


def create_simple_docx_with_highlights(
    text: str,
    output_path: Union[str, Path],
    title: str = "Optimized Content",
) -> Path:
    """
    Create a simple docx with highlighted text.

    Useful for testing the highlight functionality.

    Args:
        text: Text with [[[ADD]]]...[[[ENDADD]]] markers.
        output_path: Output file path.
        title: Document title.

    Returns:
        Path to created document.
    """
    doc = Document()

    # Add title
    doc.add_heading(title, level=1)

    # Add content with highlights
    writer = DocxWriter()
    para = doc.add_paragraph()
    writer._add_text_with_highlights_to_paragraph(para, text)

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".docx":
        output_path = output_path.with_suffix(".docx")

    doc.save(str(output_path))
    return output_path
