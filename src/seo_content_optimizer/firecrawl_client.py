"""
FireCrawl integration for structured web content extraction.

This module provides web scraping capabilities using FireCrawl API,
which extracts content while preserving the visual structure of web pages
including headings, tables, lists, and proper content hierarchy.
"""

import os
import re
from typing import Optional

from .models import (
    ContentBlock,
    ContentBlockType,
    HeadingLevel,
    ListBlock,
    ListItem,
    PageMeta,
    ParagraphBlock,
    TableBlock,
    TableCell,
    TableRow,
)


class FireCrawlError(Exception):
    """Raised when FireCrawl operations fail."""
    pass


class FireCrawlClient:
    """
    Client for extracting structured web content using FireCrawl.

    FireCrawl returns markdown-formatted content that preserves the
    visual structure of web pages. This client parses that markdown
    into structured ContentBlock objects.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FireCrawl client.

        Args:
            api_key: FireCrawl API key. Falls back to FIRECRAWL_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")

        # Lazy import to avoid dependency issues if not installed
        try:
            from firecrawl import Firecrawl
            self._firecrawl_available = True
            if self.api_key:
                self._client = Firecrawl(api_key=self.api_key)
            else:
                self._client = None
        except ImportError:
            self._firecrawl_available = False
            self._client = None

    @property
    def is_available(self) -> bool:
        """Check if FireCrawl is available and configured."""
        return self._firecrawl_available and self._client is not None

    def scrape_url(self, url: str, timeout: int = 60) -> PageMeta:
        """
        Scrape a URL and extract structured content.

        Args:
            url: URL to scrape.
            timeout: Request timeout in seconds.

        Returns:
            PageMeta with structured content blocks.

        Raises:
            FireCrawlError: If scraping fails or FireCrawl is unavailable.
        """
        if not self.is_available:
            raise FireCrawlError(
                "FireCrawl is not available. Please install firecrawl-py and "
                "set the FIRECRAWL_API_KEY environment variable."
            )

        try:
            # Scrape with markdown and HTML formats
            result = self._client.scrape(
                url,
                formats=["markdown", "html"],
            )

            # Extract metadata
            metadata = result.get("metadata", {})
            title = metadata.get("title")
            meta_description = metadata.get("description")

            # Get markdown content
            markdown = result.get("markdown", "")

            # Parse markdown into structured blocks
            structured_blocks = self._parse_markdown_to_blocks(markdown)

            # Extract H1 from structured blocks
            h1 = None
            for block in structured_blocks:
                if (block.block_type == ContentBlockType.HEADING and
                    block.paragraph and
                    block.paragraph.heading_level == HeadingLevel.H1):
                    h1 = block.paragraph.text
                    break

            # Also keep plain text content blocks for backward compatibility
            content_blocks = self._extract_text_blocks(structured_blocks)

            return PageMeta(
                title=title,
                meta_description=meta_description,
                h1=h1,
                content_blocks=content_blocks,
                url=url,
                structured_blocks=structured_blocks,
            )

        except Exception as e:
            raise FireCrawlError(f"Failed to scrape URL with FireCrawl: {e}")

    def _parse_markdown_to_blocks(self, markdown: str) -> list[ContentBlock]:
        """
        Parse markdown content into structured ContentBlock objects.

        Args:
            markdown: Markdown-formatted content from FireCrawl.

        Returns:
            List of ContentBlock objects preserving the document structure.
        """
        blocks: list[ContentBlock] = []
        lines = markdown.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Skip empty lines
            if not line.strip():
                i += 1
                continue

            # Check for headings (# Heading)
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                heading_level = HeadingLevel(level) if 1 <= level <= 6 else HeadingLevel.H6
                blocks.append(ContentBlock.from_paragraph(text, heading_level))
                i += 1
                continue

            # Check for table (starts with |)
            if line.strip().startswith("|"):
                table_block, consumed = self._parse_table(lines, i)
                if table_block:
                    blocks.append(ContentBlock(
                        block_type=ContentBlockType.TABLE,
                        table=table_block,
                    ))
                i += consumed
                continue

            # Check for unordered list (starts with - or * or +)
            if re.match(r"^\s*[-*+]\s+", line):
                list_block, consumed = self._parse_list(lines, i, ordered=False)
                if list_block:
                    blocks.append(ContentBlock(
                        block_type=ContentBlockType.LIST,
                        list_block=list_block,
                    ))
                i += consumed
                continue

            # Check for ordered list (starts with number.)
            if re.match(r"^\s*\d+\.\s+", line):
                list_block, consumed = self._parse_list(lines, i, ordered=True)
                if list_block:
                    blocks.append(ContentBlock(
                        block_type=ContentBlockType.LIST,
                        list_block=list_block,
                    ))
                i += consumed
                continue

            # Regular paragraph - collect consecutive non-empty, non-special lines
            para_lines = []
            while i < len(lines):
                current = lines[i]
                # Stop at empty line, heading, table, or list
                if (not current.strip() or
                    re.match(r"^#{1,6}\s+", current) or
                    current.strip().startswith("|") or
                    re.match(r"^\s*[-*+]\s+", current) or
                    re.match(r"^\s*\d+\.\s+", current)):
                    break
                para_lines.append(current)
                i += 1

            if para_lines:
                text = " ".join(line.strip() for line in para_lines)
                blocks.append(ContentBlock.from_paragraph(text, HeadingLevel.BODY))

        return blocks

    def _parse_table(self, lines: list[str], start: int) -> tuple[Optional[TableBlock], int]:
        """
        Parse a markdown table starting at the given line.

        Args:
            lines: All lines of the markdown.
            start: Starting line index.

        Returns:
            Tuple of (TableBlock or None, number of lines consumed).
        """
        rows: list[TableRow] = []
        i = start
        is_first_row = True

        while i < len(lines) and lines[i].strip().startswith("|"):
            line = lines[i].strip()

            # Skip separator row (|---|---|)
            if re.match(r"^\|[\s\-:]+\|$", line.replace(" ", "")):
                i += 1
                continue

            # Parse row cells
            cells_text = [c.strip() for c in line.split("|")[1:-1]]  # Remove empty first/last
            cells = [
                TableCell(text=text, is_header=is_first_row)
                for text in cells_text
            ]

            if cells:
                rows.append(TableRow(cells=cells))
                is_first_row = False

            i += 1

        if rows:
            return TableBlock(rows=rows), i - start
        return None, 1

    def _parse_list(
        self,
        lines: list[str],
        start: int,
        ordered: bool,
    ) -> tuple[Optional[ListBlock], int]:
        """
        Parse a markdown list starting at the given line.

        Args:
            lines: All lines of the markdown.
            start: Starting line index.
            ordered: Whether this is an ordered list.

        Returns:
            Tuple of (ListBlock or None, number of lines consumed).
        """
        items: list[ListItem] = []
        i = start

        # Pattern for list items
        pattern = r"^\s*\d+\.\s+" if ordered else r"^\s*[-*+]\s+"

        while i < len(lines):
            line = lines[i]
            match = re.match(pattern, line)
            if not match:
                break

            # Calculate nesting level from indentation
            indent = len(line) - len(line.lstrip())
            level = indent // 2

            # Extract text after the bullet/number
            text = re.sub(pattern, "", line).strip()
            items.append(ListItem(text=text, level=level))
            i += 1

        if items:
            return ListBlock(items=items, ordered=ordered), i - start
        return None, 1

    def _extract_text_blocks(self, structured_blocks: list[ContentBlock]) -> list[str]:
        """
        Extract plain text blocks from structured content for backward compatibility.

        Args:
            structured_blocks: List of ContentBlock objects.

        Returns:
            List of plain text strings.
        """
        text_blocks: list[str] = []

        for block in structured_blocks:
            if block.block_type in (ContentBlockType.PARAGRAPH, ContentBlockType.HEADING):
                if block.paragraph:
                    text_blocks.append(block.paragraph.text)
            elif block.block_type == ContentBlockType.TABLE:
                if block.table:
                    # Convert table to text representation
                    for row in block.table.rows:
                        row_text = " | ".join(cell.text for cell in row.cells)
                        text_blocks.append(row_text)
            elif block.block_type == ContentBlockType.LIST:
                if block.list_block:
                    for item in block.list_block.items:
                        prefix = "- " if not block.list_block.ordered else "1. "
                        text_blocks.append(f"{prefix}{item.text}")

        return text_blocks


def scrape_url_with_firecrawl(url: str, api_key: Optional[str] = None) -> PageMeta:
    """
    Convenience function to scrape a URL using FireCrawl.

    Args:
        url: URL to scrape.
        api_key: Optional FireCrawl API key.

    Returns:
        PageMeta with structured content.

    Raises:
        FireCrawlError: If scraping fails.
    """
    client = FireCrawlClient(api_key=api_key)
    return client.scrape_url(url)
