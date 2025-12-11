"""
Content extraction from various sources (URLs and Word documents).

This module handles fetching and parsing content from:
- Web URLs (using requests + trafilatura + BeautifulSoup)
- Word documents (.docx files using python-docx)
"""

import re
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from docx import Document
from docx.enum.style import WD_STYLE_TYPE

try:
    import trafilatura
except ImportError:
    trafilatura = None  # type: ignore

from .models import DocxContent, HeadingLevel, PageMeta, ParagraphBlock


# Default headers for web requests
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Mapping of Word heading styles to HeadingLevel
HEADING_STYLE_MAP = {
    "Heading 1": HeadingLevel.H1,
    "Heading 2": HeadingLevel.H2,
    "Heading 3": HeadingLevel.H3,
    "Heading 4": HeadingLevel.H4,
    "Heading 5": HeadingLevel.H5,
    "Heading 6": HeadingLevel.H6,
    "Title": HeadingLevel.H1,
}


class ContentExtractionError(Exception):
    """Raised when content extraction fails."""
    pass


def fetch_url_content(url: str, timeout: int = 30) -> PageMeta:
    """
    Fetch and extract content from a URL.

    Args:
        url: The URL to fetch content from.
        timeout: Request timeout in seconds.

    Returns:
        PageMeta object containing extracted content and metadata.

    Raises:
        ContentExtractionError: If fetching or parsing fails.
    """
    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ContentExtractionError(f"Invalid URL: {url}")

    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ContentExtractionError(f"Failed to fetch URL: {e}")

    html = response.text

    # Parse HTML with BeautifulSoup for meta extraction
    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title = None
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Extract meta description
    meta_description = None
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    if meta_desc_tag and meta_desc_tag.get("content"):
        meta_description = meta_desc_tag["content"].strip()

    # Extract H1
    h1 = None
    h1_tag = soup.find("h1")
    if h1_tag:
        h1 = h1_tag.get_text(strip=True)

    # Extract main content using trafilatura (if available)
    content_blocks: list[str] = []

    if trafilatura:
        # trafilatura extracts main article content, removing boilerplate
        main_content = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )
        if main_content:
            # Split into paragraphs
            paragraphs = [p.strip() for p in main_content.split("\n\n") if p.strip()]
            content_blocks = paragraphs
    else:
        # Fallback: extract text from common content containers
        content_blocks = _extract_content_fallback(soup)

    return PageMeta(
        title=title,
        meta_description=meta_description,
        h1=h1,
        content_blocks=content_blocks,
        url=url,
    )


def _extract_content_fallback(soup: BeautifulSoup) -> list[str]:
    """
    Fallback content extraction when trafilatura is not available.

    Args:
        soup: BeautifulSoup parsed HTML.

    Returns:
        List of content blocks (paragraphs).
    """
    content_blocks: list[str] = []

    # Remove script, style, nav, header, footer elements
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Look for common content containers
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|post|entry|article", re.I))
        or soup.find("div", id=re.compile(r"content|post|entry|article", re.I))
    )

    if main_content:
        source = main_content
    else:
        source = soup.body or soup

    # Extract paragraphs and headings
    for element in source.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
        text = element.get_text(strip=True)
        if text and len(text) > 20:  # Skip very short snippets
            content_blocks.append(text)

    return content_blocks


def load_docx_content(file_path: Union[str, Path]) -> DocxContent:
    """
    Load and extract content from a Word document.

    Args:
        file_path: Path to the .docx file.

    Returns:
        DocxContent object containing extracted paragraphs and headings.

    Raises:
        ContentExtractionError: If the file cannot be read or parsed.
    """
    path = Path(file_path)

    if not path.exists():
        raise ContentExtractionError(f"File not found: {file_path}")

    if not path.suffix.lower() == ".docx":
        raise ContentExtractionError(f"File must be a .docx file: {file_path}")

    try:
        doc = Document(str(path))
    except Exception as e:
        raise ContentExtractionError(f"Failed to open Word document: {e}")

    paragraphs: list[ParagraphBlock] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Determine heading level from style
        heading_level = HeadingLevel.BODY
        style_name = None

        if para.style:
            style_name = para.style.name
            if style_name in HEADING_STYLE_MAP:
                heading_level = HEADING_STYLE_MAP[style_name]
            elif style_name and style_name.startswith("Heading"):
                # Try to parse numeric heading level
                try:
                    level = int(style_name.replace("Heading ", "").strip())
                    if 1 <= level <= 6:
                        heading_level = HeadingLevel(level)
                except ValueError:
                    pass

        paragraphs.append(
            ParagraphBlock(
                text=text,
                heading_level=heading_level,
                style_name=style_name,
            )
        )

    return DocxContent(paragraphs=paragraphs, source_path=str(path))


def load_content(source: str) -> Union[PageMeta, DocxContent]:
    """
    Load content from either a URL or a file path.

    Args:
        source: URL or file path to load content from.

    Returns:
        PageMeta (for URLs) or DocxContent (for .docx files).

    Raises:
        ContentExtractionError: If the source is invalid or cannot be loaded.
    """
    # Check if it's a URL
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return fetch_url_content(source)

    # Check if it's a file path
    path = Path(source)
    if path.suffix.lower() == ".docx":
        return load_docx_content(path)

    raise ContentExtractionError(
        f"Invalid source: {source}. Must be a URL (http/https) or a .docx file path."
    )


def convert_page_meta_to_blocks(page_meta: PageMeta) -> list[ParagraphBlock]:
    """
    Convert PageMeta content blocks to ParagraphBlock list for uniform processing.

    Args:
        page_meta: PageMeta object from URL extraction.

    Returns:
        List of ParagraphBlock objects.
    """
    blocks: list[ParagraphBlock] = []

    # Add H1 if present
    if page_meta.h1:
        blocks.append(ParagraphBlock(text=page_meta.h1, heading_level=HeadingLevel.H1))

    # Add content blocks as body paragraphs
    # In real scenarios, we might want to detect headings within content_blocks
    for block in page_meta.content_blocks:
        # Simple heuristic: short text that looks like a heading
        if len(block) < 100 and not block.endswith((".", "!", "?")):
            # Might be a heading - default to H2
            blocks.append(ParagraphBlock(text=block, heading_level=HeadingLevel.H2))
        else:
            blocks.append(ParagraphBlock(text=block, heading_level=HeadingLevel.BODY))

    return blocks
