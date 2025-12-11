"""
Intuitive filename generation for optimized documents.

Generates descriptive filenames based on the source content:
- For URLs: Extracts slug from URL path or uses domain name
- For DOCX files: Prefixes with 'optimized-'
"""

import re
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse


def generate_output_filename(
    source: Union[str, Path],
    primary_keyword: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Generate an intuitive output filename based on the source.

    Args:
        source: URL string or Path to source DOCX file.
        primary_keyword: Optional primary keyword to include in filename.
        output_dir: Optional output directory. Defaults to current directory.

    Returns:
        Path object for the output file with .docx extension.

    Examples:
        >>> generate_output_filename("https://example.com/about-us")
        Path('optimized-about-us.docx')

        >>> generate_output_filename("https://example.com/services/plumbing")
        Path('optimized-plumbing.docx')

        >>> generate_output_filename(Path("my-document.docx"))
        Path('optimized-my-document.docx')
    """
    output_dir = output_dir or Path.cwd()

    # Determine if source is URL or file path
    source_str = str(source)

    if source_str.startswith(("http://", "https://")):
        base_name = _generate_from_url(source_str)
    else:
        base_name = _generate_from_file(Path(source))

    # Optionally incorporate primary keyword
    if primary_keyword:
        keyword_slug = _slugify(primary_keyword)
        # Only add keyword if it's not already in the name
        if keyword_slug.lower() not in base_name.lower():
            base_name = f"{base_name}-{keyword_slug}"

    # Ensure the name isn't too long (max 100 chars before extension)
    if len(base_name) > 100:
        base_name = base_name[:100]

    filename = f"optimized-{base_name}.docx"

    return output_dir / filename


def _generate_from_url(url: str) -> str:
    """
    Generate a base filename from a URL.

    Strategy:
    1. Try to use the last path segment (if meaningful)
    2. Fall back to domain name
    3. Fall back to 'content' if nothing else works

    Args:
        url: URL to extract filename from.

    Returns:
        Base filename string (without 'optimized-' prefix or extension).
    """
    parsed = urlparse(url)

    # Try to get a meaningful path segment
    path = parsed.path.strip("/")
    if path:
        # Get the last segment of the path
        segments = path.split("/")
        last_segment = segments[-1]

        # Clean up the segment
        # Remove common file extensions
        last_segment = re.sub(r"\.(html?|php|aspx?|jsp)$", "", last_segment, flags=re.IGNORECASE)

        # If segment is meaningful (not just index, empty, or numeric)
        if last_segment and not re.match(r"^(index|default|\d+)$", last_segment, re.IGNORECASE):
            return _slugify(last_segment)

        # Try the second-to-last segment if available
        if len(segments) > 1:
            second_last = segments[-2]
            if second_last and not re.match(r"^(index|default|\d+)$", second_last, re.IGNORECASE):
                return _slugify(second_last)

    # Fall back to domain name
    domain = parsed.netloc
    if domain:
        # Remove www. prefix and common TLDs for cleaner names
        domain = re.sub(r"^www\.", "", domain)
        # Extract just the domain name part
        domain_parts = domain.split(".")
        if len(domain_parts) > 1:
            # Use the main domain name (not TLD)
            main_domain = domain_parts[0]
            if main_domain and main_domain != "www":
                return _slugify(main_domain)

    # Ultimate fallback
    return "content"


def _generate_from_file(file_path: Path) -> str:
    """
    Generate a base filename from a file path.

    Args:
        file_path: Path to the source file.

    Returns:
        Base filename string (without 'optimized-' prefix or extension).
    """
    # Get the stem (filename without extension)
    stem = file_path.stem

    # If already starts with 'optimized', strip it to avoid 'optimized-optimized-'
    stem = re.sub(r"^optimized[-_]?", "", stem, flags=re.IGNORECASE)

    if not stem:
        return "document"

    return _slugify(stem)


def _slugify(text: str) -> str:
    """
    Convert text to a URL/filename-safe slug.

    Args:
        text: Text to convert.

    Returns:
        Slugified text.
    """
    # Convert to lowercase
    text = text.lower()

    # Replace spaces and underscores with hyphens
    text = re.sub(r"[\s_]+", "-", text)

    # Remove any characters that aren't alphanumeric or hyphens
    text = re.sub(r"[^a-z0-9-]", "", text)

    # Remove multiple consecutive hyphens
    text = re.sub(r"-+", "-", text)

    # Remove leading/trailing hyphens
    text = text.strip("-")

    return text or "content"


def suggest_filename_for_download(
    source: Union[str, Path],
    original_filename: Optional[str] = None,
) -> str:
    """
    Generate a suggested filename for HTTP Content-Disposition header.

    Args:
        source: URL or file path that was the source.
        original_filename: Original filename if from file upload.

    Returns:
        Suggested filename string with .docx extension.
    """
    source_str = str(source)

    # If we have an original filename (file upload case), use it
    if original_filename:
        stem = Path(original_filename).stem
        # Remove existing 'optimized' prefix if present
        stem = re.sub(r"^optimized[-_]?", "", stem, flags=re.IGNORECASE)
        if stem:
            return f"optimized-{_slugify(stem)}.docx"

    # For URLs, generate from the URL
    if source_str.startswith(("http://", "https://")):
        base = _generate_from_url(source_str)
        return f"optimized-{base}.docx"

    # Fallback
    return "optimized-content.docx"
