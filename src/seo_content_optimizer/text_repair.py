# -*- coding: utf-8 -*-
"""
Text repair and encoding normalization module.

Handles:
- Mojibake detection and repair (curly quotes -> ASCII, etc.)
- Whitespace normalization
- HTML entity unescaping
- Character encoding correction
"""

import html
import re
import unicodedata
from typing import Tuple, Optional, List

# Common mojibake byte sequences (as raw bytes for safety)
# These are checked as substrings in text
MOJIBAKE_MARKERS_BYTES = [
    b'\xc3\xa2\xe2\x82\xac',  # Common UTF-8/Windows corruption
    b'\xc3\x83',              # UTF-8 decoded as Latin-1
    b'\xc2\xa0',              # Non-breaking space corruption
    b'\xef\xbf\xbd',          # Replacement character
]

# Simple string markers to detect
MOJIBAKE_MARKERS_STR = [
    "â€",      # Most common mojibake prefix
    "Â",       # Non-breaking space corruption
    "Ã",       # UTF-8 decoded as Latin-1
]

# Smart quotes to normalize (using Unicode code points)
SMART_QUOTE_MAP = {
    '\u2018': "'",   # Left single quotation mark
    '\u2019': "'",   # Right single quotation mark
    '\u201a': "'",   # Single low-9 quotation mark
    '\u201b': "'",   # Single high-reversed-9 quotation mark
    '\u201c': '"',   # Left double quotation mark
    '\u201d': '"',   # Right double quotation mark
    '\u201e': '"',   # Double low-9 quotation mark
    '\u201f': '"',   # Double high-reversed-9 quotation mark
    '\u00ab': '"',   # Left-pointing double angle quotation mark
    '\u00bb': '"',   # Right-pointing double angle quotation mark
    '\u2039': "'",   # Single left-pointing angle quotation mark
    '\u203a': "'",   # Single right-pointing angle quotation mark
}

# Dash variants to normalize
DASH_MAP = {
    '\u2014': " - ",  # Em dash -> spaced hyphen
    '\u2013': "-",    # En dash
    '\u2212': "-",    # Minus sign
    '\u00ad': "",     # Soft hyphen (remove)
}


def detect_mojibake(text: str) -> bool:
    """
    Detect if text contains mojibake corruption.

    Args:
        text: Text to check.

    Returns:
        True if mojibake markers detected.
    """
    if not text:
        return False

    # Check string markers
    for marker in MOJIBAKE_MARKERS_STR:
        if marker in text:
            return True

    # Check for replacement character
    if '\ufffd' in text:
        return True

    return False


def repair_mojibake(text: str) -> str:
    """
    Attempt to repair mojibake corruption.

    Args:
        text: Text with potential mojibake.

    Returns:
        Repaired text.
    """
    if not text:
        return text

    result = text

    # Try ftfy first if available (best mojibake fixer)
    try:
        import ftfy
        result = ftfy.fix_text(result)
    except ImportError:
        # Manual repair of common patterns
        # These are safe ASCII replacements
        replacements = [
            # Quote corruptions
            ("â€™", "'"),
            ("â€˜", "'"),
            ("â€œ", '"'),
            ("â€", '"'),
            # Ellipsis
            ("â€¦", "..."),
            # Copyright/trademark
            ("Â©", "(c)"),
            ("Â®", "(R)"),
            # Orphan bytes (remove)
            ("Ã", ""),
            ("Â", ""),
        ]
        for pattern, replacement in replacements:
            result = result.replace(pattern, replacement)

    return result


def normalize_quotes(text: str, to_ascii: bool = True) -> str:
    """
    Normalize smart quotes and other typographic characters.

    Args:
        text: Text to normalize.
        to_ascii: If True, convert to ASCII quotes/dashes.

    Returns:
        Normalized text.
    """
    if not text:
        return text

    result = text

    if to_ascii:
        # Convert smart quotes to ASCII
        for smart, ascii_char in SMART_QUOTE_MAP.items():
            result = result.replace(smart, ascii_char)

        # Convert dashes
        for fancy, simple in DASH_MAP.items():
            result = result.replace(fancy, simple)

    return result


def unescape_html(text: str) -> str:
    """
    Unescape HTML entities.

    Args:
        text: Text with potential HTML entities.

    Returns:
        Unescaped text.
    """
    if not text:
        return text

    # Standard HTML unescape
    result = html.unescape(text)

    # Handle numeric entities
    result = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), result)
    result = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), result)

    return result


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Fixes:
    - Multiple spaces -> single space
    - Non-breaking spaces -> regular spaces
    - Missing spaces after periods (word.Word -> word. Word)
    - Tab/newline normalization
    - U+2028 LINE SEPARATOR -> newline (common mojibake source)
    - U+2029 PARAGRAPH SEPARATOR -> double newline

    Args:
        text: Text to normalize.

    Returns:
        Normalized text.
    """
    if not text:
        return text

    result = text

    # CRITICAL: Normalize Unicode line/paragraph separators FIRST
    # U+2028 LINE SEPARATOR causes "â¨" artifacts when mis-decoded
    # U+2029 PARAGRAPH SEPARATOR can also cause issues
    result = result.replace('\u2028', '\n')   # Line separator -> newline
    result = result.replace('\u2029', '\n\n') # Paragraph separator -> double newline

    # Convert non-breaking spaces and other space variants
    result = result.replace('\u00a0', ' ')  # Non-breaking space
    result = result.replace('\u2002', ' ')  # En space
    result = result.replace('\u2003', ' ')  # Em space
    result = result.replace('\u2009', ' ')  # Thin space
    result = result.replace('\u200a', ' ')  # Hair space
    result = result.replace('\u200b', '')   # Zero-width space (remove)
    result = result.replace('\ufeff', '')   # BOM (remove)

    # Remove other problematic Unicode characters
    result = result.replace('\u200c', '')   # Zero-width non-joiner
    result = result.replace('\u200d', '')   # Zero-width joiner
    result = result.replace('\u2060', '')   # Word joiner

    # Normalize tabs and newlines for inline text
    result = result.replace('\t', ' ')
    result = result.replace('\r\n', '\n')
    result = result.replace('\r', '\n')

    # Fix missing space after punctuation: "word.Word" -> "word. Word"
    result = re.sub(r'([.!?])([A-Z])', r'\1 \2', result)

    # Collapse multiple spaces (but preserve single spaces!)
    result = re.sub(r'  +', ' ', result)

    # IMPORTANT: Only strip if the entire string, not boundaries
    # This preserves whitespace needed for word separation
    return result.strip()


def repair_text(text: str, aggressive: bool = False) -> Tuple[str, bool]:
    """
    Full text repair pipeline.

    Applies all repair steps in order:
    1. HTML unescape
    2. Mojibake repair
    3. Quote normalization
    4. Whitespace normalization

    Args:
        text: Text to repair.
        aggressive: If True, convert all to ASCII-safe.

    Returns:
        Tuple of (repaired_text, was_corrupted).
    """
    if not text:
        return text, False

    original = text
    was_corrupted = detect_mojibake(text)

    # Step 1: HTML unescape
    result = unescape_html(text)

    # Step 2: Mojibake repair
    result = repair_mojibake(result)

    # Step 3: Quote normalization
    result = normalize_quotes(result, to_ascii=aggressive)

    # Step 4: Whitespace normalization
    result = normalize_whitespace(result)

    # Step 5: Unicode normalization (NFC form)
    result = unicodedata.normalize('NFC', result)

    # Check if we made changes
    if result != original:
        was_corrupted = True

    return result, was_corrupted


def validate_text_quality(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that text is clean and uncorrupted.

    Returns False with error message if:
    - Mojibake markers detected
    - Replacement characters present
    - High ratio of non-printable characters

    Args:
        text: Text to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not text:
        return True, None

    # Check for mojibake markers
    if detect_mojibake(text):
        return False, "Text contains mojibake corruption markers"

    # Check for replacement characters
    if '\ufffd' in text:
        return False, "Text contains replacement characters"

    # Check for high ratio of control characters
    control_count = sum(1 for c in text if unicodedata.category(c).startswith('C') and c not in '\n\t\r')
    if control_count > len(text) * 0.05:  # More than 5% control chars
        return False, "Text contains {} control characters".format(control_count)

    return True, None


def repair_content_blocks(blocks: List) -> Tuple[List, int]:
    """
    Repair a list of content blocks.

    Args:
        blocks: List of text blocks.

    Returns:
        Tuple of (repaired_blocks, corruption_count).
    """
    repaired = []
    corruption_count = 0

    for block in blocks:
        if isinstance(block, str):
            fixed, was_corrupted = repair_text(block)
            repaired.append(fixed)
            if was_corrupted:
                corruption_count += 1
        else:
            repaired.append(block)

    return repaired, corruption_count
