# -*- coding: utf-8 -*-
"""
Locked Token Protection for URLs, Emails, and Phone Numbers.

This module provides a robust system for protecting sensitive tokens
(URLs, emails, phone numbers) during LLM rewriting and diff operations.

The Problem:
- LLMs can corrupt URLs like "fortell.com/warranty" → "Fortell. com/warranty"
- Diff algorithms can split URLs into multiple tokens, causing partial highlighting
- Emails and phone numbers face similar corruption risks

The Solution:
1. BEFORE any LLM/diff processing: Replace sensitive tokens with placeholders
2. DURING processing: Placeholders are treated as atomic, unmutable tokens
3. AFTER processing: Restore original tokens verbatim

Usage:
    protector = LockedTokenProtector()
    protected_text, token_map = protector.protect(original_text)
    # ... do LLM rewriting, diffing, etc. on protected_text ...
    final_text = protector.restore(processed_text, token_map)
"""

import re
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# URL PATTERN (Comprehensive)
# =============================================================================
# Matches:
# - Full URLs: https://example.com/path?query=1
# - Domain-only URLs: example.com/path, www.example.com
# - Protocol-prefixed: http://, https://, ftp://
#
# Does NOT match:
# - Single words without TLD (to avoid false positives)
# - File paths (C:\path\to\file)

URL_PATTERN = re.compile(
    r"""
    (?:
        # Full URL with protocol
        (?:https?|ftp)://
        (?:[\w-]+\.)+[\w-]+
        (?:/[^\s<>"']*)?
    )
    |
    (?:
        # Domain-style URL without protocol (must have recognizable TLD)
        (?:www\.)?
        (?:[\w-]+\.)+
        (?:com|org|net|edu|gov|io|co|ai|app|dev|info|biz|us|uk|ca|au|de|fr|es|it|nl|be|ch|at|pl|ru|cn|jp|kr|in|br|mx|ar|cl|nz|za|ie|se|no|dk|fi|pt|cz|hu|ro|gr|tr|il|ae|sg|hk|tw|my|ph|th|vn|id)
        (?:/[^\s<>"']*)?
    )
    """,
    re.VERBOSE | re.IGNORECASE
)

# =============================================================================
# EMAIL PATTERN
# =============================================================================
# Matches standard email formats: user@domain.tld

EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE
)

# =============================================================================
# PHONE PATTERN (US/International formats)
# =============================================================================
# Matches:
# - US formats: (123) 456-7890, 123-456-7890, 123.456.7890
# - International: +1-123-456-7890, +44 20 7123 4567
# - Simple: 1234567890 (10+ digits)

PHONE_PATTERN = re.compile(
    r"""
    (?:
        # International format with +
        \+\d{1,3}[\s.-]?\d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,9}
    )
    |
    (?:
        # US format with parentheses
        \(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4}
    )
    |
    (?:
        # US format without parentheses
        \d{3}[\s.-]\d{3}[\s.-]\d{4}
    )
    """,
    re.VERBOSE
)


@dataclass
class TokenMapping:
    """Stores the mapping between placeholder and original token."""
    placeholder: str
    original: str
    token_type: str  # "url", "email", or "phone"
    start_pos: int  # Position in original text (for ordering)


class LockedTokenProtector:
    """
    Protects sensitive tokens (URLs, emails, phones) during text processing.

    This class provides bidirectional protection:
    - protect(): Replaces sensitive tokens with safe placeholders
    - restore(): Restores original tokens from placeholders

    Thread-safe for single protect/restore cycles.
    """

    # Placeholder format: __URL_0__, __EMAIL_1__, __PHONE_2__
    PLACEHOLDER_PREFIX = "__"
    PLACEHOLDER_SUFFIX = "__"

    def __init__(
        self,
        protect_urls: bool = True,
        protect_emails: bool = True,
        protect_phones: bool = True,
    ):
        """
        Initialize the protector with configurable token types.

        Args:
            protect_urls: Whether to protect URLs.
            protect_emails: Whether to protect email addresses.
            protect_phones: Whether to protect phone numbers.
        """
        self.protect_urls = protect_urls
        self.protect_emails = protect_emails
        self.protect_phones = protect_phones

    def protect(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Replace sensitive tokens with placeholders.

        Args:
            text: The text containing sensitive tokens.

        Returns:
            Tuple of (protected_text, token_map) where token_map
            maps placeholders back to original tokens.
        """
        if not text:
            return text, {}

        token_map: dict[str, str] = {}
        mappings: list[TokenMapping] = []

        # Find all tokens to protect (collect first, replace later to handle overlaps)
        if self.protect_urls:
            for match in URL_PATTERN.finditer(text):
                mappings.append(TokenMapping(
                    placeholder="",  # Will be assigned later
                    original=match.group(0),
                    token_type="URL",
                    start_pos=match.start(),
                ))

        if self.protect_emails:
            for match in EMAIL_PATTERN.finditer(text):
                # Skip if this overlaps with a URL (emails in URLs)
                if not self._overlaps_existing(match.start(), match.end(), mappings):
                    mappings.append(TokenMapping(
                        placeholder="",
                        original=match.group(0),
                        token_type="EMAIL",
                        start_pos=match.start(),
                    ))

        if self.protect_phones:
            for match in PHONE_PATTERN.finditer(text):
                # Skip if this overlaps with existing tokens
                if not self._overlaps_existing(match.start(), match.end(), mappings):
                    mappings.append(TokenMapping(
                        placeholder="",
                        original=match.group(0),
                        token_type="PHONE",
                        start_pos=match.start(),
                    ))

        # Sort by position (process from end to start to preserve indices)
        mappings.sort(key=lambda m: m.start_pos, reverse=True)

        # Assign placeholders and replace in text
        result = text
        counters = {"URL": 0, "EMAIL": 0, "PHONE": 0}

        for mapping in mappings:
            # Create unique placeholder
            placeholder = f"{self.PLACEHOLDER_PREFIX}{mapping.token_type}_{counters[mapping.token_type]}{self.PLACEHOLDER_SUFFIX}"
            counters[mapping.token_type] += 1

            mapping.placeholder = placeholder
            token_map[placeholder] = mapping.original

            # Replace in text (from end to preserve earlier indices)
            result = (
                result[:mapping.start_pos] +
                placeholder +
                result[mapping.start_pos + len(mapping.original):]
            )

        return result, token_map

    def restore(self, text: str, token_map: dict[str, str]) -> str:
        """
        Restore original tokens from placeholders.

        Args:
            text: Text containing placeholders.
            token_map: Map from placeholders to original tokens.

        Returns:
            Text with original tokens restored.
        """
        if not text or not token_map:
            return text

        result = text

        # Sort by placeholder length (longest first) to handle nested cases
        sorted_placeholders = sorted(token_map.keys(), key=len, reverse=True)

        for placeholder in sorted_placeholders:
            original = token_map[placeholder]
            result = result.replace(placeholder, original)

        return result

    def _overlaps_existing(
        self,
        start: int,
        end: int,
        mappings: list[TokenMapping],
    ) -> bool:
        """Check if a span overlaps with any existing mapping."""
        for m in mappings:
            m_end = m.start_pos + len(m.original)
            # Overlap check: spans overlap if neither is completely before the other
            if not (end <= m.start_pos or start >= m_end):
                return True
        return False


def protect_locked_tokens(
    text: str,
    protect_urls: bool = True,
    protect_emails: bool = True,
    protect_phones: bool = True,
) -> tuple[str, dict[str, str]]:
    """
    Convenience function to protect locked tokens in text.

    Args:
        text: Text to protect.
        protect_urls: Whether to protect URLs.
        protect_emails: Whether to protect emails.
        protect_phones: Whether to protect phones.

    Returns:
        Tuple of (protected_text, token_map).
    """
    protector = LockedTokenProtector(
        protect_urls=protect_urls,
        protect_emails=protect_emails,
        protect_phones=protect_phones,
    )
    return protector.protect(text)


def restore_locked_tokens(text: str, token_map: dict[str, str]) -> str:
    """
    Convenience function to restore locked tokens in text.

    Args:
        text: Text with placeholders.
        token_map: Map from placeholders to original tokens.

    Returns:
        Text with original tokens restored.
    """
    protector = LockedTokenProtector()
    return protector.restore(text, token_map)


def protect_and_process(
    original: str,
    optimized: str,
    process_fn,
    **kwargs,
) -> str:
    """
    Protect tokens in both texts, process, then restore.

    This is the main entry point for protected processing.
    It ensures URLs/emails/phones are preserved through any transformation.

    Args:
        original: Original text (for reference).
        optimized: Optimized text to process.
        process_fn: Function to call with (protected_original, protected_optimized).
                   Should return processed text.
        **kwargs: Additional arguments to pass to process_fn.

    Returns:
        Processed text with original tokens restored.
    """
    protector = LockedTokenProtector()

    # Protect both texts
    protected_original, orig_map = protector.protect(original)
    protected_optimized, opt_map = protector.protect(optimized)

    # Merge token maps (prefer original's tokens for restoration)
    combined_map = {**opt_map, **orig_map}

    # Process
    result = process_fn(protected_original, protected_optimized, **kwargs)

    # Restore
    return protector.restore(result, combined_map)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_urls_preserved(original: str, processed: str) -> list[str]:
    """
    Check that all URLs from original appear in processed text.

    Args:
        original: Original text with URLs.
        processed: Processed text that should contain same URLs.

    Returns:
        List of URLs that were lost or corrupted. Empty list means success.
    """
    if not original:
        return []

    original_urls = set(URL_PATTERN.findall(original))
    processed_urls = set(URL_PATTERN.findall(processed))

    lost_urls = []
    for url in original_urls:
        # Check if URL exists exactly or in slightly modified form
        if url not in processed_urls:
            # Check for common corruptions
            normalized_url = url.lower().strip()
            found = False
            for p_url in processed_urls:
                if p_url.lower().strip() == normalized_url:
                    found = True
                    break
            if not found:
                lost_urls.append(url)

    return lost_urls


def detect_url_corruption(text: str) -> list[tuple[str, str]]:
    """
    Detect common URL corruption patterns.

    Looks for patterns like:
    - "Fortell. com" (space before TLD)
    - "example .com" (space before dot)
    - "HTTPS://Example.Com" (case changes that might affect display)

    Args:
        text: Text to check for URL corruptions.

    Returns:
        List of (corrupted_pattern, description) tuples.
    """
    corruptions = []

    # Pattern: space before TLD
    space_before_tld = re.findall(
        r'\b[\w-]+\s+\.(?:com|org|net|edu|gov|io|co)\b',
        text,
        re.IGNORECASE
    )
    for match in space_before_tld:
        corruptions.append((match, "Space inserted before TLD"))

    # Pattern: capitalized domain that shouldn't be
    cap_domain = re.findall(
        r'(?:Https?|HTTPS?)://[A-Z][a-z]+\.',
        text
    )
    for match in cap_domain:
        corruptions.append((match, "Unexpected capitalization in URL"))

    # Pattern: period + space in middle of URL
    period_space = re.findall(
        r'[\w-]+\.\s+[\w-]+/[\w/-]*',
        text
    )
    for match in period_space:
        corruptions.append((match, "Space after period in URL path"))

    return corruptions
