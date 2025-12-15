"""
Token-level diff highlighter for SEO Content Optimizer V2 Architecture.

This module provides precise word-level diff highlighting:
- Computes minimal edit distance between original and modified text
- Identifies exact spans of inserted/deleted/changed words
- Generates highlight spans for DOCX output
- Supports semantic diff (knows when keywords were injected vs other changes)

The goal is to show users exactly what changed at the word level.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class DiffType(Enum):
    """Type of diff operation."""
    EQUAL = "equal"
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"


@dataclass
class DiffSpan:
    """A span of text with diff information."""
    text: str
    diff_type: DiffType
    start_pos: int = 0  # Character position in output
    end_pos: int = 0
    original_text: Optional[str] = None  # For replacements, what was replaced
    is_keyword: bool = False  # Whether this is a keyword injection


@dataclass
class DiffResult:
    """Result of diff comparison between two texts."""
    original: str
    modified: str
    spans: list[DiffSpan]
    total_insertions: int = 0
    total_deletions: int = 0
    total_replacements: int = 0
    keywords_injected: list[str] = field(default_factory=list)


@dataclass
class HighlightSpan:
    """A span to be highlighted in output document."""
    start: int
    end: int
    text: str
    highlight_type: str  # "keyword", "insertion", "change"
    color: str = "green"  # Default to green


class TokenDiffer:
    """
    Computes token-level diffs between original and modified text.

    Uses difflib's SequenceMatcher for accurate alignment, then
    identifies word-level changes.
    """

    def __init__(self, keywords: Optional[list[str]] = None):
        """
        Initialize token differ.

        Args:
            keywords: Optional list of keywords to track injections.
        """
        self.keywords = set(kw.lower() for kw in (keywords or []))
        self._keyword_patterns = self._build_keyword_patterns(keywords or [])

    def diff(self, original: str, modified: str) -> DiffResult:
        """
        Compute token-level diff between original and modified text.

        Args:
            original: Original text.
            modified: Modified text.

        Returns:
            DiffResult with all diff spans.
        """
        # Tokenize both texts
        original_tokens = self._tokenize(original)
        modified_tokens = self._tokenize(modified)

        # Use SequenceMatcher for alignment
        matcher = SequenceMatcher(None, original_tokens, modified_tokens)
        opcodes = matcher.get_opcodes()

        spans = []
        current_pos = 0
        total_insertions = 0
        total_deletions = 0
        total_replacements = 0
        keywords_found = []

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                # Unchanged text
                text = ' '.join(modified_tokens[j1:j2])
                if text:
                    spans.append(DiffSpan(
                        text=text,
                        diff_type=DiffType.EQUAL,
                        start_pos=current_pos,
                        end_pos=current_pos + len(text),
                    ))
                    current_pos += len(text) + 1  # +1 for space

            elif tag == 'insert':
                # New text inserted
                text = ' '.join(modified_tokens[j1:j2])
                if text:
                    is_kw = self._is_keyword_text(text)
                    if is_kw:
                        keywords_found.append(text)

                    spans.append(DiffSpan(
                        text=text,
                        diff_type=DiffType.INSERT,
                        start_pos=current_pos,
                        end_pos=current_pos + len(text),
                        is_keyword=is_kw,
                    ))
                    current_pos += len(text) + 1
                    total_insertions += j2 - j1

            elif tag == 'delete':
                # Text deleted (we track but don't include in output)
                deleted_text = ' '.join(original_tokens[i1:i2])
                total_deletions += i2 - i1
                # Note: deleted spans don't appear in output, but we track them

            elif tag == 'replace':
                # Text replaced
                original_text = ' '.join(original_tokens[i1:i2])
                new_text = ' '.join(modified_tokens[j1:j2])

                if new_text:
                    is_kw = self._is_keyword_text(new_text)
                    if is_kw:
                        keywords_found.append(new_text)

                    spans.append(DiffSpan(
                        text=new_text,
                        diff_type=DiffType.REPLACE,
                        start_pos=current_pos,
                        end_pos=current_pos + len(new_text),
                        original_text=original_text,
                        is_keyword=is_kw,
                    ))
                    current_pos += len(new_text) + 1
                    total_replacements += max(i2 - i1, j2 - j1)

        return DiffResult(
            original=original,
            modified=modified,
            spans=spans,
            total_insertions=total_insertions,
            total_deletions=total_deletions,
            total_replacements=total_replacements,
            keywords_injected=keywords_found,
        )

    def get_highlight_spans(
        self,
        diff_result: DiffResult,
        include_equal: bool = False,
    ) -> list[HighlightSpan]:
        """
        Extract spans that should be highlighted in output.

        Args:
            diff_result: Result from diff().
            include_equal: If True, include unchanged spans too.

        Returns:
            List of HighlightSpan for document highlighting.
        """
        highlights = []

        for span in diff_result.spans:
            if span.diff_type == DiffType.EQUAL:
                if include_equal:
                    highlights.append(HighlightSpan(
                        start=span.start_pos,
                        end=span.end_pos,
                        text=span.text,
                        highlight_type="unchanged",
                        color="none",
                    ))
                continue

            # Determine highlight color/type
            if span.is_keyword:
                highlight_type = "keyword"
                color = "green"
            elif span.diff_type == DiffType.INSERT:
                highlight_type = "insertion"
                color = "lightgreen"
            elif span.diff_type == DiffType.REPLACE:
                highlight_type = "change"
                color = "yellow"
            else:
                highlight_type = "other"
                color = "gray"

            highlights.append(HighlightSpan(
                start=span.start_pos,
                end=span.end_pos,
                text=span.text,
                highlight_type=highlight_type,
                color=color,
            ))

        return highlights

    def find_keyword_positions(
        self,
        text: str,
    ) -> list[tuple[int, int, str]]:
        """
        Find all keyword positions in text.

        Args:
            text: Text to search.

        Returns:
            List of (start, end, keyword) tuples.
        """
        positions = []
        text_lower = text.lower()

        for kw in self.keywords:
            # Find all occurrences
            pattern = r'\b' + re.escape(kw) + r'\b'
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                # Get actual text from original (preserve case)
                actual_text = text[match.start():match.end()]
                positions.append((match.start(), match.end(), actual_text))

        # Sort by position and remove overlaps
        positions.sort(key=lambda x: x[0])
        non_overlapping = []
        last_end = -1
        for start, end, kw in positions:
            if start >= last_end:
                non_overlapping.append((start, end, kw))
                last_end = end

        return non_overlapping

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        # Split on whitespace but preserve punctuation attached to words
        tokens = re.findall(r'\S+', text)
        return tokens

    def _is_keyword_text(self, text: str) -> bool:
        """Check if text contains a keyword."""
        text_lower = text.lower()

        for pattern in self._keyword_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _build_keyword_patterns(self, keywords: list[str]) -> list[re.Pattern]:
        """Build regex patterns for keyword detection."""
        patterns = []
        for kw in keywords:
            # Create pattern that matches the keyword as whole word
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            patterns.append(re.compile(pattern, re.IGNORECASE))
        return patterns


def compute_diff(
    original: str,
    modified: str,
    keywords: Optional[list[str]] = None,
) -> DiffResult:
    """
    Convenience function for computing diff.

    Args:
        original: Original text.
        modified: Modified text.
        keywords: Optional keywords to track.

    Returns:
        DiffResult with all changes.
    """
    differ = TokenDiffer(keywords)
    return differ.diff(original, modified)


def get_changes_summary(diff_result: DiffResult) -> dict:
    """
    Generate a summary of changes from diff result.

    Args:
        diff_result: DiffResult from diff operation.

    Returns:
        Dictionary with change summary.
    """
    insertions = [s for s in diff_result.spans if s.diff_type == DiffType.INSERT]
    replacements = [s for s in diff_result.spans if s.diff_type == DiffType.REPLACE]
    keywords = [s for s in diff_result.spans if s.is_keyword]

    return {
        "total_changes": len(insertions) + len(replacements),
        "insertions": len(insertions),
        "replacements": len(replacements),
        "keywords_injected": len(keywords),
        "keyword_texts": diff_result.keywords_injected,
        "insertion_texts": [s.text for s in insertions],
        "replacement_texts": [
            {"old": s.original_text, "new": s.text}
            for s in replacements
        ],
    }


def highlight_text_html(
    diff_result: DiffResult,
    keyword_color: str = "#90EE90",  # Light green
    change_color: str = "#FFFF00",  # Yellow
) -> str:
    """
    Generate HTML with highlighted changes.

    Args:
        diff_result: DiffResult from diff operation.
        keyword_color: Color for keyword highlights.
        change_color: Color for other changes.

    Returns:
        HTML string with highlighted spans.
    """
    parts = []

    for span in diff_result.spans:
        if span.diff_type == DiffType.EQUAL:
            parts.append(span.text)
        elif span.is_keyword:
            parts.append(f'<span style="background-color:{keyword_color}">{span.text}</span>')
        elif span.diff_type in (DiffType.INSERT, DiffType.REPLACE):
            parts.append(f'<span style="background-color:{change_color}">{span.text}</span>')

    return ' '.join(parts)


def find_new_keywords_in_text(
    original: str,
    modified: str,
    keywords: list[str],
) -> list[str]:
    """
    Find keywords that appear in modified but not in original.

    Args:
        original: Original text.
        modified: Modified text.
        keywords: Keywords to search for.

    Returns:
        List of keywords that were newly injected.
    """
    original_lower = original.lower()
    modified_lower = modified.lower()

    new_keywords = []
    for kw in keywords:
        kw_lower = kw.lower()
        pattern = r'\b' + re.escape(kw_lower) + r'\b'

        in_original = bool(re.search(pattern, original_lower))
        in_modified = bool(re.search(pattern, modified_lower))

        if in_modified and not in_original:
            new_keywords.append(kw)

    return new_keywords
