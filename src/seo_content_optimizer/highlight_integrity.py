# -*- coding: utf-8 -*-
"""
Highlight Integrity Validation for Insert-Only Mode.

This module validates that highlighting is accurate:
1. Unhighlighted text must exist verbatim in source
2. URLs/emails/phones must be preserved
3. Markers must be balanced
4. Added text must be genuinely new

These checks catch cases where:
- Text was changed but not highlighted (false black)
- Text was highlighted but wasn't actually changed (false green)
- URLs were corrupted during processing
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .diff_markers import (
    MARK_START,
    MARK_END,
    strip_markers,
    normalize_sentence,
)
from .locked_tokens import validate_urls_preserved, detect_url_corruption


@dataclass
class HighlightIssue:
    """Represents a single highlight integrity issue."""
    severity: str  # "error" or "warning"
    category: str  # "unhighlighted_mismatch", "url_corruption", "marker_imbalance", etc.
    description: str
    span_text: Optional[str] = None  # The problematic text span
    expected: Optional[str] = None  # What was expected
    actual: Optional[str] = None  # What was found


@dataclass
class HighlightIntegrityReport:
    """Complete report of highlight integrity validation."""
    is_valid: bool
    issues: list[HighlightIssue] = field(default_factory=list)

    # Summary flags
    unhighlighted_valid: bool = True
    urls_preserved: bool = True
    markers_balanced: bool = True
    no_false_additions: bool = True

    # Statistics
    total_unhighlighted_spans: int = 0
    total_highlighted_spans: int = 0
    mismatched_spans: int = 0

    def add_error(self, category: str, description: str, **kwargs):
        """Add an error issue."""
        self.issues.append(HighlightIssue(
            severity="error",
            category=category,
            description=description,
            **kwargs,
        ))
        self.is_valid = False

    def add_warning(self, category: str, description: str, **kwargs):
        """Add a warning issue."""
        self.issues.append(HighlightIssue(
            severity="warning",
            category=category,
            description=description,
            **kwargs,
        ))

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.is_valid:
            return f"Highlight integrity OK: {self.total_highlighted_spans} highlights, {self.total_unhighlighted_spans} unchanged spans"

        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]

        lines = [f"Highlight integrity FAILED: {len(errors)} errors, {len(warnings)} warnings"]
        for issue in errors[:5]:  # Show first 5 errors
            lines.append(f"  ERROR: {issue.description}")
        if len(errors) > 5:
            lines.append(f"  ... and {len(errors) - 5} more errors")

        return "\n".join(lines)


def extract_unhighlighted_spans(marked_text: str) -> list[str]:
    """
    Extract all unhighlighted (non-marked) text spans.

    Args:
        marked_text: Text with [[[ADD]]]/[[[ENDADD]]] markers.

    Returns:
        List of text spans that are NOT inside markers.
    """
    if not marked_text:
        return []

    result = []
    current_pos = 0
    text = marked_text

    while True:
        start_pos = text.find(MARK_START, current_pos)

        if start_pos == -1:
            remaining = text[current_pos:].strip()
            if remaining:
                result.append(remaining)
            break

        span = text[current_pos:start_pos].strip()
        if span:
            result.append(span)

        end_pos = text.find(MARK_END, start_pos)
        if end_pos == -1:
            break

        current_pos = end_pos + len(MARK_END)

    return result


def extract_highlighted_spans(marked_text: str) -> list[str]:
    """
    Extract all highlighted (marked) text spans.

    Args:
        marked_text: Text with [[[ADD]]]/[[[ENDADD]]] markers.

    Returns:
        List of text spans that ARE inside markers.
    """
    if not marked_text or MARK_START not in marked_text:
        return []

    pattern = re.compile(
        rf"{re.escape(MARK_START)}(.*?){re.escape(MARK_END)}",
        re.DOTALL
    )

    return [match.group(1) for match in pattern.finditer(marked_text)]


def validate_unhighlighted_in_source(
    original: str,
    marked_output: str,
    strict: bool = False,
) -> tuple[bool, list[tuple[str, str]]]:
    """
    Validate that unhighlighted spans exist in original source.

    Args:
        original: Original source text.
        marked_output: Output text with markers.
        strict: If True, require exact substring match.
                If False, use normalized matching.

    Returns:
        Tuple of (all_valid, mismatches) where mismatches is
        list of (span, reason) tuples.
    """
    if not original or not marked_output:
        return True, []

    unhighlighted = extract_unhighlighted_spans(marked_output)
    mismatches = []

    # Normalize original for comparison
    original_normalized = normalize_sentence(original)
    original_words = set(original_normalized.split())

    for span in unhighlighted:
        if not span.strip():
            continue

        if strict:
            # Exact substring match
            if span not in original:
                mismatches.append((span, "Not found as exact substring"))
        else:
            # Normalized word-based matching
            span_normalized = normalize_sentence(span)
            span_words = span_normalized.split()

            # Check if all significant words exist in original
            # Skip very short spans (single words likely exist)
            if len(span_words) >= 3:
                missing_words = [w for w in span_words if w not in original_words and len(w) > 2]
                # Allow small number of missing words (could be articles, etc.)
                if len(missing_words) > len(span_words) * 0.3:
                    mismatches.append((span[:60] + "...", f"Words not in source: {missing_words[:3]}"))

    return len(mismatches) == 0, mismatches


def validate_highlighted_is_new(
    original: str,
    marked_output: str,
) -> tuple[bool, list[str]]:
    """
    Validate that highlighted spans are actually new/changed.

    This catches false positives where unchanged text was marked.

    Args:
        original: Original source text.
        marked_output: Output text with markers.

    Returns:
        Tuple of (all_valid, false_highlights) where false_highlights
        are spans that exist verbatim in original.
    """
    if not original or not marked_output:
        return True, []

    highlighted = extract_highlighted_spans(marked_output)
    false_highlights = []

    original_normalized = normalize_sentence(original)

    for span in highlighted:
        if not span.strip():
            continue

        # Check if this span exists as-is in original
        span_normalized = normalize_sentence(span)

        # If the normalized span is found in original, it's a false highlight
        if span_normalized in original_normalized:
            # Exception: very short spans (1-2 words) might legitimately repeat
            if len(span.split()) > 2:
                false_highlights.append(span[:60] + "...")

    return len(false_highlights) == 0, false_highlights


def validate_marker_balance(text: str) -> tuple[bool, int, int]:
    """
    Check that markers are properly balanced.

    Args:
        text: Text with markers.

    Returns:
        Tuple of (is_balanced, start_count, end_count).
    """
    start_count = text.count(MARK_START)
    end_count = text.count(MARK_END)
    return start_count == end_count, start_count, end_count


def validate_no_nested_markers(text: str) -> tuple[bool, list[str]]:
    """
    Check for improperly nested markers.

    Args:
        text: Text with markers.

    Returns:
        Tuple of (no_nesting, nested_examples).
    """
    # Pattern: MARK_START followed by another MARK_START before MARK_END
    pattern = re.compile(
        rf"{re.escape(MARK_START)}[^{re.escape(MARK_END[0])}]*{re.escape(MARK_START)}"
    )

    matches = pattern.findall(text)
    return len(matches) == 0, matches[:3]


def run_highlight_integrity_check(
    original: str,
    marked_output: str,
    strict: bool = False,
) -> HighlightIntegrityReport:
    """
    Run complete highlight integrity validation.

    This is the main entry point for highlight validation.

    Checks performed:
    1. All unhighlighted text exists in original
    2. Highlighted text is actually new (no false positives)
    3. URLs are preserved
    4. Markers are balanced
    5. No nested markers

    Args:
        original: Original source text.
        marked_output: Output text with markers.
        strict: If True, use strict matching for unhighlighted spans.

    Returns:
        HighlightIntegrityReport with all findings.
    """
    report = HighlightIntegrityReport(is_valid=True)

    # Count spans
    report.total_unhighlighted_spans = len(extract_unhighlighted_spans(marked_output))
    report.total_highlighted_spans = len(extract_highlighted_spans(marked_output))

    # Check 1: Unhighlighted text exists in source
    valid, mismatches = validate_unhighlighted_in_source(original, marked_output, strict)
    if not valid:
        report.unhighlighted_valid = False
        report.mismatched_spans = len(mismatches)
        for span, reason in mismatches:
            report.add_error(
                "unhighlighted_mismatch",
                f"Unhighlighted text not in source: {reason}",
                span_text=span,
            )

    # Check 2: Highlighted text is actually new
    valid, false_highlights = validate_highlighted_is_new(original, marked_output)
    if not valid:
        report.no_false_additions = False
        for span in false_highlights:
            report.add_warning(
                "false_highlight",
                "Text highlighted but exists in original",
                span_text=span,
            )

    # Check 3: URL preservation
    stripped = strip_markers(marked_output)
    lost_urls = validate_urls_preserved(original, stripped)
    if lost_urls:
        report.urls_preserved = False
        for url in lost_urls:
            report.add_error(
                "url_lost",
                f"URL lost or corrupted: {url}",
                expected=url,
            )

    # Check 3b: URL corruption patterns
    corruptions = detect_url_corruption(stripped)
    if corruptions:
        report.urls_preserved = False
        for pattern, desc in corruptions:
            report.add_error(
                "url_corruption",
                f"URL corruption detected: {desc}",
                span_text=pattern,
            )

    # Check 4: Marker balance
    balanced, start_count, end_count = validate_marker_balance(marked_output)
    if not balanced:
        report.markers_balanced = False
        report.add_error(
            "marker_imbalance",
            f"Unbalanced markers: {start_count} starts, {end_count} ends",
        )

    # Check 5: Nested markers
    no_nesting, nested = validate_no_nested_markers(marked_output)
    if not no_nesting:
        report.markers_balanced = False
        report.add_error(
            "marker_nesting",
            f"Nested markers detected: {len(nested)} occurrences",
        )

    return report


def get_highlight_diff_summary(
    original: str,
    marked_output: str,
) -> dict:
    """
    Get a summary of changes between original and marked output.

    Useful for debugging and understanding what was changed.

    Args:
        original: Original source text.
        marked_output: Output text with markers.

    Returns:
        Dict with summary statistics.
    """
    stripped = strip_markers(marked_output)

    original_words = len(original.split()) if original else 0
    final_words = len(stripped.split()) if stripped else 0

    highlighted_spans = extract_highlighted_spans(marked_output)
    highlighted_words = sum(len(span.split()) for span in highlighted_spans)

    unhighlighted_spans = extract_unhighlighted_spans(marked_output)
    unhighlighted_words = sum(len(span.split()) for span in unhighlighted_spans)

    return {
        "original_word_count": original_words,
        "final_word_count": final_words,
        "words_added": max(0, final_words - original_words),
        "highlighted_span_count": len(highlighted_spans),
        "highlighted_word_count": highlighted_words,
        "unhighlighted_span_count": len(unhighlighted_spans),
        "unhighlighted_word_count": unhighlighted_words,
        "change_ratio": highlighted_words / max(1, final_words),
    }
