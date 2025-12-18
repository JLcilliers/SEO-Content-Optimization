# -*- coding: utf-8 -*-
"""
Post-optimization enforcement for insert-only mode.

This module enforces strict constraints after all insertions:
1. Keyword caps - Remove excess keyword occurrences (inserted ones first)
2. Budget limits - Drop insertions if over max sentences/words
3. Relevance check - Ensure added text contains target keywords

These constraints prevent keyword stuffing and ensure minimal, targeted changes.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .config import OptimizationConfig
from .diff_markers import MARK_START, MARK_END, strip_markers


@dataclass
class MarkerSpan:
    """Represents a marked (inserted) span of text."""
    start: int  # Start position in marked text
    end: int  # End position in marked text (after MARK_END)
    content: str  # The text content inside markers
    word_count: int  # Number of words in content
    sentence_count: int  # Number of sentences in content
    keywords_found: set[str]  # Keywords found in this span


@dataclass
class DeltaBudgetResult:
    """Result of delta budget enforcement for a single keyword."""
    keyword: str
    source_count: int  # Count in original source (before optimization)
    final_count: int  # Count after optimization
    new_additions: int  # final_count - source_count
    allowed_new: int  # Maximum allowed new additions
    within_budget: bool  # new_additions <= allowed_new
    removed_count: int  # How many were removed by enforcement


@dataclass
class EnforcementResult:
    """Result of enforcement pass."""
    text: str  # Enforced text
    caps_removed: int  # Number of keyword occurrences removed for caps
    budget_removed_spans: int  # Number of spans removed for budget
    budget_removed_words: int  # Words removed for budget
    budget_removed_sentences: int  # Sentences removed for budget
    warnings: list[str]  # Any warnings generated
    original_keyword_counts: dict[str, int]  # Counts before enforcement
    final_keyword_counts: dict[str, int]  # Counts after enforcement
    # Delta budget tracking
    delta_budget_results: list[DeltaBudgetResult] = None  # Results per keyword
    delta_removed: int = 0  # Total removed for delta budget enforcement


def count_sentences(text: str) -> int:
    """
    Count sentences in text.

    Args:
        text: Text to count sentences in.

    Returns:
        Number of sentences.
    """
    if not text:
        return 0
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings
    return len([s for s in sentences if s.strip()])


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Text to count words in.

    Returns:
        Number of words.
    """
    if not text:
        return 0
    return len(text.split())


def extract_marker_spans(text: str) -> list[MarkerSpan]:
    """
    Extract all marker spans from text with their positions.

    Args:
        text: Text with [[[ADD]]]/[[[ENDADD]]] markers.

    Returns:
        List of MarkerSpan objects with positions and content.
    """
    if not text or MARK_START not in text:
        return []

    spans = []
    pattern = re.compile(
        rf"({re.escape(MARK_START)})(.*?)({re.escape(MARK_END)})",
        re.DOTALL
    )

    for match in pattern.finditer(text):
        content = match.group(2)
        spans.append(MarkerSpan(
            start=match.start(),
            end=match.end(),
            content=content,
            word_count=count_words(content),
            sentence_count=count_sentences(content),
            keywords_found=set(),  # Populated later
        ))

    return spans


def find_keywords_in_text(text: str, keywords: list[str]) -> dict[str, int]:
    """
    Count occurrences of each keyword in text.

    Uses case-insensitive whole-word matching.

    Args:
        text: Text to search.
        keywords: List of keywords to find.

    Returns:
        Dict mapping keyword to occurrence count.
    """
    counts = {}
    text_lower = text.lower()

    for keyword in keywords:
        if not keyword:
            continue
        kw_lower = keyword.lower().strip()
        # Use word boundary matching for accurate counts
        pattern = re.compile(rf'\b{re.escape(kw_lower)}\b', re.IGNORECASE)
        matches = pattern.findall(text_lower)
        counts[kw_lower] = len(matches)

    return counts


def find_keyword_positions(text: str, keyword: str) -> list[tuple[int, int]]:
    """
    Find all positions of a keyword in text.

    Args:
        text: Text to search.
        keyword: Keyword to find.

    Returns:
        List of (start, end) position tuples.
    """
    if not text or not keyword:
        return []

    positions = []
    pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)

    for match in pattern.finditer(text):
        positions.append((match.start(), match.end()))

    return positions


def is_position_in_marker(pos: int, marker_spans: list[MarkerSpan]) -> Optional[MarkerSpan]:
    """
    Check if a position falls within a marker span.

    Args:
        pos: Position to check.
        marker_spans: List of marker spans.

    Returns:
        The MarkerSpan containing the position, or None.
    """
    for span in marker_spans:
        # Account for MARK_START length when checking content position
        content_start = span.start + len(MARK_START)
        content_end = span.end - len(MARK_END)
        if content_start <= pos < content_end:
            return span
    return None


def enforce_keyword_caps(
    text: str,
    keywords: list[str],
    primary_cap: int = 1,
    secondary_cap: int = 1,
    primary_keyword: Optional[str] = None,
) -> tuple[str, int, dict[str, int], dict[str, int]]:
    """
    Enforce keyword occurrence caps by removing excess inserted occurrences.

    Strategy:
    1. Count keyword occurrences in final text
    2. For each over-cap keyword, identify which occurrences are inside markers
    3. Remove inserted occurrences (inside markers) first, from last to first
    4. If still over cap, log warning (original text exceeded cap)

    Args:
        text: Text with markers.
        keywords: All keywords to enforce caps on.
        primary_cap: Max occurrences for primary keyword.
        secondary_cap: Max occurrences for each secondary keyword.
        primary_keyword: The primary keyword (others are secondary).

    Returns:
        Tuple of (enforced_text, removed_count, original_counts, final_counts).
    """
    if not text or not keywords:
        return text, 0, {}, {}

    # Get marker spans
    marker_spans = extract_marker_spans(text)

    # Count original occurrences
    original_counts = find_keywords_in_text(strip_markers(text), keywords)

    result = text
    total_removed = 0

    # Process each keyword
    for keyword in keywords:
        kw_lower = keyword.lower().strip()
        if not kw_lower:
            continue

        # Determine cap for this keyword
        cap = primary_cap if (primary_keyword and kw_lower == primary_keyword.lower()) else secondary_cap

        current_count = original_counts.get(kw_lower, 0)
        if current_count <= cap:
            continue  # Under cap, no action needed

        # Need to remove excess occurrences
        excess = current_count - cap

        # Find all positions of this keyword in current result
        positions = find_keyword_positions(strip_markers(result), kw_lower)

        # Find which positions are inside markers (inserted content)
        # We need to work with the marked text to identify inserted positions
        marked_positions = find_keyword_positions(result, kw_lower)
        current_spans = extract_marker_spans(result)

        inserted_positions = []
        for start, end in marked_positions:
            span = is_position_in_marker(start, current_spans)
            if span:
                inserted_positions.append((start, end, span))

        # Remove inserted occurrences from last to first (to preserve earlier positions)
        inserted_positions.reverse()
        removed_this_keyword = 0

        for start, end, span in inserted_positions:
            if removed_this_keyword >= excess:
                break

            # Remove this occurrence from the text
            # We need to be careful - we're removing from inside a marker span
            # Find the keyword in the span's content and remove it

            # Extract the full span text from result
            full_span = result[span.start:span.end]
            span_content = span.content

            # Find and remove the keyword from span content
            pattern = re.compile(rf'\b{re.escape(kw_lower)}\b', re.IGNORECASE)
            # Only remove one occurrence at a time
            new_content, subs = pattern.subn('', span_content, count=1)

            if subs > 0:
                # Clean up any double spaces from removal
                new_content = re.sub(r' +', ' ', new_content).strip()

                # If span is now empty, remove the whole marker block
                if not new_content:
                    result = result[:span.start] + result[span.end:]
                else:
                    # Replace span content
                    new_span = f"{MARK_START}{new_content}{MARK_END}"
                    result = result[:span.start] + new_span + result[span.end:]

                removed_this_keyword += 1
                total_removed += 1

                # Re-extract spans since positions changed
                current_spans = extract_marker_spans(result)

    # Calculate final counts
    final_counts = find_keywords_in_text(strip_markers(result), keywords)

    return result, total_removed, original_counts, final_counts


def enforce_keyword_delta_budgets(
    text: str,
    keywords: list[str],
    source_counts: dict[str, int],
    allowed_new_primary: int = 1,
    allowed_new_secondary: int = 1,
    primary_keyword: Optional[str] = None,
) -> tuple[str, int, list[DeltaBudgetResult]]:
    """
    Enforce keyword DELTA budgets (not total caps).

    Unlike enforce_keyword_caps which limits total occurrences,
    this enforces that new_additions = final_count - source_count <= allowed_new.

    This allows original content to have any number of keywords,
    but strictly limits how many NEW ones can be added.

    Strategy:
    1. For each keyword, calculate: new_additions = current_count - source_count
    2. If new_additions > allowed_new, remove excess inserted occurrences
    3. Track results per keyword for debugging

    Args:
        text: Text with markers (after optimization).
        keywords: All keywords to enforce budgets on.
        source_counts: Keyword counts from ORIGINAL source (before optimization).
        allowed_new_primary: Max new additions for primary keyword.
        allowed_new_secondary: Max new additions for secondary keywords.
        primary_keyword: The primary keyword (others are secondary).

    Returns:
        Tuple of (enforced_text, total_removed, delta_budget_results).
    """
    if not text or not keywords:
        return text, 0, []

    # Get marker spans for identifying inserted content
    marker_spans = extract_marker_spans(text)

    # Get current counts in the optimized text
    current_counts = find_keywords_in_text(strip_markers(text), keywords)

    result = text
    total_removed = 0
    delta_results = []

    # Process each keyword
    for keyword in keywords:
        kw_lower = keyword.lower().strip()
        if not kw_lower:
            continue

        # Determine allowed_new for this keyword
        is_primary = primary_keyword and kw_lower == primary_keyword.lower()
        allowed_new = allowed_new_primary if is_primary else allowed_new_secondary

        # Calculate delta
        source_count = source_counts.get(kw_lower, 0)
        current_count = current_counts.get(kw_lower, 0)
        new_additions = max(0, current_count - source_count)

        # Check if over budget
        within_budget = new_additions <= allowed_new
        excess = max(0, new_additions - allowed_new)
        removed_for_kw = 0

        if excess > 0:
            # Need to remove excess inserted occurrences
            # Find all positions of this keyword in current result
            marked_positions = find_keyword_positions(result, kw_lower)
            current_spans = extract_marker_spans(result)

            # Find which positions are inside markers (inserted content)
            inserted_positions = []
            for start, end in marked_positions:
                span = is_position_in_marker(start, current_spans)
                if span:
                    inserted_positions.append((start, end, span))

            # Remove inserted occurrences from last to first (to preserve positions)
            inserted_positions.reverse()

            for start, end, span in inserted_positions:
                if removed_for_kw >= excess:
                    break

                # Extract the full span text from result
                full_span = result[span.start:span.end]
                span_content = span.content

                # Find and remove the keyword from span content
                pattern = re.compile(rf'\b{re.escape(kw_lower)}\b', re.IGNORECASE)
                new_content, subs = pattern.subn('', span_content, count=1)

                if subs > 0:
                    # Clean up any double spaces from removal
                    new_content = re.sub(r' +', ' ', new_content).strip()

                    # If span is now empty, remove the whole marker block
                    if not new_content:
                        result = result[:span.start] + result[span.end:]
                    else:
                        # Replace span content
                        new_span = f"{MARK_START}{new_content}{MARK_END}"
                        result = result[:span.start] + new_span + result[span.end:]

                    removed_for_kw += 1
                    total_removed += 1

                    # Re-extract spans since positions changed
                    current_spans = extract_marker_spans(result)

        # Record result for this keyword
        final_count = current_count - removed_for_kw
        delta_results.append(DeltaBudgetResult(
            keyword=kw_lower,
            source_count=source_count,
            final_count=final_count,
            new_additions=max(0, final_count - source_count),
            allowed_new=allowed_new,
            within_budget=max(0, final_count - source_count) <= allowed_new,
            removed_count=removed_for_kw,
        ))

    return result, total_removed, delta_results


def enforce_budget_limits(
    text: str,
    max_sentences: Optional[int] = None,
    max_words: Optional[int] = None,
    keywords: Optional[list[str]] = None,
) -> tuple[str, int, int, int]:
    """
    Enforce insertion budget limits by dropping least important insertions.

    Strategy:
    1. Calculate total new sentences and words from marker spans
    2. If over budget, rank spans by importance (keyword presence, length)
    3. Remove least important spans until under budget

    Args:
        text: Text with markers.
        max_sentences: Maximum new sentences allowed. None = no limit.
        max_words: Maximum new words allowed. None = no limit.
        keywords: Keywords to check for importance ranking.

    Returns:
        Tuple of (enforced_text, spans_removed, words_removed, sentences_removed).
    """
    if not text or MARK_START not in text:
        return text, 0, 0, 0

    # No limits set
    if max_sentences is None and max_words is None:
        return text, 0, 0, 0

    # Extract and analyze spans
    spans = extract_marker_spans(text)
    if not spans:
        return text, 0, 0, 0

    # Populate keywords found in each span
    if keywords:
        keywords_lower = {kw.lower().strip() for kw in keywords if kw}
        for span in spans:
            content_lower = span.content.lower()
            span.keywords_found = {
                kw for kw in keywords_lower
                if re.search(rf'\b{re.escape(kw)}\b', content_lower)
            }

    # Calculate current totals
    total_sentences = sum(s.sentence_count for s in spans)
    total_words = sum(s.word_count for s in spans)

    # Check if over budget
    over_sentences = max_sentences is not None and total_sentences > max_sentences
    over_words = max_words is not None and total_words > max_words

    if not over_sentences and not over_words:
        return text, 0, 0, 0

    # Rank spans by importance (lower = less important = remove first)
    # Importance factors:
    # - Contains keywords: +10 per keyword
    # - Word count: longer spans may be more important (capped)
    def importance_score(span: MarkerSpan) -> float:
        score = len(span.keywords_found) * 10
        score += min(span.word_count, 10)  # Cap contribution from length
        return score

    # Sort spans by importance (ascending = least important first)
    ranked_spans = sorted(spans, key=importance_score)

    # Track removals
    result = text
    spans_removed = 0
    words_removed = 0
    sentences_removed = 0

    # Remove spans until under budget
    for span in ranked_spans:
        # Check current totals
        current_sentences = total_sentences - sentences_removed
        current_words = total_words - words_removed

        still_over_sentences = max_sentences is not None and current_sentences > max_sentences
        still_over_words = max_words is not None and current_words > max_words

        if not still_over_sentences and not still_over_words:
            break

        # Remove this span
        # Find its current position in result (positions shift as we remove)
        pattern = re.compile(
            rf"{re.escape(MARK_START)}{re.escape(span.content)}{re.escape(MARK_END)}",
            re.DOTALL
        )

        match = pattern.search(result)
        if match:
            # Remove the span (keep any surrounding whitespace reasonable)
            before = result[:match.start()]
            after = result[match.end():]

            # Clean up double spaces
            if before.endswith(' ') and after.startswith(' '):
                after = after[1:]

            result = before + after

            spans_removed += 1
            words_removed += span.word_count
            sentences_removed += span.sentence_count

    return result, spans_removed, words_removed, sentences_removed


def validate_insertions_have_keywords(
    text: str,
    keywords: list[str],
    require_all: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validate that inserted content contains target keywords.

    This ensures we're not adding irrelevant fluff text.

    Args:
        text: Text with markers.
        keywords: Keywords to check for.
        require_all: If True, every insertion must contain a keyword.
                    If False, at least one insertion must contain a keyword.

    Returns:
        Tuple of (is_valid, issues_list).
    """
    if not text or MARK_START not in text:
        return True, []

    if not keywords:
        return True, []

    spans = extract_marker_spans(text)
    if not spans:
        return True, []

    keywords_lower = {kw.lower().strip() for kw in keywords if kw}
    issues = []
    any_has_keyword = False

    for i, span in enumerate(spans):
        content_lower = span.content.lower()
        found_keywords = {
            kw for kw in keywords_lower
            if re.search(rf'\b{re.escape(kw)}\b', content_lower)
        }

        if found_keywords:
            any_has_keyword = True
        elif require_all:
            issues.append(f"Span {i+1} has no target keywords: '{span.content[:50]}...'")

    if require_all:
        return len(issues) == 0, issues
    else:
        if not any_has_keyword:
            issues.append("No insertions contain any target keywords")
        return any_has_keyword, issues


def run_enforcement(
    text: str,
    config: OptimizationConfig,
    keywords: list[str],
    primary_keyword: Optional[str] = None,
    source_keyword_counts: Optional[dict[str, int]] = None,
) -> EnforcementResult:
    """
    Run full enforcement pipeline on optimized text.

    This is the main entry point for post-optimization enforcement.

    Enforcement steps (in order):
    1a. Keyword DELTA budget enforcement (if source_counts provided) - PREFERRED
    1b. OR Keyword caps enforcement (legacy fallback)
    2. Budget limits enforcement (if configured)
    3. Insertion relevance validation (warning only)

    The delta budget approach is preferred because it allows original content
    to have any number of keywords while strictly limiting new additions.
    Formula: new_additions = final_count - source_count <= allowed_new

    Args:
        text: Text with markers after optimization.
        config: OptimizationConfig with enforcement settings.
        keywords: All keywords used in optimization.
        primary_keyword: The primary keyword.
        source_keyword_counts: Original keyword counts from source BEFORE optimization.
                              If provided, delta budget enforcement is used instead of caps.

    Returns:
        EnforcementResult with enforced text and statistics.
    """
    result_text = text
    warnings = []
    caps_removed = 0
    budget_removed_spans = 0
    budget_removed_words = 0
    budget_removed_sentences = 0
    original_counts = {}
    final_counts = {}
    delta_budget_results = []
    delta_removed = 0

    # Step 1: Keyword enforcement (prefer delta budgets over absolute caps)
    if config.should_enforce_keyword_caps and keywords:
        if source_keyword_counts is not None:
            # Use DELTA budget enforcement (preferred for insert-only mode)
            # This allows original content to have any count, but limits NEW additions
            result_text, delta_removed, delta_budget_results = enforce_keyword_delta_budgets(
                text=result_text,
                keywords=keywords,
                source_counts=source_keyword_counts,
                allowed_new_primary=config.allowed_new_primary,
                allowed_new_secondary=config.allowed_new_secondary,
                primary_keyword=primary_keyword,
            )

            if delta_removed > 0:
                warnings.append(f"Removed {delta_removed} excess keyword insertions to meet delta budget")

            # Track counts from delta results
            for dr in delta_budget_results:
                original_counts[dr.keyword] = dr.source_count
                final_counts[dr.keyword] = dr.final_count
        else:
            # Legacy caps enforcement (total count <= cap)
            result_text, caps_removed, original_counts, final_counts = enforce_keyword_caps(
                text=result_text,
                keywords=keywords,
                primary_cap=config.primary_keyword_body_cap,
                secondary_cap=config.secondary_keyword_body_cap,
                primary_keyword=primary_keyword,
            )

            if caps_removed > 0:
                warnings.append(f"Removed {caps_removed} excess keyword occurrences to meet caps")

    # Step 2: Budget limits enforcement
    if config.max_new_sentences_total is not None or config.max_new_words_total is not None:
        result_text, budget_removed_spans, budget_removed_words, budget_removed_sentences = enforce_budget_limits(
            text=result_text,
            max_sentences=config.max_new_sentences_total,
            max_words=config.max_new_words_total,
            keywords=keywords,
        )

        if budget_removed_spans > 0:
            warnings.append(
                f"Removed {budget_removed_spans} insertion(s) to meet budget "
                f"({budget_removed_words} words, {budget_removed_sentences} sentences)"
            )

    # Step 3: Insertion relevance validation (warning only)
    if keywords:
        is_valid, issues = validate_insertions_have_keywords(
            result_text,
            keywords,
            require_all=config.is_insert_only_mode,  # Stricter in insert-only mode
        )

        if not is_valid:
            for issue in issues:
                warnings.append(f"Relevance warning: {issue}")

    # Recalculate final counts if not done by caps or delta enforcement
    if not final_counts and keywords:
        final_counts = find_keywords_in_text(strip_markers(result_text), keywords)

    return EnforcementResult(
        text=result_text,
        caps_removed=caps_removed,
        budget_removed_spans=budget_removed_spans,
        budget_removed_words=budget_removed_words,
        budget_removed_sentences=budget_removed_sentences,
        warnings=warnings,
        original_keyword_counts=original_counts,
        final_keyword_counts=final_counts,
        delta_budget_results=delta_budget_results if delta_budget_results else None,
        delta_removed=delta_removed,
    )


def get_enforcement_summary(result: EnforcementResult) -> str:
    """
    Generate a human-readable summary of enforcement actions.

    Args:
        result: EnforcementResult from run_enforcement.

    Returns:
        Multi-line summary string.
    """
    lines = ["Enforcement Summary:"]

    if result.caps_removed > 0:
        lines.append(f"  - Keyword caps: Removed {result.caps_removed} excess occurrences")

    if result.budget_removed_spans > 0:
        lines.append(
            f"  - Budget limits: Removed {result.budget_removed_spans} spans "
            f"({result.budget_removed_words} words, {result.budget_removed_sentences} sentences)"
        )

    if result.warnings:
        lines.append("  Warnings:")
        for warning in result.warnings:
            lines.append(f"    - {warning}")

    if len(lines) == 1:
        lines.append("  - No enforcement actions needed")

    return "\n".join(lines)


# =============================================================================
# STRIP-ADDITIONS VALIDATOR
# =============================================================================


@dataclass
class StripAdditionsResult:
    """Result of strip-additions validation."""
    valid: bool  # True if stripping additions produces original content
    stripped_text: str  # Text after removing all marked spans
    source_text: str  # Original source text for comparison
    differences: list[str]  # List of difference descriptions
    missing_from_stripped: list[str]  # Text segments missing after stripping
    extra_in_stripped: list[str]  # Text segments that shouldn't be there


def strip_marked_additions(text: str) -> str:
    """
    Strip all marked additions from text, leaving only original content.

    Removes everything between [[[ADD]]] and [[[ENDADD]]] markers,
    including the markers themselves.

    Args:
        text: Text with markers.

    Returns:
        Text with all marked spans removed.
    """
    if not text or MARK_START not in text:
        return text

    # Remove all marked spans
    pattern = re.compile(
        rf"{re.escape(MARK_START)}.*?{re.escape(MARK_END)}",
        re.DOTALL
    )
    result = pattern.sub("", text)

    # Clean up any double spaces left behind
    result = re.sub(r"  +", " ", result)

    # Clean up space before punctuation (e.g., " ." -> ".")
    result = re.sub(r"\s+([.!?,;:])", r"\1", result)

    return result.strip()


def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for comparison (collapse whitespace, strip markers).

    Args:
        text: Text to normalize.

    Returns:
        Normalized text suitable for comparison.
    """
    if not text:
        return ""

    # Strip any remaining markers
    text = strip_markers(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def validate_strip_additions(
    optimized_text: str,
    source_text: str,
    strict: bool = False,
) -> StripAdditionsResult:
    """
    Validate that stripping all additions produces the original source content.

    This is a KEY VALIDATION for insert-only mode:
    - All green highlighted text should be ADDITIONS only
    - Removing all green text should restore the original content
    - If this fails, it means text was deleted or modified (not just added)

    Args:
        optimized_text: Text with [[[ADD]]]...[[[ENDADD]]] markers.
        source_text: Original source text before optimization.
        strict: If True, require exact match. If False, allow minor differences.

    Returns:
        StripAdditionsResult with validation details.
    """
    differences = []
    missing_from_stripped = []
    extra_in_stripped = []

    # Strip all additions from optimized text
    stripped = strip_marked_additions(optimized_text)

    # Normalize both texts for comparison
    norm_stripped = normalize_for_comparison(stripped)
    norm_source = normalize_for_comparison(source_text)

    # Check for exact match (normalized)
    if norm_stripped == norm_source:
        return StripAdditionsResult(
            valid=True,
            stripped_text=stripped,
            source_text=source_text,
            differences=[],
            missing_from_stripped=[],
            extra_in_stripped=[],
        )

    # If not exact match, analyze differences
    # Split into sentences for comparison
    source_sentences = set(
        s.strip().lower() for s in re.split(r'[.!?]+', norm_source) if s.strip()
    )
    stripped_sentences = set(
        s.strip().lower() for s in re.split(r'[.!?]+', norm_stripped) if s.strip()
    )

    # Find missing sentences (in source but not in stripped)
    missing = source_sentences - stripped_sentences
    for sent in missing:
        if len(sent) > 10:  # Ignore very short fragments
            missing_from_stripped.append(sent[:100] + "..." if len(sent) > 100 else sent)
            differences.append(f"Missing from stripped: '{sent[:50]}...'")

    # Find extra sentences (in stripped but not in source)
    extra = stripped_sentences - source_sentences
    for sent in extra:
        if len(sent) > 10:  # Ignore very short fragments
            extra_in_stripped.append(sent[:100] + "..." if len(sent) > 100 else sent)
            differences.append(f"Extra in stripped (should not exist): '{sent[:50]}...'")

    # In strict mode, any difference is a failure
    # In lenient mode, allow minor differences (e.g., punctuation changes)
    valid = len(differences) == 0 if strict else len(missing_from_stripped) == 0

    return StripAdditionsResult(
        valid=valid,
        stripped_text=stripped,
        source_text=source_text,
        differences=differences,
        missing_from_stripped=missing_from_stripped,
        extra_in_stripped=extra_in_stripped,
    )


def get_strip_additions_report(result: StripAdditionsResult) -> str:
    """
    Generate a human-readable report from strip-additions validation.

    Args:
        result: StripAdditionsResult from validate_strip_additions.

    Returns:
        Multi-line report string.
    """
    lines = ["Strip-Additions Validation:"]

    if result.valid:
        lines.append("  ✓ VALID: Stripping additions restores original content")
    else:
        lines.append("  ✗ INVALID: Stripping additions does NOT match original")

    if result.missing_from_stripped:
        lines.append(f"  Missing content ({len(result.missing_from_stripped)} segments):")
        for seg in result.missing_from_stripped[:5]:  # Limit to 5
            lines.append(f"    - {seg[:60]}...")

    if result.extra_in_stripped:
        lines.append(f"  Unexpected content ({len(result.extra_in_stripped)} segments):")
        for seg in result.extra_in_stripped[:5]:  # Limit to 5
            lines.append(f"    - {seg[:60]}...")

    return "\n".join(lines)
