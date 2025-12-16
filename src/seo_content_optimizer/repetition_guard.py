"""
Repetition and keyword stuffing prevention module.

Prevents:
- Exact duplicate sentences
- Near-duplicate sentences (semantic similarity)
- Keyword stuffing (too many mentions clustered together)
- Forced/unnatural keyword placement

Implements density-based keyword targets instead of fixed counts.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from difflib import SequenceMatcher


@dataclass
class RepetitionIssue:
    """A repetition issue found in content."""
    issue_type: str  # "duplicate", "near_duplicate", "keyword_cluster", "keyword_stuffing"
    text: str
    location: int  # Approximate character position
    severity: str  # "high", "medium", "low"
    recommendation: str


@dataclass
class KeywordDensityConfig:
    """
    Keyword density configuration (replaces fixed counts).

    Density-based approach:
    - Primary keyword: 0.5-1.5% density (1 per 67-200 words)
    - Secondary keywords: 0.2-0.8% density (1 per 125-500 words)
    """
    primary_min_density: float = 0.005  # 0.5%
    primary_max_density: float = 0.015  # 1.5%
    secondary_min_density: float = 0.002  # 0.2%
    secondary_max_density: float = 0.008  # 0.8%
    min_words_between_same_keyword: int = 50  # Minimum words between repetitions
    max_keywords_per_paragraph: int = 2  # Don't stuff multiple keywords in one paragraph

    def get_target_count(self, word_count: int, is_primary: bool = True) -> tuple[int, int]:
        """
        Get min/max target count based on word count.

        Args:
            word_count: Total document word count.
            is_primary: Whether this is the primary keyword.

        Returns:
            Tuple of (min_count, max_count).
        """
        if is_primary:
            min_count = max(1, int(word_count * self.primary_min_density))
            max_count = max(min_count + 1, int(word_count * self.primary_max_density))
        else:
            min_count = max(1, int(word_count * self.secondary_min_density))
            max_count = max(min_count, int(word_count * self.secondary_max_density))

        return min_count, max_count


def find_duplicate_sentences(text: str) -> list[RepetitionIssue]:
    """
    Find exact duplicate sentences in text.

    Args:
        text: Content to check.

    Returns:
        List of repetition issues for duplicates.
    """
    issues = []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Normalize and track sentences
    seen: dict[str, int] = {}  # normalized -> first position

    for i, sentence in enumerate(sentences):
        if len(sentence) < 10:  # Skip very short sentences
            continue

        # Normalize for comparison
        normalized = _normalize_for_comparison(sentence)

        if normalized in seen:
            issues.append(RepetitionIssue(
                issue_type="duplicate",
                text=sentence[:100] + "..." if len(sentence) > 100 else sentence,
                location=i,
                severity="high",
                recommendation=f"Remove duplicate sentence (first seen at position {seen[normalized]})",
            ))
        else:
            seen[normalized] = i

    return issues


def find_near_duplicate_sentences(text: str, threshold: float = 0.85) -> list[RepetitionIssue]:
    """
    Find near-duplicate sentences (high semantic similarity).

    Args:
        text: Content to check.
        threshold: Similarity threshold (0-1).

    Returns:
        List of repetition issues for near-duplicates.
    """
    issues = []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if len(s) > 20]  # Filter short sentences

    # Compare each pair (O(n^2) but sentences list is usually small)
    for i, sent1 in enumerate(sentences):
        for j in range(i + 1, min(i + 10, len(sentences))):  # Only check nearby sentences
            sent2 = sentences[j]

            similarity = _sentence_similarity(sent1, sent2)

            if similarity >= threshold and similarity < 1.0:  # Near but not exact duplicate
                issues.append(RepetitionIssue(
                    issue_type="near_duplicate",
                    text=f"'{sent1[:50]}...' similar to '{sent2[:50]}...'",
                    location=i,
                    severity="medium",
                    recommendation=f"Rephrase or remove - {similarity:.0%} similar to nearby sentence",
                ))

    return issues


def find_keyword_clustering(
    text: str,
    keyword: str,
    min_words_between: int = 50,
) -> list[RepetitionIssue]:
    """
    Find instances where a keyword appears too close together.

    Args:
        text: Content to check.
        keyword: Keyword to check for clustering.
        min_words_between: Minimum words required between occurrences.

    Returns:
        List of clustering issues.
    """
    issues = []

    # Find all keyword positions
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches = list(pattern.finditer(text))

    if len(matches) < 2:
        return issues

    # Check distances between occurrences
    for i in range(1, len(matches)):
        prev_match = matches[i - 1]
        curr_match = matches[i]

        # Count words between matches
        between_text = text[prev_match.end():curr_match.start()]
        words_between = len(between_text.split())

        if words_between < min_words_between:
            issues.append(RepetitionIssue(
                issue_type="keyword_cluster",
                text=f"'{keyword}' appears {words_between} words apart (minimum: {min_words_between})",
                location=curr_match.start(),
                severity="medium",
                recommendation=f"Spread keyword occurrences further apart",
            ))

    return issues


def check_keyword_density(
    text: str,
    keyword: str,
    is_primary: bool = True,
    config: Optional[KeywordDensityConfig] = None,
) -> tuple[int, int, int, str]:
    """
    Check if keyword density is within acceptable range.

    Args:
        text: Content to check.
        keyword: Keyword to count.
        is_primary: Whether this is the primary keyword.
        config: Density configuration.

    Returns:
        Tuple of (actual_count, min_target, max_target, status).
        Status is "under", "optimal", or "over".
    """
    if config is None:
        config = KeywordDensityConfig()

    word_count = len(text.split())
    min_target, max_target = config.get_target_count(word_count, is_primary)

    # Count keyword occurrences
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    actual_count = len(pattern.findall(text))

    if actual_count < min_target:
        status = "under"
    elif actual_count > max_target:
        status = "over"
    else:
        status = "optimal"

    return actual_count, min_target, max_target, status


def remove_duplicate_sentences(text: str) -> str:
    """
    Remove exact duplicate sentences from text.

    Args:
        text: Content to clean.

    Returns:
        Text with duplicates removed.
    """
    # Split into sentences while preserving delimiters
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Track seen sentences (normalized)
    seen: set[str] = set()
    unique_sentences = []

    for sentence in sentences:
        if len(sentence) < 10:
            unique_sentences.append(sentence)
            continue

        normalized = _normalize_for_comparison(sentence)

        if normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(sentence)

    return " ".join(unique_sentences)


def reduce_keyword_clustering(
    text: str,
    keyword: str,
    min_words_between: int = 50,
) -> str:
    """
    Reduce keyword clustering by removing excess occurrences.

    Only removes occurrences that are too close together,
    keeping the first occurrence in each cluster.

    Args:
        text: Content to clean.
        keyword: Keyword to check.
        min_words_between: Minimum words required between occurrences.

    Returns:
        Text with reduced clustering.
    """
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches = list(pattern.finditer(text))

    if len(matches) < 2:
        return text

    # Find positions to remove
    positions_to_remove = []
    last_kept_end = matches[0].end()

    for i in range(1, len(matches)):
        curr_match = matches[i]
        between_text = text[last_kept_end:curr_match.start()]
        words_between = len(between_text.split())

        if words_between < min_words_between:
            positions_to_remove.append((curr_match.start(), curr_match.end()))
        else:
            last_kept_end = curr_match.end()

    # Remove marked positions (in reverse to preserve indices)
    result = text
    for start, end in reversed(positions_to_remove):
        # Replace with a period if in middle of sentence, or just remove
        before = result[:start].rstrip()
        after = result[end:].lstrip()

        # Try to make the removal grammatical
        if before.endswith((",", "and", "or")):
            before = before[:-1].rstrip() if before.endswith(",") else before[:-3].rstrip()

        result = before + " " + after

    # Clean up spacing
    result = re.sub(r" +", " ", result)

    return result


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase, remove extra spaces, remove punctuation
    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def _sentence_similarity(sent1: str, sent2: str) -> float:
    """
    Calculate similarity between two sentences.

    Uses SequenceMatcher for quick comparison.
    For production, consider using embeddings.
    """
    norm1 = _normalize_for_comparison(sent1)
    norm2 = _normalize_for_comparison(sent2)

    return SequenceMatcher(None, norm1, norm2).ratio()


def validate_content_repetition(text: str, keywords: list[str]) -> list[RepetitionIssue]:
    """
    Full repetition validation pipeline.

    Args:
        text: Content to validate.
        keywords: Keywords to check for clustering.

    Returns:
        List of all repetition issues found.
    """
    issues = []

    # Check for duplicate sentences
    issues.extend(find_duplicate_sentences(text))

    # Check for near-duplicate sentences
    issues.extend(find_near_duplicate_sentences(text))

    # Check keyword clustering for each keyword
    for keyword in keywords:
        issues.extend(find_keyword_clustering(text, keyword))

    return issues


def clean_repetition(text: str, keywords: list[str]) -> str:
    """
    Clean repetition issues from text.

    Args:
        text: Content to clean.
        keywords: Keywords to de-cluster.

    Returns:
        Cleaned text.
    """
    result = text

    # Remove exact duplicates
    result = remove_duplicate_sentences(result)

    # Reduce keyword clustering
    for keyword in keywords:
        result = reduce_keyword_clustering(result, keyword)

    return result
