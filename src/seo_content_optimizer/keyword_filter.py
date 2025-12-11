"""
Keyword topical relevance filtering.

This module ensures that only keywords that are genuinely on-topic for
the source content are used for optimization. It prevents injection of
off-topic industries, spammy branded phrases, and irrelevant verticals.

The filter works by:
1. Checking if the keyword phrase appears verbatim in the content
2. Measuring lexical overlap between keyword tokens and content tokens
3. Detecting high-risk/off-topic industry terms that should be blocked
   unless they're already in the original content
"""

import re
from dataclasses import dataclass
from typing import Optional

from .models import Keyword


# High-risk industry terms that should NEVER be injected unless already in content
# These are commonly used for keyword stuffing or can create legal/compliance issues
HIGH_RISK_INDUSTRIES = {
    # Adult/gambling/vice - HIGH PRIORITY BLOCK
    "adult", "gambling", "casino", "betting", "poker", "lottery",
    "cannabis", "marijuana", "hemp", "cbd", "thc", "dispensary",
    "tobacco", "vape", "vaping", "e-cigarette", "firearms", "guns",
    "ammunition", "weapons", "alcohol", "liquor", "wine", "beer",
    "escort", "porn", "xxx",

    # Often spammed verticals - BLOCK THESE
    "hair salon", "nail salon", "barbershop", "spa", "massage",
    "tattoo", "piercing", "tanning", "beauty salon", "salon",
    "restaurant", "food truck", "cafe", "coffee shop", "bakery",
    "gym", "fitness", "yoga", "pilates", "crossfit",
    "auto repair", "car wash", "mechanic", "auto body",
    "laundromat", "dry cleaning", "laundry",
    "pet grooming", "veterinary", "vet clinic", "pet store",
    "daycare", "childcare", "preschool",
    "florist", "flower shop",
    "liquor store", "convenience store",
    "pawn shop", "pawnshop",

    # Merchant-type terms often injected wrongly
    "merchant services", "merchant account", "high risk merchant",
    "high-risk merchant", "high risk processing",

    # Financial services that require special licensing
    "forex", "cryptocurrency", "crypto", "bitcoin", "nft",
    "payday loan", "cash advance", "debt collection",
    "insurance", "mortgage", "lending",

    # Healthcare (regulated)
    "pharmacy", "medical", "healthcare", "doctor", "clinic",
    "dental", "dentist", "chiropractor", "therapy",
    "telehealth", "telemedicine",
}

# Single word blocklist - these individual words are ALWAYS blocked unless in content
# These are more aggressive blockers for common spam terms
SINGLE_WORD_BLOCKLIST = {
    "hemp", "cbd", "cannabis", "marijuana", "thc",
    "salon", "spa", "massage", "tattoo",
    "casino", "gambling", "betting",
    "firearm", "firearms", "guns", "weapons",
    "escort", "porn", "adult",
}

# Common stopwords to exclude from token matching
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "whom", "how", "when",
    "where", "why", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "also", "now", "here", "there",
    "your", "our", "their", "my", "his", "her", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "any", "up", "down", "out", "off",
}


@dataclass
class KeywordFilterResult:
    """Result of keyword filtering with explanation."""
    keyword: Keyword
    is_allowed: bool
    reason: str
    overlap_score: float = 0.0


def normalize_tokens(text: str) -> set[str]:
    """
    Extract and normalize tokens from text.

    Args:
        text: Input text to tokenize.

    Returns:
        Set of lowercase tokens with stopwords removed.
    """
    # Extract alphanumeric words
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    # Remove stopwords and very short tokens
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def contains_high_risk_term(keyword: str) -> Optional[str]:
    """
    Check if a keyword contains high-risk industry terms.

    Args:
        keyword: The keyword phrase to check.

    Returns:
        The high-risk term found, or None if clean.
    """
    keyword_lower = keyword.lower()

    # Check single-word blocklist first (most aggressive)
    for term in SINGLE_WORD_BLOCKLIST:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, keyword_lower):
            return term

    # Check multi-word high-risk phrases
    for term in HIGH_RISK_INDUSTRIES:
        # Check for term as a whole word or phrase
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, keyword_lower):
            return term

    return None


def contains_blocked_term(keyword: str, content_text: str) -> Optional[str]:
    """
    Check if a keyword contains a blocked term that's NOT in the content.

    This is the strictest check - if ANY blocked term is in the keyword
    but NOT in the original content, the keyword is rejected.

    Args:
        keyword: The keyword phrase to check.
        content_text: The original page content.

    Returns:
        The blocked term found, or None if allowed.
    """
    keyword_lower = keyword.lower()
    content_lower = content_text.lower()

    # Check all blocked terms
    all_blocked = SINGLE_WORD_BLOCKLIST | HIGH_RISK_INDUSTRIES

    for term in all_blocked:
        pattern = r'\b' + re.escape(term) + r'\b'
        # If term is in keyword but NOT in content, block it
        if re.search(pattern, keyword_lower) and not re.search(pattern, content_lower):
            return term

    return None


def keyword_on_topic(
    keyword: str,
    content_text: str,
    min_overlap: float = 0.5,
) -> tuple[bool, float, str]:
    """
    Determine if a keyword is on-topic for the given content.

    This function applies STRICT filtering:
    1. First checks if keyword contains any blocked terms not in content (instant reject)
    2. Then checks for exact phrase match (instant allow)
    3. Finally checks token overlap

    Args:
        keyword: The keyword phrase to check.
        content_text: The original page content.
        min_overlap: Minimum token overlap ratio required (0.0 to 1.0).

    Returns:
        Tuple of (is_on_topic, overlap_score, reason).
    """
    content_lower = content_text.lower()
    keyword_lower = keyword.lower()

    # CRITICAL CHECK FIRST: Block any keyword with blocked terms not in content
    # This is the most important filter - it prevents injection of off-topic industries
    blocked_term = contains_blocked_term(keyword, content_text)
    if blocked_term:
        return False, 0.0, f"BLOCKED: Contains '{blocked_term}' which is not in original content"

    # Check 2: Exact substring match (strongest signal for ALLOWING)
    if keyword_lower in content_lower:
        return True, 1.0, "Exact phrase found in content"

    # Check 3: High-risk term detection (secondary check)
    high_risk_term = contains_high_risk_term(keyword)
    if high_risk_term:
        # Only allow if the high-risk term itself appears in content
        pattern = r'\b' + re.escape(high_risk_term) + r'\b'
        if not re.search(pattern, content_lower):
            return False, 0.0, f"Contains off-topic industry term: '{high_risk_term}'"

    # Check 4: Token overlap analysis
    content_tokens = normalize_tokens(content_text)
    kw_tokens = normalize_tokens(keyword)

    if not kw_tokens:
        return False, 0.0, "Keyword has no meaningful tokens"

    # Calculate overlap
    matching_tokens = kw_tokens & content_tokens
    overlap = len(matching_tokens) / len(kw_tokens)

    if overlap >= min_overlap:
        return True, overlap, f"Token overlap {overlap:.0%} >= {min_overlap:.0%} threshold"
    else:
        missing = kw_tokens - content_tokens
        return False, overlap, f"Low token overlap {overlap:.0%}; missing: {', '.join(sorted(missing)[:5])}"


def filter_keywords_for_content(
    keywords: list[Keyword],
    content_text: str,
    min_overlap: float = 0.5,
    max_keywords: Optional[int] = None,
) -> tuple[list[Keyword], list[KeywordFilterResult]]:
    """
    Filter a list of keywords to only those relevant to the content.

    This is the main entry point for keyword filtering. It should be called
    BEFORE selecting primary/secondary keywords to ensure only on-topic
    keywords are considered.

    Args:
        keywords: List of keywords to filter.
        content_text: The original page content.
        min_overlap: Minimum token overlap ratio required.
        max_keywords: Optional limit on returned keywords.

    Returns:
        Tuple of (allowed_keywords, all_filter_results).
    """
    results: list[KeywordFilterResult] = []
    allowed: list[Keyword] = []

    for kw in keywords:
        is_on_topic, overlap, reason = keyword_on_topic(
            kw.phrase,
            content_text,
            min_overlap=min_overlap,
        )

        result = KeywordFilterResult(
            keyword=kw,
            is_allowed=is_on_topic,
            reason=reason,
            overlap_score=overlap,
        )
        results.append(result)

        if is_on_topic:
            allowed.append(kw)

    # Sort allowed keywords by overlap score (highest first), then by search volume
    allowed.sort(
        key=lambda k: (
            -next((r.overlap_score for r in results if r.keyword == k), 0),
            -(k.search_volume or 0),
        )
    )

    if max_keywords:
        allowed = allowed[:max_keywords]

    return allowed, results


def log_filter_results(
    results: list[KeywordFilterResult],
    verbose: bool = False,
) -> dict:
    """
    Generate a summary of keyword filtering results.

    Args:
        results: List of KeywordFilterResult from filter_keywords_for_content.
        verbose: If True, include details of each rejected keyword.

    Returns:
        Dict with summary statistics and optionally rejected keyword details.
    """
    allowed = [r for r in results if r.is_allowed]
    rejected = [r for r in results if not r.is_allowed]

    summary = {
        "total_keywords": len(results),
        "allowed_count": len(allowed),
        "rejected_count": len(rejected),
        "allowed_keywords": [r.keyword.phrase for r in allowed],
    }

    if verbose:
        summary["rejected_details"] = [
            {
                "keyword": r.keyword.phrase,
                "reason": r.reason,
                "overlap_score": r.overlap_score,
            }
            for r in rejected
        ]
    else:
        # Just show rejected keyword phrases for brevity
        summary["rejected_keywords"] = [r.keyword.phrase for r in rejected[:20]]
        if len(rejected) > 20:
            summary["rejected_keywords"].append(f"... and {len(rejected) - 20} more")

    return summary


def get_content_topics(content_text: str, top_n: int = 20) -> list[str]:
    """
    Extract the main topic terms from content for reference.

    This helps understand what the content is actually about,
    which can be useful for debugging and prompt construction.

    Args:
        content_text: The original page content.
        top_n: Number of top terms to return.

    Returns:
        List of most frequent meaningful terms in the content.
    """
    tokens = normalize_tokens(content_text)

    # Count token frequency in original text (case-insensitive)
    words = re.findall(r"[A-Za-z0-9]+", content_text.lower())
    word_counts = {}
    for word in words:
        if word in tokens:  # Only count non-stopwords
            word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency
    sorted_terms = sorted(word_counts.items(), key=lambda x: -x[1])

    return [term for term, count in sorted_terms[:top_n]]
