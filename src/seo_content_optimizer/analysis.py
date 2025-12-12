"""
Content and intent analysis module.

This module analyzes content to:
- Detect topic and content type/intent
- Measure current keyword usage
- Provide statistics for optimization planning
"""

import re
from collections import Counter
from typing import Optional, Union

from .models import (
    ContentAnalysis,
    ContentIntent,
    DocxContent,
    HeadingLevel,
    Keyword,
    KeywordUsageStats,
    PageMeta,
    ParagraphBlock,
)


def analyze_content(
    content: Union[PageMeta, DocxContent],
    keywords: Optional[list[Keyword]] = None,
) -> ContentAnalysis:
    """
    Analyze content to extract topic, intent, and keyword usage statistics.

    Args:
        content: PageMeta or DocxContent to analyze.
        keywords: Optional list of keywords to check usage for.

    Returns:
        ContentAnalysis object with analysis results.
    """
    # Get full text and metadata
    if isinstance(content, PageMeta):
        full_text = content.full_text
        title = content.title
        h1 = content.h1
        meta_desc = content.meta_description
        blocks = content.content_blocks
        headings = []  # PageMeta doesn't track heading structure
    else:
        full_text = content.full_text
        title = None  # DOCX has no title meta
        h1 = content.h1
        meta_desc = None
        blocks = [p.text for p in content.paragraphs]
        headings = [p.text for p in content.headings]

    # Basic statistics
    word_count = len(full_text.split())
    paragraph_count = len(blocks)
    heading_count = len(headings)

    # Detect topic from title/H1 or first paragraph
    topic = _detect_topic(title, h1, blocks)

    # Detect content intent
    intent = _detect_intent(full_text, title, h1)

    # Create summary
    summary = _create_summary(topic, intent, word_count)

    # Analyze keyword usage if keywords provided
    existing_keywords: dict[str, dict] = {}
    if keywords:
        for kw in keywords:
            stats = get_keyword_usage_stats(
                kw.phrase,
                full_text,
                title=title,
                meta_description=meta_desc,
                h1=h1,
                headings=headings,
            )
            existing_keywords[kw.phrase] = {
                "count_in_body": stats.count_in_body,
                "in_title": stats.in_title,
                "in_meta_description": stats.in_meta_description,
                "in_h1": stats.in_h1,
                "in_headings": stats.in_headings,
                "in_first_100_words": stats.in_first_100_words,
            }

    return ContentAnalysis(
        topic=topic,
        intent=intent,
        summary=summary,
        existing_keywords=existing_keywords,
        word_count=word_count,
        heading_count=heading_count,
        paragraph_count=paragraph_count,
    )


def _detect_topic(
    title: Optional[str], h1: Optional[str], blocks: list[str]
) -> str:
    """Detect the main topic from content."""
    # Priority: H1 > Title > First paragraph
    if h1:
        return h1
    if title:
        return title
    if blocks:
        # Use first non-empty block, truncated
        first = blocks[0][:200]
        return first.split(".")[0] if "." in first else first

    return "Unknown Topic"


def _detect_intent(
    full_text: str, title: Optional[str], h1: Optional[str]
) -> ContentIntent:
    """
    Detect the content intent based on text patterns.

    Returns:
        ContentIntent enum value.
    """
    text_lower = full_text.lower()
    title_h1 = f"{title or ''} {h1 or ''}".lower()

    # Transactional indicators
    transactional_patterns = [
        r"\b(buy|purchase|order|shop|price|pricing|cost|quote|hire|book|subscribe)\b",
        r"\b(get started|sign up|contact us|call now|request|free trial)\b",
        r"\b(service|services|product|products|solution|solutions)\b",
        r"\$\d+",  # Price patterns
    ]

    # Informational indicators
    informational_patterns = [
        r"\b(how to|what is|what are|why|guide|tutorial|learn|understand)\b",
        r"\b(tips|advice|best practices|complete guide|ultimate guide)\b",
        r"\b(explained|introduction|overview|basics|beginner)\b",
        r"\b(example|examples|step by step|steps)\b",
    ]

    transactional_score = 0
    informational_score = 0

    # Check patterns in title/H1 (weighted higher)
    for pattern in transactional_patterns:
        if re.search(pattern, title_h1):
            transactional_score += 2
        if re.search(pattern, text_lower):
            transactional_score += 1

    for pattern in informational_patterns:
        if re.search(pattern, title_h1):
            informational_score += 2
        if re.search(pattern, text_lower):
            informational_score += 1

    # Determine intent
    if transactional_score > informational_score * 1.5:
        return ContentIntent.TRANSACTIONAL
    elif informational_score > transactional_score * 1.5:
        return ContentIntent.INFORMATIONAL
    elif transactional_score > 0 and informational_score > 0:
        return ContentIntent.MIXED
    elif transactional_score > 0:
        return ContentIntent.TRANSACTIONAL
    elif informational_score > 0:
        return ContentIntent.INFORMATIONAL
    else:
        return ContentIntent.INFORMATIONAL  # Default


def _create_summary(topic: str, intent: ContentIntent, word_count: int) -> str:
    """Create a brief summary of the content."""
    intent_label = intent.value.capitalize()
    return f"This is {intent_label.lower()} content about '{topic}' ({word_count} words)."


def get_keyword_usage_stats(
    phrase: str,
    full_text: str,
    title: Optional[str] = None,
    meta_description: Optional[str] = None,
    h1: Optional[str] = None,
    headings: Optional[list[str]] = None,
) -> KeywordUsageStats:
    """
    Get detailed statistics about keyword usage in content.

    Args:
        phrase: The keyword phrase to analyze.
        full_text: The full content text.
        title: Optional page title.
        meta_description: Optional meta description.
        h1: Optional H1 heading.
        headings: Optional list of all headings.

    Returns:
        KeywordUsageStats with usage information.
    """
    phrase_lower = phrase.lower()
    pattern = re.compile(re.escape(phrase_lower), re.IGNORECASE)

    # Count in body
    count_in_body = len(pattern.findall(full_text))

    # Check in title
    in_title = bool(title and pattern.search(title))

    # Check in meta description
    in_meta_description = bool(meta_description and pattern.search(meta_description))

    # Check in H1
    in_h1 = bool(h1 and pattern.search(h1))

    # Check in headings
    in_headings = False
    if headings:
        for heading in headings:
            if pattern.search(heading):
                in_headings = True
                break

    # Check in first 100 words
    words = full_text.split()[:100]
    first_100 = " ".join(words)
    in_first_100_words = bool(pattern.search(first_100))

    return KeywordUsageStats(
        phrase=phrase,
        count_in_body=count_in_body,
        in_title=in_title,
        in_meta_description=in_meta_description,
        in_h1=in_h1,
        in_headings=in_headings,
        in_first_100_words=in_first_100_words,
    )


def extract_key_terms(text: str, top_n: int = 20) -> list[tuple[str, int]]:
    """
    Extract key terms from text based on frequency.

    Simple approach: count word frequencies, filter stopwords.

    Args:
        text: Text to analyze.
        top_n: Number of top terms to return.

    Returns:
        List of (term, count) tuples sorted by frequency.
    """
    # Common English stopwords
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "this",
        "that", "these", "those", "it", "its", "you", "your", "we", "our",
        "they", "their", "he", "she", "him", "her", "his", "my", "i", "me",
        "as", "if", "when", "where", "why", "how", "what", "which", "who",
        "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "also", "now", "here", "there", "then",
    }

    # Tokenize and clean
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    filtered = [w for w in words if w not in stopwords]

    # Count frequencies
    counter = Counter(filtered)

    return counter.most_common(top_n)


def calculate_keyword_density(phrase: str, text: str) -> float:
    """
    Calculate keyword density as percentage.

    Args:
        phrase: Keyword phrase.
        text: Text to analyze.

    Returns:
        Keyword density as percentage (0-100).
    """
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return 0.0

    phrase_words = len(phrase.split())
    pattern = re.compile(re.escape(phrase.lower()), re.IGNORECASE)
    occurrences = len(pattern.findall(text))

    # Density = (occurrences * phrase_word_count / total_words) * 100
    density = (occurrences * phrase_words / total_words) * 100

    return round(density, 2)


def find_missing_keywords(
    content: Union[PageMeta, DocxContent],
    keywords: list[Keyword],
) -> list[Keyword]:
    """
    Find keywords that are not present in the content.

    Args:
        content: Content to search in.
        keywords: List of keywords to check.

    Returns:
        List of keywords not found in content.
    """
    full_text = content.full_text.lower()
    missing = []

    for kw in keywords:
        pattern = re.compile(re.escape(kw.phrase.lower()), re.IGNORECASE)
        if not pattern.search(full_text):
            missing.append(kw)

    return missing


def find_underused_keywords(
    content: Union[PageMeta, DocxContent],
    keywords: list[Keyword],
    min_occurrences: int = 2,
) -> list[Keyword]:
    """
    Find keywords that appear less than the minimum threshold.

    Args:
        content: Content to search in.
        keywords: List of keywords to check.
        min_occurrences: Minimum number of times keyword should appear.

    Returns:
        List of keywords appearing fewer than min_occurrences times.
    """
    full_text = content.full_text
    underused = []

    for kw in keywords:
        pattern = re.compile(re.escape(kw.phrase.lower()), re.IGNORECASE)
        count = len(pattern.findall(full_text))
        if count < min_occurrences:
            underused.append(kw)

    return underused


# =============================================================================
# Brand Detection Heuristics
# =============================================================================

def guess_brand_tokens(
    url: Optional[str] = None,
    h1: Optional[str] = None,
    title: Optional[str] = None,
) -> set[str]:
    """
    Extract likely brand name tokens from URL domain, H1, and title.

    This heuristic identifies potential brand names by:
    1. Extracting domain name from URL (e.g., "payzli" from "payzli.com")
    2. Extracting capitalized proper nouns from H1/title
    3. Looking for common brand patterns like "CompanyName vs/vs."

    Args:
        url: The page URL (optional).
        h1: The H1 heading text (optional).
        title: The page title (optional).

    Returns:
        Set of lowercase brand token candidates.
    """
    brand_tokens: set[str] = set()

    # 1. Extract from URL domain
    if url:
        # Handle URLs with or without protocol
        domain_match = re.search(r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+)\.", url)
        if domain_match:
            domain = domain_match.group(1).lower()
            # Skip generic domains
            if domain not in {"blog", "www", "shop", "store", "app", "my", "get"}:
                brand_tokens.add(domain)

    # 2. Extract capitalized words from H1 and title (likely proper nouns/brands)
    for text in [h1, title]:
        if not text:
            continue

        # Find capitalized words that could be brand names
        # Match words starting with capital letter, possibly with numbers
        cap_words = re.findall(r"\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*)\b", text)
        for word in cap_words:
            word_lower = word.lower()
            # Skip common non-brand words that are often capitalized
            skip_words = {
                "the", "a", "an", "and", "or", "but", "for", "to", "of", "in", "on",
                "with", "by", "is", "are", "was", "were", "be", "been", "have", "has",
                "how", "what", "why", "when", "where", "which", "who", "your", "our",
                "best", "top", "guide", "review", "reviews", "complete", "ultimate",
                "new", "vs", "versus", "comparison", "compared", "alternatives",
            }
            if word_lower not in skip_words and len(word) >= 2:
                brand_tokens.add(word_lower)

    # 3. Look for "X vs Y" patterns which often contain brand names
    combined_text = f"{h1 or ''} {title or ''}".lower()
    vs_match = re.search(r"(\w+)\s+(?:vs\.?|versus)\s+(\w+)", combined_text, re.IGNORECASE)
    if vs_match:
        for group in vs_match.groups():
            if group and len(group) >= 2:
                brand_tokens.add(group.lower())

    return brand_tokens


def is_branded_phrase(
    phrase: str,
    brand_tokens: set[str],
    keyword: Optional[Keyword] = None,
) -> bool:
    """
    Check if a keyword phrase is a brand/navigational keyword.

    A phrase is considered branded if:
    1. The keyword has is_brand=True flag set (explicit from CSV/Excel)
    2. Any token in the phrase matches a known brand token
    3. The keyword has navigational intent

    Args:
        phrase: The keyword phrase to check.
        brand_tokens: Set of known brand tokens (from guess_brand_tokens).
        keyword: Optional Keyword object (to check is_brand flag and intent).

    Returns:
        True if the phrase is branded, False otherwise.
    """
    # Check explicit is_brand flag from CSV/Excel
    if keyword and keyword.is_brand:
        return True

    # Check for navigational intent (often brand-related)
    if keyword and keyword.intent and keyword.intent.lower() == "navigational":
        return True

    # Tokenize the phrase
    phrase_lower = phrase.lower()
    phrase_tokens = set(re.findall(r"[a-zA-Z0-9]+", phrase_lower))

    # Check if any phrase token matches a brand token
    overlap = phrase_tokens & brand_tokens
    if overlap:
        return True

    return False


def mark_branded_keywords(
    keywords: list[Keyword],
    url: Optional[str] = None,
    h1: Optional[str] = None,
    title: Optional[str] = None,
) -> list[Keyword]:
    """
    Update keywords with is_brand=True if they match detected brand patterns.

    This function applies heuristic brand detection to keywords that don't
    already have is_brand explicitly set. Keywords already marked as brands
    are left unchanged.

    Args:
        keywords: List of keywords to process.
        url: The page URL for brand detection.
        h1: The H1 heading text for brand detection.
        title: The page title for brand detection.

    Returns:
        The same list of keywords with is_brand flags updated.
    """
    brand_tokens = guess_brand_tokens(url=url, h1=h1, title=title)

    for kw in keywords:
        # Skip keywords already marked as brands
        if kw.is_brand:
            continue

        # Apply heuristic detection
        if is_branded_phrase(kw.phrase, brand_tokens, kw):
            kw.is_brand = True

    return keywords
