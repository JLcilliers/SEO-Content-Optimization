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
    ContentAudit,
    ContentIntent,
    DocxContent,
    HeadingLevel,
    Keyword,
    KeywordPlan,
    KeywordPlacementPlan,
    KeywordPlacementStatus,
    KeywordUsageStats,
    OptimizationPlan,
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


# =============================================================================
# 10-Part SEO Framework: Content Audit (Parts 1-4 & 6)
# =============================================================================

def audit_content(
    content: Union[PageMeta, DocxContent],
    keyword_plan: KeywordPlan,
    meta_title: Optional[str] = None,
    meta_description: Optional[str] = None,
) -> ContentAudit:
    """
    Perform a comprehensive content audit following the 10-part SEO framework.

    This implements Parts 1-4 & 6 of the framework:
    - Part 1: Understand what search engines (and readers) want
    - Part 2: Identify target keywords
    - Part 3: Audit current state
    - Part 4: Identify gaps
    - Part 6: Research/competitive context

    Args:
        content: PageMeta or DocxContent to audit.
        keyword_plan: The KeywordPlan with primary, secondary, and long-tail keywords.
        meta_title: Current meta title (if available separately from content).
        meta_description: Current meta description (if available separately).

    Returns:
        ContentAudit with comprehensive analysis results.
    """
    # Extract content details
    if isinstance(content, PageMeta):
        full_text = content.full_text
        h1 = content.h1
        title = meta_title or content.title
        meta_desc = meta_description or content.meta_description
        blocks = content.content_blocks
        headings = _extract_headings_from_page_meta(content)
    else:
        full_text = content.full_text
        h1 = content.h1
        title = meta_title
        meta_desc = meta_description
        blocks = [p.text for p in content.paragraphs]
        headings = [p.text for p in content.headings]

    # Basic stats
    word_count = len(full_text.split())

    # Detect topic and intent
    topic = _detect_topic(title, h1, blocks)
    intent = _detect_intent(full_text, title, h1)
    topic_summary = _create_topic_summary(topic, intent, word_count)

    # Build heading outline
    heading_outline = _build_heading_outline(headings, h1)

    # Analyze keyword placement for all keywords
    keyword_status = _analyze_keyword_placements(
        keyword_plan=keyword_plan,
        full_text=full_text,
        title=title,
        meta_description=meta_desc,
        h1=h1,
        headings=headings,
    )

    # Identify gaps
    depth_gaps = _identify_depth_gaps(full_text, keyword_plan, topic)
    structural_gaps = _identify_structural_gaps(content, headings, full_text)
    format_opportunities = _identify_format_opportunities(content, full_text)
    technical_opportunities = _identify_technical_opportunities(content)

    # Prioritize issues
    high_priority, medium_priority, standard_priority = _prioritize_issues(
        keyword_status=keyword_status,
        structural_gaps=structural_gaps,
        depth_gaps=depth_gaps,
        format_opportunities=format_opportunities,
        technical_opportunities=technical_opportunities,
    )

    return ContentAudit(
        topic_summary=topic_summary,
        intent=intent.value,
        word_count=word_count,
        current_meta_title=title,
        current_meta_description=meta_desc,
        current_h1=h1,
        heading_outline=heading_outline,
        keyword_status=keyword_status,
        depth_gaps=depth_gaps,
        structural_gaps=structural_gaps,
        format_opportunities=format_opportunities,
        technical_opportunities=technical_opportunities,
        high_priority_issues=high_priority,
        medium_priority_issues=medium_priority,
        standard_priority_issues=standard_priority,
    )


def _create_topic_summary(topic: str, intent: ContentIntent, word_count: int) -> str:
    """Create a 1-2 sentence summary of what the content is about."""
    intent_label = intent.value
    return f"Content about '{topic}' with {intent_label} intent ({word_count} words)."


def _extract_headings_from_page_meta(content: PageMeta) -> list[str]:
    """Extract headings from PageMeta structured blocks if available."""
    headings = []
    if content.has_structured_content:
        from .models import ContentBlockType
        for block in content.structured_blocks:
            if block.block_type == ContentBlockType.HEADING and block.paragraph:
                headings.append(block.paragraph.text)
    return headings


def _build_heading_outline(headings: list[str], h1: Optional[str]) -> list[str]:
    """Build a heading outline list with H1 first, then subheadings."""
    outline = []
    if h1:
        outline.append(f"H1: {h1}")
    for heading in headings:
        # Skip if this is the H1 (already added)
        if h1 and heading.strip() == h1.strip():
            continue
        outline.append(f"H2/H3: {heading}")
    return outline


def _analyze_keyword_placements(
    keyword_plan: KeywordPlan,
    full_text: str,
    title: Optional[str],
    meta_description: Optional[str],
    h1: Optional[str],
    headings: list[str],
) -> list[KeywordPlacementStatus]:
    """
    Analyze where each keyword appears in the content.

    Checks placement in all tiers of the hierarchy:
    - Tier 1: Title Tag
    - Tier 2: H1
    - Tier 3: First 100 words
    - Tier 4: Subheadings (H2/H3)
    - Tier 5: Body content
    - Tier 7: Conclusion (last paragraph)
    """
    placements = []

    # Get all keywords from the plan
    all_keywords = keyword_plan.all_keywords

    # Extract first 100 words and conclusion
    words = full_text.split()
    first_100_words = " ".join(words[:100])
    conclusion = _extract_conclusion(full_text)

    for kw in all_keywords:
        phrase = kw.phrase
        phrase_lower = phrase.lower()
        pattern = re.compile(re.escape(phrase_lower), re.IGNORECASE)

        # Check each placement location
        in_title = bool(title and pattern.search(title))
        in_meta_desc = bool(meta_description and pattern.search(meta_description))
        in_h1 = bool(h1 and pattern.search(h1))
        in_first_100 = bool(pattern.search(first_100_words))

        # Check subheadings
        in_subheadings = False
        for heading in headings:
            if pattern.search(heading):
                in_subheadings = True
                break

        # Check body (total count)
        body_count = len(pattern.findall(full_text))
        in_body = body_count > 0

        # Check conclusion
        in_conclusion = bool(pattern.search(conclusion))

        placements.append(KeywordPlacementStatus(
            keyword=phrase,
            in_title=in_title,
            in_meta_description=in_meta_desc,
            in_h1=in_h1,
            in_first_100_words=in_first_100,
            in_subheadings=in_subheadings,
            in_body=in_body,
            in_conclusion=in_conclusion,
            body_count=body_count,
        ))

    return placements


def _extract_conclusion(full_text: str) -> str:
    """Extract the last paragraph/conclusion of the content."""
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    if paragraphs:
        # Return last substantial paragraph (at least 50 chars)
        for para in reversed(paragraphs):
            if len(para) >= 50:
                return para
        # If all are short, just return the last one
        return paragraphs[-1]
    return ""


def _identify_depth_gaps(
    full_text: str,
    keyword_plan: KeywordPlan,
    topic: str,
) -> list[str]:
    """
    Identify missing subtopics that searchers would expect.

    This analyzes the content to find gaps in topic coverage.
    """
    gaps = []
    text_lower = full_text.lower()

    # Check if long-tail question keywords are addressed
    for kw in keyword_plan.long_tail_questions:
        if kw.phrase.lower() not in text_lower:
            # Check if the question topic is addressed even if exact phrase isn't
            question_words = set(kw.phrase.lower().split())
            # Remove common question words
            question_words -= {"how", "what", "why", "when", "where", "which", "can", "does", "is", "are", "to", "do"}
            significant_words = [w for w in question_words if len(w) > 3]
            if significant_words:
                matches = sum(1 for w in significant_words if w in text_lower)
                if matches < len(significant_words) * 0.5:
                    gaps.append(f"Missing coverage: {kw.phrase}")

    # Check for common expected subtopics based on intent
    if "guide" in topic.lower() or "how" in topic.lower():
        if "step" not in text_lower and "steps" not in text_lower:
            gaps.append("Consider adding step-by-step instructions")
        if "example" not in text_lower and "examples" not in text_lower:
            gaps.append("Consider adding practical examples")

    if "vs" in topic.lower() or "comparison" in topic.lower():
        if "pros" not in text_lower and "cons" not in text_lower:
            gaps.append("Consider adding pros and cons comparison")

    return gaps[:5]  # Limit to top 5 gaps


def _identify_structural_gaps(
    content: Union[PageMeta, DocxContent],
    headings: list[str],
    full_text: str,
) -> list[str]:
    """Identify structural issues like missing FAQ, conclusion, etc."""
    gaps = []
    text_lower = full_text.lower()

    # Check for FAQ section
    has_faq = any("faq" in h.lower() or "question" in h.lower() for h in headings)
    has_faq = has_faq or "frequently asked" in text_lower
    if not has_faq:
        gaps.append("No FAQ section detected")

    # Check for clear conclusion
    conclusion_indicators = ["conclusion", "summary", "final thoughts", "in conclusion", "to summarize"]
    has_conclusion = any(ind in text_lower for ind in conclusion_indicators)
    if not has_conclusion:
        gaps.append("No clear conclusion section")

    # Check heading structure
    if len(headings) < 3:
        gaps.append("Content has few subheadings - consider adding more H2/H3 structure")

    # Check for introduction
    intro_indicators = ["introduction", "overview", "in this article", "in this guide"]
    has_intro = any(ind in text_lower[:500] for ind in intro_indicators)
    if not has_intro and len(full_text) > 1000:
        gaps.append("Consider adding a clear introduction")

    return gaps


def _identify_format_opportunities(
    content: Union[PageMeta, DocxContent],
    full_text: str,
) -> list[str]:
    """Identify opportunities to add structured formats like tables, lists."""
    opportunities = []
    text_lower = full_text.lower()

    # Check for comparison content that could use a table
    if "vs" in text_lower or "comparison" in text_lower or "compare" in text_lower:
        # Check if there's already a table (rough heuristic)
        if isinstance(content, PageMeta) and content.has_structured_content:
            from .models import ContentBlockType
            has_table = any(b.block_type == ContentBlockType.TABLE for b in content.structured_blocks)
            if not has_table:
                opportunities.append("Add comparison table for key differences")
        else:
            opportunities.append("Consider adding comparison table")

    # Check for list content that could be formatted
    list_indicators = ["features:", "benefits:", "advantages:", "includes:"]
    for indicator in list_indicators:
        if indicator in text_lower:
            opportunities.append(f"Content after '{indicator[:-1]}' could be formatted as bullet list")
            break

    # Check for pricing/spec content that could use a table
    if "price" in text_lower or "pricing" in text_lower or "cost" in text_lower:
        opportunities.append("Consider adding pricing comparison table")

    return opportunities[:3]  # Limit to top 3


def _identify_technical_opportunities(
    content: Union[PageMeta, DocxContent],
) -> list[str]:
    """Identify technical SEO opportunities like schema, internal links."""
    opportunities = []

    # Always recommend FAQ schema for FAQ sections
    opportunities.append("Add FAQ schema markup for rich results")

    # Recommend internal links
    opportunities.append("Review internal linking opportunities")

    # For web content, check meta
    if isinstance(content, PageMeta):
        if not content.meta_description:
            opportunities.append("Add meta description")
        if not content.title:
            opportunities.append("Add title tag")

    return opportunities


def _prioritize_issues(
    keyword_status: list[KeywordPlacementStatus],
    structural_gaps: list[str],
    depth_gaps: list[str],
    format_opportunities: list[str],
    technical_opportunities: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Prioritize issues into high, medium, and standard categories.

    High priority (Part 9): Meta/H1/keyword presence issues
    Medium priority: Content depth gaps
    Standard priority: Technical polish items
    """
    high_priority = []
    medium_priority = []
    standard_priority = []

    # Primary keyword placement issues are high priority
    if keyword_status:
        primary = keyword_status[0]
        if not primary.in_title:
            high_priority.append(f"Primary keyword '{primary.keyword}' missing from title tag")
        if not primary.in_h1:
            high_priority.append(f"Primary keyword '{primary.keyword}' missing from H1")
        if not primary.in_first_100_words:
            high_priority.append(f"Primary keyword '{primary.keyword}' missing from first 100 words")
        if not primary.in_meta_description:
            high_priority.append(f"Primary keyword '{primary.keyword}' missing from meta description")

    # Secondary keyword issues
    for status in keyword_status[1:4]:  # First 3 secondary keywords
        if not status.in_body:
            medium_priority.append(f"Secondary keyword '{status.keyword}' missing from body")

    # Structural gaps are medium priority
    for gap in structural_gaps:
        if "FAQ" in gap or "conclusion" in gap.lower():
            medium_priority.append(gap)
        else:
            standard_priority.append(gap)

    # Depth gaps are medium priority
    medium_priority.extend(depth_gaps[:3])

    # Format and technical opportunities are standard priority
    standard_priority.extend(format_opportunities)
    standard_priority.extend(technical_opportunities)

    return high_priority, medium_priority, standard_priority


# =============================================================================
# 10-Part SEO Framework: Optimization Plan (Part 5)
# =============================================================================

def build_optimization_plan(
    audit: ContentAudit,
    keyword_plan: KeywordPlan,
) -> OptimizationPlan:
    """
    Build a comprehensive optimization plan based on the content audit.

    This implements Part 5 of the framework: Keyword placement hierarchy.

    Args:
        audit: The ContentAudit from audit_content().
        keyword_plan: The KeywordPlan with all keywords.

    Returns:
        OptimizationPlan with targeted optimizations.
    """
    primary = keyword_plan.primary.phrase
    secondary = [kw.phrase for kw in keyword_plan.secondary]
    long_tail = [kw.phrase for kw in keyword_plan.long_tail_questions]

    # Build placement plan with tiered hierarchy
    placement_plan = KeywordPlacementPlan(
        title=primary,  # Primary keyword in title
        meta_description=primary,  # Primary in meta description
        h1=primary,  # Primary in H1
        first_100_words=primary,  # Primary in intro
        subheadings=secondary[:3],  # Top 3 secondary in subheadings
        body_priority=[primary] + secondary,  # All keywords by priority
        faq_keywords=long_tail[:4] if long_tail else secondary[:2],  # Long-tail for FAQ
        conclusion=[primary] + secondary[:2],  # Primary + top secondary in conclusion
    )

    # Determine sections to add based on structural gaps
    sections_to_add = []
    sections_to_enhance = []

    for gap in audit.structural_gaps:
        if "FAQ" in gap:
            sections_to_add.append("FAQ section")
        elif "conclusion" in gap.lower():
            sections_to_add.append("Conclusion section")
        elif "introduction" in gap.lower():
            sections_to_enhance.append("Introduction")

    # Determine FAQ questions to generate
    faq_questions = []
    for kw in keyword_plan.long_tail_questions[:4]:
        if kw.is_question:
            faq_questions.append(kw.phrase)

    # If not enough question keywords, generate from secondary
    if len(faq_questions) < 4:
        for kw in keyword_plan.secondary[:3]:
            question = f"What is {kw.phrase}?"
            if question not in faq_questions:
                faq_questions.append(question)
            if len(faq_questions) >= 4:
                break

    # Generate target meta elements based on audit
    target_title = _generate_target_title(audit.current_meta_title, primary, audit.topic_summary)
    target_meta_desc = _generate_target_meta_desc(audit.current_meta_description, primary, audit.topic_summary)
    target_h1 = _generate_target_h1(audit.current_h1, primary, audit.topic_summary)

    return OptimizationPlan(
        primary_keyword=primary,
        secondary_keywords=secondary,
        audit=audit,
        target_meta_title=target_title,
        target_meta_description=target_meta_desc,
        target_h1=target_h1,
        sections_to_add=sections_to_add,
        sections_to_enhance=sections_to_enhance,
        faq_questions=faq_questions[:4],
        placement_plan=placement_plan,
    )


def _generate_target_title(
    current: Optional[str],
    primary_keyword: str,
    topic_summary: str,
) -> str:
    """Generate a target title placeholder based on requirements."""
    if current and primary_keyword.lower() in current.lower():
        return current  # Already has keyword
    return f"[Optimize: Include '{primary_keyword}' - Current: {current or 'None'}]"


def _generate_target_meta_desc(
    current: Optional[str],
    primary_keyword: str,
    topic_summary: str,
) -> str:
    """Generate a target meta description placeholder."""
    if current and primary_keyword.lower() in current.lower():
        return current
    return f"[Optimize: Include '{primary_keyword}' naturally - Current: {current or 'None'}]"


def _generate_target_h1(
    current: Optional[str],
    primary_keyword: str,
    topic_summary: str,
) -> str:
    """Generate a target H1 placeholder."""
    if current and primary_keyword.lower() in current.lower():
        return current
    return f"[Optimize: Include '{primary_keyword}' - Current: {current or 'None'}]"
