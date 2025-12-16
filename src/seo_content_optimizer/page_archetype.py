"""
Page archetype detection module.

Classifies pages into archetypes to enable appropriate optimization strategies:
- Homepage/Landing: CTAs, proof points, short blocks (NOT guide-style)
- Blog/Article: Guide framing, educational tone, longer content
- Product/Service: Features, benefits, conversion-focused
- About/Company: Story, mission, team info
- FAQ: Q&A format, schema markup
- Comparison: Tables, pros/cons, objective analysis

This prevents the tool from applying "guide-style" framing to landing pages
or other inappropriate content transformations.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Literal
from urllib.parse import urlparse

# Page archetype literals
PageArchetype = Literal[
    "homepage",     # Main site landing page
    "landing",      # Campaign/product landing page
    "service",      # Service description page
    "product",      # Product page
    "blog",         # Blog post / article
    "guide",        # How-to guide, tutorial
    "comparison",   # Comparison / vs page
    "pricing",      # Pricing page
    "about",        # About us / company page
    "contact",      # Contact page
    "faq",          # FAQ page
    "category",     # Category / listing page
    "legal",        # Privacy policy, terms, etc.
    "support",      # Support / help page
    "other",        # Unknown/unclassified
]

# Content style recommendations per archetype
ARCHETYPE_STYLES = {
    "homepage": {
        "tone": "confident",
        "framing": "conversion",  # NOT "guide"
        "add_faq": False,  # Conditional - only if page supports
        "guide_phrases_allowed": False,  # Block "This guide covers..."
        "cta_emphasis": True,
        "long_form_allowed": False,  # Keep content concise
    },
    "landing": {
        "tone": "persuasive",
        "framing": "conversion",
        "add_faq": False,
        "guide_phrases_allowed": False,
        "cta_emphasis": True,
        "long_form_allowed": False,
    },
    "service": {
        "tone": "professional",
        "framing": "benefits",
        "add_faq": True,  # FAQs make sense for services
        "guide_phrases_allowed": False,
        "cta_emphasis": True,
        "long_form_allowed": True,
    },
    "product": {
        "tone": "descriptive",
        "framing": "features",
        "add_faq": True,
        "guide_phrases_allowed": False,
        "cta_emphasis": True,
        "long_form_allowed": True,
    },
    "blog": {
        "tone": "educational",
        "framing": "guide",  # Guide framing IS appropriate
        "add_faq": True,
        "guide_phrases_allowed": True,
        "cta_emphasis": False,
        "long_form_allowed": True,
    },
    "guide": {
        "tone": "instructional",
        "framing": "guide",
        "add_faq": True,
        "guide_phrases_allowed": True,
        "cta_emphasis": False,
        "long_form_allowed": True,
    },
    "comparison": {
        "tone": "objective",
        "framing": "comparison",
        "add_faq": True,
        "guide_phrases_allowed": True,
        "cta_emphasis": False,
        "long_form_allowed": True,
    },
    "pricing": {
        "tone": "clear",
        "framing": "value",
        "add_faq": True,
        "guide_phrases_allowed": False,
        "cta_emphasis": True,
        "long_form_allowed": False,
    },
    "about": {
        "tone": "authentic",
        "framing": "story",
        "add_faq": False,
        "guide_phrases_allowed": False,
        "cta_emphasis": False,
        "long_form_allowed": True,
    },
    "contact": {
        "tone": "helpful",
        "framing": "action",
        "add_faq": False,
        "guide_phrases_allowed": False,
        "cta_emphasis": True,
        "long_form_allowed": False,
    },
    "faq": {
        "tone": "helpful",
        "framing": "answers",
        "add_faq": False,  # Already an FAQ page
        "guide_phrases_allowed": False,
        "cta_emphasis": False,
        "long_form_allowed": True,
    },
    "legal": {
        "tone": "formal",
        "framing": "legal",
        "add_faq": False,
        "guide_phrases_allowed": False,
        "cta_emphasis": False,
        "long_form_allowed": True,
    },
    "support": {
        "tone": "helpful",
        "framing": "solutions",
        "add_faq": True,
        "guide_phrases_allowed": True,
        "cta_emphasis": False,
        "long_form_allowed": True,
    },
    "other": {
        "tone": "neutral",
        "framing": "informational",
        "add_faq": True,  # Default to allowing
        "guide_phrases_allowed": True,
        "cta_emphasis": False,
        "long_form_allowed": True,
    },
    "category": {
        "tone": "descriptive",
        "framing": "listing",
        "add_faq": False,
        "guide_phrases_allowed": False,
        "cta_emphasis": True,
        "long_form_allowed": False,
    },
}

# Phrases that indicate "guide-style" framing (block on landing/homepage)
GUIDE_FRAMING_PHRASES = [
    "this guide covers",
    "this guide explains",
    "this guide will show",
    "this guide walks you through",
    "in this guide",
    "this comprehensive guide",
    "this complete guide",
    "everything you need to know",
    "step-by-step guide",
    "ultimate guide",
    "definitive guide",
    "beginner's guide",
    "how-to guide",
]


@dataclass
class ArchetypeResult:
    """Result of page archetype detection."""
    archetype: PageArchetype
    confidence: float  # 0.0 to 1.0
    signals: list[str] = field(default_factory=list)  # Why we classified this way
    style: dict = field(default_factory=dict)  # Style recommendations

    @property
    def allows_guide_framing(self) -> bool:
        """Check if this archetype allows guide-style framing."""
        return self.style.get("guide_phrases_allowed", False)

    @property
    def should_add_faq(self) -> bool:
        """Check if FAQ generation makes sense for this archetype."""
        return self.style.get("add_faq", True)

    @property
    def is_conversion_focused(self) -> bool:
        """Check if this archetype should emphasize CTAs."""
        return self.style.get("cta_emphasis", False)


def detect_page_archetype(
    url: Optional[str],
    title: Optional[str],
    h1: Optional[str],
    content_text: str,
    headings: list[str] = None,
) -> ArchetypeResult:
    """
    Detect page archetype from URL, title, headings, and content.

    Uses multiple signals:
    1. URL path patterns (strongest signal)
    2. Title and H1 keywords
    3. Content structure (headings, CTAs, forms)
    4. Content patterns (short blocks = landing, long content = article)

    Args:
        url: Page URL.
        title: Page title.
        h1: H1 heading.
        content_text: Full page content.
        headings: List of all headings (H1-H6).

    Returns:
        ArchetypeResult with classification and style recommendations.
    """
    signals = []
    scores: dict[PageArchetype, float] = {k: 0.0 for k in ARCHETYPE_STYLES.keys()}

    # 1. URL path analysis (strongest signal)
    if url:
        url_signals = _analyze_url(url)
        for archetype, score, signal in url_signals:
            scores[archetype] += score
            signals.append(signal)

    # 2. Title and H1 analysis
    title_text = (title or "") + " " + (h1 or "")
    title_signals = _analyze_title(title_text)
    for archetype, score, signal in title_signals:
        scores[archetype] += score
        signals.append(signal)

    # 3. Content structure analysis
    structure_signals = _analyze_structure(content_text, headings or [])
    for archetype, score, signal in structure_signals:
        scores[archetype] += score
        signals.append(signal)

    # 4. Content patterns analysis
    pattern_signals = _analyze_patterns(content_text)
    for archetype, score, signal in pattern_signals:
        scores[archetype] += score
        signals.append(signal)

    # Find best archetype
    best_archetype = max(scores.items(), key=lambda x: x[1])
    archetype = best_archetype[0]
    max_score = best_archetype[1]

    # Calculate confidence (normalize by max possible score)
    max_possible = 10.0  # Rough estimate of max score
    confidence = min(1.0, max_score / max_possible)

    # Default to "other" if confidence is very low
    if confidence < 0.2:
        archetype = "other"

    # Get style recommendations
    style = ARCHETYPE_STYLES.get(archetype, ARCHETYPE_STYLES["other"])

    return ArchetypeResult(
        archetype=archetype,
        confidence=confidence,
        signals=signals,
        style=style,
    )


def _analyze_url(url: str) -> list[tuple[PageArchetype, float, str]]:
    """Analyze URL path for archetype signals."""
    signals = []
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Homepage detection
    if path in ("/", "", "/index", "/home"):
        signals.append(("homepage", 3.0, f"URL path is homepage: {path}"))

    # Blog detection
    if "/blog" in path or "/news" in path or "/articles" in path:
        signals.append(("blog", 2.5, f"URL contains blog/news path"))

    # Guide detection
    if "/guide" in path or "/how-to" in path or "/tutorial" in path:
        signals.append(("guide", 2.5, f"URL contains guide/tutorial path"))

    # Comparison detection
    if "-vs-" in path or "/compare" in path or "/comparison" in path:
        signals.append(("comparison", 2.5, f"URL indicates comparison page"))

    # Pricing detection
    if "/pricing" in path or "/plans" in path:
        signals.append(("pricing", 3.0, f"URL contains pricing path"))

    # About detection
    if "/about" in path or "/team" in path or "/our-story" in path:
        signals.append(("about", 2.5, f"URL contains about path"))

    # Contact detection
    if "/contact" in path:
        signals.append(("contact", 3.0, f"URL contains contact path"))

    # FAQ detection
    if "/faq" in path or "/help" in path:
        signals.append(("faq", 2.5, f"URL contains FAQ/help path"))

    # Legal detection
    if "/privacy" in path or "/terms" in path or "/legal" in path:
        signals.append(("legal", 3.0, f"URL contains legal path"))

    # Service/Product detection
    if "/services" in path or "/products" in path:
        signals.append(("service", 2.0, f"URL contains services/products path"))

    # Landing page detection (common patterns)
    if "/lp/" in path or "/landing" in path or "/get-" in path:
        signals.append(("landing", 2.5, f"URL indicates landing page"))

    return signals


def _analyze_title(title_text: str) -> list[tuple[PageArchetype, float, str]]:
    """Analyze title and H1 for archetype signals."""
    signals = []
    lower_title = title_text.lower()

    # Guide indicators
    guide_words = ["guide", "how to", "tutorial", "learn", "explained", "understanding"]
    for word in guide_words:
        if word in lower_title:
            signals.append(("guide", 1.5, f"Title contains guide indicator: '{word}'"))
            break

    # Comparison indicators
    if " vs " in lower_title or "comparison" in lower_title or "compared" in lower_title:
        signals.append(("comparison", 2.0, f"Title indicates comparison"))

    # Pricing indicators
    if "pricing" in lower_title or "plans" in lower_title or "cost" in lower_title:
        signals.append(("pricing", 1.5, f"Title contains pricing indicator"))

    # FAQ indicators
    if "faq" in lower_title or "frequently asked" in lower_title:
        signals.append(("faq", 2.0, f"Title indicates FAQ"))

    # About indicators
    if "about us" in lower_title or "our team" in lower_title or "our story" in lower_title:
        signals.append(("about", 2.0, f"Title indicates about page"))

    return signals


def _analyze_structure(content_text: str, headings: list[str]) -> list[tuple[PageArchetype, float, str]]:
    """Analyze content structure for archetype signals."""
    signals = []

    # Word count analysis
    word_count = len(content_text.split())

    # Short content = likely landing/homepage
    if word_count < 500:
        signals.append(("landing", 1.5, f"Short content ({word_count} words) suggests landing page"))
    elif word_count > 2000:
        signals.append(("blog", 1.0, f"Long content ({word_count} words) suggests blog/guide"))

    # Heading count analysis
    heading_count = len(headings)
    if heading_count > 5:
        signals.append(("blog", 0.5, f"Many headings ({heading_count}) suggests blog/guide"))
    elif heading_count < 2:
        signals.append(("landing", 0.5, f"Few headings suggests landing page"))

    # CTA detection
    cta_patterns = [
        r"(get started|sign up|contact us|learn more|request|schedule|book|try|demo)",
        r"(free trial|get a quote|start now|join|subscribe)",
    ]
    cta_count = 0
    for pattern in cta_patterns:
        cta_count += len(re.findall(pattern, content_text.lower()))

    if cta_count > 3:
        signals.append(("landing", 1.0, f"Multiple CTAs ({cta_count}) suggests landing/homepage"))

    # Form/contact detection
    if re.search(r"(phone|email|address|contact form)", content_text.lower()):
        signals.append(("contact", 0.5, "Contact information detected"))

    # Q&A structure detection
    qa_pattern = r"(^\s*Q:|^\s*\?|frequently asked|common questions)"
    if re.search(qa_pattern, content_text, re.MULTILINE | re.IGNORECASE):
        signals.append(("faq", 1.5, "Q&A structure detected"))

    return signals


def _analyze_patterns(content_text: str) -> list[tuple[PageArchetype, float, str]]:
    """Analyze content patterns for archetype signals."""
    signals = []
    lower_content = content_text.lower()

    # Legal content indicators
    legal_phrases = ["privacy policy", "terms of service", "terms and conditions", "all rights reserved"]
    for phrase in legal_phrases:
        if phrase in lower_content:
            signals.append(("legal", 2.0, f"Legal phrase detected: '{phrase}'"))
            break

    # Pricing content indicators
    if re.search(r"\$\d+|per month|per year|pricing tier|subscription", lower_content):
        signals.append(("pricing", 1.5, "Pricing content detected"))

    # Testimonial/proof indicators (common on landing pages)
    if re.search(r"(testimonial|customer review|what.*say|success stor)", lower_content):
        signals.append(("landing", 0.5, "Testimonials detected"))

    # Step-by-step content (guide indicator)
    if re.search(r"(step \d+|step-by-step|first,.*second,.*third)", lower_content):
        signals.append(("guide", 1.0, "Step-by-step content detected"))

    return signals


def filter_guide_phrases(text: str, archetype_result: ArchetypeResult) -> str:
    """
    Remove guide-style phrases from text if archetype doesn't allow them.

    Args:
        text: Text to filter.
        archetype_result: Page archetype classification.

    Returns:
        Filtered text with inappropriate guide phrases removed.
    """
    if archetype_result.allows_guide_framing:
        return text  # No filtering needed

    result = text
    for phrase in GUIDE_FRAMING_PHRASES:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        result = pattern.sub("", result)

    # Clean up any resulting double spaces or empty sentences
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"\.\s*\.", ".", result)

    return result


def get_content_guidance(archetype_result: ArchetypeResult) -> dict:
    """
    Get content optimization guidance based on archetype.

    Returns a dict with guidance for:
    - tone: How to write
    - avoid: Phrases/patterns to avoid
    - emphasize: Elements to emphasize
    - structure: Structural recommendations

    Args:
        archetype_result: Page archetype classification.

    Returns:
        Dict with content guidance.
    """
    archetype = archetype_result.archetype
    style = archetype_result.style

    guidance = {
        "tone": style.get("tone", "neutral"),
        "avoid": [],
        "emphasize": [],
        "structure": [],
    }

    # Build avoid list
    if not style.get("guide_phrases_allowed"):
        guidance["avoid"].extend([
            "This guide covers...",
            "In this guide, you'll learn...",
            "This comprehensive guide...",
            "Everything you need to know...",
        ])

    if archetype in ("homepage", "landing"):
        guidance["avoid"].extend([
            "Long educational paragraphs",
            "Step-by-step instructions (unless core to offering)",
            "Academic/formal tone",
        ])
        guidance["emphasize"].extend([
            "Clear value proposition",
            "Social proof (testimonials, stats)",
            "Strong calls-to-action",
            "Concise, scannable content",
        ])
        guidance["structure"].extend([
            "Keep sections short",
            "Use bullet points for features/benefits",
            "Place CTA above the fold",
        ])

    elif archetype in ("blog", "guide"):
        guidance["emphasize"].extend([
            "Educational value",
            "Clear explanations",
            "Examples and illustrations",
            "Logical flow",
        ])
        guidance["structure"].extend([
            "Use descriptive headings",
            "Include a table of contents for long content",
            "Add an FAQ section",
        ])

    elif archetype == "comparison":
        guidance["avoid"].extend([
            "Heavily biased language",
            "Unfair comparisons",
        ])
        guidance["emphasize"].extend([
            "Objective analysis",
            "Clear criteria",
            "Feature comparison tables",
        ])

    return guidance
