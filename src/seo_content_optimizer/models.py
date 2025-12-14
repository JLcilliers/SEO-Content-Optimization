"""
Data models for SEO Content Optimizer.

This module defines all the core data structures used throughout the application.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ContentIntent(Enum):
    """Classification of content intent."""
    INFORMATIONAL = "informational"
    TRANSACTIONAL = "transactional"
    NAVIGATIONAL = "navigational"
    MIXED = "mixed"


class HeadingLevel(Enum):
    """Heading levels in content."""
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6
    BODY = 0  # Regular paragraph


@dataclass
class Keyword:
    """Represents a keyword with optional metadata."""
    phrase: str
    search_volume: Optional[int] = None
    difficulty: Optional[float] = None
    intent: Optional[str] = None
    is_brand: bool = False  # True if this is a brand/navigational keyword

    def __post_init__(self) -> None:
        """Normalize the keyword phrase."""
        self.phrase = self.phrase.strip()

    @property
    def is_question(self) -> bool:
        """Check if the keyword is a question."""
        question_starters = ('how', 'what', 'why', 'when', 'where', 'who', 'which', 'can', 'does', 'is', 'are')
        lower_phrase = self.phrase.lower()
        return lower_phrase.endswith('?') or lower_phrase.startswith(question_starters)


class ContentBlockType(Enum):
    """Types of content blocks."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    LIST = "list"
    BLOCKQUOTE = "blockquote"


@dataclass
class TableCell:
    """A single cell in a table."""
    text: str
    is_header: bool = False


@dataclass
class TableRow:
    """A row in a table."""
    cells: list[TableCell] = field(default_factory=list)


@dataclass
class TableBlock:
    """A table content block."""
    rows: list[TableRow] = field(default_factory=list)
    caption: Optional[str] = None

    @property
    def has_header(self) -> bool:
        """Check if table has a header row."""
        if not self.rows:
            return False
        return any(cell.is_header for cell in self.rows[0].cells)

    @property
    def column_count(self) -> int:
        """Get the number of columns."""
        if not self.rows:
            return 0
        return max(len(row.cells) for row in self.rows)


@dataclass
class ListItem:
    """A single list item."""
    text: str
    level: int = 0  # Nesting level


@dataclass
class ListBlock:
    """A list content block (ordered or unordered)."""
    items: list[ListItem] = field(default_factory=list)
    ordered: bool = False


@dataclass
class ParagraphBlock:
    """A block of content (paragraph or heading) from a document."""
    text: str
    heading_level: HeadingLevel = HeadingLevel.BODY
    style_name: Optional[str] = None

    @property
    def is_heading(self) -> bool:
        """Check if this block is a heading."""
        return self.heading_level != HeadingLevel.BODY


@dataclass
class ContentBlock:
    """
    A unified content block that can hold different content types.

    This is a polymorphic wrapper that can contain paragraphs, tables, lists, etc.
    """
    block_type: ContentBlockType
    paragraph: Optional[ParagraphBlock] = None
    table: Optional[TableBlock] = None
    list_block: Optional[ListBlock] = None

    @classmethod
    def from_paragraph(cls, text: str, heading_level: HeadingLevel = HeadingLevel.BODY) -> "ContentBlock":
        """Create a paragraph content block."""
        return cls(
            block_type=ContentBlockType.PARAGRAPH if heading_level == HeadingLevel.BODY else ContentBlockType.HEADING,
            paragraph=ParagraphBlock(text=text, heading_level=heading_level),
        )

    @classmethod
    def from_table(cls, rows: list[TableRow], caption: Optional[str] = None) -> "ContentBlock":
        """Create a table content block."""
        return cls(
            block_type=ContentBlockType.TABLE,
            table=TableBlock(rows=rows, caption=caption),
        )

    @classmethod
    def from_list(cls, items: list[ListItem], ordered: bool = False) -> "ContentBlock":
        """Create a list content block."""
        return cls(
            block_type=ContentBlockType.LIST,
            list_block=ListBlock(items=items, ordered=ordered),
        )


@dataclass
class PageMeta:
    """Metadata extracted from a webpage."""
    title: Optional[str] = None
    meta_description: Optional[str] = None
    h1: Optional[str] = None
    content_blocks: list[str] = field(default_factory=list)
    url: Optional[str] = None
    # Structured content blocks (new - preserves visual layout)
    structured_blocks: list[ContentBlock] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Get all content as a single string."""
        return "\n\n".join(self.content_blocks)

    @property
    def has_structured_content(self) -> bool:
        """Check if structured content is available."""
        return len(self.structured_blocks) > 0


@dataclass
class DocxContent:
    """Content extracted from a Word document."""
    paragraphs: list[ParagraphBlock] = field(default_factory=list)
    source_path: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Get all content as a single string."""
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def h1(self) -> Optional[str]:
        """Get the first H1 heading if present."""
        for p in self.paragraphs:
            if p.heading_level == HeadingLevel.H1:
                return p.text
        return None

    @property
    def headings(self) -> list[ParagraphBlock]:
        """Get all headings."""
        return [p for p in self.paragraphs if p.is_heading]


@dataclass
class ManualKeywordsConfig:
    """Manual keyword selection configuration.

    When provided, bypasses automatic keyword selection and uses
    user-specified keywords directly without filtering or scoring.

    The target_count specifies how many times the primary keyword
    should appear in the optimized content (default: 6).
    Target locations: Title (1), Meta (1), H1 (1), First para (1), Middle (1), Closing (1)
    """
    primary: str  # Required primary keyword phrase
    secondary: list[str] = field(default_factory=list)  # Up to 3 secondary keyword phrases
    target_count: int = 6  # Target occurrences for primary keyword (default: 6)


@dataclass
class KeywordPlan:
    """A plan for keyword optimization."""
    primary: Keyword
    secondary: list[Keyword] = field(default_factory=list)
    long_tail_questions: list[Keyword] = field(default_factory=list)

    @property
    def all_keywords(self) -> list[Keyword]:
        """Get all keywords in priority order."""
        return [self.primary] + self.secondary + self.long_tail_questions

    @property
    def all_phrases(self) -> list[str]:
        """Get all keyword phrases."""
        return [kw.phrase for kw in self.all_keywords]


@dataclass
class MetaElement:
    """Represents a meta element with current and optimized versions."""
    element_name: str  # "Title Tag", "Meta Description", "H1"
    current: Optional[str]
    optimized: str
    why_changed: str

    @property
    def was_changed(self) -> bool:
        """Check if the element was modified."""
        return self.current != self.optimized


@dataclass
class FAQItem:
    """A single FAQ question and answer."""
    question: str
    answer: str


@dataclass
class OptimizationResult:
    """Complete result of content optimization."""
    # Meta elements
    meta_elements: list[MetaElement] = field(default_factory=list)

    # Optimized content blocks (with [[[ADD]]]...[[[ENDADD]]] markers)
    optimized_blocks: list[ParagraphBlock] = field(default_factory=list)

    # FAQ section
    faq_items: list[FAQItem] = field(default_factory=list)

    # Summary info
    primary_keyword: Optional[str] = None
    secondary_keywords: list[str] = field(default_factory=list)

    # Keyword usage counts (phrase -> count in final output)
    keyword_usage_counts: dict[str, int] = field(default_factory=dict)

    @property
    def title_tag(self) -> Optional[MetaElement]:
        """Get the title tag meta element."""
        for elem in self.meta_elements:
            if elem.element_name == "Title Tag":
                return elem
        return None

    @property
    def meta_description(self) -> Optional[MetaElement]:
        """Get the meta description element."""
        for elem in self.meta_elements:
            if elem.element_name == "Meta Description":
                return elem
        return None

    @property
    def h1(self) -> Optional[MetaElement]:
        """Get the H1 element."""
        for elem in self.meta_elements:
            if elem.element_name == "H1":
                return elem
        return None


@dataclass
class ContentAnalysis:
    """Analysis results for content."""
    topic: str
    intent: ContentIntent
    summary: str
    existing_keywords: dict[str, dict] = field(default_factory=dict)  # phrase -> stats
    word_count: int = 0
    heading_count: int = 0
    paragraph_count: int = 0


@dataclass
class KeywordUsageStats:
    """Statistics about keyword usage in content."""
    phrase: str
    count_in_body: int = 0
    in_title: bool = False
    in_meta_description: bool = False
    in_h1: bool = False
    in_headings: bool = False
    in_first_100_words: bool = False


# =============================================================================
# NEW: 10-Part SEO Framework Models
# =============================================================================

@dataclass
class KeywordPlacementStatus:
    """
    Tracks where a specific keyword appears in the content.

    Used during content audit to identify gaps in keyword placement
    according to the tiered hierarchy:
    - Tier 1: Title Tag
    - Tier 2: H1
    - Tier 3: First 100 words
    - Tier 4: Subheadings (H2/H3)
    - Tier 5: Body content
    - Tier 6: Alt text (tracked but not always available)
    - Tier 7: Conclusion
    """
    keyword: str
    in_title: bool = False
    in_meta_description: bool = False
    in_h1: bool = False
    in_first_100_words: bool = False
    in_subheadings: bool = False
    in_body: bool = False
    in_conclusion: bool = False
    body_count: int = 0  # How many times it appears in body

    @property
    def placement_score(self) -> int:
        """
        Calculate a placement score based on tier importance.

        Higher score = better placement coverage.
        """
        score = 0
        if self.in_title:
            score += 10  # Tier 1 - most important
        if self.in_h1:
            score += 9   # Tier 2
        if self.in_first_100_words:
            score += 8   # Tier 3
        if self.in_subheadings:
            score += 6   # Tier 4
        if self.in_body:
            score += 4   # Tier 5
        if self.in_conclusion:
            score += 5   # Tier 7
        if self.in_meta_description:
            score += 7   # Important for CTR
        return score

    @property
    def missing_placements(self) -> list[str]:
        """List placements where keyword is missing."""
        missing = []
        if not self.in_title:
            missing.append("title")
        if not self.in_meta_description:
            missing.append("meta_description")
        if not self.in_h1:
            missing.append("h1")
        if not self.in_first_100_words:
            missing.append("first_100_words")
        if not self.in_subheadings:
            missing.append("subheadings")
        if not self.in_body:
            missing.append("body")
        if not self.in_conclusion:
            missing.append("conclusion")
        return missing


@dataclass
class ContentAudit:
    """
    Comprehensive audit of content following the 10-part SEO framework.

    This captures:
    - Parts 1-4: Understanding what search engines want, target keywords,
      current state audit, and gap identification
    - Part 6: Research/competitive context
    - Part 9: Prioritized recommendations by impact
    """
    # Basic content info
    topic_summary: str  # 1-2 sentence description of what the page is about
    intent: str  # "informational", "transactional", or "mixed"
    word_count: int = 0

    # Current meta state
    current_meta_title: Optional[str] = None
    current_meta_description: Optional[str] = None
    current_h1: Optional[str] = None

    # Structure analysis
    heading_outline: list[str] = field(default_factory=list)  # H1/H2/H3 text in order

    # Keyword placement analysis
    keyword_status: list[KeywordPlacementStatus] = field(default_factory=list)

    # Gap analysis (Part 4)
    depth_gaps: list[str] = field(default_factory=list)  # Missing subtopics searchers expect
    structural_gaps: list[str] = field(default_factory=list)  # e.g., "no FAQ", "no clear conclusion"
    format_opportunities: list[str] = field(default_factory=list)  # e.g., "comparison table", "bullet list"
    technical_opportunities: list[str] = field(default_factory=list)  # e.g., "FAQ schema", "internal links"

    # Prioritized issues (Part 9)
    high_priority_issues: list[str] = field(default_factory=list)  # Meta/H1/keyword presence
    medium_priority_issues: list[str] = field(default_factory=list)  # Content depth
    standard_priority_issues: list[str] = field(default_factory=list)  # Technical polish

    @property
    def primary_keyword_status(self) -> Optional[KeywordPlacementStatus]:
        """Get the placement status for the primary keyword (first in list)."""
        return self.keyword_status[0] if self.keyword_status else None

    @property
    def has_critical_gaps(self) -> bool:
        """Check if there are critical keyword placement gaps."""
        primary = self.primary_keyword_status
        if not primary:
            return True
        return not (primary.in_title and primary.in_h1 and primary.in_first_100_words)


@dataclass
class KeywordPlacementPlan:
    """
    Specifies where each keyword should be placed in the optimized content.

    Implements Part 5 of the framework: Keyword placement hierarchy.
    """
    title: str  # Which keyword goes in title (usually primary)
    meta_description: str  # Which keyword goes in meta description
    h1: str  # Which keyword goes in H1 (usually primary)
    first_100_words: str  # Which keyword must appear early (primary)
    subheadings: list[str] = field(default_factory=list)  # Keywords for H2/H3
    body_priority: list[str] = field(default_factory=list)  # Keywords by priority for body
    faq_keywords: list[str] = field(default_factory=list)  # Keywords to use in FAQ
    conclusion: list[str] = field(default_factory=list)  # Keywords for conclusion


@dataclass
class OptimizationPlan:
    """
    Complete plan for content optimization based on audit results.

    This bridges the gap between analysis and execution, ensuring
    all optimization follows the 10-part framework systematically.
    """
    # Reference to keyword plan
    primary_keyword: str
    secondary_keywords: list[str] = field(default_factory=list)

    # Reference to audit
    audit: Optional[ContentAudit] = None

    # Target meta elements
    target_meta_title: str = ""
    target_meta_description: str = ""
    target_h1: str = ""

    # Structural plan
    sections_to_add: list[str] = field(default_factory=list)  # e.g., ["FAQ section", "Comparison table"]
    sections_to_enhance: list[str] = field(default_factory=list)  # Existing sections needing work
    faq_questions: list[str] = field(default_factory=list)  # Planned FAQ questions

    # Keyword placement mapping
    placement_plan: Optional[KeywordPlacementPlan] = None

    @property
    def all_keywords(self) -> list[str]:
        """Get all keywords in priority order."""
        return [self.primary_keyword] + self.secondary_keywords
