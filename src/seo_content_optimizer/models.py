"""
Data models for SEO Content Optimizer.

This module defines all the core data structures used throughout the application.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Literal
import uuid


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
    CODE = "code"
    IMAGE = "image"


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
class PolymorphicContentBlock:
    """
    A unified content block that can hold different content types.

    This is a polymorphic wrapper that can contain paragraphs, tables, lists, etc.
    Legacy model - use ContentBlock for V2 architecture.
    """
    block_type: ContentBlockType
    paragraph: Optional[ParagraphBlock] = None
    table: Optional[TableBlock] = None
    list_block: Optional[ListBlock] = None

    @classmethod
    def from_paragraph(cls, text: str, heading_level: HeadingLevel = HeadingLevel.BODY) -> "PolymorphicContentBlock":
        """Create a paragraph content block."""
        return cls(
            block_type=ContentBlockType.PARAGRAPH if heading_level == HeadingLevel.BODY else ContentBlockType.HEADING,
            paragraph=ParagraphBlock(text=text, heading_level=heading_level),
        )

    @classmethod
    def from_table(cls, rows: list[TableRow], caption: Optional[str] = None) -> "PolymorphicContentBlock":
        """Create a table content block."""
        return cls(
            block_type=ContentBlockType.TABLE,
            table=TableBlock(rows=rows, caption=caption),
        )

    @classmethod
    def from_list(cls, items: list[ListItem], ordered: bool = False) -> "PolymorphicContentBlock":
        """Create a list content block."""
        return cls(
            block_type=ContentBlockType.LIST,
            list_block=ListBlock(items=items, ordered=ordered),
        )


@dataclass
class ContentBlock:
    """
    Simple content block for V2 architecture.

    Used by structure preservation, factuality guardrails, and change tracking.
    """
    block_id: str
    block_type: ContentBlockType
    text: str
    order: int = 0
    heading_level: Optional[int] = None

    @property
    def word_count(self) -> int:
        """Get word count of this block."""
        return len(self.text.split()) if self.text else 0


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
class KeywordUsageDetail:
    """
    Detailed keyword usage counts distinguishing existing vs added occurrences.

    This is critical for insert-only mode where we need to show:
    - How many times the keyword already existed in source
    - How many NEW occurrences were added by optimization

    Example: If source had "AI hearing aids" twice and we added once more,
    existing=2, added=1, total=3
    """
    keyword: str
    is_primary: bool = False

    # Counts by location in final output
    meta: int = 0       # Meta title + description
    headings: int = 0   # H1-H6 headings
    body: int = 0       # Body paragraphs
    faq: int = 0        # FAQ section

    # Existing counts from source (before optimization)
    existing_meta: int = 0
    existing_headings: int = 0
    existing_body: int = 0
    existing_faq: int = 0

    @property
    def total(self) -> int:
        """Total occurrences in final output."""
        return self.meta + self.headings + self.body + self.faq

    @property
    def existing_total(self) -> int:
        """Total occurrences that existed before optimization."""
        return self.existing_meta + self.existing_headings + self.existing_body + self.existing_faq

    @property
    def added_total(self) -> int:
        """Total NEW occurrences added by optimization."""
        return max(0, self.total - self.existing_total)

    @property
    def added_body(self) -> int:
        """NEW body occurrences added by optimization."""
        return max(0, self.body - self.existing_body)


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
    # Legacy format for backward compatibility
    keyword_usage_counts: dict[str, int] = field(default_factory=dict)

    # Detailed keyword usage with existing vs added breakdown
    # Maps keyword phrase to KeywordUsageDetail
    keyword_usage_detailed: dict[str, "KeywordUsageDetail"] = field(default_factory=dict)

    # Warnings generated during optimization
    warnings: list[str] = field(default_factory=list)

    # Track if FAQ was generated despite inappropriate archetype
    faq_archetype_warning: Optional[str] = None

    # AI Optimization Add-ons (Key Takeaways, Chunk Map)
    # These are stored here for inclusion in the final DOCX output
    ai_addons: Optional["AIAddonsResult"] = None

    # Debug bundle for insert-only mode troubleshooting
    # Only populated when include_debug=True is passed to optimizer
    debug_bundle: Optional[dict] = None


@dataclass
class AIAddonsResult:
    """
    AI Optimization Add-ons result container.

    Contains:
    - Key Takeaways: 3-6 bullet points summarizing the content
    - Chunk Map: Structured content chunks for AI/RAG retrieval
    - FAQs: Fallback FAQ items if LLM generation fails
    """
    key_takeaways: list[str] = field(default_factory=list)
    chunk_map_chunks: list["ChunkData"] = field(default_factory=list)
    chunk_map_stats: Optional["ChunkMapStats"] = None
    faqs: list[dict] = field(default_factory=list)  # [{"question": ..., "answer": ...}]


@dataclass
class ChunkData:
    """
    A single chunk in the Chunk Map.

    Represents a semantically meaningful section of content
    optimized for AI retrieval systems.
    """
    chunk_id: str
    heading_context: str
    summary: str
    best_question: str
    keywords_present: list[str] = field(default_factory=list)
    word_count: int = 0
    token_estimate: int = 0


@dataclass
class ChunkMapStats:
    """Statistics for the Chunk Map."""
    total_chunks: int = 0
    total_words: int = 0
    total_tokens: int = 0

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


# =============================================================================
# V2 ARCHITECTURE: ContentDocument Block Model
# =============================================================================
# This new architecture provides:
# - Typed blocks for precise content representation
# - Run-level formatting preservation for DOCX
# - Better structure for micro-edit per block approach
# - Support for tables/lists as first-class citizens
# =============================================================================

# Block type literals for type safety
BlockType = Literal[
    "title",        # Page/document title
    "meta_title",   # SEO meta title tag
    "meta_desc",    # SEO meta description
    "h1", "h2", "h3", "h4", "h5", "h6",  # Heading levels
    "p",            # Paragraph
    "ul",           # Unordered list container
    "ol",           # Ordered list container
    "li",           # List item
    "table",        # Table container
    "tr",           # Table row
    "td",           # Table cell
    "th",           # Table header cell
    "caption",      # Table/figure caption
    "image",        # Image placeholder
    "blockquote",   # Block quote
    "hr",           # Horizontal rule/section break
]

# Page type classification for content strategy
PageType = Literal[
    "guide",        # How-to, tutorial, educational content
    "service",      # Service page, product description
    "pricing",      # Pricing page
    "comparison",   # Comparison/vs page
    "faq",          # FAQ page
    "policy",       # Legal, privacy, terms
    "blog",         # Blog post, news article
    "landing",      # Landing page, conversion-focused
    "category",     # Category/listing page
    "other",        # Unclassified
]

# Intent classification for SEO strategy
IntentType = Literal[
    "informational",  # User wants to learn/understand
    "transactional",  # User wants to buy/convert
    "navigational",   # User wants to find specific page/brand
    "commercial",     # User researching before purchase
]

# Conversion goal types
ConversionGoal = Literal[
    "get_quote",
    "book_call",
    "download",
    "subscribe",
    "contact",
    "buy",
    "none",
]


@dataclass
class Run:
    """
    A run of text with consistent formatting within a block.

    This preserves DOCX run-level formatting (bold, italic, etc.)
    to enable precise token-level diff highlighting in output.
    """
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    highlight_color: Optional[str] = None  # For diff highlighting
    font_name: Optional[str] = None
    font_size: Optional[float] = None  # In points

    def __post_init__(self) -> None:
        """Ensure text is never None."""
        if self.text is None:
            self.text = ""


@dataclass
class Block:
    """
    A single content block in the document structure.

    This is the fundamental unit of the V2 architecture. Each block has:
    - A unique ID for tracking through optimization
    - A type indicating its semantic role (p, h1, li, table, etc.)
    - Text content (for simple blocks) or runs (for formatted content)
    - Children blocks (for containers like lists and tables)
    - Attributes for type-specific metadata

    The block model supports both flat (URL extraction) and nested
    (DOCX extraction) document structures.
    """
    id: str
    type: BlockType
    text: Optional[str] = None  # Plain text for simple blocks
    runs: Optional[list[Run]] = None  # Formatted text runs (DOCX)
    children: list["Block"] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)  # Type-specific attributes

    # Optimization tracking
    original_text: Optional[str] = None  # Preserved for diff comparison
    was_modified: bool = False  # Track if block was changed during optimization
    skip_optimization: bool = False  # If true, block should not be modified

    def __post_init__(self) -> None:
        """Generate ID if not provided and normalize text."""
        if not self.id:
            self.id = f"block_{uuid.uuid4().hex[:12]}"
        if self.text:
            self.text = self.text.strip()

    @classmethod
    def create(
        cls,
        block_type: BlockType,
        text: Optional[str] = None,
        runs: Optional[list[Run]] = None,
        children: Optional[list["Block"]] = None,
        attrs: Optional[dict] = None,
        block_id: Optional[str] = None,
    ) -> "Block":
        """Factory method for creating blocks with auto-generated IDs."""
        return cls(
            id=block_id or f"block_{uuid.uuid4().hex[:12]}",
            type=block_type,
            text=text,
            runs=runs,
            children=children or [],
            attrs=attrs or {},
        )

    @property
    def plain_text(self) -> str:
        """Get plain text content, combining runs if necessary."""
        if self.text:
            return self.text
        if self.runs:
            return "".join(run.text for run in self.runs)
        return ""

    @property
    def is_heading(self) -> bool:
        """Check if this block is a heading."""
        return self.type in ("h1", "h2", "h3", "h4", "h5", "h6")

    @property
    def is_container(self) -> bool:
        """Check if this block is a container for other blocks."""
        return self.type in ("ul", "ol", "table", "tr")

    @property
    def is_meta(self) -> bool:
        """Check if this block is a meta element (title, description)."""
        return self.type in ("meta_title", "meta_desc", "title")

    @property
    def heading_level(self) -> int:
        """Get numeric heading level (1-6) or 0 for non-headings."""
        level_map = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
        return level_map.get(self.type, 0)

    @property
    def word_count(self) -> int:
        """Get approximate word count for this block and children."""
        count = len(self.plain_text.split())
        for child in self.children:
            count += child.word_count
        return count

    def get_all_text(self) -> str:
        """Get all text including from children, for search purposes."""
        texts = [self.plain_text]
        for child in self.children:
            texts.append(child.get_all_text())
        return " ".join(filter(None, texts))

    def mark_modified(self, new_text: str) -> None:
        """Mark this block as modified and store original."""
        if self.original_text is None:
            self.original_text = self.plain_text
        self.text = new_text
        self.runs = None  # Clear runs when text is directly modified
        self.was_modified = True


@dataclass
class BlockBasedDocument:
    """
    A complete document with typed blocks and metadata.

    This is the root container using Block type. For V2 simple ContentBlock
    usage, see ContentDocument at the end of this file.
    """
    blocks: list[Block] = field(default_factory=list)
    source_url: Optional[str] = None
    source_docx_path: Optional[str] = None
    extracted_title: Optional[str] = None
    extracted_meta_desc: Optional[str] = None
    language: str = "en"

    # Additional metadata
    extraction_timestamp: Optional[str] = None
    original_word_count: int = 0

    def __post_init__(self) -> None:
        """Calculate initial word count."""
        if not self.original_word_count:
            self.original_word_count = self.word_count

    @property
    def word_count(self) -> int:
        """Get total word count across all blocks."""
        return sum(block.word_count for block in self.blocks)

    @property
    def full_text(self) -> str:
        """Get all text content as a single string."""
        texts = []
        for block in self.blocks:
            text = block.get_all_text()
            if text:
                texts.append(text)
        return "\n\n".join(texts)

    @property
    def h1_block(self) -> Optional[Block]:
        """Get the first H1 block if present."""
        for block in self.blocks:
            if block.type == "h1":
                return block
        return None

    @property
    def h1_text(self) -> Optional[str]:
        """Get the H1 text if present."""
        h1 = self.h1_block
        return h1.plain_text if h1 else None

    @property
    def headings(self) -> list[Block]:
        """Get all heading blocks in document order."""
        return [b for b in self.blocks if b.is_heading]

    @property
    def paragraphs(self) -> list[Block]:
        """Get all paragraph blocks."""
        return [b for b in self.blocks if b.type == "p"]

    @property
    def tables(self) -> list[Block]:
        """Get all table blocks."""
        return [b for b in self.blocks if b.type == "table"]

    @property
    def lists(self) -> list[Block]:
        """Get all list blocks (ul and ol)."""
        return [b for b in self.blocks if b.type in ("ul", "ol")]

    def get_block_by_id(self, block_id: str) -> Optional[Block]:
        """Find a block by its ID."""
        for block in self.blocks:
            if block.id == block_id:
                return block
            # Check children
            for child in block.children:
                if child.id == block_id:
                    return child
        return None

    def get_first_n_words(self, n: int = 100) -> str:
        """Get the first N words of body content (excluding meta)."""
        words = []
        for block in self.blocks:
            if block.is_meta:
                continue
            block_words = block.plain_text.split()
            words.extend(block_words)
            if len(words) >= n:
                break
        return " ".join(words[:n])

    def get_conclusion_blocks(self, last_n: int = 3) -> list[Block]:
        """Get the last N paragraph blocks as conclusion."""
        paragraphs = self.paragraphs
        return paragraphs[-last_n:] if len(paragraphs) >= last_n else paragraphs

    def get_modified_blocks(self) -> list[Block]:
        """Get all blocks that were modified during optimization."""
        return [b for b in self.blocks if b.was_modified]

    def iter_all_blocks(self) -> list[Block]:
        """Iterate through all blocks including children (flattened)."""
        result = []
        for block in self.blocks:
            result.append(block)
            result.extend(self._flatten_children(block))
        return result

    def _flatten_children(self, block: Block) -> list[Block]:
        """Recursively flatten children blocks."""
        result = []
        for child in block.children:
            result.append(child)
            result.extend(self._flatten_children(child))
        return result


@dataclass
class TopicFingerprint:
    """
    Semantic understanding of the content's topic and purpose.

    This captures the essence of what the content is about, enabling:
    - Semantic keyword relevance scoring
    - Appropriate content strategy selection
    - Factuality guardrails (what claims are appropriate)
    """
    keyphrases: list[str] = field(default_factory=list)  # Core topic phrases
    entities: list[str] = field(default_factory=list)  # Named entities (brands, products, etc.)
    summary: str = ""  # 1-2 sentence topic summary
    page_type: PageType = "other"

    # Additional semantic info
    industry: Optional[str] = None  # e.g., "healthcare", "finance", "technology"
    target_audience: Optional[str] = None  # e.g., "B2B decision makers", "consumers"
    content_depth: Literal["shallow", "medium", "deep"] = "medium"

    @property
    def is_ymyl(self) -> bool:
        """Check if content is Your Money Your Life (requires extra care)."""
        ymyl_industries = {"healthcare", "finance", "legal", "insurance", "medical"}
        if self.industry and self.industry.lower() in ymyl_industries:
            return True
        # Check for YMYL signals in keyphrases
        ymyl_signals = {"health", "medical", "doctor", "investment", "loan", "legal", "lawyer"}
        for phrase in self.keyphrases:
            if any(signal in phrase.lower() for signal in ymyl_signals):
                return True
        return False


@dataclass
class IntentClassification:
    """
    Detailed intent classification for SEO strategy.

    Goes beyond simple informational/transactional to provide
    actionable insights for content optimization.
    """
    primary_intent: IntentType = "informational"
    secondary_intent: Optional[IntentType] = None
    confidence: float = 0.0  # 0.0 to 1.0
    conversion_goal: ConversionGoal = "none"
    notes: str = ""

    # Intent signals detected
    transactional_signals: list[str] = field(default_factory=list)  # e.g., "pricing", "buy now"
    informational_signals: list[str] = field(default_factory=list)  # e.g., "how to", "guide"

    @property
    def is_conversion_focused(self) -> bool:
        """Check if content should focus on conversion."""
        return self.primary_intent in ("transactional", "commercial") or self.conversion_goal != "none"

    @property
    def needs_cta(self) -> bool:
        """Check if content should include call-to-action."""
        return self.conversion_goal != "none"


@dataclass
class KeywordVariant:
    """
    A keyword with its variants for flexible matching.

    Supports pluralization, acronym expansion, and synonyms
    to enable smarter keyword detection and placement.
    """
    canonical: str  # The primary form of the keyword
    variants: list[str] = field(default_factory=list)  # Alternative forms

    # Variant types
    plurals: list[str] = field(default_factory=list)
    acronyms: list[str] = field(default_factory=list)  # e.g., "SEO" <-> "search engine optimization"
    synonyms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Build variants list from all sources."""
        all_variants = set(self.variants)
        all_variants.update(self.plurals)
        all_variants.update(self.acronyms)
        all_variants.update(self.synonyms)
        all_variants.discard(self.canonical)
        self.variants = list(all_variants)

    @property
    def all_forms(self) -> list[str]:
        """Get canonical form plus all variants."""
        return [self.canonical] + self.variants

    def matches(self, text: str) -> bool:
        """Check if any form of this keyword appears in text."""
        text_lower = text.lower()
        for form in self.all_forms:
            if form.lower() in text_lower:
                return True
        return False

    def count_in_text(self, text: str) -> int:
        """Count total occurrences of any form in text."""
        text_lower = text.lower()
        count = 0
        for form in self.all_forms:
            count += text_lower.count(form.lower())
        return count


@dataclass
class SemanticKeyword:
    """
    A keyword with semantic relevance scoring.

    Replaces simple lexical matching with embedding-based
    relevance to prevent off-topic keyword acceptance.
    """
    phrase: str
    relevance_score: float = 0.0  # 0.0 to 1.0, based on embedding similarity
    search_volume: Optional[int] = None
    difficulty: Optional[float] = None
    intent: Optional[str] = None
    is_brand: bool = False

    # Semantic analysis
    cluster_id: Optional[str] = None  # For grouping similar keywords
    variants: Optional[KeywordVariant] = None

    # Selection status
    selected: bool = False
    skip_reason: Optional[str] = None  # Why keyword was not selected

    def __post_init__(self) -> None:
        """Normalize the phrase."""
        self.phrase = self.phrase.strip()

    @property
    def is_relevant(self) -> bool:
        """Check if keyword meets relevance threshold."""
        return self.relevance_score >= 0.5  # Configurable threshold

    @property
    def is_question(self) -> bool:
        """Check if the keyword is a question."""
        question_starters = ('how', 'what', 'why', 'when', 'where', 'who', 'which', 'can', 'does', 'is', 'are')
        lower_phrase = self.phrase.lower()
        return lower_phrase.endswith('?') or lower_phrase.startswith(question_starters)


@dataclass
class KeywordCluster:
    """
    A cluster of semantically similar keywords.

    Used to avoid selecting multiple keywords that are essentially
    the same concept (e.g., "SEO tools" and "SEO software").
    """
    id: str
    keywords: list[SemanticKeyword] = field(default_factory=list)
    centroid_phrase: str = ""  # Representative phrase for the cluster

    @property
    def primary_keyword(self) -> Optional[SemanticKeyword]:
        """Get the best keyword from this cluster."""
        if not self.keywords:
            return None
        # Prefer highest relevance, then highest search volume
        return max(
            self.keywords,
            key=lambda k: (k.relevance_score, k.search_volume or 0)
        )

    @property
    def size(self) -> int:
        """Number of keywords in cluster."""
        return len(self.keywords)


@dataclass
class SemanticKeywordPlan:
    """
    Enhanced keyword plan with semantic scoring and clustering.

    Replaces the simple KeywordPlan with smarter selection
    that prevents off-topic keywords and respects clustering.
    """
    primary: SemanticKeyword
    secondary: list[SemanticKeyword] = field(default_factory=list)
    questions: list[SemanticKeyword] = field(default_factory=list)

    # Clustering info
    clusters: list[KeywordCluster] = field(default_factory=list)

    # Selection tracking
    skipped_keywords: list[SemanticKeyword] = field(default_factory=list)  # Keywords not used
    skip_reasons: dict[str, str] = field(default_factory=dict)  # phrase -> reason

    @property
    def all_selected(self) -> list[SemanticKeyword]:
        """Get all selected keywords in priority order."""
        return [self.primary] + self.secondary + self.questions

    @property
    def all_phrases(self) -> list[str]:
        """Get all selected keyword phrases."""
        return [kw.phrase for kw in self.all_selected]

    def get_keyword(self, phrase: str) -> Optional[SemanticKeyword]:
        """Find a keyword by phrase."""
        for kw in self.all_selected + self.skipped_keywords:
            if kw.phrase.lower() == phrase.lower():
                return kw
        return None


@dataclass
class FactualityClaim:
    """
    A potentially factual claim detected in content.

    Used by the factuality guardrail to identify claims
    that need validation or removal.
    """
    claim_text: str  # The claim text
    claim_type: str  # Type of claim (percentage, year, certification, etc.)
    source_block_id: str = ""  # Which block contains this claim
    source_sentence: str = ""  # The context/sentence containing the claim
    severity: str = "medium"  # Severity level (high, medium, low)
    is_new: bool = False  # Whether this is a newly detected claim (not in original)

    # Validation
    is_validated: bool = False
    validation_source: Optional[str] = None  # Where the claim was verified
    should_remove: bool = False  # If true, claim should be removed
    removal_reason: Optional[str] = None

    # Replacement (if claim should be modified rather than removed)
    replacement_text: Optional[str] = None

    @property
    def text(self) -> str:
        """Alias for claim_text for compatibility."""
        return self.claim_text

    @property
    def block_id(self) -> str:
        """Alias for source_block_id for compatibility."""
        return self.source_block_id


@dataclass
class OptimizationReport:
    """
    Comprehensive report of optimization actions taken.

    Provides transparency into what was changed and why,
    supporting the "explain everything" philosophy.
    """
    # Summary stats
    blocks_analyzed: int = 0
    blocks_modified: int = 0
    blocks_skipped: int = 0

    # Keyword tracking
    keywords_selected: list[str] = field(default_factory=list)
    keywords_skipped: list[str] = field(default_factory=list)
    keyword_skip_reasons: dict[str, str] = field(default_factory=dict)

    # Factuality tracking
    claims_detected: int = 0
    claims_validated: int = 0
    claims_removed: int = 0
    removed_claims: list[FactualityClaim] = field(default_factory=list)

    # Block-level changes
    block_changes: list[dict] = field(default_factory=list)  # {block_id, original, modified, reason}

    # Warnings and notes
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_block_change(
        self,
        block_id: str,
        original: str,
        modified: str,
        reason: str
    ) -> None:
        """Record a block modification."""
        self.block_changes.append({
            "block_id": block_id,
            "original": original,
            "modified": modified,
            "reason": reason,
        })
        self.blocks_modified += 1

    def add_skipped_keyword(self, phrase: str, reason: str) -> None:
        """Record a skipped keyword with reason."""
        self.keywords_skipped.append(phrase)
        self.keyword_skip_reasons[phrase] = reason

    def add_warning(self, warning: str) -> None:
        """Add a warning to the report."""
        self.warnings.append(warning)

    @property
    def modification_rate(self) -> float:
        """Get the percentage of blocks that were modified."""
        if self.blocks_analyzed == 0:
            return 0.0
        return (self.blocks_modified / self.blocks_analyzed) * 100


@dataclass
class ContentDocument:
    """
    V2-compatible content document with simple ContentBlock.

    Used by structure preservation, factuality guardrails, and change tracking.
    """
    title: str
    blocks: list[ContentBlock] = field(default_factory=list)
    source_url: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Get all text content as a single string."""
        return "\n\n".join(block.text for block in self.blocks if block.text)

    @property
    def paragraphs(self) -> list[ContentBlock]:
        """Get all paragraph blocks."""
        return [b for b in self.blocks if b.block_type == ContentBlockType.PARAGRAPH]

    @property
    def headings(self) -> list[ContentBlock]:
        """Get all heading blocks."""
        return [b for b in self.blocks if b.block_type == ContentBlockType.HEADING]

    @property
    def word_count(self) -> int:
        """Get total word count."""
        return sum(block.word_count for block in self.blocks)
