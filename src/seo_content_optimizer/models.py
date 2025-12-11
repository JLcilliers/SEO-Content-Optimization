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

    def __post_init__(self) -> None:
        """Normalize the keyword phrase."""
        self.phrase = self.phrase.strip()

    @property
    def is_question(self) -> bool:
        """Check if the keyword is a question."""
        question_starters = ('how', 'what', 'why', 'when', 'where', 'who', 'which', 'can', 'does', 'is', 'are')
        lower_phrase = self.phrase.lower()
        return lower_phrase.endswith('?') or lower_phrase.startswith(question_starters)


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
class PageMeta:
    """Metadata extracted from a webpage."""
    title: Optional[str] = None
    meta_description: Optional[str] = None
    h1: Optional[str] = None
    content_blocks: list[str] = field(default_factory=list)
    url: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Get all content as a single string."""
        return "\n\n".join(self.content_blocks)


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
