"""
SEO Content Optimizer

A fully automated SEO content optimization tool that:
- Accepts content from URLs or Word documents
- Analyzes keywords from CSV/Excel files
- Produces optimized Word documents with green-highlighted changes
"""

__version__ = "1.0.0"
__author__ = "SEO Content Optimizer Team"

from .models import (
    Keyword,
    PageMeta,
    DocxContent,
    ParagraphBlock,
    KeywordPlan,
    OptimizationResult,
    MetaElement,
)

__all__ = [
    "Keyword",
    "PageMeta",
    "DocxContent",
    "ParagraphBlock",
    "KeywordPlan",
    "OptimizationResult",
    "MetaElement",
]
