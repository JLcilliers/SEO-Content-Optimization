"""
SEO Content Optimizer

A fully automated SEO content optimization tool that:
- Accepts content from URLs or Word documents
- Analyzes keywords from CSV/Excel files
- Produces optimized Word documents with green-highlighted changes
"""

__version__ = "1.0.0"
__author__ = "SEO Content Optimizer Team"

from .config import OptimizationConfig

from .models import (
    Keyword,
    PageMeta,
    DocxContent,
    ParagraphBlock,
    KeywordPlan,
    OptimizationResult,
    MetaElement,
    # V2 Architecture models
    ContentBlock,
    ContentBlockType,
    ContentDocument,
    SemanticKeyword,
    SemanticKeywordPlan,
    FactualityClaim,
)

# V2 Architecture modules
from .structure_preservation import (
    StructurePreserver,
    StructurePolicy,
    PreservationPolicy,
    get_modifiable_blocks,
)

from .factuality_guardrails import (
    FactualityChecker,
    validate_no_new_facts,
)

from .diff_highlighter import (
    TokenDiffer,
    DiffResult,
    DiffType,
    compute_diff,
    get_changes_summary,
)

from .change_summary import (
    ChangeSummaryBuilder,
    OptimizationSummary,
    format_summary_text,
    format_summary_dict,
)

# V2 Optimizer
from .optimizer_v2 import (
    ContentOptimizerV2,
    V2OptimizationConfig,
    optimize_content_v2,
)

# Pipeline improvement modules (2024 reliability fixes)
from .text_repair import (
    repair_text,
    repair_mojibake,
    normalize_whitespace,
    validate_text_quality,
    repair_content_blocks,
)

from .page_archetype import (
    detect_page_archetype,
    filter_guide_phrases,
    get_content_guidance,
    ArchetypeResult,
)

from .claim_validator import (
    extract_facts_ledger,
    validate_generated_content,
    remove_hallucinated_claims,
    get_allowed_claims,
    FactsLedger,
)

from .repetition_guard import (
    find_duplicate_sentences,
    find_near_duplicate_sentences,
    remove_duplicate_sentences,
    check_keyword_density,
    clean_repetition,
    KeywordDensityConfig,
)

__all__ = [
    # Configuration
    "OptimizationConfig",
    # Legacy models
    "Keyword",
    "PageMeta",
    "DocxContent",
    "ParagraphBlock",
    "KeywordPlan",
    "OptimizationResult",
    "MetaElement",
    # V2 Architecture models
    "ContentBlock",
    "ContentBlockType",
    "ContentDocument",
    "SemanticKeyword",
    "SemanticKeywordPlan",
    "FactualityClaim",
    # V2 Structure preservation
    "StructurePreserver",
    "StructurePolicy",
    "PreservationPolicy",
    "get_modifiable_blocks",
    # V2 Factuality
    "FactualityChecker",
    "validate_no_new_facts",
    # V2 Diff highlighting
    "TokenDiffer",
    "DiffResult",
    "DiffType",
    "compute_diff",
    "get_changes_summary",
    # V2 Change summary
    "ChangeSummaryBuilder",
    "OptimizationSummary",
    "format_summary_text",
    "format_summary_dict",
    # V2 Optimizer
    "ContentOptimizerV2",
    "V2OptimizationConfig",
    "optimize_content_v2",
    # Pipeline improvements (2024 reliability fixes)
    # Text repair
    "repair_text",
    "repair_mojibake",
    "normalize_whitespace",
    "validate_text_quality",
    "repair_content_blocks",
    # Page archetype
    "detect_page_archetype",
    "filter_guide_phrases",
    "get_content_guidance",
    "ArchetypeResult",
    # Claim validation
    "extract_facts_ledger",
    "validate_generated_content",
    "remove_hallucinated_claims",
    "get_allowed_claims",
    "FactsLedger",
    # Repetition guard
    "find_duplicate_sentences",
    "find_near_duplicate_sentences",
    "remove_duplicate_sentences",
    "check_keyword_density",
    "clean_repetition",
    "KeywordDensityConfig",
]
