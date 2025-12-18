"""
SEO Content Optimizer

A fully automated SEO content optimization tool that:
- Accepts content from URLs or Word documents
- Analyzes keywords from CSV/Excel files
- Produces optimized Word documents with green-highlighted changes
"""

__version__ = "1.0.0"
__author__ = "SEO Content Optimizer Team"

from .config import OptimizationConfig, OptimizationMode

from .models import (
    Keyword,
    PageMeta,
    DocxContent,
    ParagraphBlock,
    KeywordPlan,
    OptimizationResult,
    MetaElement,
    KeywordUsageDetail,
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

# Locked token protection (URLs, emails, phones)
from .locked_tokens import (
    LockedTokenProtector,
    protect_locked_tokens,
    restore_locked_tokens,
    validate_urls_preserved,
    detect_url_corruption,
)

# Highlight integrity validation
from .diff_markers import (
    compute_title_markers,
    compute_h1_markers,
    compute_meta_desc_markers,
    add_markers_with_url_protection,
    validate_unhighlighted_matches_source,
    get_highlight_integrity_report,
    # Token-level diff (precise highlighting)
    compute_markers_token_level,
    compute_markers_unified,
)

# Cap validation and debug bundle (insert-only mode)
from .optimizer import (
    KeywordCapValidationResult,
    CapValidationReport,
    DebugBundle,
    DebugBundleConfig,
    DebugBundleKeyword,
    RunManifest,
)

# Post-optimization enforcement (insert-only mode)
from .enforcement import (
    run_enforcement,
    enforce_keyword_caps,
    enforce_keyword_delta_budgets,
    enforce_budget_limits,
    validate_insertions_have_keywords,
    EnforcementResult,
    MarkerSpan,
    DeltaBudgetResult,
    # Strip-additions validator
    strip_marked_additions,
    validate_strip_additions,
    get_strip_additions_report,
    StripAdditionsResult,
)

# Highlight integrity validation
from .highlight_integrity import (
    run_highlight_integrity_check,
    get_highlight_diff_summary,
    HighlightIntegrityReport,
    HighlightIssue,
)

__all__ = [
    # Configuration
    "OptimizationConfig",
    "OptimizationMode",
    # Legacy models
    "Keyword",
    "PageMeta",
    "DocxContent",
    "ParagraphBlock",
    "KeywordPlan",
    "OptimizationResult",
    "MetaElement",
    "KeywordUsageDetail",
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
    # Highlight integrity (diff markers)
    "compute_title_markers",
    "compute_h1_markers",
    "compute_meta_desc_markers",
    "add_markers_with_url_protection",
    "validate_unhighlighted_matches_source",
    "get_highlight_integrity_report",
    "compute_markers_token_level",
    "compute_markers_unified",
    # Cap validation and debug bundle (insert-only mode)
    "KeywordCapValidationResult",
    "CapValidationReport",
    "DebugBundle",
    "DebugBundleConfig",
    "DebugBundleKeyword",
    "RunManifest",
    # Post-optimization enforcement (insert-only mode)
    "run_enforcement",
    "enforce_keyword_caps",
    "enforce_keyword_delta_budgets",
    "enforce_budget_limits",
    "validate_insertions_have_keywords",
    "EnforcementResult",
    "MarkerSpan",
    "DeltaBudgetResult",
    # Strip-additions validator
    "strip_marked_additions",
    "validate_strip_additions",
    "get_strip_additions_report",
    "StripAdditionsResult",
    # Highlight integrity validation
    "run_highlight_integrity_check",
    "get_highlight_diff_summary",
    "HighlightIntegrityReport",
    "HighlightIssue",
]
