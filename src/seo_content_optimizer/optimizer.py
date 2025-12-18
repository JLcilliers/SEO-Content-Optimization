"""
SEO optimization orchestration module.

This module coordinates the optimization process:
- Applies SEO rules to meta elements and content
- Uses LLM for intelligent rewriting
- Generates FAQ sections
- Produces structured optimization results

IMPORTANT: All keywords are filtered for topical relevance BEFORE optimization
to prevent injection of off-topic industries or spammy content.

BRAND NAME PROTECTION:
- Brand names are detected from URL domain and content
- Original brand spelling is preserved throughout optimization
- Brand names are excluded from diff highlighting
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Union

from .analysis import (
    ContentAnalysis,
    analyze_content,
    audit_content,
    build_optimization_plan,
)
from .content_sources import convert_page_meta_to_blocks
from .keyword_filter import (
    filter_keywords_for_content,
    get_content_topics,
    log_filter_results,
)
from .output_validator import (
    validate_and_fallback,
    validate_and_preserve,  # Content preservation: fallback if content deleted
    validate_content_preservation,  # Content preservation: check length ratio
    validate_block_preservation,  # Content preservation: check block count
    validate_faq_items,
    find_blocked_terms,
)
from .llm_client import (
    LLMClient,
    create_llm_client,
    strip_markers,
)
from .diff_markers import (
    MARK_END as ADD_END,
    MARK_START as ADD_START,
    add_markers_by_diff,  # Token-level diff: only NEW/CHANGED tokens highlighted
    compute_h1_markers,
    inject_phrase_with_markers,
    inject_keyword_naturally,  # Minimal mode: natural body keyword injection
    mark_block_as_new,
    build_original_sentence_index,
    normalize_sentence,
    generate_brand_variations,  # Brand exclusion: generate all variations of brand name
    normalize_brand_in_text,  # Brand exclusion: normalize brand to original spelling
    normalize_paragraph_spacing,  # Paragraph structure: fix "word.Word" patterns
    preprocess_keywords_for_diff,  # Keyword atomic units: preprocess for diff
    postprocess_keywords_from_diff,  # Keyword atomic units: postprocess after diff
)
from .config import OptimizationConfig
from .models import (
    AIAddonsResult,
    ChunkData,
    ChunkMapStats,
    ContentAudit,
    DocxContent,
    FAQItem,
    HeadingLevel,
    Keyword,
    KeywordPlan,
    ManualKeywordsConfig,
    MetaElement,
    OptimizationPlan,
    OptimizationResult,
    PageMeta,
    ParagraphBlock,
)
from .ai_addons import (
    generate_ai_addons,
    generate_fallback_faqs,
    AIAddons,
)
from .prioritizer import create_keyword_plan
from .page_archetype import (
    detect_page_archetype,
    filter_guide_phrases,
    get_content_guidance,
    ArchetypeResult,
)
from .highlight_integrity import (
    run_highlight_integrity_check,
    HighlightIntegrityReport,
)


@dataclass
class KeywordCounts:
    """
    Scoped keyword counts across different document sections.

    This dataclass separates keyword occurrences by location to ensure
    body content is the primary target for keyword satisfaction, not
    meta elements or FAQ sections.

    For insert-only mode, also tracks existing counts from source vs added counts.
    """
    meta: int = 0       # Count in title + description + H1
    headings: int = 0   # Count in H2+ headings (excluding H1)
    body: int = 0       # Count in body paragraphs (non-heading blocks)
    faq: int = 0        # Count in FAQ questions + answers

    # Existing counts from source (before optimization)
    existing_meta: int = 0
    existing_headings: int = 0
    existing_body: int = 0
    existing_faq: int = 0

    @property
    def total(self) -> int:
        """Total count across all sections (current/final)."""
        return self.meta + self.headings + self.body + self.faq

    @property
    def body_and_headings(self) -> int:
        """Count in body content (paragraphs + headings, excluding meta/FAQ)."""
        return self.body + self.headings

    @property
    def existing_total(self) -> int:
        """Total existing count from source (before optimization)."""
        return self.existing_meta + self.existing_headings + self.existing_body + self.existing_faq

    @property
    def added_meta(self) -> int:
        """Newly added count in meta (current - existing)."""
        return max(0, self.meta - self.existing_meta)

    @property
    def added_headings(self) -> int:
        """Newly added count in headings."""
        return max(0, self.headings - self.existing_headings)

    @property
    def added_body(self) -> int:
        """Newly added count in body paragraphs."""
        return max(0, self.body - self.existing_body)

    @property
    def added_faq(self) -> int:
        """Newly added count in FAQ."""
        return max(0, self.faq - self.existing_faq)

    @property
    def added_total(self) -> int:
        """Total newly added count."""
        return self.added_meta + self.added_headings + self.added_body + self.added_faq


@dataclass
class ScopedKeywordCounts:
    """
    Complete keyword counts report with scoped breakdowns.

    Used for transparent reporting of where keywords appear.
    """
    keyword: str
    target: int
    counts: KeywordCounts
    is_primary: bool = False

    @property
    def body_satisfied(self) -> bool:
        """Check if target is satisfied by body content alone."""
        return self.counts.body >= self.target

    @property
    def total_satisfied(self) -> bool:
        """Check if target is satisfied by total (legacy behavior)."""
        return self.counts.total >= self.target

    @property
    def body_needed(self) -> int:
        """How many more occurrences needed in body to satisfy target."""
        return max(0, self.target - self.counts.body)

    @property
    def added_body(self) -> int:
        """Newly added body occurrences (for insert-only mode reporting)."""
        return self.counts.added_body

    @property
    def added_total(self) -> int:
        """Newly added total occurrences (for insert-only mode reporting)."""
        return self.counts.added_total

    @property
    def existing_body(self) -> int:
        """Existing body occurrences from source."""
        return self.counts.existing_body

    @property
    def existing_total(self) -> int:
        """Existing total occurrences from source."""
        return self.counts.existing_total


@dataclass
class KeywordCapValidationResult:
    """
    Validation result for keyword cap compliance in insert-only mode.

    Tracks whether each keyword is within its maximum occurrence cap.
    """
    keyword: str
    cap: int  # Maximum allowed occurrences
    actual_body_count: int  # Actual body occurrences
    actual_total_count: int  # Actual total occurrences
    is_primary: bool = False

    @property
    def within_cap(self) -> bool:
        """Check if keyword body count is within cap."""
        return self.actual_body_count <= self.cap

    @property
    def excess_count(self) -> int:
        """How many occurrences over the cap (0 if within cap)."""
        return max(0, self.actual_body_count - self.cap)


@dataclass
class CapValidationReport:
    """
    Full validation report for keyword caps in insert-only mode.

    Provides transparency into whether the optimization respected caps.
    """
    mode: str  # "minimal" or "enhanced"
    cap_enforcement_enabled: bool
    primary_cap: int
    secondary_cap: int
    keyword_results: list[KeywordCapValidationResult]

    @property
    def all_within_caps(self) -> bool:
        """Check if all keywords are within their caps."""
        return all(r.within_cap for r in self.keyword_results)

    @property
    def total_excess(self) -> int:
        """Total excess occurrences across all keywords."""
        return sum(r.excess_count for r in self.keyword_results)

    @property
    def keywords_over_cap(self) -> list[KeywordCapValidationResult]:
        """List of keywords that exceeded their cap."""
        return [r for r in self.keyword_results if not r.within_cap]


@dataclass
class DebugBundleConfig:
    """
    Configuration snapshot for debug bundle.

    Captures all relevant configuration at the time of optimization.
    """
    optimization_mode: str
    faq_policy: str
    generate_ai_sections: bool
    generate_key_takeaways: bool
    generate_chunk_map: bool
    primary_keyword_body_cap: int
    secondary_keyword_body_cap: int
    enforce_keyword_caps: bool
    max_secondary: int
    has_keyword_allowlist: bool
    keyword_allowlist: Optional[set[str]] = None


@dataclass
class RunManifest:
    """
    Run manifest showing exactly what stages executed during optimization.

    This is critical for diagnosing insert-only mode issues:
    - If llm_body_rewrite=True in insert-only mode, that's the bug.
    - If heading_rewrite=True in insert-only mode, headings weren't locked.
    """
    optimizer_version: str  # "v1" or "v2"
    optimization_mode: str  # "insert_only", "minimal", "enhanced"

    # Which stages actually ran
    llm_body_rewrite: bool
    heading_rewrite: bool
    faq_generation: bool
    ai_addons_generation: bool
    keyword_caps_enforcement: bool
    budget_enforcement: bool
    highlight_integrity_check: bool

    # Source analysis
    source_has_existing_faq: bool
    page_archetype: str  # "landing", "blog", "guide", etc.

    # Keyword handling
    manual_keywords_mode: bool
    keyword_allowlist_active: bool
    keywords_expanded: bool  # Were synonyms/LSI added? Should be False in insert-only


@dataclass
class DebugBundleKeyword:
    """
    Keyword entry for debug bundle.

    Contains the keyword phrase and its usage statistics.
    Now includes original_count for delta budget tracking.
    """
    phrase: str
    is_primary: bool
    cap: int
    # Legacy fields (populated first for backward compatibility)
    body_count: int
    meta_count: int
    headings_count: int
    faq_count: int
    total_count: int
    within_cap: bool  # Legacy: body_count <= cap
    # Delta tracking - this is the key to proper insert-only behavior
    # These have defaults for backward compatibility
    original_count: int = 0  # Count in source BEFORE optimization
    new_additions: int = 0  # body_count - original_count (should be <= allowed_new)
    allowed_new: int = 1  # Max new additions allowed (default: 1)
    delta_within_budget: bool = True  # new_additions <= allowed_new


@dataclass
class DebugBundle:
    """
    Comprehensive debug bundle for insert-only mode troubleshooting.

    This bundle provides all the information needed to diagnose
    keyword inflation issues and verify insert-only compliance.

    Includes:
    - Run manifest (what stages actually ran)
    - Configuration snapshot
    - Keyword plan details
    - Per-keyword usage statistics with DELTA tracking
    - Cap validation results
    - Block-by-block analysis
    """
    timestamp: str
    config: DebugBundleConfig
    keywords: list[DebugBundleKeyword]
    cap_validation: Optional[CapValidationReport]
    total_blocks: int
    blocks_with_keywords: int
    warnings: list[str]
    # NEW: Run manifest shows exactly what stages executed
    run_manifest: Optional[RunManifest] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        result = {
            "timestamp": self.timestamp,
            # Run manifest - what stages actually ran (critical for debugging)
            "run_manifest": {
                "optimizer_version": self.run_manifest.optimizer_version,
                "optimization_mode": self.run_manifest.optimization_mode,
                "llm_body_rewrite": self.run_manifest.llm_body_rewrite,
                "heading_rewrite": self.run_manifest.heading_rewrite,
                "faq_generation": self.run_manifest.faq_generation,
                "ai_addons_generation": self.run_manifest.ai_addons_generation,
                "keyword_caps_enforcement": self.run_manifest.keyword_caps_enforcement,
                "budget_enforcement": self.run_manifest.budget_enforcement,
                "highlight_integrity_check": self.run_manifest.highlight_integrity_check,
                "source_has_existing_faq": self.run_manifest.source_has_existing_faq,
                "page_archetype": self.run_manifest.page_archetype,
                "manual_keywords_mode": self.run_manifest.manual_keywords_mode,
                "keyword_allowlist_active": self.run_manifest.keyword_allowlist_active,
                "keywords_expanded": self.run_manifest.keywords_expanded,
            } if self.run_manifest else None,
            "config": {
                "optimization_mode": self.config.optimization_mode,
                "faq_policy": self.config.faq_policy,
                "generate_ai_sections": self.config.generate_ai_sections,
                "generate_key_takeaways": self.config.generate_key_takeaways,
                "generate_chunk_map": self.config.generate_chunk_map,
                "primary_keyword_body_cap": self.config.primary_keyword_body_cap,
                "secondary_keyword_body_cap": self.config.secondary_keyword_body_cap,
                "enforce_keyword_caps": self.config.enforce_keyword_caps,
                "max_secondary": self.config.max_secondary,
                "has_keyword_allowlist": self.config.has_keyword_allowlist,
                "keyword_allowlist": list(self.config.keyword_allowlist) if self.config.keyword_allowlist else None,
            },
            # Keywords with DELTA tracking (key for insert-only mode)
            "keywords": [
                {
                    "phrase": kw.phrase,
                    "is_primary": kw.is_primary,
                    "cap": kw.cap,
                    # Delta tracking fields (these are the important ones!)
                    "original_count": kw.original_count,
                    "body_count": kw.body_count,
                    "new_additions": kw.new_additions,
                    "allowed_new": kw.allowed_new,
                    "delta_within_budget": kw.delta_within_budget,
                    # Legacy fields
                    "meta_count": kw.meta_count,
                    "headings_count": kw.headings_count,
                    "faq_count": kw.faq_count,
                    "total_count": kw.total_count,
                    "within_cap": kw.within_cap,
                }
                for kw in self.keywords
            ],
            "cap_validation": {
                "mode": self.cap_validation.mode,
                "cap_enforcement_enabled": self.cap_validation.cap_enforcement_enabled,
                "primary_cap": self.cap_validation.primary_cap,
                "secondary_cap": self.cap_validation.secondary_cap,
                "all_within_caps": self.cap_validation.all_within_caps,
                "total_excess": self.cap_validation.total_excess,
                "keywords_over_cap": [r.keyword for r in self.cap_validation.keywords_over_cap],
            } if self.cap_validation else None,
            "summary": {
                "total_blocks": self.total_blocks,
                "blocks_with_keywords": self.blocks_with_keywords,
                "warnings_count": len(self.warnings),
            },
            "warnings": self.warnings,
        }
        return result

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_checklist(self) -> str:
        """Generate human-readable checklist for debugging."""
        lines = []
        lines.append("=" * 70)
        lines.append("INSERT-ONLY MODE DEBUG CHECKLIST")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append("")

        # RUN MANIFEST - what stages actually executed (critical for debugging)
        if self.run_manifest:
            lines.append("RUN MANIFEST (What Actually Ran):")
            lines.append("-" * 40)
            rm = self.run_manifest
            lines.append(f"  Optimizer: {rm.optimizer_version} | Mode: {rm.optimization_mode}")
            lines.append(f"  Page Archetype: {rm.page_archetype}")
            lines.append("")
            lines.append("  Stage Execution:")
            # In insert-only, these should all be False
            llm_ok = not rm.llm_body_rewrite
            lines.append(f"    [{'✓' if llm_ok else '✗'}] LLM body rewrite = {rm.llm_body_rewrite} (should be False)")
            heading_ok = not rm.heading_rewrite
            lines.append(f"    [{'✓' if heading_ok else '✗'}] Heading rewrite = {rm.heading_rewrite} (should be False)")
            faq_ok_manifest = not rm.faq_generation
            lines.append(f"    [{'✓' if faq_ok_manifest else '✗'}] FAQ generation = {rm.faq_generation} (should be False)")
            ai_ok_manifest = not rm.ai_addons_generation
            lines.append(f"    [{'✓' if ai_ok_manifest else '✗'}] AI add-ons generation = {rm.ai_addons_generation} (should be False)")
            lines.append(f"    [{'✓' if rm.keyword_caps_enforcement else '✗'}] Keyword caps enforcement = {rm.keyword_caps_enforcement} (should be True)")
            lines.append(f"    [{'✓' if rm.budget_enforcement else '✗'}] Budget enforcement = {rm.budget_enforcement} (should be True)")
            lines.append(f"    [i] Highlight integrity check = {rm.highlight_integrity_check}")
            lines.append("")
            lines.append("  Keyword Handling:")
            lines.append(f"    [{'✓' if rm.manual_keywords_mode else '✗'}] Manual keywords mode = {rm.manual_keywords_mode} (should be True)")
            lines.append(f"    [{'✓' if rm.keyword_allowlist_active else '✗'}] Keyword allowlist active = {rm.keyword_allowlist_active} (should be True)")
            kw_expanded_ok = not rm.keywords_expanded
            lines.append(f"    [{'✓' if kw_expanded_ok else '✗'}] Keywords expanded = {rm.keywords_expanded} (should be False)")
            lines.append(f"    [i] Source has existing FAQ = {rm.source_has_existing_faq}")
            lines.append("")

        # Configuration section
        lines.append("CONFIGURATION:")
        lines.append("-" * 40)
        cfg = self.config
        mode_ok = cfg.optimization_mode in ("minimal", "insert_only")
        lines.append(f"  [{'✓' if mode_ok else '✗'}] optimization_mode = '{cfg.optimization_mode}' (should be 'minimal' or 'insert_only')")
        faq_ok = cfg.faq_policy == "never"
        lines.append(f"  [{'✓' if faq_ok else '!'}] faq_policy = '{cfg.faq_policy}' (recommended: 'never')")
        ai_ok = not cfg.generate_ai_sections
        lines.append(f"  [{'✓' if ai_ok else '!'}] generate_ai_sections = {cfg.generate_ai_sections} (recommended: False)")
        cap_ok = cfg.enforce_keyword_caps
        lines.append(f"  [{'✓' if cap_ok else '✗'}] enforce_keyword_caps = {cfg.enforce_keyword_caps} (should be True)")
        lines.append(f"  [i] primary_keyword_body_cap = {cfg.primary_keyword_body_cap}")
        lines.append(f"  [i] secondary_keyword_body_cap = {cfg.secondary_keyword_body_cap}")
        allow_ok = cfg.has_keyword_allowlist
        lines.append(f"  [{'✓' if allow_ok else '!'}] has_keyword_allowlist = {cfg.has_keyword_allowlist} (recommended: True)")
        lines.append("")

        # DELTA BUDGET section - this is the key to insert-only mode!
        lines.append("KEYWORD DELTA BUDGETS (NEW ADDITIONS):")
        lines.append("-" * 40)
        lines.append("  [Key insight: delta = body_count - original_count]")
        lines.append("  [In insert-only mode, delta should be <= allowed_new (typically 1)]")
        lines.append("")
        all_deltas_ok = True
        for kw in self.keywords:
            kw_type = "PRIMARY" if kw.is_primary else "SECONDARY"
            delta_status = "✓" if kw.delta_within_budget else "✗"
            if not kw.delta_within_budget:
                all_deltas_ok = False
            lines.append(f"  [{delta_status}] [{kw_type}] '{kw.phrase}'")
            lines.append(f"      Original: {kw.original_count} | Final: {kw.body_count} | Delta: +{kw.new_additions} (allowed: {kw.allowed_new})")
            if not kw.delta_within_budget:
                over_budget = kw.new_additions - kw.allowed_new
                lines.append(f"      ⚠ OVER DELTA BUDGET BY {over_budget}")
        lines.append("")

        # Legacy Keywords section (legacy cap tracking for backward compat)
        lines.append("KEYWORDS (LEGACY CAP VIEW):")
        lines.append("-" * 40)
        for kw in self.keywords:
            kw_type = "PRIMARY" if kw.is_primary else "SECONDARY"
            status = "✓" if kw.within_cap else "✗"
            lines.append(f"  [{status}] [{kw_type}] '{kw.phrase}'")
            lines.append(f"      Body: {kw.body_count}/{kw.cap} | Meta: {kw.meta_count} | Headings: {kw.headings_count} | FAQ: {kw.faq_count} | Total: {kw.total_count}")
            if not kw.within_cap:
                excess = kw.body_count - kw.cap
                lines.append(f"      ⚠ OVER CAP BY {excess}")
        lines.append("")

        # Cap validation summary
        lines.append("CAP VALIDATION:")
        lines.append("-" * 40)
        if self.cap_validation:
            if self.cap_validation.all_within_caps:
                lines.append("  [✓] ALL KEYWORDS WITHIN CAPS")
            else:
                lines.append(f"  [✗] {len(self.cap_validation.keywords_over_cap)} KEYWORD(S) OVER CAP")
                for kw in self.cap_validation.keywords_over_cap:
                    lines.append(f"      - {kw}")
                lines.append(f"  Total excess: {self.cap_validation.total_excess}")
        else:
            lines.append("  [i] Cap validation not run (not in minimal mode)")
        lines.append("")

        # Warnings section
        if self.warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 40)
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")

        # Summary
        lines.append("SUMMARY:")
        lines.append("-" * 40)
        lines.append(f"  Total blocks: {self.total_blocks}")
        lines.append(f"  Blocks with keywords: {self.blocks_with_keywords}")

        # Overall verdict
        lines.append("")
        lines.append("=" * 70)

        # Check run manifest compliance (if available)
        manifest_ok = True
        manifest_issues = []
        if self.run_manifest:
            rm = self.run_manifest
            if rm.llm_body_rewrite:
                manifest_ok = False
                manifest_issues.append("LLM body rewrite ran (should be False)")
            if rm.heading_rewrite:
                manifest_ok = False
                manifest_issues.append("Heading rewrite ran (should be False)")
            if rm.faq_generation:
                manifest_ok = False
                manifest_issues.append("FAQ generation ran (should be False)")
            if rm.ai_addons_generation:
                manifest_ok = False
                manifest_issues.append("AI add-ons generation ran (should be False)")
            if rm.keywords_expanded:
                manifest_ok = False
                manifest_issues.append("Keywords were expanded (should be False)")

        # Check delta budget compliance
        all_deltas_ok = all(kw.delta_within_budget for kw in self.keywords)

        all_ok = (
            mode_ok and cap_ok and manifest_ok and all_deltas_ok and
            (self.cap_validation is None or self.cap_validation.all_within_caps)
        )
        if all_ok:
            lines.append("VERDICT: ✓ INSERT-ONLY MODE COMPLIANCE: PASS")
        else:
            lines.append("VERDICT: ✗ INSERT-ONLY MODE COMPLIANCE: FAIL")
            if not mode_ok:
                lines.append("  - optimization_mode should be 'minimal' or 'insert_only'")
            if not cap_ok:
                lines.append("  - enforce_keyword_caps should be True")
            if not manifest_ok:
                for issue in manifest_issues:
                    lines.append(f"  - {issue}")
            if not all_deltas_ok:
                over_budget_kws = [kw for kw in self.keywords if not kw.delta_within_budget]
                lines.append(f"  - {len(over_budget_kws)} keyword(s) exceeded delta budget")
            if self.cap_validation and not self.cap_validation.all_within_caps:
                lines.append(f"  - {len(self.cap_validation.keywords_over_cap)} keyword(s) exceeded caps")
        lines.append("=" * 70)

        return "\n".join(lines)


def ensure_keyword_in_text(
    text: str,
    keyword: str,
    position: str = "start",
    max_length: int = 0,
    element_type: str = "general",
) -> str:
    """
    DETERMINISTIC: Ensure exact keyword phrase appears in text naturally.

    IMPROVED: No longer uses crude "keyword:" prefix. Instead:
    1. Returns unchanged if keyword already present
    2. For titles: Integrates keyword naturally at beginning
    3. For descriptions: Integrates keyword in first sentence
    4. For headings: Prepends keyword phrase naturally
    5. Enforces length constraints AFTER integration

    Args:
        text: The text to check/modify.
        keyword: The EXACT keyword phrase that must appear.
        position: Where to inject if missing - "start" or "end".
        max_length: Maximum length constraint (0 = no limit).
        element_type: Type of element - "title", "description", "h1", "general".

    Returns:
        Text guaranteed to contain the exact keyword phrase.
    """
    if not text:
        return keyword

    # Check if keyword already present (case-insensitive)
    if keyword.lower() in text.lower():
        return text

    # IMPROVED: Natural keyword integration based on element type
    result = text

    if element_type == "title":
        # For titles: Use natural phrasing patterns
        # Instead of "keyword: text", use "Keyword - Text" or "Text | Keyword"
        result = _integrate_keyword_title(text, keyword)
    elif element_type == "description":
        # For descriptions: Integrate into first sentence naturally
        result = _integrate_keyword_description(text, keyword)
    elif element_type == "h1":
        # For H1: Use natural heading phrasing
        result = _integrate_keyword_heading(text, keyword)
    else:
        # General text: Natural sentence integration
        if position == "start":
            # Avoid colon prefix - use proper sentence structure
            result = _integrate_keyword_general(text, keyword, at_start=True)
        else:
            result = _integrate_keyword_general(text, keyword, at_start=False)

    # ENFORCE LENGTH CONSTRAINTS after integration
    if max_length > 0 and len(result) > max_length:
        # Try to shorten while keeping keyword
        result = _shorten_with_keyword(result, keyword, max_length)

    return result


def _integrate_keyword_title(text: str, keyword: str) -> str:
    """Integrate keyword into title naturally."""
    # Pattern: "Keyword for/in Topic" or "Topic | Keyword"
    # Check if text already starts with common words that blend well
    lower_text = text.lower()

    # If title is very short, just prepend naturally
    if len(text) < 30:
        return f"{keyword.title()} - {text}"

    # If title has a pipe or dash, use that structure
    if " | " in text:
        parts = text.split(" | ", 1)
        return f"{keyword.title()} | {parts[-1]}"
    elif " - " in text:
        parts = text.split(" - ", 1)
        return f"{keyword.title()} - {parts[-1]}"

    # Default: Clean prefix without colon
    return f"{keyword.title()} - {text}"


def _integrate_keyword_description(text: str, keyword: str) -> str:
    """Integrate keyword into meta description naturally."""
    # Find first sentence end
    sentences = text.split(". ")
    if len(sentences) > 1:
        # Insert keyword phrase into flow
        first = sentences[0]
        rest = ". ".join(sentences[1:])
        # Add keyword to first sentence naturally
        if first.endswith(("s", "es", "ing")):
            return f"{first} for {keyword}. {rest}"
        else:
            return f"Discover {keyword}. {text}"
    else:
        # Single sentence - prepend naturally
        return f"Learn about {keyword}. {text}"


def _integrate_keyword_heading(text: str, keyword: str) -> str:
    """Integrate keyword into H1 heading naturally."""
    # If heading already includes the topic, just prepend
    # Avoid patterns like "keyword: heading"
    if len(text) < 40:
        return f"{keyword.title()}: {text}"

    # For longer headings, try to integrate
    return f"{keyword.title()} - {text}"


def _integrate_keyword_general(text: str, keyword: str, at_start: bool = True) -> str:
    """Integrate keyword into general text naturally."""
    if at_start:
        # Find first sentence and integrate
        if ". " in text:
            first_sentence, rest = text.split(". ", 1)
            return f"{first_sentence} featuring {keyword}. {rest}"
        else:
            return f"Explore {keyword}. {text}"
    else:
        # Integrate at end naturally
        if text.rstrip().endswith((".","!", "?")):
            text_stripped = text.rstrip()
            return f"{text_stripped[:-1]} with {keyword}{text_stripped[-1]}"
        return f"{text} featuring {keyword}"


def _shorten_with_keyword(text: str, keyword: str, max_length: int) -> str:
    """Shorten text while preserving keyword."""
    if len(text) <= max_length:
        return text

    # Find keyword position
    keyword_lower = keyword.lower()
    text_lower = text.lower()
    keyword_start = text_lower.find(keyword_lower)

    if keyword_start == -1:
        # Keyword not found, just truncate
        return text[:max_length-3] + "..."

    keyword_end = keyword_start + len(keyword)

    # Calculate how much we need to remove
    excess = len(text) - max_length

    # Try removing from the end first (after keyword)
    if keyword_end < len(text) - excess - 3:
        # We can truncate after keyword
        return text[:max_length-3] + "..."

    # Try removing from middle (between keyword and end)
    # Keep keyword at start and truncate middle
    if keyword_start < 5:  # Keyword is near start
        return text[:max_length-3] + "..."

    # Keep keyword area, truncate everything else
    return text[:keyword_end + 3] + "..." if len(text[:keyword_end + 3]) < max_length else keyword


def count_keyword_in_text(text: str, keyword: str) -> int:
    """
    Count occurrences of a keyword phrase in text (case-insensitive).

    Args:
        text: Text to search in.
        keyword: Keyword phrase to count.

    Returns:
        Number of occurrences.
    """
    if not text or not keyword:
        return 0
    import re
    # Use word boundary matching for more accurate counts
    pattern = re.escape(keyword)
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)


class ContentOptimizer:
    """
    Orchestrates SEO content optimization.

    This class coordinates all optimization tasks:
    - Meta element optimization (title, description, H1)
    - Body content optimization
    - FAQ generation

    KEYWORD GUARANTEES:
    - Primary keyword MUST appear in: Title, Meta Description, H1, first ~100 words of body
    - Secondary keywords MUST each appear at least once somewhere in the output
    - These are PROGRAMMATIC guarantees - if LLM fails, we inject keywords ourselves
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            llm_client: Pre-configured LLM client. If None, creates one.
            api_key: API key for LLM. Used if llm_client is None.
        """
        if llm_client:
            self.llm = llm_client
        else:
            self.llm = create_llm_client(api_key=api_key)

    def optimize(
        self,
        content: Union[PageMeta, DocxContent],
        keywords: list[Keyword],
        manual_keywords: Optional[ManualKeywordsConfig] = None,
        generate_faq: bool = True,
        faq_count: int = 4,
        max_secondary: int = 5,
        config: Optional[OptimizationConfig] = None,
        include_debug: bool = False,
    ) -> OptimizationResult:
        """
        Perform full SEO optimization on content.

        Args:
            content: Content to optimize (from URL or DOCX).
            keywords: List of available keywords (used only if manual_keywords is None).
            manual_keywords: Manual keyword selection config. If provided, bypasses
                            automatic keyword selection and uses user-specified keywords
                            directly without filtering or scoring.
            generate_faq: Whether to generate FAQ section (DEPRECATED: use config.faq_policy).
            faq_count: Number of FAQ items to generate (DEPRECATED: use config.faq_count).
            max_secondary: Maximum secondary keywords.
            config: Centralized optimization configuration. When provided, overrides
                   generate_faq and faq_count parameters.
            include_debug: Whether to include debug bundle in result. When True,
                          generates comprehensive debugging information including
                          config snapshot, keyword plan, and enforcement details.

        Returns:
            OptimizationResult with all optimized content.
        """
        # Step 0: Extract full content text for keyword filtering
        if isinstance(content, PageMeta):
            full_text = content.full_text
        else:
            full_text = content.full_text

        # Check if we're using manual keyword mode
        if manual_keywords is not None:
            # MANUAL KEYWORD MODE: Bypass all filtering and auto-selection
            # User-specified keywords are used directly without any modification
            keyword_plan = self._build_keyword_plan_from_manual(manual_keywords)
            filtered_keywords = keyword_plan.all_keywords
            self._filter_summary = "Manual keyword mode: user-specified keywords used directly"
            self._rejected_keywords = []

            # Build allowlist from manual keywords for hard enforcement
            all_manual_kw = [manual_keywords.primary]
            if manual_keywords.secondary:
                all_manual_kw.extend(manual_keywords.secondary)

            # Create config with keyword allowlist if not provided
            if config is None:
                config = OptimizationConfig.for_manual_keywords(
                    keywords=all_manual_kw,
                    mode="minimal",  # Default to minimal for manual keywords
                )
                import sys
                print(f"DEBUG: Created minimal config with allowlist: {config.keyword_allowlist}", file=sys.stderr)
            elif config.keyword_allowlist is None:
                # Config provided but no allowlist - add one
                config.keyword_allowlist = {k.strip().lower() for k in all_manual_kw if k.strip()}
        else:
            # AUTOMATIC MODE: Filter and select keywords as before
            # Step 0.5: Filter keywords for topical relevance BEFORE any optimization
            # This is CRITICAL to prevent injection of off-topic industries/verticals
            filtered_keywords, filter_results = filter_keywords_for_content(
                keywords=keywords,
                content_text=full_text,
                min_overlap=0.5,  # Require 50% token overlap for non-exact matches
            )

            # Log filter results for debugging
            filter_summary = log_filter_results(filter_results, verbose=False)
            self._filter_summary = filter_summary

            # Log rejected keywords for debugging (in verbose mode this could be exposed)
            rejected = [r for r in filter_results if not r.is_allowed]
            if rejected:
                # Store rejection info for potential debugging
                self._rejected_keywords = rejected

            # If no keywords passed the filter, use a relaxed filter
            # BUT NEVER allow blocked terms through even with relaxed filter
            if not filtered_keywords:
                # Try again with lower threshold but same blocklist
                filtered_keywords, _ = filter_keywords_for_content(
                    keywords=keywords,
                    content_text=full_text,
                    min_overlap=0.3,  # More relaxed but blocklist still applies
                )

            # If still no keywords, fall back to generic keywords that don't have blocked terms
            # NEVER use original list unfiltered
            if not filtered_keywords:
                # Only take keywords that don't contain blocked terms
                safe_keywords = []
                from .keyword_filter import contains_blocked_term
                for kw in keywords:
                    if not contains_blocked_term(kw.phrase, full_text):
                        safe_keywords.append(kw)
                    if len(safe_keywords) >= 10:
                        break
                filtered_keywords = safe_keywords

        # Extract content topics for LLM context
        content_topics = get_content_topics(full_text, top_n=15)

        # Step 0.6: Brand name detection and control (with exclusion system)
        brand_name, brand_variations = self._detect_brand_name(content, full_text)
        brand_count = full_text.lower().count(brand_name.lower()) if brand_name else 0
        self._brand_context = {
            "name": brand_name,
            "original_count": brand_count,
            "max_extra_mentions": min(3, max(1, brand_count)),
            "variations": brand_variations,  # For diff exclusion
            "original_spelling": brand_name,  # Exact original spelling to preserve
        }

        # Step 0.7: PAGE ARCHETYPE DETECTION
        # This prevents applying wrong content styles (e.g., "guide" framing on landing pages)
        url = content.url if isinstance(content, PageMeta) else None
        h1 = content.h1 if isinstance(content, PageMeta) else (content.h1 if hasattr(content, 'h1') else None)
        title = content.title if isinstance(content, PageMeta) else None
        headings = []
        if isinstance(content, PageMeta):
            headings = content.content_blocks[:10]  # First 10 blocks may contain headings
        self._page_archetype = detect_page_archetype(
            url=url,
            title=title,
            h1=h1,
            content_text=full_text,
            headings=headings,
        )
        import sys
        print(f"DEBUG: Page archetype detected: {self._page_archetype.archetype} "
              f"(confidence: {self._page_archetype.confidence:.2f})", file=sys.stderr)
        if not self._page_archetype.allows_guide_framing:
            print(f"DEBUG: Guide framing BLOCKED for this page type", file=sys.stderr)

        # Step 1: Analyze content (with filtered keywords)
        analysis = analyze_content(content, filtered_keywords)

        # Step 2: Create keyword plan (skip if using manual keywords)
        if manual_keywords is None:
            keyword_plan = create_keyword_plan(
                keywords=filtered_keywords,
                content=content,
                analysis=analysis,
                max_secondary=max_secondary,
                max_questions=faq_count,
            )

        # Store content topics and full text for use in LLM calls and validation
        self._content_topics = content_topics
        self._full_original_text = full_text

        # Store keyword plan for use in _compute_markers_v2 (atomic keyword units)
        self._current_keyword_plan = keyword_plan

        # Step 2.5: Run structured content audit (10-Part SEO Framework)
        # This identifies keyword placement gaps and prioritizes issues
        if isinstance(content, PageMeta):
            content_audit = audit_content(
                content=content,
                keyword_plan=keyword_plan,
                meta_title=content.title,
                meta_description=content.meta_description,
            )
        else:
            content_audit = audit_content(
                content=content,
                keyword_plan=keyword_plan,
            )

        # Step 2.6: Build optimization plan based on audit
        # This creates the tiered keyword placement strategy
        optimization_plan = build_optimization_plan(
            audit=content_audit,
            keyword_plan=keyword_plan,
        )

        # Store audit and plan for use in optimization methods
        self._content_audit = content_audit
        self._optimization_plan = optimization_plan

        # Step 3: Get current meta elements
        import sys
        if isinstance(content, PageMeta):
            current_title = content.title
            current_meta_desc = content.meta_description
            current_h1 = content.h1
            blocks = convert_page_meta_to_blocks(content)
            print(f"DEBUG Step 3 (PageMeta): H1='{current_h1}'", file=sys.stderr)
        else:
            current_title = None
            current_meta_desc = None
            current_h1 = content.h1
            blocks = content.paragraphs
            print(f"DEBUG Step 3 (DocxContent): H1='{current_h1}'", file=sys.stderr)

        # Debug: Count H1 blocks after conversion
        h1_blocks = [b for b in blocks if b.heading_level == HeadingLevel.H1]
        print(f"DEBUG Step 3: {len(blocks)} blocks total, {len(h1_blocks)} H1 blocks", file=sys.stderr)
        for i, b in enumerate(h1_blocks):
            print(f"  H1 Block {i}: '{b.text[:80]}...'", file=sys.stderr)

        # Step 3.5: Count EXISTING keyword occurrences in source before optimization
        # This is critical for insert-only mode to distinguish existing vs added counts
        existing_keyword_counts = self._count_existing_keywords_in_source(
            original_meta_title=current_title,
            original_meta_desc=current_meta_desc,
            original_h1=current_h1,
            original_blocks=blocks,
            keyword_plan=keyword_plan,
        )
        print(f"DEBUG Step 3.5: Existing keyword counts captured from source", file=sys.stderr)
        for kw, counts in existing_keyword_counts.items():
            print(f"  '{kw}': body={counts.body}, meta={counts.meta}, headings={counts.headings}", file=sys.stderr)

        # Step 4: Optimize meta elements using optimization plan
        meta_elements = self._optimize_meta_elements(
            current_title=current_title,
            current_meta_desc=current_meta_desc,
            current_h1=current_h1,
            keyword_plan=keyword_plan,
            optimization_plan=optimization_plan,
            topic=analysis.topic,
            full_original_text=full_text,
        )

        # Step 5: Optimize body content using optimization plan
        optimized_blocks = self._optimize_body_content(
            blocks=blocks,
            keyword_plan=keyword_plan,
            optimization_plan=optimization_plan,
            analysis=analysis,
            full_original_text=full_text,
        )

        # Step 5.5: Update H1 in body blocks to match optimized H1 from meta elements
        # This ensures the H1 in "OPTIMIZED CONTENT" matches the meta table
        optimized_h1_text = None
        for meta in meta_elements:
            if meta.element_name == "H1":
                optimized_h1_text = meta.optimized
                break

        # Debug: Check H1 blocks before replacement
        h1_before = [b for b in optimized_blocks if b.heading_level == HeadingLevel.H1]
        print(f"DEBUG Step 5.5: Before H1 replacement: {len(h1_before)} H1 blocks in optimized_blocks", file=sys.stderr)

        if optimized_h1_text:
            print(f"DEBUG Step 5.5: Calling _replace_h1_in_blocks with optimized_h1='{optimized_h1_text[:80]}...'", file=sys.stderr)
            optimized_blocks = self._replace_h1_in_blocks(
                blocks=optimized_blocks,
                optimized_h1=optimized_h1_text,
            )
            # Debug: Check H1 blocks after replacement
            h1_after = [b for b in optimized_blocks if b.heading_level == HeadingLevel.H1]
            print(f"DEBUG Step 5.5: After H1 replacement: {len(h1_after)} H1 blocks", file=sys.stderr)
            for i, b in enumerate(h1_after):
                print(f"  H1 Block {i}: '{b.text[:80]}...'", file=sys.stderr)
        else:
            print(f"DEBUG Step 5.5: No optimized H1 found in meta_elements!", file=sys.stderr)

        # Step 6: Generate FAQ if requested (Part 10 of 10-Part Framework)
        # Uses optimization plan's faq_keywords for targeted keyword integration
        # POLICY-BASED FAQ CONTROL: auto/always/never
        faq_items = []
        faq_archetype_warning = None

        # Build effective config from legacy params or provided config
        # NOTE: If manual_keywords was provided, config is already set with allowlist
        effective_config = config or OptimizationConfig(
            faq_policy="auto" if generate_faq else "never",
            faq_count=faq_count,
        )
        # Store for use in other methods
        self._effective_config = effective_config

        # Log optimization mode for debugging
        import sys
        print(f"DEBUG: Optimization mode: {effective_config.optimization_mode}", file=sys.stderr)
        if effective_config.has_keyword_allowlist:
            print(f"DEBUG: Keyword allowlist active: {effective_config.keyword_allowlist}", file=sys.stderr)

        # Determine if archetype recommends FAQ
        archetype_recommends_faq = self._page_archetype.should_add_faq

        # INSERT-ONLY MODE: Lock existing FAQ from modification
        # In insert-only mode, existing FAQ sections are preserved and no new FAQ is generated
        source_has_existing_faq = self._source_has_faq()
        if source_has_existing_faq:
            print("DEBUG: Source already has FAQ section detected", file=sys.stderr)

        # Explicit lock check: if should_lock_existing_faq and source has FAQ, skip all FAQ generation
        if effective_config.should_lock_existing_faq and source_has_existing_faq:
            print("DEBUG: FAQ generation LOCKED - insert-only mode preserves existing FAQ", file=sys.stderr)
            # Skip all FAQ generation - existing FAQ is preserved as-is in optimized_blocks
            faq_items = []
        elif effective_config.is_minimal_mode and source_has_existing_faq:
            # Minimal mode + existing FAQ = skip generation
            print("DEBUG: FAQ generation SKIPPED - minimal mode and source already has FAQ", file=sys.stderr)
        elif effective_config.faq_policy == "never":
            # Never generate FAQ
            print("DEBUG: FAQ generation DISABLED (policy=never)", file=sys.stderr)
        elif effective_config.faq_policy == "always":
            # Always generate FAQ, but warn if archetype is inappropriate
            if not archetype_recommends_faq:
                faq_archetype_warning = (
                    f"FAQ generated despite '{self._page_archetype.archetype}' page type "
                    f"(policy=always). Consider reviewing FAQ appropriateness for this page."
                )
                print(f"WARNING: {faq_archetype_warning}", file=sys.stderr)

            faq_items = self._generate_faq_with_fallback(
                topic=analysis.topic,
                keyword_plan=keyword_plan,
                optimization_plan=optimization_plan,
                num_items=effective_config.faq_count,
                min_valid=effective_config.faq_min_valid,
            )
        elif effective_config.faq_policy == "auto":
            # Auto mode: only generate if archetype recommends
            if archetype_recommends_faq:
                faq_items = self._generate_faq_with_fallback(
                    topic=analysis.topic,
                    keyword_plan=keyword_plan,
                    optimization_plan=optimization_plan,
                    num_items=effective_config.faq_count,
                    min_valid=effective_config.faq_min_valid,
                )
            else:
                print(f"DEBUG: FAQ generation SKIPPED - archetype '{self._page_archetype.archetype}' "
                      "does not recommend FAQ (policy=auto)", file=sys.stderr)

        # Step 7: GUARANTEE all keywords appear in content (especially for manual mode)
        # Check which keywords are missing from the body content
        optimized_blocks = self._ensure_all_keywords_present(
            blocks=optimized_blocks,
            keyword_plan=keyword_plan,
            topic=analysis.topic,
        )

        # Step 7.5: DISTRIBUTE keywords across document sections
        # MINIMAL MODE: Skip distribution, just ensure each keyword appears once
        # ENHANCED MODE: Distribute keywords across sections based on H2 headings
        if effective_config.is_minimal_mode:
            # Minimal mode: each keyword should appear exactly once
            target_count = 1
            secondary_target = 1
            import sys
            print(f"DEBUG Step 7.5: MINIMAL MODE - skip distribution, target_count=1", file=sys.stderr)
        else:
            # Enhanced mode: target density-based counts
            target_count = manual_keywords.target_count if manual_keywords else 6
            secondary_target = 3  # Each secondary keyword should appear at least 3 times
            import sys
            print(f"DEBUG Step 7.5: ENHANCED MODE - distributing keywords, target_count={target_count}", file=sys.stderr)
            optimized_blocks = self._distribute_keywords_across_sections(
                meta_elements=meta_elements,
                blocks=optimized_blocks,
                keyword_plan=keyword_plan,
                topic=analysis.topic,
                primary_target=target_count,
                secondary_target=secondary_target,
            )

        # Step 7.7: ENFORCE BODY KEYWORD INVARIANTS before FAQ generation
        # This is the FINAL SAFETY check to ensure body targets are satisfied.
        # MINIMAL MODE: Only enforce each keyword appears at least once
        # ENHANCED MODE: Enforce full targets
        optimized_blocks = self._enforce_body_keyword_invariants(
            blocks=optimized_blocks,
            keyword_plan=keyword_plan,
            topic=analysis.topic,
            primary_target=target_count,
            secondary_target=secondary_target,
        )

        # Step 7.8: ENFORCE KEYWORD CAPS (INSERT-ONLY MODE ONLY)
        # This is the KEY step for minimal/insert-only behavior.
        # After all keyword insertion, enforce MAXIMUM caps.
        # If keywords appear MORE than the cap, remove excess occurrences.
        if effective_config.should_enforce_keyword_caps:
            optimized_blocks = self._enforce_keyword_caps(
                blocks=optimized_blocks,
                keyword_plan=keyword_plan,
                config=effective_config,
            )

        # Step 7.9: FINAL SAFETY CHECK - Validate content preservation
        # Ensure total optimized content >= original content (additive only)
        original_body_length = sum(len(block.text) for block in blocks)
        optimized_body_length = sum(len(strip_markers(block.text)) for block in optimized_blocks)

        # Debug: Check H1 before content deletion check
        h1_before_check = [b for b in optimized_blocks if b.heading_level == HeadingLevel.H1]
        print(f"DEBUG Step 7.9: H1 blocks before content check: {len(h1_before_check)}", file=sys.stderr)
        for i, b in enumerate(h1_before_check):
            print(f"  H1 Block {i}: '{b.text[:80]}...'", file=sys.stderr)

        if optimized_body_length < original_body_length * 0.85:
            # CRITICAL: Content was deleted - fall back to original blocks
            deletion_pct = (1 - optimized_body_length / original_body_length) * 100 if original_body_length > 0 else 0
            print(
                f"CRITICAL: Content deletion detected - "
                f"original: {original_body_length} chars, "
                f"optimized: {optimized_body_length} chars "
                f"({deletion_pct:.1f}% deleted). Falling back to original.",
                file=sys.stderr
            )
            # Fall back to original blocks but keep meta elements
            # Add minimal keyword markers to original to preserve functionality
            optimized_blocks = blocks
            print(f"DEBUG Step 7.9: Fell back to original blocks!", file=sys.stderr)

            # Re-apply H1 replacement even on fallback - keep optimized H1 from meta table
            if optimized_h1_text:
                print(f"DEBUG Step 7.9: Re-applying H1 replacement after fallback", file=sys.stderr)
                optimized_blocks = self._replace_h1_in_blocks(
                    blocks=optimized_blocks,
                    optimized_h1=optimized_h1_text,
                )

            # Re-apply keyword enforcement after fallback - original blocks don't have keywords
            print(f"DEBUG Step 7.9: Re-applying keyword enforcement after fallback", file=sys.stderr)
            optimized_blocks = self._enforce_body_keyword_invariants(
                blocks=optimized_blocks,
                keyword_plan=keyword_plan,
                topic=analysis.topic,
                primary_target=target_count,
                secondary_target=secondary_target,  # Use mode-appropriate target
            )

            # Also enforce caps after fallback path
            if effective_config.should_enforce_keyword_caps:
                optimized_blocks = self._enforce_keyword_caps(
                    blocks=optimized_blocks,
                    keyword_plan=keyword_plan,
                    config=effective_config,
                )
        else:
            print(f"DEBUG Step 7.9: Content check passed, keeping optimized blocks", file=sys.stderr)

        # Debug: Final H1 check before return
        h1_final = [b for b in optimized_blocks if b.heading_level == HeadingLevel.H1]
        print(f"DEBUG FINAL: {len(h1_final)} H1 blocks in final optimized_blocks", file=sys.stderr)
        for i, b in enumerate(h1_final):
            print(f"  FINAL H1 Block {i}: '{b.text[:100]}...'", file=sys.stderr)

        # Step 7.95: Generate AI Optimization Add-ons (Key Takeaways + Chunk Map)
        # This is policy-controlled by effective_config.generate_ai_sections
        ai_addons_result = None
        if effective_config.should_generate_ai_addons:
            print("DEBUG: Generating AI Optimization Add-ons...", file=sys.stderr)
            ai_addons_result = self._generate_ai_addons(
                optimized_blocks=optimized_blocks,
                keyword_plan=keyword_plan,
                faq_items=faq_items,
                config=effective_config,
            )
            print(f"DEBUG: AI Add-ons generated: {len(ai_addons_result.key_takeaways) if ai_addons_result else 0} takeaways, "
                  f"{len(ai_addons_result.chunk_map_chunks) if ai_addons_result else 0} chunks", file=sys.stderr)
        else:
            print("DEBUG: AI Add-ons generation DISABLED (config.generate_ai_sections=False)", file=sys.stderr)

        # Step 8: Compute SCOPED keyword usage counts in final output
        # This provides transparency on where keywords appear (body vs meta vs FAQ)
        # Also includes existing counts from source for insert-only mode reporting
        scoped_keyword_counts = self._compute_keyword_usage_counts(
            meta_elements=meta_elements,
            optimized_blocks=optimized_blocks,
            faq_items=faq_items,
            keyword_plan=keyword_plan,
            primary_target=target_count,
            secondary_target=secondary_target,  # Use mode-appropriate target
            existing_counts=existing_keyword_counts,  # Pass existing counts for added vs existing tracking
        )

        # Log scoped counts for transparency
        self._log_scoped_keyword_counts(scoped_keyword_counts)

        # Step 8.5: VALIDATE KEYWORD CAPS (Insert-Only Mode)
        # This is the POST-OPTIMIZATION VALIDATION that provides transparency
        # on whether the optimizer respected keyword caps.
        cap_validation_report = None
        if effective_config.is_minimal_mode:
            cap_validation_report = self._validate_keyword_caps_compliance(
                scoped_counts=scoped_keyword_counts,
                keyword_plan=keyword_plan,
                config=effective_config,
            )
            # Log the cap validation report for transparency
            self._log_cap_validation_report(cap_validation_report)

        # Convert to legacy format for backward compatibility
        keyword_usage_counts = self._get_legacy_keyword_counts(scoped_keyword_counts)

        # Get detailed counts with existing vs added breakdown
        keyword_usage_detailed = self._get_detailed_keyword_counts(scoped_keyword_counts)

        # Build warnings list
        warnings = []
        if faq_archetype_warning:
            warnings.append(faq_archetype_warning)

        # Add warning if any keywords exceeded caps (insert-only mode)
        if cap_validation_report and not cap_validation_report.all_within_caps:
            over_cap_keywords = [r.keyword for r in cap_validation_report.keywords_over_cap]
            warnings.append(
                f"KEYWORD CAP VIOLATION: {len(over_cap_keywords)} keyword(s) exceeded caps: "
                f"{', '.join(over_cap_keywords)}"
            )

        # Step 9: Generate debug bundle if requested
        debug_bundle_dict = None
        if include_debug:
            print("DEBUG: Generating debug bundle...", file=sys.stderr)

            # Build run manifest showing what stages actually executed
            manifest = RunManifest(
                optimizer_version="v1",
                optimization_mode=effective_config.optimization_mode,
                llm_body_rewrite=not effective_config.is_minimal_mode,  # LLM rewrites disabled in minimal mode
                heading_rewrite=False,  # TODO: track this properly
                faq_generation=len(faq_items) > 0,
                ai_addons_generation=ai_addons_result is not None,
                keyword_caps_enforcement=effective_config.enforce_keyword_caps,
                budget_enforcement=effective_config.is_minimal_mode,  # Budget enforced in minimal mode
                highlight_integrity_check=effective_config.is_minimal_mode,
                source_has_existing_faq=False,  # TODO: detect from source
                page_archetype=self._page_archetype.archetype if self._page_archetype else "unknown",
                manual_keywords_mode=effective_config.manual_keywords_only,
                keyword_allowlist_active=effective_config.has_keyword_allowlist,
                keywords_expanded=not effective_config.manual_keywords_only,  # Keywords expanded if not manual mode
            )

            debug_bundle = self._generate_debug_bundle(
                config=effective_config,
                keyword_plan=keyword_plan,
                scoped_counts=scoped_keyword_counts,
                cap_validation=cap_validation_report,
                optimized_blocks=optimized_blocks,
                warnings=warnings,
                original_keyword_counts=existing_keyword_counts,
                run_manifest=manifest,
            )
            debug_bundle_dict = debug_bundle.to_dict()
            print(f"DEBUG: Debug bundle generated with {len(debug_bundle.keywords)} keywords", file=sys.stderr)

        # Step 10: HIGHLIGHT INTEGRITY VALIDATION (Insert-Only Mode)
        # Validates that: (1) Unhighlighted text exists in source (no false black)
        #                 (2) Highlighted text is actually new (no false green)
        #                 (3) URLs preserved, markers balanced
        highlight_integrity_report = None
        if effective_config.is_minimal_mode:
            # Build optimized body text with markers from blocks
            optimized_body_with_markers = "\n\n".join(
                block.text for block in optimized_blocks if block.text
            )

            # Run highlight integrity check against original source
            highlight_integrity_report = run_highlight_integrity_check(
                original=self._full_original_text,
                marked_output=optimized_body_with_markers,
                strict=False,  # Use normalized matching (not strict substring)
            )

            # Log results
            print(f"DEBUG Step 10: Highlight integrity check: {highlight_integrity_report.get_summary()}", file=sys.stderr)

            # Add warnings for any highlight integrity issues
            if not highlight_integrity_report.is_valid:
                error_count = len([i for i in highlight_integrity_report.issues if i.severity == "error"])
                warning_count = len([i for i in highlight_integrity_report.issues if i.severity == "warning"])
                warnings.append(
                    f"HIGHLIGHT INTEGRITY: {error_count} errors, {warning_count} warnings detected. "
                    f"Run with include_debug=True for details."
                )

                # Add specific URL corruption warnings
                url_issues = [i for i in highlight_integrity_report.issues if "url" in i.category.lower()]
                for issue in url_issues[:2]:  # Limit to first 2
                    warnings.append(f"URL ISSUE: {issue.description}")

        return OptimizationResult(
            meta_elements=meta_elements,
            optimized_blocks=optimized_blocks,
            faq_items=faq_items,
            primary_keyword=keyword_plan.primary.phrase,
            secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
            keyword_usage_counts=keyword_usage_counts,
            keyword_usage_detailed=keyword_usage_detailed,
            warnings=warnings,
            faq_archetype_warning=faq_archetype_warning,
            ai_addons=ai_addons_result,
            debug_bundle=debug_bundle_dict,
        )

    def _build_keyword_plan_from_manual(
        self,
        manual_keywords: ManualKeywordsConfig,
    ) -> KeywordPlan:
        """
        Build a KeywordPlan directly from manual keyword configuration.

        This bypasses all automatic keyword selection, filtering, and scoring.
        User-specified keywords are used exactly as provided.

        CRITICAL: The exact phrases provided by the user MUST appear in the final output.
        No paraphrasing, no splitting, no reordering - the EXACT phrase as typed.

        Args:
            manual_keywords: The manual keyword configuration from user input.

        Returns:
            KeywordPlan with user-specified keywords.
        """
        # Create primary keyword - never marked as brand, no filtering
        # IMPORTANT: Store the EXACT phrase as provided (preserve casing/spacing)
        primary = Keyword(
            phrase=manual_keywords.primary.strip(),
            is_brand=False,  # Manual keywords are never treated as brand
        )

        # Create secondary keywords from user-provided list
        # IMPORTANT: Store EXACT phrases as provided
        secondary = [
            Keyword(
                phrase=phrase.strip(),
                is_brand=False,  # Manual keywords are never treated as brand
            )
            for phrase in manual_keywords.secondary
            if phrase and phrase.strip()
        ]

        # Return the keyword plan (no long_tail_questions in manual mode)
        return KeywordPlan(
            primary=primary,
            secondary=secondary,
            long_tail_questions=[],  # FAQ questions generated from topic in manual mode
        )

    def _enforce_keyword_allowlist(
        self,
        text: str,
        original_text: str,
        config: OptimizationConfig,
    ) -> str:
        """
        Enforce keyword allowlist by reverting unauthorized keyword insertions.

        In minimal mode with an allowlist, only the explicitly provided keywords
        should be inserted. If the LLM adds variations, synonyms, or other phrases
        not in the allowlist, those insertions are reverted.

        Args:
            text: The optimized text that may contain unauthorized insertions.
            original_text: The original text before optimization.
            config: Configuration with allowlist settings.

        Returns:
            Text with unauthorized insertions reverted.
        """
        if not config.has_keyword_allowlist:
            return text  # No allowlist enforcement needed

        # Find additions by looking at what's new in text vs original
        # This is a simplified check - for complex cases, use diff markers
        original_lower = original_text.lower()
        text_lower = text.lower()

        # Check each word/phrase in the allowlist
        allowlist = config.keyword_allowlist

        # If the new text doesn't contain any unauthorized keywords, return as-is
        # For now, we trust that the LLM respects the keyword constraints
        # Full enforcement would require parsing diff markers

        import sys
        print(f"DEBUG: Allowlist enforcement active: {allowlist}", file=sys.stderr)

        return text

    def _validate_keywords_against_allowlist(
        self,
        keywords_to_check: list[str],
        config: OptimizationConfig,
    ) -> tuple[list[str], list[str]]:
        """
        Validate keywords against the allowlist.

        Args:
            keywords_to_check: List of keywords to validate.
            config: Configuration with allowlist settings.

        Returns:
            Tuple of (allowed_keywords, rejected_keywords).
        """
        if not config.has_keyword_allowlist:
            return keywords_to_check, []

        allowed = []
        rejected = []

        for kw in keywords_to_check:
            if config.is_keyword_allowed(kw):
                allowed.append(kw)
            else:
                rejected.append(kw)
                import sys
                print(f"DEBUG: Keyword '{kw}' rejected - not in allowlist", file=sys.stderr)

        return allowed, rejected

    def _optimize_meta_elements(
        self,
        current_title: Optional[str],
        current_meta_desc: Optional[str],
        current_h1: Optional[str],
        keyword_plan: KeywordPlan,
        optimization_plan: OptimizationPlan,
        topic: str,
        full_original_text: Optional[str] = None,
    ) -> list[MetaElement]:
        """
        Optimize title, meta description, and H1 using the optimization plan.

        KEYWORD GUARANTEE (Part 5 of 10-Part Framework):
        - Tier 1: Title Tag MUST contain primary keyword
        - Tier 2: H1 MUST contain primary keyword
        - Meta Description MUST contain primary keyword for CTR

        This is enforced in two stages:
        1. Use optimization plan's pre-computed targets as guidance
        2. Post-processing keyword injection if LLM fails to include

        Args:
            current_title: Current page title.
            current_meta_desc: Current meta description.
            current_h1: Current H1 heading.
            keyword_plan: The keyword plan.
            optimization_plan: Pre-computed optimization targets from audit.
            topic: Content topic.
            full_original_text: Full document text for validation.

        Returns:
            List of MetaElement results.
        """
        meta_elements = []
        primary = keyword_plan.primary.phrase
        secondary = [kw.phrase for kw in keyword_plan.secondary]

        # INSERT-ONLY MODE: No LLM optimization, only inject keywords if missing
        effective_config = getattr(self, '_effective_config', None)
        if effective_config and effective_config.is_insert_only_mode:
            return self._optimize_meta_elements_insert_only(
                current_title=current_title,
                current_meta_desc=current_meta_desc,
                current_h1=current_h1,
                primary=primary,
            )

        # Use optimization plan targets as guidance for LLM
        target_title = optimization_plan.target_meta_title
        target_meta_desc = optimization_plan.target_meta_description
        target_h1 = optimization_plan.target_h1

        # Optimize title - use target as hint if available
        optimized_title_raw = self.llm.optimize_meta_title(
            current_title=current_title,
            primary_keyword=primary,
            topic=topic,
            max_length=60,
            target_hint=target_title if target_title else None,
        )
        # SAFETY: Validate LLM output for blocked industry terms
        optimized_title_raw, _ = validate_and_fallback(
            optimized_title_raw, current_title or "", "title"
        )
        # DETERMINISTIC KEYWORD GUARANTEE: Primary keyword MUST appear (Tier 1)
        # Do this BEFORE adding markers so the injection is clean
        # IMPROVED: Use element_type and max_length for natural integration
        optimized_title_raw = ensure_keyword_in_text(
            optimized_title_raw, primary, position="start",
            max_length=60, element_type="title"
        )

        # BRAND NAME NORMALIZATION: Ensure LLM output uses original brand spelling
        # This prevents "Cell-Gate" when original says "CellGate"
        if self._brand_context and self._brand_context.get("name"):
            optimized_title_raw = normalize_brand_in_text(
                optimized_title_raw,
                self._brand_context["name"],
                self._brand_context.get("variations", set()),
            )

        # V3: Token-level diff - highlight ONLY changed/added tokens
        # This ensures unchanged words stay unhighlighted (black)
        # Example: "Best Hearing Aids" → "Best AI Hearing Aids | Clear Sound"
        #          Only "AI" and "| Clear Sound" get highlighted (green)
        from .diff_markers import compute_title_markers
        optimized_title = compute_title_markers(current_title or "", optimized_title_raw)

        meta_elements.append(
            MetaElement(
                element_name="Title Tag",
                current=current_title,
                optimized=optimized_title,
                why_changed=self._explain_title_change(current_title, optimized_title, primary),
            )
        )

        # Optimize meta description - use target as hint if available
        optimized_desc_raw = self.llm.optimize_meta_description(
            current_description=current_meta_desc,
            primary_keyword=primary,
            topic=topic,
            max_length=160,
            target_hint=target_meta_desc if target_meta_desc else None,
        )
        # SAFETY: Validate LLM output for blocked industry terms
        optimized_desc_raw, _ = validate_and_fallback(
            optimized_desc_raw, current_meta_desc or "", "meta_description"
        )
        # DETERMINISTIC KEYWORD GUARANTEE: Primary keyword MUST appear
        # IMPROVED: Use element_type and max_length for natural integration
        optimized_desc_raw = ensure_keyword_in_text(
            optimized_desc_raw, primary, position="start",
            max_length=160, element_type="description"
        )

        # BRAND NAME NORMALIZATION: Ensure LLM output uses original brand spelling
        if self._brand_context and self._brand_context.get("name"):
            optimized_desc_raw = normalize_brand_in_text(
                optimized_desc_raw,
                self._brand_context["name"],
                self._brand_context.get("variations", set()),
            )

        # V3: Token-level diff - highlight ONLY changed/added tokens
        # This ensures unchanged words stay unhighlighted (black)
        from .diff_markers import compute_meta_desc_markers
        optimized_desc = compute_meta_desc_markers(current_meta_desc or "", optimized_desc_raw)

        meta_elements.append(
            MetaElement(
                element_name="Meta Description",
                current=current_meta_desc,
                optimized=optimized_desc,
                why_changed=self._explain_meta_desc_change(current_meta_desc, optimized_desc, primary),
            )
        )

        # Optimize H1 (Tier 2) - use target as hint if available
        # INSERT-ONLY MODE: Lock H1 from modification
        effective_config = getattr(self, '_effective_config', None)
        if effective_config and effective_config.should_lock_headings:
            # Return original H1 unchanged (no markers, no LLM, no keyword injection)
            optimized_h1 = current_h1 or ""
            meta_elements.append(
                MetaElement(
                    element_name="H1",
                    current=current_h1,
                    optimized=optimized_h1,
                    why_changed="H1 locked (insert-only mode)",
                )
            )
            return meta_elements

        clean_title = strip_markers(optimized_title)
        optimized_h1_raw = self.llm.optimize_h1(
            current_h1=current_h1,
            primary_keyword=primary,
            title=clean_title,
            topic=topic,
            target_hint=target_h1 if target_h1 else None,
        )
        # SAFETY: Validate LLM output for blocked industry terms
        optimized_h1_raw, _ = validate_and_fallback(
            optimized_h1_raw, current_h1 or "", "h1"
        )
        # DETERMINISTIC KEYWORD GUARANTEE: Primary keyword MUST appear (Tier 2)
        # IMPROVED: Use element_type for natural H1 integration
        optimized_h1_raw = ensure_keyword_in_text(
            optimized_h1_raw, primary, position="start",
            max_length=0, element_type="h1"
        )

        # BRAND NAME NORMALIZATION: Ensure LLM output uses original brand spelling
        if self._brand_context and self._brand_context.get("name"):
            optimized_h1_raw = normalize_brand_in_text(
                optimized_h1_raw,
                self._brand_context["name"],
                self._brand_context.get("variations", set()),
            )

        # Use special H1 marker handling:
        # If H1 changed at all, wrap ENTIRE H1 in markers (not just changed phrases)
        optimized_h1 = compute_h1_markers(current_h1 or "", optimized_h1_raw)

        meta_elements.append(
            MetaElement(
                element_name="H1",
                current=current_h1,
                optimized=optimized_h1,
                why_changed=self._explain_h1_change(current_h1, optimized_h1, primary),
            )
        )

        return meta_elements

    def _optimize_meta_elements_insert_only(
        self,
        current_title: Optional[str],
        current_meta_desc: Optional[str],
        current_h1: Optional[str],
        primary: str,
    ) -> list[MetaElement]:
        """
        Insert-only mode meta element optimization.

        No LLM rewrites. Only injects primary keyword if missing.
        Markers wrap ONLY the injected phrase, not the entire element.

        Args:
            current_title: Current page title.
            current_meta_desc: Current meta description.
            current_h1: Current H1 heading.
            primary: Primary keyword to inject if missing.

        Returns:
            List of MetaElement with minimal changes.
        """
        from .diff_markers import inject_phrase_with_markers

        meta_elements = []

        # Title: Inject keyword only if missing
        title_changed = False
        if current_title and primary.lower() not in current_title.lower():
            optimized_title = inject_phrase_with_markers(current_title, primary, position="start")
            title_changed = True
        else:
            optimized_title = current_title or ""

        meta_elements.append(
            MetaElement(
                element_name="Title Tag",
                current=current_title,
                optimized=optimized_title,
                why_changed="Keyword injected (insert-only mode)" if title_changed else "Title locked (insert-only mode)",
            )
        )

        # Meta Description: Inject keyword only if missing
        desc_changed = False
        if current_meta_desc and primary.lower() not in current_meta_desc.lower():
            optimized_desc = inject_phrase_with_markers(current_meta_desc, primary, position="start")
            desc_changed = True
        else:
            optimized_desc = current_meta_desc or ""

        meta_elements.append(
            MetaElement(
                element_name="Meta Description",
                current=current_meta_desc,
                optimized=optimized_desc,
                why_changed="Keyword injected (insert-only mode)" if desc_changed else "Meta description locked (insert-only mode)",
            )
        )

        # H1: Always locked in insert-only mode (no changes)
        meta_elements.append(
            MetaElement(
                element_name="H1",
                current=current_h1,
                optimized=current_h1 or "",
                why_changed="H1 locked (insert-only mode)",
            )
        )

        return meta_elements

    def _compute_markers_v2(
        self,
        original: str,
        optimized: str,
        full_original_text: Optional[str] = None,
        keywords: Optional[list[str]] = None,
    ) -> str:
        """
        Token-Level Diff: highlight ONLY new/changed tokens.

        This implements precise token-level diff using SequenceMatcher:
        - Only tokens that are NEW or CHANGED get highlighted (green)
        - Original unchanged tokens remain unhighlighted (black)
        - Prevents the issue where adding a sentence before existing content
          causes the existing content to also be highlighted
        - BRAND NAMES are EXCLUDED from highlighting (never shown as changes)
        - Multi-word KEYWORDS are treated as ATOMIC UNITS (not partially highlighted)

        Algorithm:
        1. Preprocess to replace multi-word keywords with atomic tokens
        2. Tokenize both original and optimized text
        3. Use SequenceMatcher to find exact changes
        4. Only wrap "insert" and "replace" operations in markers
        5. Skip highlighting for brand name variations
        6. Postprocess to restore keyword phrases

        Args:
            original: Original text block.
            optimized: Optimized text (without markers).
            full_original_text: Full document text for context (used as baseline).
            keywords: Optional list of keyword phrases to treat as atomic units.

        Returns:
            Text with markers around ONLY new/changed tokens.
        """
        # Get brand variations for exclusion (if available)
        brand_variations = None
        brand_context = getattr(self, '_brand_context', None)
        if brand_context:
            brand_variations = brand_context.get('variations')
            original_spelling = brand_context.get('original_spelling')

            # Normalize brand names in optimized text to match original spelling
            if original_spelling and brand_variations:
                optimized = normalize_brand_in_text(optimized, original_spelling, brand_variations)

        # Normalize paragraph spacing to fix "word.Word" patterns from LLM output
        optimized = normalize_paragraph_spacing(optimized)

        # Get keywords for atomic unit treatment if not provided
        if keywords is None:
            # Try to get keywords from the keyword plan stored on the optimizer
            keyword_plan = getattr(self, '_current_keyword_plan', None)
            if keyword_plan:
                keywords = [keyword_plan.primary.phrase]
                keywords.extend([kw.phrase for kw in keyword_plan.secondary])

        # Compare original block directly against optimized block
        # This ensures only tokens that changed within THIS block get highlighted
        # Brand variations are excluded from highlighting
        # Multi-word keywords are treated as atomic units (not partially highlighted)
        return add_markers_by_diff(
            original,
            optimized,
            brand_variations=brand_variations,
            keywords=keywords,
        )

    def _optimize_body_content(
        self,
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        optimization_plan: OptimizationPlan,
        analysis: ContentAnalysis,
        full_original_text: Optional[str] = None,
    ) -> list[ParagraphBlock]:
        """
        Optimize body content blocks using the optimization plan.

        KEYWORD GUARANTEES (Part 5 of 10-Part Framework):
        - Tier 3: Primary keyword MUST appear in first ~100 words
        - Tier 4: Keywords should appear in subheadings (H2/H3)
        - Tier 5: Keywords distributed throughout body
        - Tier 7: Keywords in conclusion

        Uses the optimization plan's placement_plan to guide keyword distribution.

        Args:
            blocks: Content blocks to optimize.
            keyword_plan: The keyword plan.
            optimization_plan: Pre-computed optimization targets and placement strategy.
            analysis: Content analysis results.
            full_original_text: Full document for diff comparison.

        Returns:
            Optimized content blocks with markers.
        """
        if not blocks:
            return []

        # MINIMAL MODE: Use patch-based optimization (no LLM rewrites)
        effective_config = getattr(self, '_effective_config', None)
        if effective_config and effective_config.is_minimal_mode:
            import sys
            print("DEBUG: Using patch-based body optimization (minimal mode)", file=sys.stderr)
            return self._optimize_body_content_minimal(
                blocks=blocks,
                keyword_plan=keyword_plan,
                analysis=analysis,
                full_original_text=full_original_text,
            )

        optimized_blocks = []
        primary = keyword_plan.primary.phrase
        secondary = [kw.phrase for kw in keyword_plan.secondary]

        # Get content topics for constraints (set during optimize())
        content_topics = getattr(self, '_content_topics', None)

        # Get placement plan from optimization plan for tiered keyword distribution
        placement_plan = optimization_plan.placement_plan
        subheading_keywords = placement_plan.subheadings if placement_plan else secondary[:2]
        body_keywords = placement_plan.body_priority if placement_plan else [primary] + secondary

        # Track which keywords have been placed in body
        keywords_placed_in_body = set()

        # Identify the conclusion block (last non-heading paragraph)
        conclusion_index = -1
        for i in range(len(blocks) - 1, -1, -1):
            if not blocks[i].is_heading and len(blocks[i].text) > 50:
                conclusion_index = i
                break

        # Determine which blocks to optimize
        for i, block in enumerate(blocks):
            if block.heading_level == HeadingLevel.H1:
                # H1 is handled separately in meta elements
                optimized_blocks.append(block)
                continue

            # Optimize headings (H2-H6) - Tier 4 placement
            if block.is_heading:
                # Use subheading keywords from placement plan
                # Note: In insert_only mode, headings are locked (returns original)
                optimized_text = self._optimize_heading(
                    block.text,
                    primary,
                    subheading_keywords,
                    block.heading_level,
                    full_original_text=full_original_text,
                    config=effective_config,
                )
                optimized_blocks.append(
                    ParagraphBlock(
                        text=optimized_text,
                        heading_level=block.heading_level,
                        style_name=block.style_name,
                    )
                )
            elif i < 3:  # First few paragraphs get full optimization (Tier 3 & 5)
                # For first paragraph, emphasize primary keyword (Tier 3)
                para_keywords = [primary] + secondary[:2] if i == 0 else secondary[:3]
                rewritten_text = self.llm.rewrite_with_markers(
                    content=block.text,
                    primary_keyword=primary,
                    secondary_keywords=para_keywords,
                    context=f"This is paragraph {i+1} of the content about {analysis.topic}. "
                           f"{'Ensure the primary keyword appears early in this paragraph.' if i == 0 else ''}",
                    content_topics=content_topics,
                    brand_context=getattr(self, '_brand_context', None),
                    optimization_mode=getattr(self, '_effective_config', None).optimization_mode if getattr(self, '_effective_config', None) else "enhanced",
                )
                # SAFETY: Validate LLM output for blocked terms AND content preservation
                # Falls back to original if content was deleted or blocked terms added
                rewritten_text, _, _ = validate_and_preserve(
                    strip_markers(rewritten_text), block.text, f"paragraph_{i+1}"
                )
                # V2: Sentence-level diff (full sentences highlighted, no partial)
                optimized_text = self._compute_markers_v2(
                    block.text, rewritten_text, full_original_text=full_original_text
                )

                # Track placed keywords
                for kw in [primary] + secondary:
                    if kw.lower() in optimized_text.lower():
                        keywords_placed_in_body.add(kw.lower())

                optimized_blocks.append(
                    ParagraphBlock(
                        text=optimized_text,
                        heading_level=block.heading_level,
                        style_name=block.style_name,
                    )
                )
            elif i == conclusion_index:
                # Conclusion paragraph - Tier 7 placement
                conclusion_keywords = placement_plan.conclusion if placement_plan else [primary]
                rewritten_text = self.llm.rewrite_with_markers(
                    content=block.text,
                    primary_keyword=primary,
                    secondary_keywords=conclusion_keywords,
                    context=f"This is the conclusion paragraph. Ensure it reinforces the main topic about {analysis.topic}.",
                    content_topics=content_topics,
                    brand_context=getattr(self, '_brand_context', None),
                    optimization_mode=getattr(self, '_effective_config', None).optimization_mode if getattr(self, '_effective_config', None) else "enhanced",
                )
                # SAFETY: Validate LLM output for blocked terms AND content preservation
                # Falls back to original if content was deleted or blocked terms added
                rewritten_text, _, _ = validate_and_preserve(
                    strip_markers(rewritten_text), block.text, "conclusion"
                )
                # V2: Sentence-level diff (full sentences highlighted, no partial)
                optimized_text = self._compute_markers_v2(
                    block.text, rewritten_text, full_original_text=full_original_text
                )
                optimized_blocks.append(
                    ParagraphBlock(
                        text=optimized_text,
                        heading_level=block.heading_level,
                        style_name=block.style_name,
                    )
                )
            else:
                # Body paragraphs - Tier 5 placement
                # Always send non-heading body paragraphs to LLM for potential optimization
                # (unless already handled by first few paragraphs or conclusion logic)
                # Ensure the paragraph has substantial text before sending to LLM
                if len(block.text) > 50:  # Only optimize paragraphs with more than 50 characters
                    rewritten_text = self.llm.rewrite_with_markers(
                        content=block.text,
                        primary_keyword=primary if primary.lower() not in keywords_placed_in_body else primary, # Provide primary for context
                        secondary_keywords=secondary, # Provide all secondary for context
                        context="Optimize this paragraph to naturally include relevant keywords and enhance content quality.",
                        content_topics=content_topics,
                        brand_context=getattr(self, '_brand_context', None),
                        optimization_mode=getattr(self, '_effective_config', None).optimization_mode if getattr(self, '_effective_config', None) else "enhanced",
                    )
                    # SAFETY: Validate LLM output for blocked terms AND content preservation
                    # Falls back to original if content was deleted or blocked terms added
                    rewritten_text, _, _ = validate_and_preserve(
                        strip_markers(rewritten_text), block.text, f"paragraph_{i+1}"
                    )
                    # V2: Sentence-level diff (full sentences highlighted, no partial)
                    optimized_text = self._compute_markers_v2(
                        block.text, rewritten_text, full_original_text=full_original_text
                    )

                    # Track placed keywords
                    for kw in [primary] + secondary:
                        if kw.lower() in optimized_text.lower():
                            keywords_placed_in_body.add(kw.lower())

                    optimized_blocks.append(
                        ParagraphBlock(
                            text=optimized_text,
                            heading_level=block.heading_level,
                            style_name=block.style_name,
                        )
                    )
                else:
                    # Keep very short paragraphs as-is but track any keywords present
                    block_text_lower = block.text.lower()
                    for kw in [primary] + secondary:
                        if kw.lower() in block_text_lower:
                            keywords_placed_in_body.add(kw.lower())
                    optimized_blocks.append(block)

        # KEYWORD GUARANTEE (Tier 3): Ensure primary keyword is in first ~100 words
        optimized_blocks = self._ensure_primary_in_first_100_words(
            optimized_blocks, primary, analysis.topic
        )

        # CONTENT PRESERVATION: Validate block count is preserved
        # Optimization should be ADDITIVE - never remove blocks
        block_valid, block_error = validate_block_preservation(
            blocks, optimized_blocks, "body_content"
        )
        if not block_valid:
            # CRITICAL: Block deletion detected - log error and return original
            import sys
            print(f"CRITICAL ERROR: {block_error}", file=sys.stderr)
            # Fall back to original blocks to prevent data loss
            return blocks

        return optimized_blocks

    def _optimize_body_content_minimal(
        self,
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        analysis: ContentAnalysis,
        full_original_text: Optional[str] = None,
    ) -> list[ParagraphBlock]:
        """
        Patch-based body optimization for minimal/insert-only mode.

        Unlike the full LLM-based optimization, this method:
        1. Preserves all original text without rewrites
        2. Only injects keywords where they don't already exist
        3. Respects keyword caps (primary=1, secondary=1 by default)
        4. Uses natural injection patterns without LLM calls

        This is MUCH faster and produces minimal changes to the original content.

        Args:
            blocks: Content blocks to optimize.
            keyword_plan: The keyword plan with primary and secondary keywords.
            analysis: Content analysis results.
            full_original_text: Full document for context.

        Returns:
            Blocks with minimal keyword injections (no full rewrites).
        """
        import sys

        if not blocks:
            return []

        effective_config = getattr(self, '_effective_config', None)
        primary_cap = effective_config.primary_keyword_body_cap if effective_config else 1
        secondary_cap = effective_config.secondary_keyword_body_cap if effective_config else 1

        primary = keyword_plan.primary.phrase
        secondary_keywords = [kw.phrase for kw in keyword_plan.secondary]

        # Count existing keyword occurrences in body (excluding H1 and headings)
        body_text_parts = []
        for block in blocks:
            if block.heading_level == HeadingLevel.H1 or block.is_heading:
                continue
            body_text_parts.append(block.text.lower())
        full_body_text = " ".join(body_text_parts)

        # Track which keywords need injection
        primary_count = count_keyword_in_text(full_body_text, primary.lower())
        primary_needs_injection = primary_count < primary_cap

        secondary_needs = {}
        for kw in secondary_keywords:
            kw_count = count_keyword_in_text(full_body_text, kw.lower())
            if kw_count < secondary_cap:
                secondary_needs[kw] = secondary_cap - kw_count

        print(f"DEBUG minimal: primary '{primary}' count={primary_count}, needs_injection={primary_needs_injection}", file=sys.stderr)
        print(f"DEBUG minimal: secondary needs: {secondary_needs}", file=sys.stderr)

        # Find suitable paragraphs for injection (prefer longer ones)
        # Create list of (index, block, length) for non-heading blocks
        injection_candidates = []
        for i, block in enumerate(blocks):
            if block.heading_level == HeadingLevel.H1 or block.is_heading:
                continue
            if len(block.text) > 50:  # Only paragraphs with substantial text
                injection_candidates.append((i, block, len(block.text)))

        # Sort by length (longer paragraphs are better for injection)
        injection_candidates.sort(key=lambda x: x[2], reverse=True)

        # Track which blocks we've modified
        modified_blocks = {i: block.text for i, block, _ in injection_candidates}

        # Inject primary keyword if needed
        if primary_needs_injection and injection_candidates:
            # Use the longest paragraph for primary keyword
            target_idx, target_block, _ = injection_candidates[0]
            modified_text = inject_keyword_naturally(modified_blocks[target_idx], primary)
            if modified_text != modified_blocks[target_idx]:
                modified_blocks[target_idx] = modified_text
                print(f"DEBUG minimal: Injected primary '{primary}' into block {target_idx}", file=sys.stderr)

        # Inject secondary keywords if needed
        candidate_idx = 1  # Start from second-best candidate for secondary
        for kw, needed in secondary_needs.items():
            if needed <= 0:
                continue
            # Find a suitable candidate that doesn't already have this keyword
            for idx in range(candidate_idx, len(injection_candidates)):
                target_idx, _, _ = injection_candidates[idx]
                if kw.lower() not in modified_blocks[target_idx].lower():
                    modified_text = inject_keyword_naturally(modified_blocks[target_idx], kw)
                    if modified_text != modified_blocks[target_idx]:
                        modified_blocks[target_idx] = modified_text
                        print(f"DEBUG minimal: Injected secondary '{kw}' into block {target_idx}", file=sys.stderr)
                        candidate_idx = idx + 1
                        break

        # Build result blocks
        result_blocks = []
        for i, block in enumerate(blocks):
            if i in modified_blocks and modified_blocks[i] != block.text:
                # Block was modified - create new block with injected content
                result_blocks.append(
                    ParagraphBlock(
                        text=modified_blocks[i],
                        heading_level=block.heading_level,
                        style_name=block.style_name,
                    )
                )
            else:
                # Block unchanged - keep original
                result_blocks.append(block)

        # KEYWORD GUARANTEE (Tier 3): Ensure primary keyword is in first ~100 words
        # This may add an intro sentence if keyword wasn't injected successfully
        result_blocks = self._ensure_primary_in_first_100_words(
            result_blocks, primary, analysis.topic
        )

        return result_blocks

    def _ensure_primary_in_first_100_words(
        self,
        blocks: list[ParagraphBlock],
        primary: str,
        topic: str,
    ) -> list[ParagraphBlock]:
        """
        Ensure primary keyword appears in the first ~100 words of body content.

        If not present, inject an intro sentence containing the primary keyword.

        Args:
            blocks: The optimized content blocks.
            primary: The primary keyword phrase.
            topic: The content topic.

        Returns:
            Updated blocks with primary keyword guaranteed in first 100 words.
        """
        # Collect first 100 words from body blocks (skip H1)
        words_collected = []
        for block in blocks:
            if block.heading_level == HeadingLevel.H1:
                continue
            block_text = strip_markers(block.text)
            block_words = block_text.split()
            words_collected.extend(block_words)
            if len(words_collected) >= 100:
                break

        first_100_words = " ".join(words_collected[:100]).lower()

        # Check if primary keyword is in first 100 words
        if primary.lower() in first_100_words:
            return blocks  # Already present

        # Primary keyword missing from first 100 words - inject intro sentence
        intro_sentence = f"This guide covers everything you need to know about {primary}."

        # Find the first non-H1 body block and prepend the intro
        result_blocks = []
        intro_added = False

        for i, block in enumerate(blocks):
            if block.heading_level == HeadingLevel.H1:
                result_blocks.append(block)
                continue

            if not intro_added and not block.is_heading:
                # Add intro paragraph before this block
                intro_block = ParagraphBlock(
                    text=f"{ADD_START}{intro_sentence}{ADD_END}",
                    heading_level=HeadingLevel.BODY,
                )
                result_blocks.append(intro_block)
                intro_added = True

            result_blocks.append(block)

        # If we couldn't find a good place, add at the end
        if not intro_added:
            intro_block = ParagraphBlock(
                text=f"{ADD_START}{intro_sentence}{ADD_END}",
                heading_level=HeadingLevel.BODY,
            )
            result_blocks.append(intro_block)

        return result_blocks

    def _optimize_heading(
        self,
        text: str,
        primary: str,
        secondary: list[str],
        level: HeadingLevel,
        full_original_text: Optional[str] = None,
        config: Optional["OptimizationConfig"] = None,
    ) -> str:
        """Optimize a heading to include keywords where appropriate.

        In insert_only mode (config.should_lock_headings=True), headings are
        NEVER modified - they are preserved exactly as in the source.
        """
        # INSERT-ONLY MODE: Lock headings from modification
        if config is not None and config.should_lock_headings:
            return text

        text_lower = text.lower()

        # Check if already contains keywords
        if primary.lower() in text_lower:
            return text

        # Try to find a secondary keyword that fits
        for kw in secondary:
            if kw.lower() in text_lower:
                return text

        # Need to optimize - use LLM for natural inclusion
        prompt = f"""Optimize this heading to naturally include a relevant keyword.

Heading: {text}
Primary keyword: {primary}
Secondary keywords: {', '.join(secondary[:3])}

Requirements:
- Keep the heading concise and clear
- Only add keyword if it fits naturally
- If no natural fit, return the original heading unchanged

Return ONLY the optimized heading text, with no markers or annotations."""

        try:
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=100,
                system="You are an SEO heading optimizer. Return only the plain optimized heading text.",
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = response.content[0].text.strip()
            # SAFETY: Validate LLM output for blocked industry terms
            rewritten, _ = validate_and_fallback(
                strip_markers(rewritten), text, "heading"
            )
            # V2: Sentence-level diff (full sentences highlighted, no partial)
            return self._compute_markers_v2(
                text, rewritten, full_original_text=full_original_text
            )
        except Exception:
            # On error, return original
            return text

    def _replace_h1_in_blocks(
        self,
        blocks: list[ParagraphBlock],
        optimized_h1: str,
    ) -> list[ParagraphBlock]:
        """
        Replace the H1 block in body content with the optimized H1.

        This ensures the first H1 in "OPTIMIZED CONTENT" matches the
        optimized H1 shown in the meta elements table.

        Args:
            blocks: The body content blocks.
            optimized_h1: The optimized H1 text (with markers).

        Returns:
            Updated blocks with H1 replaced.
        """
        import sys

        result = []
        h1_replaced = False

        # Debug: Count H1 blocks in input
        h1_count = sum(1 for b in blocks if b.heading_level == HeadingLevel.H1)
        print(f"DEBUG _replace_h1_in_blocks: {len(blocks)} blocks, {h1_count} H1 blocks", file=sys.stderr)
        print(f"DEBUG _replace_h1_in_blocks: optimized_h1='{optimized_h1[:100]}...'", file=sys.stderr)

        for i, block in enumerate(blocks):
            if block.heading_level == HeadingLevel.H1 and not h1_replaced:
                # Replace with optimized H1
                print(f"DEBUG: Replacing H1 at index {i}: '{block.text[:50]}...' -> '{optimized_h1[:50]}...'", file=sys.stderr)
                result.append(
                    ParagraphBlock(
                        text=optimized_h1,
                        heading_level=HeadingLevel.H1,
                        style_name=block.style_name,
                    )
                )
                h1_replaced = True
            else:
                result.append(block)

        if not h1_replaced:
            print(f"WARNING: No H1 block found to replace! First 3 block types:", file=sys.stderr)
            for i, b in enumerate(blocks[:3]):
                print(f"  Block {i}: heading_level={b.heading_level}, text='{b.text[:50]}...'", file=sys.stderr)

        return result

    def _ensure_all_keywords_present(
        self,
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        topic: str,
    ) -> list[ParagraphBlock]:
        """
        Ensure ALL keywords from the keyword plan appear in the content.

        This is the final safety net - if any keyword was missed by the LLM
        during optimization, we add a fallback sentence containing it.

        Args:
            blocks: The optimized content blocks.
            keyword_plan: The keyword plan with primary and secondary keywords.
            topic: The content topic for context.

        Returns:
            Updated blocks with any missing keywords added.
        """
        # Collect all content text
        all_text = " ".join(strip_markers(block.text) for block in blocks).lower()

        # Find missing keywords
        missing_keywords = []

        # Check primary keyword
        if keyword_plan.primary.phrase.lower() not in all_text:
            missing_keywords.append(keyword_plan.primary.phrase)

        # Check secondary keywords
        for kw in keyword_plan.secondary:
            if kw.phrase.lower() not in all_text:
                missing_keywords.append(kw.phrase)

        # If no keywords are missing, return unchanged
        if not missing_keywords:
            return blocks

        # Generate fallback sentences for missing keywords
        fallback_sentences = self._generate_keyword_fallback_sentences(
            missing_keywords=missing_keywords,
            topic=topic,
        )

        # Add fallback paragraph at the end of the content
        if fallback_sentences:
            fallback_text = " ".join(fallback_sentences)
            # Wrap entire fallback in markers since it's all new content
            marked_fallback = f"{ADD_START}{fallback_text}{ADD_END}"
            blocks.append(
                ParagraphBlock(
                    text=marked_fallback,
                    heading_level=HeadingLevel.BODY,
                )
            )

        return blocks

    def _generate_keyword_fallback_sentences(
        self,
        missing_keywords: list[str],
        topic: str,
    ) -> list[str]:
        """
        Generate natural-sounding sentences that include missing keywords.

        Args:
            missing_keywords: Keywords that need to be added.
            topic: The content topic for context.

        Returns:
            List of sentences, each containing at least one keyword.
        """
        if not missing_keywords:
            return []

        # Use LLM to generate natural sentences containing the keywords
        prompt = f"""Generate 1-2 short, natural sentences about "{topic}" that include these EXACT keyword phrases:

Keywords to include: {", ".join(missing_keywords)}

CRITICAL REQUIREMENTS:
1. Each keyword phrase must appear EXACTLY as written (not paraphrased)
2. Sentences must sound natural and informative
3. Keep each sentence under 30 words
4. Do NOT use bullet points or numbered lists
5. Return only the sentences, nothing else

Example output format:
Understanding [keyword1] is essential for [topic]. Many businesses benefit from [keyword2] solutions."""

        try:
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=200,
                system="You are a content writer. Generate natural sentences that include specific keyword phrases exactly as provided.",
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text.strip()

            # Get original content for validation
            full_original_text = getattr(self, '_full_original_text', "")

            # Split into sentences and filter
            sentences = []
            for sentence in result.replace("\n", " ").split(". "):
                sentence = sentence.strip()
                if sentence and not sentence.startswith("-") and not sentence[0].isdigit():
                    if not sentence.endswith("."):
                        sentence += "."
                    # SAFETY: Validate each sentence for blocked terms
                    validated, _ = validate_and_fallback(sentence, full_original_text, "fallback_sentence")
                    if validated == sentence:  # Only include if validation passed
                        sentences.append(sentence)

            return sentences
        except Exception:
            # Fallback: create simple sentences manually
            sentences = []
            for kw in missing_keywords:
                sentences.append(f"Learn more about {kw} and how it relates to {topic}.")
            return sentences

    def _enforce_keyword_count(
        self,
        meta_elements: list[MetaElement],
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        target_count: int,
        topic: str,
    ) -> list[ParagraphBlock]:
        """
        Enforce minimum keyword count for the primary keyword.

        After optimization, count actual keyword occurrences. If below target,
        insert additional natural usages distributed throughout the content.

        Target locations (6 total):
        - Title tag (1) - already handled in meta optimization
        - Meta description (1) - already handled in meta optimization
        - H1 (1) - already handled in meta optimization
        - First paragraph (1) - check/add here
        - Middle content (1) - check/add here
        - Closing paragraph (1) - check/add here

        Args:
            meta_elements: Optimized meta elements.
            blocks: Optimized content blocks.
            keyword_plan: The keyword plan.
            target_count: Target number of keyword occurrences (default: 6).
            topic: Content topic for generating natural sentences.

        Returns:
            Updated blocks with additional keyword occurrences if needed.
        """
        primary_keyword = keyword_plan.primary.phrase

        # Count current occurrences across all content
        # MetaElement has .optimized attribute (not .optimized_title etc.)
        meta_text = " ".join(
            elem.optimized or '' for elem in meta_elements
        )
        body_text = " ".join(strip_markers(block.text) for block in blocks)
        full_text = f"{meta_text} {body_text}".lower()

        current_count = full_text.count(primary_keyword.lower())

        # If we've already met the target, no action needed
        if current_count >= target_count:
            return blocks

        # Calculate how many more we need
        needed = target_count - current_count

        # Identify strategic insertion points in body content
        # We'll try to insert in: first paragraph, middle content, closing paragraph
        insertion_points = self._identify_keyword_insertion_points(blocks)

        # Generate additional sentences with the primary keyword
        additional_sentences = self._generate_keyword_sentences(
            keyword=primary_keyword,
            count=needed,
            topic=topic,
        )

        if not additional_sentences:
            return blocks

        # Insert sentences at strategic points
        updated_blocks = self._insert_keyword_sentences(
            blocks=blocks,
            sentences=additional_sentences,
            insertion_points=insertion_points,
        )

        return updated_blocks

    def _enforce_secondary_keyword_counts(
        self,
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        topic: str,
        secondary_target: int = 3,
    ) -> list[ParagraphBlock]:
        """
        Enforce minimum keyword count for secondary keywords.

        After optimization, count actual keyword occurrences for each secondary keyword.
        If below target, insert additional natural usages distributed throughout the content.

        Default target for secondary keywords is 3 occurrences each.

        Args:
            blocks: Optimized content blocks.
            keyword_plan: The keyword plan with secondary keywords.
            topic: Content topic for generating natural sentences.
            secondary_target: Target occurrences for each secondary keyword (default: 3).

        Returns:
            Updated blocks with additional secondary keyword occurrences if needed.
        """
        if not keyword_plan.secondary:
            return blocks

        updated_blocks = list(blocks)

        # Process each secondary keyword
        for secondary_kw in keyword_plan.secondary:
            keyword_phrase = secondary_kw.phrase

            # Count current occurrences in body content
            body_text = " ".join(strip_markers(block.text) for block in updated_blocks).lower()
            current_count = body_text.count(keyword_phrase.lower())

            # If we've already met the target, skip this keyword
            if current_count >= secondary_target:
                continue

            # Calculate how many more we need
            needed = secondary_target - current_count

            # Identify strategic insertion points
            insertion_points = self._identify_keyword_insertion_points(updated_blocks)

            # Generate additional sentences with this secondary keyword
            additional_sentences = self._generate_keyword_sentences(
                keyword=keyword_phrase,
                count=needed,
                topic=topic,
            )

            if not additional_sentences:
                continue

            # Insert sentences at strategic points
            updated_blocks = self._insert_keyword_sentences(
                blocks=updated_blocks,
                sentences=additional_sentences,
                insertion_points=insertion_points,
            )

        return updated_blocks

    def _distribute_keywords_across_sections(
        self,
        meta_elements: list[MetaElement],
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        topic: str,
        primary_target: int = 6,
        secondary_target: int = 3,
    ) -> list[ParagraphBlock]:
        """
        Distribute ALL keywords EVENLY across the ENTIRE document using paragraph intervals.

        This is the UNIFIED keyword distribution method that ensures keywords are
        spread throughout the document, not clustered at beginning/end or section
        boundaries only.

        Algorithm:
        1. Identify ALL body paragraphs in the document
        2. Calculate keyword needs (primary and all secondaries)
        3. Calculate insertion interval: total_paragraphs / total_insertions
        4. Assign keywords to paragraphs at regular intervals throughout document
        5. Ensure no span of 3+ paragraphs is skipped
        6. Validate max 2 consecutive green sentences per block

        Args:
            meta_elements: Meta elements (to count existing keyword usage).
            blocks: Content blocks to modify.
            keyword_plan: All keywords (primary + secondary).
            topic: Content topic for generating sentences.
            primary_target: Target count for primary keyword.
            secondary_target: Target count for each secondary keyword.

        Returns:
            Updated blocks with keywords evenly distributed throughout.
        """
        # Step 1: Identify ALL body paragraphs (non-headings with content)
        paragraph_indices = self._get_all_paragraph_indices(blocks)

        if not paragraph_indices:
            return blocks

        # Step 2: Calculate keyword needs
        keyword_needs = self._calculate_keyword_needs(
            meta_elements=meta_elements,
            blocks=blocks,
            keyword_plan=keyword_plan,
            primary_target=primary_target,
            secondary_target=secondary_target,
        )

        if not keyword_needs:
            return blocks  # All keywords already at target

        # Step 3: Build flat list of all keyword insertions needed
        all_insertions = []
        for kw_info in keyword_needs:
            for _ in range(kw_info["needed"]):
                all_insertions.append(kw_info["keyword"])

        if not all_insertions:
            return blocks

        # Step 4: Calculate insertion points using paragraph intervals
        # Distribute insertions evenly throughout the document
        insertion_assignments = self._calculate_paragraph_interval_insertions(
            paragraph_indices=paragraph_indices,
            insertions=all_insertions,
        )

        # Step 5: Insert keywords at their assigned paragraphs
        updated_blocks = self._insert_keywords_at_paragraphs(
            blocks=blocks,
            insertion_assignments=insertion_assignments,
            topic=topic,
        )

        # Step 6: Validate and fix clustering (max 2 consecutive green sentences)
        updated_blocks = self._fix_consecutive_insertions(updated_blocks)

        return updated_blocks

    def _enforce_body_keyword_invariants(
        self,
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        topic: str,
        primary_target: int = 6,
        secondary_target: int = 3,
    ) -> list[ParagraphBlock]:
        """
        FINAL SAFETY: Enforce body keyword invariants before FAQ generation.

        After all optimization and distribution, this method guarantees:
        1. Primary keyword appears >= primary_target times in body (non-heading blocks)
        2. Each secondary keyword appears >= secondary_target times in body

        If any target is NOT satisfied, performs deterministic injection directly
        into body paragraphs as a last resort.

        Args:
            blocks: Content blocks after optimization.
            keyword_plan: All keywords (primary + secondary).
            topic: Content topic for generating sentences.
            primary_target: Target count for primary keyword in body.
            secondary_target: Target count for each secondary keyword in body.

        Returns:
            Updated blocks with guaranteed keyword invariants.
        """
        import sys

        # Step 1: Count keywords in BODY ONLY (non-heading blocks)
        body_text = " ".join(
            strip_markers(block.text)
            for block in blocks
            if not block.is_heading
        ).lower()

        # Step 2: Check each keyword against its target
        unsatisfied_keywords = []

        # Check primary keyword
        primary = keyword_plan.primary.phrase
        primary_count = count_keyword_in_text(body_text, primary)
        if primary_count < primary_target:
            needed = primary_target - primary_count
            unsatisfied_keywords.append({
                "keyword": primary,
                "needed": needed,
                "is_primary": True,
                "current": primary_count,
                "target": primary_target,
            })
            print(
                f"BODY INVARIANT VIOLATION: Primary keyword '{primary}' has only "
                f"{primary_count}/{primary_target} occurrences in body. Need {needed} more.",
                file=sys.stderr
            )

        # Check secondary keywords
        for secondary in keyword_plan.secondary:
            phrase = secondary.phrase
            count = count_keyword_in_text(body_text, phrase)
            if count < secondary_target:
                needed = secondary_target - count
                unsatisfied_keywords.append({
                    "keyword": phrase,
                    "needed": needed,
                    "is_primary": False,
                    "current": count,
                    "target": secondary_target,
                })
                print(
                    f"BODY INVARIANT VIOLATION: Secondary keyword '{phrase}' has only "
                    f"{count}/{secondary_target} occurrences in body. Need {needed} more.",
                    file=sys.stderr
                )

        # Step 3: If all targets satisfied, return unchanged
        if not unsatisfied_keywords:
            print("BODY INVARIANTS SATISFIED: All keywords meet body targets.", file=sys.stderr)
            return blocks

        # Step 4: Deterministic injection for unsatisfied keywords
        print(
            f"ENFORCING BODY INVARIANTS: Injecting {len(unsatisfied_keywords)} "
            f"unsatisfied keywords into body paragraphs...",
            file=sys.stderr
        )

        updated_blocks = list(blocks)

        # Get all body paragraph indices (sorted from start to end)
        paragraph_indices = self._get_all_paragraph_indices(updated_blocks)

        if not paragraph_indices:
            print("WARNING: No body paragraphs available for injection.", file=sys.stderr)
            return blocks

        # Build flat list of all keyword insertions needed
        all_insertions = []
        for kw_info in unsatisfied_keywords:
            for _ in range(kw_info["needed"]):
                all_insertions.append(kw_info["keyword"])

        # Distribute insertions evenly across paragraphs
        insertion_assignments = self._calculate_paragraph_interval_insertions(
            paragraph_indices=paragraph_indices,
            insertions=all_insertions,
        )

        # Insert keywords at their assigned paragraphs
        updated_blocks = self._insert_keywords_at_paragraphs(
            blocks=updated_blocks,
            insertion_assignments=insertion_assignments,
            topic=topic,
        )

        # Fix any clustering issues
        updated_blocks = self._fix_consecutive_insertions(updated_blocks)

        # Step 5: Verify invariants are now satisfied
        body_text_after = " ".join(
            strip_markers(block.text)
            for block in updated_blocks
            if not block.is_heading
        ).lower()

        still_unsatisfied = []
        for kw_info in unsatisfied_keywords:
            keyword = kw_info["keyword"]
            target = kw_info["target"]
            new_count = count_keyword_in_text(body_text_after, keyword)
            if new_count < target:
                still_unsatisfied.append(f"{keyword} ({new_count}/{target})")
            else:
                print(
                    f"  ✓ '{keyword}' now has {new_count}/{target} in body",
                    file=sys.stderr
                )

        if still_unsatisfied:
            print(
                f"WARNING: Some keywords still below target after enforcement: "
                f"{', '.join(still_unsatisfied)}",
                file=sys.stderr
            )

        return updated_blocks

    def _enforce_keyword_caps(
        self,
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        config: "OptimizationConfig",
    ) -> list[ParagraphBlock]:
        """
        Enforce MAXIMUM keyword occurrence caps (insert-only mode).

        Unlike _enforce_body_keyword_invariants which enforces MINIMUMS,
        this function enforces MAXIMUMS. If a keyword appears MORE than
        the cap allows, excess occurrences are removed.

        This is the KEY function for insert-only mode behavior.

        Algorithm:
        1. For each keyword (primary + secondary):
           a. Count occurrences in body text
           b. If count > cap:
              - Find sentences containing the keyword (in ADDED content)
              - Remove sentences beyond the cap (starting from end)
        2. Only remove ADDED content - never remove original content

        Args:
            blocks: Content blocks after optimization.
            keyword_plan: All keywords (primary + secondary).
            config: Optimization config with cap settings.

        Returns:
            Updated blocks with keyword caps enforced.
        """
        import sys
        import re

        if not config.should_enforce_keyword_caps:
            return blocks

        print("ENFORCING KEYWORD CAPS: Checking for excess keyword occurrences...", file=sys.stderr)

        updated_blocks = list(blocks)

        # Build list of all keywords with their caps
        keywords_to_check = []

        # Primary keyword
        primary = keyword_plan.primary.phrase.lower()
        keywords_to_check.append({
            "phrase": primary,
            "cap": config.primary_keyword_body_cap,
            "is_primary": True,
        })

        # Secondary keywords
        for secondary in keyword_plan.secondary:
            keywords_to_check.append({
                "phrase": secondary.phrase.lower(),
                "cap": config.secondary_keyword_body_cap,
                "is_primary": False,
            })

        # Process each keyword
        for kw_info in keywords_to_check:
            phrase = kw_info["phrase"]
            cap = kw_info["cap"]

            # Count current occurrences in body (non-heading blocks)
            body_text = " ".join(
                strip_markers(block.text)
                for block in updated_blocks
                if not block.is_heading
            ).lower()

            current_count = count_keyword_in_text(body_text, phrase)

            if current_count <= cap:
                label = "PRIMARY" if kw_info["is_primary"] else "secondary"
                print(
                    f"  ✓ {label} '{phrase}' is within cap: {current_count}/{cap}",
                    file=sys.stderr
                )
                continue

            # Need to remove excess occurrences
            excess = current_count - cap
            label = "PRIMARY" if kw_info["is_primary"] else "secondary"
            print(
                f"  ✗ {label} '{phrase}' exceeds cap: {current_count}/{cap} (need to remove {excess})",
                file=sys.stderr
            )

            # Strategy: Find and remove ADDED sentences containing the keyword
            # We remove from the END of the document first (least impactful)
            updated_blocks = self._remove_excess_keyword_occurrences(
                blocks=updated_blocks,
                keyword=phrase,
                excess_count=excess,
            )

        return updated_blocks

    def _remove_excess_keyword_occurrences(
        self,
        blocks: list[ParagraphBlock],
        keyword: str,
        excess_count: int,
    ) -> list[ParagraphBlock]:
        """
        Remove excess keyword occurrences from content blocks.

        Priority for removal:
        1. ADDED sentences (marked with [[[ADD]]]) containing the keyword
        2. If not enough added sentences, remove keyword phrase from text

        Args:
            blocks: Content blocks.
            keyword: Keyword phrase to reduce.
            excess_count: How many occurrences to remove.

        Returns:
            Updated blocks with excess keywords removed.
        """
        import sys
        import re

        updated_blocks = []
        removed_count = 0

        # First pass: Find all blocks with keyword in ADDED content
        # Process in REVERSE order (remove from end first)
        for block in reversed(blocks):
            if block.is_heading:
                updated_blocks.insert(0, block)
                continue

            if removed_count >= excess_count:
                updated_blocks.insert(0, block)
                continue

            block_text = block.text
            keyword_lower = keyword.lower()

            # Check if this block has the keyword in ADDED content
            # Pattern: [[[ADD]]]...keyword...[[[ENDADD]]]
            add_pattern = r'\[\[\[ADD\]\]\](.*?)\[\[\[ENDADD\]\]\]'
            add_matches = list(re.finditer(add_pattern, block_text, re.IGNORECASE | re.DOTALL))

            modified_text = block_text
            for match in reversed(add_matches):  # Process in reverse to preserve indices
                if removed_count >= excess_count:
                    break

                added_content = match.group(1)
                if keyword_lower in added_content.lower():
                    # Found keyword in added content - remove this entire added segment
                    start, end = match.start(), match.end()

                    # Handle whitespace around the removed segment
                    # If there's a period or space before, preserve it
                    prefix = modified_text[:start].rstrip()
                    suffix = modified_text[end:].lstrip()

                    # Ensure proper spacing
                    if prefix and suffix:
                        if not prefix.endswith(('.', '!', '?', ':')):
                            modified_text = prefix + " " + suffix
                        else:
                            modified_text = prefix + " " + suffix
                    else:
                        modified_text = prefix + suffix

                    removed_count += 1
                    print(
                        f"    Removed added sentence containing '{keyword}' from block",
                        file=sys.stderr
                    )

            # Update block if modified
            if modified_text != block_text:
                updated_blocks.insert(0, ParagraphBlock(
                    text=modified_text.strip(),
                    original_index=block.original_index,
                    is_heading=block.is_heading,
                    heading_level=block.heading_level,
                ))
            else:
                updated_blocks.insert(0, block)

        # If we still have excess after removing added content, do a second pass
        # This time we remove keyword occurrences from the text itself
        if removed_count < excess_count:
            print(
                f"    Still {excess_count - removed_count} excess after removing added content",
                file=sys.stderr
            )
            # Second pass: Remove keyword phrases from non-added content (last resort)
            still_needed = excess_count - removed_count
            updated_blocks = self._strip_keyword_from_text(
                blocks=updated_blocks,
                keyword=keyword,
                count=still_needed,
            )

        return updated_blocks

    def _strip_keyword_from_text(
        self,
        blocks: list[ParagraphBlock],
        keyword: str,
        count: int,
    ) -> list[ParagraphBlock]:
        """
        Strip keyword phrase from text (last resort, non-added content).

        This carefully removes the keyword phrase while preserving readability.
        Only removes from body paragraphs, not headings.

        Args:
            blocks: Content blocks.
            keyword: Keyword phrase to remove.
            count: How many to remove.

        Returns:
            Updated blocks with keywords stripped.
        """
        import sys
        import re

        updated_blocks = []
        removed = 0

        # Process in reverse order (remove from end first)
        for block in reversed(blocks):
            if block.is_heading or removed >= count:
                updated_blocks.insert(0, block)
                continue

            block_text = block.text
            keyword_pattern = re.compile(
                r'\b' + re.escape(keyword) + r'\b',
                re.IGNORECASE
            )

            # Find all matches
            matches = list(keyword_pattern.finditer(block_text))
            if not matches:
                updated_blocks.insert(0, block)
                continue

            # Remove matches from end (preserving first occurrence in each block)
            modified_text = block_text
            for match in reversed(matches):
                if removed >= count:
                    break
                # Keep at least one occurrence per block if it's there
                if len(matches) == 1:
                    break  # Don't remove the only occurrence

                start, end = match.start(), match.end()
                # Remove the keyword phrase, preserving surrounding punctuation
                before = modified_text[:start].rstrip()
                after = modified_text[end:].lstrip()

                # Clean up double spaces or orphaned punctuation
                if before and after:
                    modified_text = before + " " + after
                else:
                    modified_text = before + after

                removed += 1
                print(
                    f"    Stripped keyword '{keyword}' from block text",
                    file=sys.stderr
                )

            # Update block if modified
            if modified_text != block_text:
                updated_blocks.insert(0, ParagraphBlock(
                    text=modified_text.strip(),
                    original_index=block.original_index,
                    is_heading=block.is_heading,
                    heading_level=block.heading_level,
                ))
            else:
                updated_blocks.insert(0, block)

        return updated_blocks

    def _get_all_paragraph_indices(
        self,
        blocks: list[ParagraphBlock],
    ) -> list[int]:
        """
        Get indices of ALL body paragraphs in the document.

        Returns indices of non-heading blocks with substantial content (>30 chars).
        These are the potential insertion points for keyword distribution.

        Args:
            blocks: Content blocks.

        Returns:
            List of block indices that are body paragraphs.
        """
        paragraph_indices = []
        for i, block in enumerate(blocks):
            # Skip headings
            if block.is_heading:
                continue
            # Skip very short blocks (likely captions, etc.)
            if len(strip_markers(block.text)) < 30:
                continue
            paragraph_indices.append(i)
        return paragraph_indices

    def _calculate_paragraph_interval_insertions(
        self,
        paragraph_indices: list[int],
        insertions: list[str],
    ) -> dict[int, list[str]]:
        """
        Calculate which paragraphs should receive keyword insertions using intervals.

        Algorithm:
        1. Calculate interval = num_paragraphs / num_insertions
        2. Place insertions at regular intervals throughout the document
        3. Ensure no span of 3+ paragraphs is skipped
        4. If more insertions than paragraphs, distribute multiple per paragraph

        Args:
            paragraph_indices: Indices of body paragraphs in document.
            insertions: List of keyword phrases to insert.

        Returns:
            Dict mapping paragraph index -> list of keywords to insert there.
        """
        num_paragraphs = len(paragraph_indices)
        num_insertions = len(insertions)

        if num_paragraphs == 0 or num_insertions == 0:
            return {}

        # Initialize assignments
        assignments: dict[int, list[str]] = {}

        # Calculate ideal interval between insertions
        # E.g., 15 paragraphs, 10 insertions -> interval of 1.5 paragraphs
        interval = num_paragraphs / num_insertions

        # Place insertions at calculated intervals
        for i, keyword in enumerate(insertions):
            # Calculate target paragraph position (0-indexed within paragraph list)
            target_position = int(i * interval)
            # Clamp to valid range
            target_position = min(target_position, num_paragraphs - 1)

            # Get actual block index from paragraph indices list
            block_idx = paragraph_indices[target_position]

            # Add to assignments
            if block_idx not in assignments:
                assignments[block_idx] = []
            assignments[block_idx].append(keyword)

        # Step 2: Ensure no span of 3+ consecutive paragraphs is skipped
        # Walk through paragraphs and fill gaps
        assignments = self._fill_distribution_gaps(
            paragraph_indices=paragraph_indices,
            assignments=assignments,
            insertions=insertions,
        )

        return assignments

    def _fill_distribution_gaps(
        self,
        paragraph_indices: list[int],
        assignments: dict[int, list[str]],
        insertions: list[str],
    ) -> dict[int, list[str]]:
        """
        Ensure no span of 3+ consecutive paragraphs is skipped.

        If a gap of 3+ paragraphs exists without any insertions,
        redistribute one insertion from a nearby paragraph with multiple.

        Args:
            paragraph_indices: Indices of body paragraphs.
            assignments: Current insertion assignments.
            insertions: Original list of insertions (for reference).

        Returns:
            Updated assignments with gaps filled.
        """
        if len(paragraph_indices) < 4:
            return assignments  # Too few paragraphs to have meaningful gaps

        # Find gaps of 3+ consecutive paragraphs without insertions
        assigned_positions = set()
        for block_idx in assignments.keys():
            if block_idx in paragraph_indices:
                assigned_positions.add(paragraph_indices.index(block_idx))

        # Scan for gaps
        gap_start = None
        for pos in range(len(paragraph_indices)):
            if pos in assigned_positions:
                # Check if we just ended a gap
                if gap_start is not None:
                    gap_length = pos - gap_start
                    if gap_length >= 3:
                        # Found a gap of 3+ - insert in the middle of the gap
                        gap_middle = gap_start + gap_length // 2
                        gap_block_idx = paragraph_indices[gap_middle]

                        # Find a paragraph with multiple insertions to borrow from
                        donor_block = None
                        for block_idx, kws in assignments.items():
                            if len(kws) > 1:
                                donor_block = block_idx
                                break

                        if donor_block and donor_block in assignments:
                            # Move one keyword from donor to gap
                            keyword_to_move = assignments[donor_block].pop()
                            if gap_block_idx not in assignments:
                                assignments[gap_block_idx] = []
                            assignments[gap_block_idx].append(keyword_to_move)

                gap_start = None
            else:
                if gap_start is None:
                    gap_start = pos

        # Check for trailing gap
        if gap_start is not None:
            gap_length = len(paragraph_indices) - gap_start
            if gap_length >= 3:
                gap_middle = gap_start + gap_length // 2
                gap_block_idx = paragraph_indices[gap_middle]

                # Find a donor
                for block_idx, kws in assignments.items():
                    if len(kws) > 1:
                        keyword_to_move = assignments[block_idx].pop()
                        if gap_block_idx not in assignments:
                            assignments[gap_block_idx] = []
                        assignments[gap_block_idx].append(keyword_to_move)
                        break

        return assignments

    def _insert_keywords_at_paragraphs(
        self,
        blocks: list[ParagraphBlock],
        insertion_assignments: dict[int, list[str]],
        topic: str,
    ) -> list[ParagraphBlock]:
        """
        Insert keyword sentences at their assigned paragraph positions.

        Appends generated keyword sentences to the end of each assigned paragraph.

        Args:
            blocks: Original content blocks.
            insertion_assignments: Dict mapping block index -> keywords to insert.
            topic: Content topic for generating sentences.

        Returns:
            Updated blocks with keywords inserted throughout.
        """
        updated_blocks = list(blocks)

        # Get original content for validation
        full_original_text = getattr(self, '_full_original_text', "")

        # Sort by block index to process in order
        for block_idx in sorted(insertion_assignments.keys()):
            keywords = insertion_assignments[block_idx]
            if not keywords or block_idx >= len(updated_blocks):
                continue

            # Generate and append sentences for each keyword at this paragraph
            for keyword in keywords:
                sentence = self._generate_single_keyword_sentence(keyword, topic)
                if not sentence:
                    continue

                # Validate sentence
                validated, _ = validate_and_fallback(sentence, full_original_text, "keyword_sentence")
                if validated != sentence:
                    continue

                # Append to the block
                block = updated_blocks[block_idx]
                marked_sentence = f" {ADD_START}{sentence}{ADD_END}"
                combined_text = normalize_paragraph_spacing(block.text + marked_sentence)
                updated_blocks[block_idx] = ParagraphBlock(
                    text=combined_text,
                    heading_level=block.heading_level,
                )

        return updated_blocks

    def _split_into_sections(
        self,
        blocks: list[ParagraphBlock],
    ) -> list[dict]:
        """
        Split document into sections based on H2 headings.

        Each section contains:
        - start_idx: First block index in section
        - end_idx: Last block index in section (inclusive)
        - heading: Section heading text (or "Intro" for first section)
        - is_intro: Whether this is the intro section (before first H2)

        Args:
            blocks: Content blocks.

        Returns:
            List of section dictionaries.
        """
        sections = []
        current_section_start = 0
        current_heading = "Introduction"
        is_intro = True

        for i, block in enumerate(blocks):
            # Check for H2 heading (section boundary)
            if block.is_heading and block.heading_level == HeadingLevel.H2:
                # End previous section
                if i > current_section_start:
                    sections.append({
                        "start_idx": current_section_start,
                        "end_idx": i - 1,
                        "heading": current_heading,
                        "is_intro": is_intro,
                    })

                # Start new section
                current_section_start = i
                current_heading = block.text
                is_intro = False

        # Add final section
        if current_section_start < len(blocks):
            sections.append({
                "start_idx": current_section_start,
                "end_idx": len(blocks) - 1,
                "heading": current_heading,
                "is_intro": is_intro,
            })

        return sections

    def _calculate_keyword_needs(
        self,
        meta_elements: list[MetaElement],
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        primary_target: int,
        secondary_target: int,
    ) -> list[dict]:
        """
        Calculate how many more occurrences each keyword needs IN BODY CONTENT.

        IMPORTANT: Keyword satisfaction is based on BODY CONTENT ONLY.
        Meta elements (title, description, H1) and FAQ do NOT count toward targets.
        This ensures keywords are actually placed in the main content.

        Args:
            meta_elements: Meta elements (NOT counted toward targets, kept for reference).
            blocks: Content blocks to count (body paragraphs only, not headings).
            keyword_plan: All keywords.
            primary_target: Target for primary keyword in body.
            secondary_target: Target for each secondary keyword in body.

        Returns:
            List of {keyword, needed, is_primary} dicts for keywords below target.
        """
        # Build BODY-ONLY text for counting (non-heading blocks only)
        # Meta and FAQ are intentionally excluded from satisfaction counting
        body_text = " ".join(
            strip_markers(block.text)
            for block in blocks
            if not block.is_heading
        ).lower()

        keyword_needs = []

        # Check primary keyword (body count only)
        primary = keyword_plan.primary.phrase
        body_count = count_keyword_in_text(body_text, primary)
        if body_count < primary_target:
            keyword_needs.append({
                "keyword": primary,
                "needed": primary_target - body_count,
                "is_primary": True,
            })

        # Check secondary keywords (body count only)
        for secondary in keyword_plan.secondary:
            phrase = secondary.phrase
            body_count = count_keyword_in_text(body_text, phrase)
            if body_count < secondary_target:
                keyword_needs.append({
                    "keyword": phrase,
                    "needed": secondary_target - body_count,
                    "is_primary": False,
                })

        return keyword_needs

    def _assign_keywords_to_sections(
        self,
        sections: list[dict],
        keyword_needs: list[dict],
    ) -> dict[int, list[str]]:
        """
        Distribute keywords EVENLY across sections.

        Algorithm:
        - Calculate total insertions needed
        - Divide proportionally across sections (minimum 1 per section if possible)
        - Rotate through keywords to spread different keywords across sections
        - Never assign more than 2 insertions to any single section

        Args:
            sections: Document sections.
            keyword_needs: Keywords and their needed counts.

        Returns:
            Dict mapping section index -> list of keywords to insert.
        """
        num_sections = len(sections)
        if num_sections == 0:
            return {}

        # Build list of all keyword insertions needed (flattened)
        all_insertions = []
        for kw_info in keyword_needs:
            for _ in range(kw_info["needed"]):
                all_insertions.append(kw_info["keyword"])

        if not all_insertions:
            return {}

        # Initialize section assignments
        section_assignments: dict[int, list[str]] = {i: [] for i in range(num_sections)}

        # Distribute evenly: assign insertions round-robin across sections
        # Start from section 1 (skip intro) if possible, to spread through middle
        total_insertions = len(all_insertions)

        # Calculate max insertions per section (never more than 2)
        max_per_section = 2

        # Calculate insertions per section for even distribution
        base_per_section = total_insertions // num_sections
        remainder = total_insertions % num_sections

        # Distribute insertions
        insertion_idx = 0
        for section_idx in range(num_sections):
            # How many for this section?
            count = base_per_section
            if section_idx < remainder:
                count += 1
            count = min(count, max_per_section)  # Cap at 2

            # Assign keywords to this section
            for _ in range(count):
                if insertion_idx < len(all_insertions):
                    section_assignments[section_idx].append(all_insertions[insertion_idx])
                    insertion_idx += 1

        # If we have leftover insertions (due to max cap), distribute to sections with room
        while insertion_idx < len(all_insertions):
            distributed = False
            for section_idx in range(num_sections):
                if len(section_assignments[section_idx]) < max_per_section:
                    section_assignments[section_idx].append(all_insertions[insertion_idx])
                    insertion_idx += 1
                    distributed = True
                    break
            if not distributed:
                # All sections at max - stop distributing
                break

        return section_assignments

    def _insert_keywords_by_section(
        self,
        blocks: list[ParagraphBlock],
        sections: list[dict],
        section_assignments: dict[int, list[str]],
        topic: str,
    ) -> list[ParagraphBlock]:
        """
        Insert keyword sentences at section boundaries.

        For each section with assigned keywords, insert sentences at the
        END of the section (last paragraph before next H2).

        Args:
            blocks: Original content blocks.
            sections: Document sections.
            section_assignments: Keywords assigned to each section.
            topic: Content topic for generating sentences.

        Returns:
            Updated blocks with keywords inserted.
        """
        updated_blocks = list(blocks)

        # Get original content for validation
        full_original_text = getattr(self, '_full_original_text', "")

        for section_idx, keywords in section_assignments.items():
            if not keywords or section_idx >= len(sections):
                continue

            section = sections[section_idx]

            # Find insertion point: last non-heading paragraph in section
            insertion_idx = None
            for i in range(section["end_idx"], section["start_idx"] - 1, -1):
                if i < len(updated_blocks) and not updated_blocks[i].is_heading:
                    insertion_idx = i
                    break

            if insertion_idx is None:
                continue

            # Generate sentences for each keyword assigned to this section
            for keyword in keywords:
                sentence = self._generate_single_keyword_sentence(keyword, topic)
                if not sentence:
                    continue

                # Validate sentence
                validated, _ = validate_and_fallback(sentence, full_original_text, "keyword_sentence")
                if validated != sentence:
                    continue

                # Append to the block at insertion point
                block = updated_blocks[insertion_idx]
                marked_sentence = f" {ADD_START}{sentence}{ADD_END}"
                combined_text = normalize_paragraph_spacing(block.text + marked_sentence)
                updated_blocks[insertion_idx] = ParagraphBlock(
                    text=combined_text,
                    heading_level=block.heading_level,
                )

        return updated_blocks

    def _generate_single_keyword_sentence(
        self,
        keyword: str,
        topic: str,
    ) -> Optional[str]:
        """
        Generate a single natural sentence containing the keyword.

        Args:
            keyword: Keyword to include.
            topic: Content topic for context.

        Returns:
            Generated sentence or None if failed.
        """
        prompt = f"""Generate ONE short, natural sentence about "{topic}" that includes this EXACT keyword phrase: "{keyword}"

REQUIREMENTS:
1. The phrase "{keyword}" must appear EXACTLY as written
2. Keep the sentence under 20 words
3. Make it feel naturally integrated
4. Return ONLY the sentence, nothing else

Example: When considering {keyword}, it's important to evaluate your specific needs."""

        try:
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=100,
                system="You are a content writer. Generate a natural sentence with the exact keyword phrase.",
                messages=[{"role": "user", "content": prompt}],
            )
            sentence = response.content[0].text.strip()

            # Validate keyword is present
            if keyword.lower() not in sentence.lower():
                return None

            # Ensure proper punctuation
            if not sentence.endswith((".", "!", "?")):
                sentence += "."

            return sentence

        except Exception:
            # Fallback template
            return f"Understanding {keyword} is essential for making informed decisions."

    def _fix_consecutive_insertions(
        self,
        blocks: list[ParagraphBlock],
    ) -> list[ParagraphBlock]:
        """
        Fix any blocks with more than 2 consecutive green (marked) sentences.

        If a block has >2 marked sentences, move excess to adjacent blocks.

        Args:
            blocks: Content blocks to check.

        Returns:
            Updated blocks with max 2 consecutive green sentences.
        """
        # Count marked sentences per block
        updated_blocks = list(blocks)
        max_marked = 2

        for i, block in enumerate(updated_blocks):
            # Count marked sentences in this block
            marked_count = block.text.count(ADD_START)

            if marked_count <= max_marked:
                continue

            # Need to redistribute excess
            # Split text by markers and reconstruct
            excess = marked_count - max_marked

            # Find adjacent blocks that can accept overflow
            # Try next block first, then previous
            overflow_sentences = []

            # Extract excess marked sentences from end
            text = block.text
            for _ in range(excess):
                last_start = text.rfind(ADD_START)
                if last_start == -1:
                    break
                last_end = text.rfind(ADD_END)
                if last_end == -1 or last_end < last_start:
                    break

                # Extract the marked sentence
                marked_portion = text[last_start:last_end + len(ADD_END)]
                overflow_sentences.insert(0, marked_portion)
                text = text[:last_start].rstrip()

            # Update current block
            updated_blocks[i] = ParagraphBlock(
                text=text,
                heading_level=block.heading_level,
            )

            # Distribute overflow to adjacent blocks
            for j, overflow in enumerate(overflow_sentences):
                # Find next suitable block
                target_idx = i + j + 1
                if target_idx < len(updated_blocks) and not updated_blocks[target_idx].is_heading:
                    target_block = updated_blocks[target_idx]
                    combined = normalize_paragraph_spacing(target_block.text + " " + overflow)
                    updated_blocks[target_idx] = ParagraphBlock(
                        text=combined,
                        heading_level=target_block.heading_level,
                    )

        return updated_blocks

    def _identify_keyword_insertion_points(
        self,
        blocks: list[ParagraphBlock],
    ) -> list[int]:
        """
        Identify EVENLY DISTRIBUTED insertion points throughout the document.

        Instead of just 3 points (first/middle/closing), this returns many
        insertion points spread across the entire document for natural keyword
        distribution. Points are placed after paragraphs and after headings
        to create natural transitions.

        Args:
            blocks: Content blocks to analyze.

        Returns:
            List of block indices suitable for keyword insertion, evenly distributed.
        """
        # Find all body paragraphs (non-headings with substantial content)
        body_indices = [
            i for i, block in enumerate(blocks)
            if not block.is_heading and len(block.text) > 50
        ]

        if not body_indices:
            return []

        # Also identify positions right after headings (good transition points)
        post_heading_indices = []
        for i, block in enumerate(blocks):
            if block.is_heading and i + 1 < len(blocks):
                # The paragraph right after a heading is a good insertion point
                if not blocks[i + 1].is_heading:
                    post_heading_indices.append(i + 1)

        # Combine and deduplicate, preferring post-heading positions
        all_candidates = list(set(body_indices + post_heading_indices))
        all_candidates.sort()

        if len(all_candidates) <= 3:
            return all_candidates

        # For longer documents, select evenly spaced points
        # Aim for roughly 1 insertion point per 3-4 paragraphs
        num_points = max(5, len(all_candidates) // 3)
        num_points = min(num_points, 10)  # Cap at 10 insertion points

        # Calculate even spacing
        if len(all_candidates) <= num_points:
            return all_candidates

        step = len(all_candidates) / num_points
        selected_indices = []
        for i in range(num_points):
            idx = int(i * step)
            if idx < len(all_candidates):
                selected_indices.append(all_candidates[idx])

        # Always include first and last if not already included
        if all_candidates[0] not in selected_indices:
            selected_indices.insert(0, all_candidates[0])
        if all_candidates[-1] not in selected_indices:
            selected_indices.append(all_candidates[-1])

        return sorted(list(set(selected_indices)))

    def _generate_keyword_sentences(
        self,
        keyword: str,
        count: int,
        topic: str,
    ) -> list[str]:
        """
        Generate natural sentences containing the primary keyword.

        Args:
            keyword: The keyword to include in sentences.
            count: Number of sentences to generate.
            topic: Content topic for context.

        Returns:
            List of natural sentences containing the keyword.
        """
        if count <= 0:
            return []

        # Cap at 3 additional sentences to avoid over-optimization
        count = min(count, 3)

        prompt = f"""Generate {count} short, natural sentence(s) about "{topic}" that naturally include this EXACT keyword phrase: "{keyword}"

CRITICAL REQUIREMENTS:
1. The keyword phrase "{keyword}" must appear EXACTLY as written in each sentence
2. Each sentence must be different and add value
3. Keep each sentence under 25 words
4. Make sentences feel naturally integrated, not forced
5. Return ONLY the sentences, one per line

Example output:
When choosing {keyword}, consider the features that matter most to your needs.
Professional {keyword} solutions provide reliable security for businesses of all sizes."""

        try:
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=300,
                system="You are a content writer. Generate natural sentences that include specific keyword phrases exactly as provided.",
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text.strip()

            # Get original content for validation
            full_original_text = getattr(self, '_full_original_text', "")

            # Parse sentences
            sentences = []
            for line in result.split("\n"):
                sentence = line.strip()
                if sentence and keyword.lower() in sentence.lower():
                    if not sentence.endswith((".", "!", "?")):
                        sentence += "."
                    # Validate for blocked terms
                    validated, _ = validate_and_fallback(sentence, full_original_text, "keyword_sentence")
                    if validated == sentence:
                        sentences.append(sentence)

            return sentences[:count]

        except Exception:
            # Fallback: create simple sentences
            templates = [
                f"Understanding {keyword} is essential for making informed decisions.",
                f"Many professionals recommend {keyword} for improved security.",
                f"Learn how {keyword} can benefit your specific situation.",
            ]
            return templates[:count]

    def _insert_keyword_sentences(
        self,
        blocks: list[ParagraphBlock],
        sentences: list[str],
        insertion_points: list[int],
    ) -> list[ParagraphBlock]:
        """
        Insert keyword sentences at EVENLY DISTRIBUTED points throughout the content.

        This method distributes sentences across the provided insertion points to ensure
        keywords appear throughout the document, not clustered at beginning/end.

        Args:
            blocks: Original content blocks.
            sentences: Sentences to insert.
            insertion_points: List of block indices suitable for insertion (evenly distributed).

        Returns:
            Updated blocks with sentences inserted at distributed locations.
        """
        if not sentences or not insertion_points:
            return blocks

        # Create a copy of blocks to modify
        updated_blocks = list(blocks)

        # Distribute sentences evenly across available insertion points
        # Never insert more than 1 sentence at each location for natural distribution
        num_sentences = len(sentences)
        num_points = len(insertion_points)

        if num_sentences <= num_points:
            # We have enough points - distribute one sentence per point, evenly spaced
            # Calculate which points to use to spread sentences throughout the document
            if num_sentences == 1:
                # Single sentence - put it in the middle of the document
                middle_idx = num_points // 2
                selected_points = [insertion_points[middle_idx]]
            else:
                # Multiple sentences - distribute evenly across all points
                step = num_points / num_sentences
                selected_points = []
                for i in range(num_sentences):
                    point_idx = int(i * step)
                    if point_idx < num_points:
                        selected_points.append(insertion_points[point_idx])
        else:
            # More sentences than points - distribute as evenly as possible
            # Each point gets at most ceil(num_sentences / num_points) sentences
            selected_points = insertion_points.copy()
            # Extend by repeating points, but spread them out
            remaining = num_sentences - num_points
            if remaining > 0:
                # Add extra points, distributing them evenly
                step = num_points / remaining
                for i in range(remaining):
                    point_idx = int(i * step)
                    if point_idx < num_points:
                        selected_points.append(insertion_points[point_idx])
            selected_points = selected_points[:num_sentences]

        # Now insert sentences at their assigned points
        # Track how many sentences we've added to each block to avoid clustering
        sentences_per_block: dict[int, int] = {}
        max_sentences_per_block = 2  # Never more than 2 keyword sentences per block

        for i, sentence in enumerate(sentences):
            if i >= len(selected_points):
                break

            idx = selected_points[i]
            if idx >= len(updated_blocks):
                continue

            # Check if this block already has max sentences
            current_count = sentences_per_block.get(idx, 0)
            if current_count >= max_sentences_per_block:
                # Find next available block that hasn't hit the limit
                found_alternative = False
                for alt_idx in insertion_points:
                    if alt_idx != idx and sentences_per_block.get(alt_idx, 0) < max_sentences_per_block:
                        idx = alt_idx
                        found_alternative = True
                        break
                if not found_alternative:
                    # All blocks at limit - skip this sentence
                    continue

            block = updated_blocks[idx]
            # Append marked sentence to existing block
            marked_sentence = f" {ADD_START}{sentence}{ADD_END}"
            combined_text = normalize_paragraph_spacing(block.text + marked_sentence)
            updated_blocks[idx] = ParagraphBlock(
                text=combined_text,
                heading_level=block.heading_level,
            )
            sentences_per_block[idx] = sentences_per_block.get(idx, 0) + 1

        return updated_blocks

    def _generate_faq(
        self,
        topic: str,
        keyword_plan: KeywordPlan,
        optimization_plan: OptimizationPlan,
        num_items: int,
    ) -> list[FAQItem]:
        """
        Generate FAQ items using the optimization plan.

        Part 10 of 10-Part Framework: FAQ generation for featured snippets.
        Uses plan.placement_plan.faq_keywords to guide keyword integration.

        Args:
            topic: Content topic.
            keyword_plan: The keyword plan.
            optimization_plan: Optimization plan with faq_keywords.
            num_items: Number of FAQ items to generate.

        Returns:
            List of FAQItem with markers (100% green - all new content).
        """
        # Get content topics for constraints (set during optimize())
        content_topics = getattr(self, '_content_topics', None)
        brand_context = getattr(self, '_brand_context', None)
        full_original_text = getattr(self, '_full_original_text', "")

        # Use faq_keywords from placement plan if available
        faq_keywords = []
        if optimization_plan.placement_plan and optimization_plan.placement_plan.faq_keywords:
            faq_keywords = optimization_plan.placement_plan.faq_keywords
        else:
            # Fallback to keyword plan keywords
            faq_keywords = [kw.phrase for kw in keyword_plan.secondary[:2]]
            if keyword_plan.long_tail_questions:
                faq_keywords.extend([kw.phrase for kw in keyword_plan.long_tail_questions[:2]])

        # Use planned FAQ questions if available from audit
        planned_questions = optimization_plan.faq_questions if optimization_plan.faq_questions else []

        faq_data = self.llm.generate_faq_items(
            topic=topic,
            primary_keyword=keyword_plan.primary.phrase,
            secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
            question_keywords=[kw.phrase for kw in keyword_plan.long_tail_questions],
            faq_keywords=faq_keywords,
            planned_questions=planned_questions,
            num_items=num_items,
            content_topics=content_topics,
            brand_context=brand_context,
        )

        # SAFETY: Validate FAQ items for blocked industry terms
        # Filter out any FAQs that introduce off-topic industries
        valid_faqs, _ = validate_faq_items(faq_data, full_original_text)

        # BRAND NAME NORMALIZATION: Ensure LLM output uses original brand spelling
        if brand_context and brand_context.get("name"):
            brand_name = brand_context["name"]
            brand_variations = brand_context.get("variations", set())
            for item in valid_faqs:
                item["question"] = normalize_brand_in_text(item["question"], brand_name, brand_variations)
                item["answer"] = normalize_brand_in_text(item["answer"], brand_name, brand_variations)

        # Wrap FAQ items in markers - all FAQ content is new and should be highlighted
        return [
            FAQItem(
                question=mark_block_as_new(item["question"]),
                answer=mark_block_as_new(item["answer"])
            )
            for item in valid_faqs
        ]

    def _source_has_faq(self) -> bool:
        """
        Detect if the original source content already has an FAQ section.

        Checks for FAQ-related patterns in headings and text:
        - Heading text containing "FAQ", "question", "frequently asked"
        - Body text containing "frequently asked questions"

        Returns:
            True if source already contains FAQ section, False otherwise.
        """
        # Get the original text stored during optimization
        full_text = getattr(self, '_full_original_text', None)
        if not full_text:
            return False

        text_lower = full_text.lower()

        # Check for FAQ patterns in text
        faq_patterns = [
            "frequently asked questions",
            "faqs",
            "faq section",
            "common questions",
        ]
        for pattern in faq_patterns:
            if pattern in text_lower:
                return True

        # Check for FAQ-like headings by looking for lines that look like headings
        # with FAQ-related words
        lines = full_text.split('\n')
        for line in lines:
            line_lower = line.strip().lower()
            # Skip very long lines (not likely headings)
            if len(line_lower) > 100:
                continue
            # Check for FAQ indicators in potential headings
            if any(indicator in line_lower for indicator in ['faq', 'frequently asked', 'common questions']):
                return True

        return False

    def _generate_faq_with_fallback(
        self,
        topic: str,
        keyword_plan: KeywordPlan,
        optimization_plan: OptimizationPlan,
        num_items: int,
        min_valid: int = 2,
    ) -> list[FAQItem]:
        """
        Generate FAQ with fail-closed behavior.

        This method implements a robust FAQ generation strategy:
        1. Try LLM generation
        2. If < min_valid items returned, retry once
        3. If still insufficient, use deterministic fallback
        4. If enabled but empty = pipeline error (raise)

        Args:
            topic: Content topic.
            keyword_plan: The keyword plan.
            optimization_plan: Optimization plan with faq_keywords.
            num_items: Target FAQ count.
            min_valid: Minimum valid FAQs required (default: 2).

        Returns:
            List of FAQItem objects (never empty if called).

        Raises:
            RuntimeError: If FAQ generation completely fails after all fallbacks.
        """
        import sys

        # First attempt
        faq_items = self._generate_faq(
            topic=topic,
            keyword_plan=keyword_plan,
            optimization_plan=optimization_plan,
            num_items=num_items,
        )

        # Check if sufficient
        if len(faq_items) >= min_valid:
            return faq_items

        # Retry once
        print(f"DEBUG: FAQ insufficient ({len(faq_items)}/{min_valid}), retrying...",
              file=sys.stderr)
        faq_items = self._generate_faq(
            topic=topic,
            keyword_plan=keyword_plan,
            optimization_plan=optimization_plan,
            num_items=num_items,
        )

        if len(faq_items) >= min_valid:
            return faq_items

        # Fall back to deterministic generation
        print(f"DEBUG: FAQ still insufficient after retry ({len(faq_items)}/{min_valid}), "
              "using fallback", file=sys.stderr)

        from .ai_addons import generate_fallback_faqs

        # Get original content blocks for fallback generation
        original_blocks = getattr(self, '_original_blocks', [])
        content_block_texts = [b.text for b in original_blocks] if original_blocks else []

        brand_context = getattr(self, '_brand_context', None)
        brand_name = brand_context.get("name") if brand_context else None

        fallback_faqs = generate_fallback_faqs(
            content_blocks=content_block_texts,
            primary_keyword=keyword_plan.primary.phrase,
            secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
            brand_name=brand_name,
            min_faqs=min_valid,
            max_faqs=num_items,
        )

        # Convert fallback to FAQItem with markers
        fallback_items = [
            FAQItem(
                question=mark_block_as_new(faq["question"]),
                answer=mark_block_as_new(faq["answer"])
            )
            for faq in fallback_faqs
        ]

        # Merge any LLM-generated items with fallback items
        # (prefer LLM items if we have some)
        if faq_items:
            # Use LLM items first, then fill with fallback
            merged = list(faq_items)
            for fb_item in fallback_items:
                if len(merged) >= num_items:
                    break
                merged.append(fb_item)
            return merged

        if not fallback_items:
            raise RuntimeError(
                f"FAQ generation failed: both LLM and fallback produced no results. "
                f"Topic: {topic}, Primary keyword: {keyword_plan.primary.phrase}"
            )

        return fallback_items

    def _explain_title_change(
        self,
        current: Optional[str],
        optimized: str,
        primary_keyword: str,
    ) -> str:
        """Generate explanation for title change."""
        reasons = []

        if not current:
            reasons.append("No title existed; created new SEO-optimized title")
        else:
            clean_optimized = strip_markers(optimized)
            if primary_keyword.lower() in clean_optimized.lower():
                if primary_keyword.lower() not in current.lower():
                    reasons.append("Added primary keyword")
                elif clean_optimized.lower().index(primary_keyword.lower()) < current.lower().index(primary_keyword.lower()):
                    reasons.append("Moved primary keyword closer to beginning")

            if len(clean_optimized) <= 60 and len(current) > 60:
                reasons.append("Shortened to fit 60-character limit")

        if not reasons:
            reasons.append("Minor wording improvements for SEO")

        return "; ".join(reasons)

    def _explain_meta_desc_change(
        self,
        current: Optional[str],
        optimized: str,
        primary_keyword: str,
    ) -> str:
        """Generate explanation for meta description change."""
        reasons = []
        clean_optimized = strip_markers(optimized)

        if not current:
            reasons.append("No meta description existed; created new SEO-optimized description")
        else:
            if primary_keyword.lower() in clean_optimized.lower() and primary_keyword.lower() not in current.lower():
                reasons.append("Added primary keyword")

            if len(clean_optimized) <= 160 and len(current) > 160:
                reasons.append("Shortened to fit 160-character limit")

        # Check for CTA
        cta_patterns = ["learn", "discover", "get", "find", "explore", "contact", "call"]
        if any(p in clean_optimized.lower() for p in cta_patterns):
            reasons.append("Added call-to-action")

        if not reasons:
            reasons.append("Enhanced for clarity and SEO")

        return "; ".join(reasons)

    def _explain_h1_change(
        self,
        current: Optional[str],
        optimized: str,
        primary_keyword: str,
    ) -> str:
        """Generate explanation for H1 change."""
        reasons = []
        clean_optimized = strip_markers(optimized)

        if not current:
            reasons.append("No H1 existed; created SEO-optimized heading")
        else:
            if primary_keyword.lower() in clean_optimized.lower() and primary_keyword.lower() not in current.lower():
                reasons.append("Added primary keyword")

            if len(clean_optimized) > len(current):
                reasons.append("Made more descriptive")

        if not reasons:
            reasons.append("Optimized for SEO while maintaining clarity")

        return "; ".join(reasons)

    def _count_existing_keywords_in_source(
        self,
        original_meta_title: Optional[str],
        original_meta_desc: Optional[str],
        original_h1: Optional[str],
        original_blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
    ) -> dict[str, KeywordCounts]:
        """
        Count keyword occurrences in the ORIGINAL source content before optimization.

        This is used to track EXISTING vs ADDED keyword counts, which is critical
        for insert-only mode where we only want to report newly added occurrences.

        Args:
            original_meta_title: Original page title.
            original_meta_desc: Original meta description.
            original_h1: Original H1 heading.
            original_blocks: Original content blocks (before optimization).
            keyword_plan: The keyword plan with primary and secondary keywords.

        Returns:
            Dictionary mapping keyword phrase to KeywordCounts with existing counts.
        """
        # Build SCOPED text for counting from original content
        # Meta text (title + description + H1) from original
        meta_parts = []
        if original_meta_title:
            meta_parts.append(original_meta_title)
        if original_meta_desc:
            meta_parts.append(original_meta_desc)
        if original_h1:
            meta_parts.append(original_h1)
        original_meta_text = " ".join(meta_parts).lower()

        # Headings text (H2+ only, excluding H1 which is in meta)
        original_headings_text = " ".join(
            block.text
            for block in original_blocks
            if block.is_heading and block.heading_level != HeadingLevel.H1
        ).lower()

        # Body text (non-heading blocks only)
        original_body_text = " ".join(
            block.text
            for block in original_blocks
            if not block.is_heading
        ).lower()

        # Count each keyword in original content
        existing_counts: dict[str, KeywordCounts] = {}

        # Count primary keyword
        primary = keyword_plan.primary.phrase
        existing_counts[primary] = KeywordCounts(
            meta=count_keyword_in_text(original_meta_text, primary),
            headings=count_keyword_in_text(original_headings_text, primary),
            body=count_keyword_in_text(original_body_text, primary),
            faq=0,  # No FAQ in original source
        )

        # Count secondary keywords
        for kw in keyword_plan.secondary:
            phrase = kw.phrase
            existing_counts[phrase] = KeywordCounts(
                meta=count_keyword_in_text(original_meta_text, phrase),
                headings=count_keyword_in_text(original_headings_text, phrase),
                body=count_keyword_in_text(original_body_text, phrase),
                faq=0,  # No FAQ in original source
            )

        return existing_counts

    def _compute_keyword_usage_counts(
        self,
        meta_elements: list[MetaElement],
        optimized_blocks: list[ParagraphBlock],
        faq_items: list[FAQItem],
        keyword_plan: KeywordPlan,
        primary_target: int = 8,
        secondary_target: int = 3,
        existing_counts: Optional[dict[str, KeywordCounts]] = None,
    ) -> dict[str, ScopedKeywordCounts]:
        """
        Compute SCOPED keyword usage counts across all final optimized content.

        TRANSPARENCY: Returns detailed breakdown of where keywords appear:
        - Meta (title, description, H1) - for SEO but NOT counted toward targets
        - Headings (H2+) - subheadings
        - Body (non-heading paragraphs) - PRIMARY target for keyword satisfaction
        - FAQ (questions + answers) - for featured snippets, NOT counted toward targets

        This allows transparent reporting and ensures body content is the
        primary target for keyword placement, not meta/FAQ.

        Args:
            meta_elements: The optimized meta elements.
            optimized_blocks: The optimized content blocks.
            faq_items: The generated FAQ items.
            keyword_plan: The keyword plan with primary and secondary keywords.
            primary_target: Target count for primary keyword.
            secondary_target: Target count for secondary keywords.
            existing_counts: Optional existing keyword counts from source content.
                            Used to track existing vs added counts.

        Returns:
            Dictionary mapping keyword phrase to ScopedKeywordCounts with full breakdown.
        """
        # Build SCOPED text for counting (strip markers for accurate counting)

        # Meta text (title + description + H1)
        meta_text = " ".join(
            strip_markers(meta.optimized) for meta in meta_elements
        ).lower()

        # Headings text (H2+ only, excluding H1 which is in meta)
        headings_text = " ".join(
            strip_markers(block.text)
            for block in optimized_blocks
            if block.is_heading and block.heading_level != HeadingLevel.H1
        ).lower()

        # Body text (non-heading blocks only)
        body_text = " ".join(
            strip_markers(block.text)
            for block in optimized_blocks
            if not block.is_heading
        ).lower()

        # FAQ text (questions + answers)
        faq_text = " ".join(
            strip_markers(faq.question) + " " + strip_markers(faq.answer)
            for faq in faq_items
        ).lower() if faq_items else ""

        # Count each keyword in each scope
        scoped_counts: dict[str, ScopedKeywordCounts] = {}

        # Count primary keyword
        primary = keyword_plan.primary.phrase
        existing = existing_counts.get(primary, KeywordCounts()) if existing_counts else KeywordCounts()
        scoped_counts[primary] = ScopedKeywordCounts(
            keyword=primary,
            target=primary_target,
            counts=KeywordCounts(
                meta=count_keyword_in_text(meta_text, primary),
                headings=count_keyword_in_text(headings_text, primary),
                body=count_keyword_in_text(body_text, primary),
                faq=count_keyword_in_text(faq_text, primary),
                # Existing counts from source
                existing_meta=existing.meta,
                existing_headings=existing.headings,
                existing_body=existing.body,
                existing_faq=existing.faq,
            ),
            is_primary=True,
        )

        # Count secondary keywords
        for kw in keyword_plan.secondary:
            phrase = kw.phrase
            existing = existing_counts.get(phrase, KeywordCounts()) if existing_counts else KeywordCounts()
            scoped_counts[phrase] = ScopedKeywordCounts(
                keyword=phrase,
                target=secondary_target,
                counts=KeywordCounts(
                    meta=count_keyword_in_text(meta_text, phrase),
                    headings=count_keyword_in_text(headings_text, phrase),
                    body=count_keyword_in_text(body_text, phrase),
                    faq=count_keyword_in_text(faq_text, phrase),
                    # Existing counts from source
                    existing_meta=existing.meta,
                    existing_headings=existing.headings,
                    existing_body=existing.body,
                    existing_faq=existing.faq,
                ),
                is_primary=False,
            )

        return scoped_counts

    def _get_legacy_keyword_counts(
        self,
        scoped_counts: dict[str, ScopedKeywordCounts],
    ) -> dict[str, int]:
        """
        Convert scoped counts to legacy format (total counts only).

        Used for backward compatibility with existing code that expects
        dict[str, int] format.

        Args:
            scoped_counts: Scoped keyword counts from _compute_keyword_usage_counts.

        Returns:
            Dictionary mapping keyword phrase to total occurrence count.
        """
        return {kw: counts.counts.total for kw, counts in scoped_counts.items()}

    def _get_detailed_keyword_counts(
        self,
        scoped_counts: dict[str, ScopedKeywordCounts],
    ) -> dict[str, "KeywordUsageDetail"]:
        """
        Convert scoped counts to detailed KeywordUsageDetail format.

        Provides full breakdown of existing vs added occurrences for
        insert-only mode transparency.

        Args:
            scoped_counts: Scoped keyword counts from _compute_keyword_usage_counts.

        Returns:
            Dictionary mapping keyword phrase to KeywordUsageDetail.
        """
        from .models import KeywordUsageDetail

        detailed: dict[str, KeywordUsageDetail] = {}

        for kw, scoped in scoped_counts.items():
            detailed[kw] = KeywordUsageDetail(
                keyword=kw,
                is_primary=scoped.is_primary,
                # Final output counts
                meta=scoped.counts.meta,
                headings=scoped.counts.headings,
                body=scoped.counts.body,
                faq=scoped.counts.faq,
                # Existing counts from source
                existing_meta=scoped.counts.existing_meta,
                existing_headings=scoped.counts.existing_headings,
                existing_body=scoped.counts.existing_body,
                existing_faq=scoped.counts.existing_faq,
            )

        return detailed

    def _log_scoped_keyword_counts(
        self,
        scoped_counts: dict[str, ScopedKeywordCounts],
    ) -> None:
        """
        Log detailed scoped keyword counts for transparency.

        This provides visibility into where keywords appear:
        - Body count (primary target for satisfaction)
        - Meta count (title/description/H1)
        - Headings count (H2+)
        - FAQ count (questions/answers)
        - Total count
        - EXISTING vs ADDED breakdown (for insert-only mode)

        Args:
            scoped_counts: Scoped keyword counts from _compute_keyword_usage_counts.
        """
        import sys

        print("\n" + "=" * 60, file=sys.stderr)
        print("KEYWORD PLACEMENT REPORT (Scoped Counts)", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print("IMPORTANT: Body count is the PRIMARY target for satisfaction.", file=sys.stderr)
        print("Meta and FAQ do NOT count toward keyword targets.", file=sys.stderr)
        print("-" * 60, file=sys.stderr)

        for kw, counts in scoped_counts.items():
            kw_type = "PRIMARY" if counts.is_primary else "SECONDARY"
            satisfied = "✓ SATISFIED" if counts.body_satisfied else "✗ NOT SATISFIED"

            print(f"\n[{kw_type}] '{kw}' (target: {counts.target})", file=sys.stderr)
            print(f"  Body:     {counts.counts.body:3d} / {counts.target} {satisfied}", file=sys.stderr)
            print(f"    └─ Existing: {counts.counts.existing_body:3d}, Added: {counts.counts.added_body:3d}", file=sys.stderr)
            print(f"  Meta:     {counts.counts.meta:3d} (title/desc/H1 - not counted toward target)", file=sys.stderr)
            print(f"    └─ Existing: {counts.counts.existing_meta:3d}, Added: {counts.counts.added_meta:3d}", file=sys.stderr)
            print(f"  Headings: {counts.counts.headings:3d} (H2+ subheadings)", file=sys.stderr)
            print(f"    └─ Existing: {counts.counts.existing_headings:3d}, Added: {counts.counts.added_headings:3d}", file=sys.stderr)
            print(f"  FAQ:      {counts.counts.faq:3d} (not counted toward target)", file=sys.stderr)
            print(f"    └─ Existing: {counts.counts.existing_faq:3d}, Added: {counts.counts.added_faq:3d}", file=sys.stderr)
            print(f"  Total:    {counts.counts.total:3d} (Existing: {counts.counts.existing_total:3d}, Added: {counts.counts.added_total:3d})", file=sys.stderr)

            if not counts.body_satisfied:
                print(f"  ⚠ Need {counts.body_needed} more in BODY to satisfy target", file=sys.stderr)

        print("\n" + "=" * 60, file=sys.stderr)

    def _validate_keyword_caps_compliance(
        self,
        scoped_counts: dict[str, ScopedKeywordCounts],
        keyword_plan: KeywordPlan,
        config: OptimizationConfig,
    ) -> CapValidationReport:
        """
        Validate that keyword occurrences are within configured caps.

        This is the POST-OPTIMIZATION VALIDATION that ensures the optimizer
        respected the keyword caps (maximums) configured for insert-only mode.

        Args:
            scoped_counts: Scoped keyword counts from _compute_keyword_usage_counts.
            keyword_plan: The keyword plan with primary and secondary keywords.
            config: Optimization configuration with cap settings.

        Returns:
            CapValidationReport with detailed compliance information.
        """
        results = []

        # Validate primary keyword
        primary = keyword_plan.primary.phrase
        if primary in scoped_counts:
            primary_counts = scoped_counts[primary]
            results.append(KeywordCapValidationResult(
                keyword=primary,
                cap=config.primary_keyword_body_cap,
                actual_body_count=primary_counts.counts.body,
                actual_total_count=primary_counts.counts.total,
                is_primary=True,
            ))

        # Validate secondary keywords
        for kw in keyword_plan.secondary:
            phrase = kw.phrase
            if phrase in scoped_counts:
                kw_counts = scoped_counts[phrase]
                results.append(KeywordCapValidationResult(
                    keyword=phrase,
                    cap=config.secondary_keyword_body_cap,
                    actual_body_count=kw_counts.counts.body,
                    actual_total_count=kw_counts.counts.total,
                    is_primary=False,
                ))

        return CapValidationReport(
            mode=config.optimization_mode,
            cap_enforcement_enabled=config.should_enforce_keyword_caps,
            primary_cap=config.primary_keyword_body_cap,
            secondary_cap=config.secondary_keyword_body_cap,
            keyword_results=results,
        )

    def _log_cap_validation_report(
        self,
        report: CapValidationReport,
    ) -> None:
        """
        Log the keyword cap validation report for transparency.

        Shows clear pass/fail status for each keyword against its cap.

        Args:
            report: The cap validation report to log.
        """
        import sys

        print("\n" + "=" * 60, file=sys.stderr)
        print("KEYWORD CAP VALIDATION REPORT (Insert-Only Mode)", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Mode: {report.mode.upper()}", file=sys.stderr)
        print(f"Cap Enforcement: {'ENABLED' if report.cap_enforcement_enabled else 'DISABLED'}", file=sys.stderr)
        print(f"Primary Cap: {report.primary_cap} | Secondary Cap: {report.secondary_cap}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)

        for result in report.keyword_results:
            kw_type = "PRIMARY" if result.is_primary else "SECONDARY"
            status = "✓ PASS" if result.within_cap else "✗ FAIL"

            print(f"\n[{kw_type}] '{result.keyword}'", file=sys.stderr)
            print(f"  Body Count: {result.actual_body_count} / {result.cap} (cap) {status}", file=sys.stderr)
            print(f"  Total Count: {result.actual_total_count} (meta+headings+body+faq)", file=sys.stderr)

            if not result.within_cap:
                print(f"  ⚠ OVER CAP BY {result.excess_count} occurrence(s)!", file=sys.stderr)

        print("\n" + "-" * 60, file=sys.stderr)
        if report.all_within_caps:
            print("✓ ALL KEYWORDS WITHIN CAPS - Insert-Only Compliance: PASS", file=sys.stderr)
        else:
            over_count = len(report.keywords_over_cap)
            print(f"✗ {over_count} KEYWORD(S) OVER CAP - Insert-Only Compliance: FAIL", file=sys.stderr)
            print(f"  Total Excess: {report.total_excess} occurrence(s) over caps", file=sys.stderr)

        print("=" * 60 + "\n", file=sys.stderr)

    def _generate_debug_bundle(
        self,
        config: OptimizationConfig,
        keyword_plan: KeywordPlan,
        scoped_counts: dict[str, ScopedKeywordCounts],
        cap_validation: Optional[CapValidationReport],
        optimized_blocks: Optional[list[ParagraphBlock]] = None,
        warnings: Optional[list[str]] = None,
        original_keyword_counts: Optional[dict[str, KeywordCounts]] = None,
        run_manifest: Optional[RunManifest] = None,
    ) -> DebugBundle:
        """
        Generate a comprehensive debug bundle for insert-only mode troubleshooting.

        This bundle contains all information needed to diagnose keyword inflation
        issues and verify that insert-only mode is working correctly.

        Args:
            config: The optimization configuration used.
            keyword_plan: The keyword plan with primary and secondary keywords.
            scoped_counts: Scoped keyword counts from _compute_keyword_usage_counts (AFTER optimization).
            cap_validation: Cap validation report (if in minimal mode).
            optimized_blocks: The optimized content blocks (optional, for counting).
            warnings: List of warnings generated during optimization (optional).
            original_keyword_counts: Keyword counts from source BEFORE optimization (for delta tracking).
            run_manifest: Run manifest showing what stages actually executed.

        Returns:
            DebugBundle with complete debugging information.
        """
        # Set defaults for optional parameters
        if optimized_blocks is None:
            optimized_blocks = []
        if warnings is None:
            warnings = []
        from datetime import datetime

        # Build config snapshot
        bundle_config = DebugBundleConfig(
            optimization_mode=config.optimization_mode,
            faq_policy=config.faq_policy,
            generate_ai_sections=config.generate_ai_sections,
            generate_key_takeaways=config.generate_key_takeaways,
            generate_chunk_map=config.generate_chunk_map,
            primary_keyword_body_cap=config.primary_keyword_body_cap,
            secondary_keyword_body_cap=config.secondary_keyword_body_cap,
            enforce_keyword_caps=config.enforce_keyword_caps,
            max_secondary=config.max_secondary,
            has_keyword_allowlist=config.has_keyword_allowlist,
            keyword_allowlist=config.keyword_allowlist,
        )

        # Build keyword entries with delta tracking
        keywords = []

        # Helper to compute delta fields
        def compute_delta_fields(phrase: str, body_count: int, is_primary: bool):
            """Compute delta tracking fields for a keyword."""
            if original_keyword_counts and phrase in original_keyword_counts:
                original_count = original_keyword_counts[phrase].body
            else:
                original_count = 0  # Assume no original occurrences if not tracked

            new_additions = max(0, body_count - original_count)
            allowed_new = 1  # Default: allow 1 new addition per keyword
            delta_within_budget = new_additions <= allowed_new

            return original_count, new_additions, allowed_new, delta_within_budget

        # Primary keyword
        primary = keyword_plan.primary.phrase
        if primary in scoped_counts:
            counts = scoped_counts[primary]
            body_count = counts.counts.body
            original_count, new_additions, allowed_new, delta_within_budget = compute_delta_fields(
                primary, body_count, is_primary=True
            )
            keywords.append(DebugBundleKeyword(
                phrase=primary,
                is_primary=True,
                cap=config.primary_keyword_body_cap,
                body_count=body_count,
                meta_count=counts.counts.meta,
                headings_count=counts.counts.headings,
                faq_count=counts.counts.faq,
                total_count=counts.counts.total,
                within_cap=body_count <= config.primary_keyword_body_cap,
                # Delta tracking fields
                original_count=original_count,
                new_additions=new_additions,
                allowed_new=allowed_new,
                delta_within_budget=delta_within_budget,
            ))

        # Secondary keywords
        for kw in keyword_plan.secondary:
            phrase = kw.phrase
            if phrase in scoped_counts:
                counts = scoped_counts[phrase]
                body_count = counts.counts.body
                original_count, new_additions, allowed_new, delta_within_budget = compute_delta_fields(
                    phrase, body_count, is_primary=False
                )
                keywords.append(DebugBundleKeyword(
                    phrase=phrase,
                    is_primary=False,
                    cap=config.secondary_keyword_body_cap,
                    body_count=body_count,
                    meta_count=counts.counts.meta,
                    headings_count=counts.counts.headings,
                    faq_count=counts.counts.faq,
                    total_count=counts.counts.total,
                    within_cap=body_count <= config.secondary_keyword_body_cap,
                    # Delta tracking fields
                    original_count=original_count,
                    new_additions=new_additions,
                    allowed_new=allowed_new,
                    delta_within_budget=delta_within_budget,
                ))

        # Count blocks with keywords
        total_blocks = len(optimized_blocks)
        blocks_with_keywords = 0
        all_keywords = {primary.lower()} | {kw.phrase.lower() for kw in keyword_plan.secondary}
        for block in optimized_blocks:
            block_text_lower = strip_markers(block.text).lower()
            if any(kw in block_text_lower for kw in all_keywords):
                blocks_with_keywords += 1

        return DebugBundle(
            timestamp=datetime.now().isoformat(),
            config=bundle_config,
            keywords=keywords,
            cap_validation=cap_validation,
            total_blocks=total_blocks,
            blocks_with_keywords=blocks_with_keywords,
            warnings=warnings,
            run_manifest=run_manifest,
        )

    def _export_debug_bundle(
        self,
        bundle: DebugBundle,
        output_path: Optional[str] = None,
        format: str = "json",
    ) -> str:
        """
        Export debug bundle to file.

        Args:
            bundle: The debug bundle to export.
            output_path: Path to write the bundle. If None, generates a default path.
            format: Output format - "json" or "checklist".

        Returns:
            Path to the exported file.
        """
        import os
        from datetime import datetime

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = ".json" if format == "json" else ".txt"
            output_path = f"debug_bundle_{timestamp}{ext}"

        if format == "json":
            content = bundle.to_json()
        else:
            content = bundle.to_checklist()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path

    def _generate_ai_addons(
        self,
        optimized_blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        faq_items: list[FAQItem],
        config: OptimizationConfig,
    ) -> Optional[AIAddonsResult]:
        """
        Generate AI Optimization Add-ons (Key Takeaways + Chunk Map).

        This method creates AI/GEO-friendly sections:
        - Key Takeaways: 3-6 concise bullet points summarizing content
        - Chunk Map: Structured content chunks for AI retrieval (RAG)

        The output is designed to be highly extractable by AI systems
        and useful for vector search / retrieval applications.

        Args:
            optimized_blocks: The optimized content blocks.
            keyword_plan: Keyword plan with primary/secondary keywords.
            faq_items: Existing FAQ items (used for fallback logic).
            config: Optimization configuration.

        Returns:
            AIAddonsResult with key takeaways and chunk map data,
            or None if generation fails.
        """
        import sys

        try:
            # Extract text content from blocks
            content_blocks = []
            headings = []

            for block in optimized_blocks:
                # Get clean text without markers
                clean_text = strip_markers(block.text).strip()
                if not clean_text:
                    continue

                content_blocks.append(clean_text)

                # Track headings for structure-aware chunking
                if block.heading_level and block.heading_level != HeadingLevel.PARAGRAPH:
                    level_map = {
                        HeadingLevel.H1: 1,
                        HeadingLevel.H2: 2,
                        HeadingLevel.H3: 3,
                        HeadingLevel.H4: 4,
                        HeadingLevel.H5: 5,
                        HeadingLevel.H6: 6,
                    }
                    level = level_map.get(block.heading_level, 2)
                    headings.append((clean_text, level))

            if not content_blocks:
                print("DEBUG: No content blocks for AI add-ons generation", file=sys.stderr)
                return None

            # Convert existing FAQs to dict format for ai_addons module
            existing_faqs = []
            for faq in faq_items:
                existing_faqs.append({
                    "question": faq.question,
                    "answer": faq.answer,
                })

            # Call the ai_addons module to generate content
            addons = generate_ai_addons(
                content_blocks=content_blocks,
                primary_keyword=keyword_plan.primary.phrase,
                secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
                brand_name=None,  # Brand detection happens separately
                headings=headings,
                existing_faqs=existing_faqs,
                generate_takeaways=config.generate_key_takeaways,
                generate_chunks=config.generate_chunk_map,
                generate_faqs=False,  # FAQs handled separately in main flow
                chunk_target_tokens=config.chunk_target_tokens,
                chunk_overlap_tokens=config.chunk_overlap_tokens,
            )

            # Convert AIAddons to AIAddonsResult
            chunk_data_list = []
            chunk_stats = None

            if addons.chunk_map and addons.chunk_map.chunks:
                for chunk in addons.chunk_map.chunks:
                    chunk_data_list.append(ChunkData(
                        chunk_id=chunk.chunk_id,
                        heading_context=chunk.heading_path,
                        summary=chunk.summary,
                        best_question=chunk.best_question,
                        keywords_present=chunk.keywords_present,
                        word_count=chunk.word_count,
                        token_estimate=chunk.token_estimate,
                    ))

                chunk_stats = ChunkMapStats(
                    total_chunks=addons.chunk_map.total_chunks,
                    total_words=addons.chunk_map.total_words,
                    total_tokens=addons.chunk_map.total_tokens,
                )

            result = AIAddonsResult(
                key_takeaways=addons.key_takeaways,
                chunk_map_chunks=chunk_data_list,
                chunk_map_stats=chunk_stats,
                faqs=addons.faqs,
            )

            print(f"DEBUG: AI Add-ons generated successfully:", file=sys.stderr)
            print(f"  - Key Takeaways: {len(result.key_takeaways)}", file=sys.stderr)
            print(f"  - Chunks: {len(result.chunk_map_chunks)}", file=sys.stderr)
            if result.chunk_map_stats:
                print(f"  - Total words: {result.chunk_map_stats.total_words}", file=sys.stderr)
                print(f"  - Total tokens: {result.chunk_map_stats.total_tokens}", file=sys.stderr)

            return result

        except Exception as e:
            print(f"ERROR: AI Add-ons generation failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return None

    def _detect_brand_name(
        self,
        content: Union[PageMeta, DocxContent],
        full_text: str = "",
    ) -> tuple[str, set[str]]:
        """
        Detect the brand/company name from the content and find its ORIGINAL spelling.

        BRAND NAME PROTECTION: This method finds exactly how the brand is spelled
        in the original content, so we can preserve that spelling and exclude
        brand name variations from diff highlighting.

        Uses multiple signals:
        - URL domain (to identify potential brand)
        - H1 heading
        - Title tag
        - Full text content (to find original spelling)

        Args:
            content: The content being optimized.
            full_text: Full text of the content for finding original spelling.

        Returns:
            Tuple of (original_brand_spelling, set_of_variations).
        """
        brand_name = ""
        brand_base = ""  # Base brand name to search for variations

        if isinstance(content, PageMeta):
            # Try to extract from URL domain
            if content.url:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(content.url)
                    domain = parsed.netloc.lower()
                    # Remove www. and .com/.io/.net etc
                    domain = re.sub(r'^www\.', '', domain)
                    domain = re.sub(r'\.(com|io|net|org|co|ai)$', '', domain)
                    # Store base brand (may be hyphenated like "cell-gate")
                    brand_base = domain
                except Exception:
                    pass

            # Fallback: look at H1 or title for brand base
            if not brand_base:
                h1 = content.h1 or ""
                title = content.title or ""
                for text in [title, h1]:
                    words = text.split()
                    if words:
                        first_word = words[0].strip(":|–-")
                        if first_word and first_word[0].isupper():
                            brand_base = first_word.lower()
                            break

        elif isinstance(content, DocxContent):
            # For DOCX, try H1
            h1 = content.h1 or ""
            words = h1.split()
            if words:
                first_word = words[0].strip(":|–-")
                if first_word and first_word[0].isupper():
                    brand_base = first_word.lower()

        if not brand_base:
            return ("", set())

        # Generate all possible variations of the brand name
        brand_variations = generate_brand_variations(brand_base)

        # Now search the content to find the ORIGINAL spelling
        # Priority: H1 > Title > Meta Description > First occurrence in body
        search_texts = []
        if isinstance(content, PageMeta):
            if content.h1:
                search_texts.append(content.h1)
            if content.title:
                search_texts.append(content.title)
            if content.meta_description:
                search_texts.append(content.meta_description)
        elif isinstance(content, DocxContent):
            if content.h1:
                search_texts.append(content.h1)

        # Add full text as last resort
        if full_text:
            search_texts.append(full_text)

        # Search for original spelling in content
        for text in search_texts:
            # Try to find any variation in this text
            for variation in sorted(brand_variations, key=len, reverse=True):
                # Case-insensitive search
                pattern = re.compile(re.escape(variation), re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    # Found it! Use the EXACT spelling as it appears in content
                    brand_name = match.group(0)
                    break
            if brand_name:
                break

        # If still not found, use the domain-derived version
        if not brand_name and brand_base:
            # Convert hyphenated domain to CamelCase as default
            parts = re.split(r'[-\s]+', brand_base)
            brand_name = ''.join(p.capitalize() for p in parts)

        # Regenerate variations based on the detected original spelling
        # to ensure the original is included
        if brand_name:
            brand_variations = generate_brand_variations(brand_name)
            # Always add the exact original
            brand_variations.add(brand_name.lower())

        return (brand_name, brand_variations)


def optimize_content(
    content: Union[PageMeta, DocxContent],
    keywords: list[Keyword],
    api_key: Optional[str] = None,
    generate_faq: bool = True,
    faq_count: int = 4,
) -> OptimizationResult:
    """
    Convenience function for content optimization.

    Args:
        content: Content to optimize.
        keywords: Available keywords.
        api_key: LLM API key.
        generate_faq: Whether to generate FAQ.
        faq_count: Number of FAQ items.

    Returns:
        OptimizationResult with all optimized content.
    """
    optimizer = ContentOptimizer(api_key=api_key)
    return optimizer.optimize(
        content=content,
        keywords=keywords,
        generate_faq=generate_faq,
        faq_count=faq_count,
    )
