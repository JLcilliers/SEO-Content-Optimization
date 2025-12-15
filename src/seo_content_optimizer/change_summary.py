"""
Change summary reporting for SEO Content Optimizer V2 Architecture.

This module generates comprehensive reports of all changes made during optimization:
- Block-by-block change tracking
- Keyword injection summary
- Factuality check results
- Structure preservation audit
- Overall optimization metrics

The goal is to provide transparency into what changes were made and why.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from enum import Enum

from .models import ContentBlock, ContentBlockType, SemanticKeywordPlan
from .diff_highlighter import DiffResult, compute_diff
from .factuality_guardrails import FactualityCheckResult

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes made."""
    KEYWORD_INJECTION = "keyword_injection"
    REPHRASING = "rephrasing"
    STRUCTURE_PRESERVED = "structure_preserved"
    NO_CHANGE = "no_change"
    BLOCKED = "blocked"


@dataclass
class BlockChange:
    """Record of changes to a single block."""
    block_id: str
    block_type: ContentBlockType
    original_text: str
    modified_text: str
    change_type: ChangeType
    keywords_injected: list[str] = field(default_factory=list)
    diff_result: Optional[DiffResult] = None
    factuality_valid: bool = True
    structure_valid: bool = True
    warnings: list[str] = field(default_factory=list)
    change_percent: float = 0.0


@dataclass
class OptimizationSummary:
    """Complete summary of optimization changes."""
    # Document info
    document_title: str
    document_url: Optional[str]
    optimization_timestamp: datetime

    # Block changes
    block_changes: list[BlockChange]
    total_blocks: int
    modified_blocks: int
    preserved_blocks: int
    blocked_blocks: int

    # Keyword stats
    keyword_plan: Optional[SemanticKeywordPlan]
    total_keyword_injections: int
    unique_keywords_injected: set[str]
    primary_keyword_count: int
    secondary_keyword_count: int

    # Validation stats
    factuality_issues: int
    structure_issues: int
    total_warnings: list[str]

    # Metrics
    original_word_count: int
    modified_word_count: int
    change_ratio: float  # Percentage of content changed


@dataclass
class SummaryReportConfig:
    """Configuration for summary report generation."""
    include_diff_details: bool = True
    include_unchanged_blocks: bool = False
    max_warnings_per_block: int = 5
    include_keyword_positions: bool = True


class ChangeSummaryBuilder:
    """
    Builds comprehensive change summaries for optimization runs.

    Tracks all changes made during optimization and generates reports.
    """

    def __init__(self, config: Optional[SummaryReportConfig] = None):
        """
        Initialize summary builder.

        Args:
            config: Report configuration.
        """
        self.config = config or SummaryReportConfig()
        self._block_changes: list[BlockChange] = []
        self._warnings: list[str] = []
        self._keyword_counts: dict[str, int] = {}
        self._document_title = ""
        self._document_url: Optional[str] = None
        self._keyword_plan: Optional[SemanticKeywordPlan] = None

    def set_document_info(
        self,
        title: str,
        url: Optional[str] = None,
    ) -> None:
        """Set document information."""
        self._document_title = title
        self._document_url = url

    def set_keyword_plan(self, plan: SemanticKeywordPlan) -> None:
        """Set the keyword plan used for optimization."""
        self._keyword_plan = plan

    def record_block_change(
        self,
        block: ContentBlock,
        modified_text: str,
        keywords_injected: Optional[list[str]] = None,
        factuality_result: Optional[FactualityCheckResult] = None,
        structure_valid: bool = True,
        structure_warnings: Optional[list[str]] = None,
    ) -> BlockChange:
        """
        Record a change to a block.

        Args:
            block: Original ContentBlock.
            modified_text: Modified text.
            keywords_injected: Keywords that were injected.
            factuality_result: Result of factuality check.
            structure_valid: Whether structure was preserved.
            structure_warnings: Structure-related warnings.

        Returns:
            BlockChange record.
        """
        keywords = keywords_injected or []

        # Compute diff
        diff_result = None
        if self.config.include_diff_details:
            diff_result = compute_diff(
                block.text,
                modified_text,
                keywords,
            )

        # Determine change type
        if block.text.strip() == modified_text.strip():
            change_type = ChangeType.NO_CHANGE
        elif keywords:
            change_type = ChangeType.KEYWORD_INJECTION
        elif structure_valid and block.block_type in (
            ContentBlockType.TABLE,
            ContentBlockType.LIST,
            ContentBlockType.CODE,
        ):
            change_type = ChangeType.STRUCTURE_PRESERVED
        else:
            change_type = ChangeType.REPHRASING

        # Calculate change percentage
        original_words = len(block.text.split())
        modified_words = len(modified_text.split())
        if original_words > 0:
            word_diff = abs(modified_words - original_words)
            change_percent = (word_diff / original_words) * 100
        else:
            change_percent = 100.0 if modified_words > 0 else 0.0

        # Collect warnings
        warnings = []
        if factuality_result and not factuality_result.is_valid:
            warnings.extend(factuality_result.warnings[:self.config.max_warnings_per_block])
        if structure_warnings:
            warnings.extend(structure_warnings[:self.config.max_warnings_per_block])

        # Create change record
        change = BlockChange(
            block_id=block.block_id,
            block_type=block.block_type,
            original_text=block.text,
            modified_text=modified_text,
            change_type=change_type,
            keywords_injected=keywords,
            diff_result=diff_result,
            factuality_valid=factuality_result.is_valid if factuality_result else True,
            structure_valid=structure_valid,
            warnings=warnings,
            change_percent=change_percent,
        )

        self._block_changes.append(change)

        # Track keyword counts
        for kw in keywords:
            self._keyword_counts[kw] = self._keyword_counts.get(kw, 0) + 1

        return change

    def record_blocked_block(
        self,
        block: ContentBlock,
        reason: str,
    ) -> BlockChange:
        """
        Record a block that was blocked from modification.

        Args:
            block: ContentBlock that was blocked.
            reason: Reason for blocking.

        Returns:
            BlockChange record.
        """
        change = BlockChange(
            block_id=block.block_id,
            block_type=block.block_type,
            original_text=block.text,
            modified_text=block.text,  # Unchanged
            change_type=ChangeType.BLOCKED,
            warnings=[reason],
        )
        self._block_changes.append(change)
        return change

    def add_warning(self, warning: str) -> None:
        """Add a global warning."""
        self._warnings.append(warning)

    def build_summary(self) -> OptimizationSummary:
        """
        Build the complete optimization summary.

        Returns:
            OptimizationSummary with all changes and metrics.
        """
        # Count changes
        modified = [c for c in self._block_changes if c.change_type not in (
            ChangeType.NO_CHANGE, ChangeType.BLOCKED
        )]
        preserved = [c for c in self._block_changes if c.change_type == ChangeType.NO_CHANGE]
        blocked = [c for c in self._block_changes if c.change_type == ChangeType.BLOCKED]

        # Count keywords
        total_injections = sum(len(c.keywords_injected) for c in self._block_changes)
        unique_keywords = set()
        for c in self._block_changes:
            unique_keywords.update(c.keywords_injected)

        primary_count = 0
        secondary_count = 0
        if self._keyword_plan:
            primary_kw = self._keyword_plan.primary.phrase.lower()
            primary_count = self._keyword_counts.get(primary_kw, 0)
            for kw in self._keyword_plan.secondary:
                secondary_count += self._keyword_counts.get(kw.phrase.lower(), 0)

        # Count validation issues
        factuality_issues = sum(1 for c in self._block_changes if not c.factuality_valid)
        structure_issues = sum(1 for c in self._block_changes if not c.structure_valid)

        # Calculate word counts
        original_words = sum(len(c.original_text.split()) for c in self._block_changes)
        modified_words = sum(len(c.modified_text.split()) for c in self._block_changes)

        if original_words > 0:
            change_ratio = abs(modified_words - original_words) / original_words * 100
        else:
            change_ratio = 0.0

        # Collect all warnings
        all_warnings = self._warnings.copy()
        for change in self._block_changes:
            all_warnings.extend(change.warnings)

        return OptimizationSummary(
            document_title=self._document_title,
            document_url=self._document_url,
            optimization_timestamp=datetime.now(),
            block_changes=self._block_changes,
            total_blocks=len(self._block_changes),
            modified_blocks=len(modified),
            preserved_blocks=len(preserved),
            blocked_blocks=len(blocked),
            keyword_plan=self._keyword_plan,
            total_keyword_injections=total_injections,
            unique_keywords_injected=unique_keywords,
            primary_keyword_count=primary_count,
            secondary_keyword_count=secondary_count,
            factuality_issues=factuality_issues,
            structure_issues=structure_issues,
            total_warnings=all_warnings,
            original_word_count=original_words,
            modified_word_count=modified_words,
            change_ratio=change_ratio,
        )


def format_summary_text(summary: OptimizationSummary) -> str:
    """
    Format summary as human-readable text.

    Args:
        summary: OptimizationSummary to format.

    Returns:
        Formatted text report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SEO CONTENT OPTIMIZATION SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # Document info
    lines.append(f"Document: {summary.document_title}")
    if summary.document_url:
        lines.append(f"URL: {summary.document_url}")
    lines.append(f"Timestamp: {summary.optimization_timestamp.isoformat()}")
    lines.append("")

    # Block stats
    lines.append("-" * 40)
    lines.append("BLOCK STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total blocks: {summary.total_blocks}")
    lines.append(f"Modified: {summary.modified_blocks}")
    lines.append(f"Preserved: {summary.preserved_blocks}")
    lines.append(f"Blocked: {summary.blocked_blocks}")
    lines.append("")

    # Keyword stats
    lines.append("-" * 40)
    lines.append("KEYWORD STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total injections: {summary.total_keyword_injections}")
    lines.append(f"Unique keywords: {len(summary.unique_keywords_injected)}")
    if summary.keyword_plan:
        lines.append(f"Primary keyword: {summary.keyword_plan.primary.phrase}")
        lines.append(f"  - Injected {summary.primary_keyword_count} times")
    lines.append(f"Secondary keywords injected: {summary.secondary_keyword_count} times")
    lines.append("")

    # Word count stats
    lines.append("-" * 40)
    lines.append("CONTENT METRICS")
    lines.append("-" * 40)
    lines.append(f"Original word count: {summary.original_word_count}")
    lines.append(f"Modified word count: {summary.modified_word_count}")
    lines.append(f"Change ratio: {summary.change_ratio:.1f}%")
    lines.append("")

    # Validation stats
    if summary.factuality_issues or summary.structure_issues:
        lines.append("-" * 40)
        lines.append("VALIDATION ISSUES")
        lines.append("-" * 40)
        lines.append(f"Factuality issues: {summary.factuality_issues}")
        lines.append(f"Structure issues: {summary.structure_issues}")
        lines.append("")

    # Warnings
    if summary.total_warnings:
        lines.append("-" * 40)
        lines.append("WARNINGS")
        lines.append("-" * 40)
        for warning in summary.total_warnings[:10]:
            lines.append(f"  - {warning}")
        if len(summary.total_warnings) > 10:
            lines.append(f"  ... and {len(summary.total_warnings) - 10} more")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_summary_dict(summary: OptimizationSummary) -> dict[str, Any]:
    """
    Format summary as dictionary for JSON export.

    Args:
        summary: OptimizationSummary to format.

    Returns:
        Dictionary representation.
    """
    return {
        "document": {
            "title": summary.document_title,
            "url": summary.document_url,
            "timestamp": summary.optimization_timestamp.isoformat(),
        },
        "blocks": {
            "total": summary.total_blocks,
            "modified": summary.modified_blocks,
            "preserved": summary.preserved_blocks,
            "blocked": summary.blocked_blocks,
        },
        "keywords": {
            "total_injections": summary.total_keyword_injections,
            "unique_count": len(summary.unique_keywords_injected),
            "unique_keywords": list(summary.unique_keywords_injected),
            "primary_keyword": summary.keyword_plan.primary.phrase if summary.keyword_plan else None,
            "primary_count": summary.primary_keyword_count,
            "secondary_count": summary.secondary_keyword_count,
        },
        "content": {
            "original_words": summary.original_word_count,
            "modified_words": summary.modified_word_count,
            "change_ratio": round(summary.change_ratio, 2),
        },
        "validation": {
            "factuality_issues": summary.factuality_issues,
            "structure_issues": summary.structure_issues,
            "warnings": summary.total_warnings,
        },
        "changes": [
            {
                "block_id": c.block_id,
                "block_type": c.block_type.value,
                "change_type": c.change_type.value,
                "keywords_injected": c.keywords_injected,
                "change_percent": round(c.change_percent, 1),
            }
            for c in summary.block_changes
            if c.change_type != ChangeType.NO_CHANGE
        ],
    }


def log_summary(summary: OptimizationSummary) -> None:
    """
    Log optimization summary.

    Args:
        summary: Summary to log.
    """
    logger.info(f"Optimization complete: {summary.document_title}")
    logger.info(f"  Blocks: {summary.modified_blocks}/{summary.total_blocks} modified")
    logger.info(f"  Keywords: {summary.total_keyword_injections} injections")
    logger.info(f"  Change ratio: {summary.change_ratio:.1f}%")

    if summary.factuality_issues:
        logger.warning(f"  Factuality issues: {summary.factuality_issues}")
    if summary.structure_issues:
        logger.warning(f"  Structure issues: {summary.structure_issues}")
