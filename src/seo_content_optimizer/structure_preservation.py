"""
Structure preservation policies for SEO Content Optimizer V2 Architecture.

This module defines how structured content (tables, lists, code blocks) should
be handled during optimization:
- Tables: Preserve structure, only allow keyword injection in header/caption
- Lists: Preserve items, allow keyword injection only in intro text
- Code blocks: Never modify
- Block quotes: Preserve attribution, allow light keyword work

The goal is to maintain structural integrity while still enabling SEO optimization.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from .models import ContentBlock, ContentBlockType

logger = logging.getLogger(__name__)


class PreservationPolicy(Enum):
    """Preservation policy levels."""
    STRICT = "strict"  # Never modify (code blocks)
    STRUCTURAL = "structural"  # Preserve structure, minor text tweaks allowed
    INTRO_ONLY = "intro_only"  # Only modify introduction/caption text
    RELAXED = "relaxed"  # Allow modifications with constraints


@dataclass
class StructurePolicy:
    """Policy definition for a content structure type."""
    block_type: ContentBlockType
    preservation: PreservationPolicy
    allow_keyword_injection: bool = False
    injection_locations: list[str] = field(default_factory=list)  # e.g., ["header", "caption"]
    max_changes_percent: float = 0.0  # Max percentage of content that can change
    preserve_items: bool = True  # For lists - preserve individual items
    preserve_formatting: bool = True  # Preserve bold, italic, etc.
    allow_synonym_replacement: bool = False
    description: str = ""


# Default policies for each block type
DEFAULT_POLICIES: dict[ContentBlockType, StructurePolicy] = {
    ContentBlockType.TABLE: StructurePolicy(
        block_type=ContentBlockType.TABLE,
        preservation=PreservationPolicy.STRICT,
        allow_keyword_injection=False,
        injection_locations=[],
        max_changes_percent=0.0,
        preserve_items=True,
        preserve_formatting=True,
        allow_synonym_replacement=False,
        description="Tables are data - never modify cell content",
    ),
    ContentBlockType.LIST: StructurePolicy(
        block_type=ContentBlockType.LIST,
        preservation=PreservationPolicy.INTRO_ONLY,
        allow_keyword_injection=True,
        injection_locations=["intro"],  # Only in intro paragraph before list
        max_changes_percent=10.0,
        preserve_items=True,
        preserve_formatting=True,
        allow_synonym_replacement=False,
        description="Lists should preserve items; keywords in intro only",
    ),
    ContentBlockType.HEADING: StructurePolicy(
        block_type=ContentBlockType.HEADING,
        preservation=PreservationPolicy.RELAXED,
        allow_keyword_injection=True,
        injection_locations=["text"],
        max_changes_percent=30.0,
        preserve_items=False,
        preserve_formatting=True,
        allow_synonym_replacement=True,
        description="Headings can be optimized for keywords",
    ),
    ContentBlockType.PARAGRAPH: StructurePolicy(
        block_type=ContentBlockType.PARAGRAPH,
        preservation=PreservationPolicy.RELAXED,
        allow_keyword_injection=True,
        injection_locations=["text"],
        max_changes_percent=20.0,
        preserve_items=False,
        preserve_formatting=True,
        allow_synonym_replacement=True,
        description="Paragraphs are the main target for optimization",
    ),
    ContentBlockType.BLOCKQUOTE: StructurePolicy(
        block_type=ContentBlockType.BLOCKQUOTE,
        preservation=PreservationPolicy.STRICT,
        allow_keyword_injection=False,
        injection_locations=[],
        max_changes_percent=0.0,
        preserve_items=True,
        preserve_formatting=True,
        allow_synonym_replacement=False,
        description="Blockquotes are citations - never modify",
    ),
    ContentBlockType.CODE: StructurePolicy(
        block_type=ContentBlockType.CODE,
        preservation=PreservationPolicy.STRICT,
        allow_keyword_injection=False,
        injection_locations=[],
        max_changes_percent=0.0,
        preserve_items=True,
        preserve_formatting=True,
        allow_synonym_replacement=False,
        description="Code blocks are never modified",
    ),
    ContentBlockType.IMAGE: StructurePolicy(
        block_type=ContentBlockType.IMAGE,
        preservation=PreservationPolicy.INTRO_ONLY,
        allow_keyword_injection=True,
        injection_locations=["alt", "caption"],
        max_changes_percent=50.0,
        preserve_items=False,
        preserve_formatting=False,
        allow_synonym_replacement=True,
        description="Image alt/caption can be optimized",
    ),
}


@dataclass
class StructureValidationResult:
    """Result of structure validation."""
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    change_percent: float = 0.0


class StructurePreserver:
    """
    Validates and enforces structure preservation policies.

    Used by BlockRewriter to ensure modifications respect content structure.
    """

    def __init__(self, policies: Optional[dict[ContentBlockType, StructurePolicy]] = None):
        """
        Initialize structure preserver.

        Args:
            policies: Optional custom policies. Uses defaults if not provided.
        """
        self.policies = policies or DEFAULT_POLICIES.copy()

    def get_policy(self, block_type: ContentBlockType) -> StructurePolicy:
        """Get the policy for a block type."""
        return self.policies.get(block_type, StructurePolicy(
            block_type=block_type,
            preservation=PreservationPolicy.RELAXED,
            allow_keyword_injection=True,
            description="Default permissive policy",
        ))

    def can_modify(self, block: ContentBlock) -> bool:
        """
        Check if a block can be modified at all.

        Args:
            block: ContentBlock to check.

        Returns:
            True if modifications are allowed.
        """
        policy = self.get_policy(block.block_type)
        return policy.preservation != PreservationPolicy.STRICT

    def can_inject_keyword(self, block: ContentBlock) -> bool:
        """
        Check if keywords can be injected into this block.

        Args:
            block: ContentBlock to check.

        Returns:
            True if keyword injection is allowed.
        """
        policy = self.get_policy(block.block_type)
        return policy.allow_keyword_injection

    def validate_modification(
        self,
        original_block: ContentBlock,
        modified_text: str,
    ) -> StructureValidationResult:
        """
        Validate a proposed modification against the block's policy.

        Args:
            original_block: Original ContentBlock.
            modified_text: Proposed new text for the block.

        Returns:
            StructureValidationResult with validation details.
        """
        policy = self.get_policy(original_block.block_type)
        violations = []
        warnings = []

        # Check if modification is allowed at all
        if policy.preservation == PreservationPolicy.STRICT:
            if modified_text.strip() != original_block.text.strip():
                violations.append(
                    f"STRICT policy violation: {original_block.block_type.value} cannot be modified"
                )
                return StructureValidationResult(
                    is_valid=False,
                    violations=violations,
                    change_percent=100.0,
                )

        # Calculate change percentage
        change_percent = self._calculate_change_percent(original_block.text, modified_text)

        # Check max change percentage
        if policy.max_changes_percent > 0 and change_percent > policy.max_changes_percent:
            violations.append(
                f"Change percent {change_percent:.1f}% exceeds max {policy.max_changes_percent:.1f}%"
            )

        # Check structure preservation for lists
        if policy.preserve_items and original_block.block_type == ContentBlockType.LIST:
            structure_valid, structure_msg = self._validate_list_structure(
                original_block.text, modified_text
            )
            if not structure_valid:
                violations.append(structure_msg)

        # Check formatting preservation
        if policy.preserve_formatting:
            format_valid, format_msg = self._validate_formatting(
                original_block.text, modified_text
            )
            if not format_valid:
                warnings.append(format_msg)

        return StructureValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            change_percent=change_percent,
        )

    def _calculate_change_percent(self, original: str, modified: str) -> float:
        """Calculate percentage of text that changed."""
        if not original:
            return 100.0 if modified else 0.0

        original_words = set(original.lower().split())
        modified_words = set(modified.lower().split())

        if not original_words:
            return 100.0 if modified_words else 0.0

        # Count words that were added or removed
        added = modified_words - original_words
        removed = original_words - modified_words
        total_changes = len(added) + len(removed)

        # Calculate as percentage of original
        return (total_changes / len(original_words)) * 100

    def _validate_list_structure(
        self,
        original: str,
        modified: str,
    ) -> tuple[bool, str]:
        """Validate that list structure is preserved."""
        # Extract list items from both texts
        original_items = self._extract_list_items(original)
        modified_items = self._extract_list_items(modified)

        # Check item count
        if len(modified_items) != len(original_items):
            return False, f"List item count changed: {len(original_items)} -> {len(modified_items)}"

        # Check that items are similar (allow minor text changes)
        for i, (orig, mod) in enumerate(zip(original_items, modified_items)):
            similarity = self._text_similarity(orig, mod)
            if similarity < 0.7:  # 70% similar required
                return False, f"List item {i+1} changed too much"

        return True, ""

    def _extract_list_items(self, text: str) -> list[str]:
        """Extract list items from text."""
        items = []

        # Match bullet points (-, *, •) or numbered lists
        pattern = r'^[\s]*(?:[-*•]|\d+\.)\s*(.+)$'
        for line in text.split('\n'):
            match = re.match(pattern, line)
            if match:
                items.append(match.group(1).strip())

        return items

    def _validate_formatting(
        self,
        original: str,
        modified: str,
    ) -> tuple[bool, str]:
        """Validate that formatting markers are preserved."""
        # Check for markdown formatting
        original_bold = len(re.findall(r'\*\*[^*]+\*\*', original))
        modified_bold = len(re.findall(r'\*\*[^*]+\*\*', modified))

        original_italic = len(re.findall(r'\*[^*]+\*', original))
        modified_italic = len(re.findall(r'\*[^*]+\*', modified))

        if original_bold > modified_bold:
            return False, f"Bold formatting reduced: {original_bold} -> {modified_bold}"

        if original_italic > modified_italic:
            return False, f"Italic formatting reduced: {original_italic} -> {modified_italic}"

        return True, ""

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (0-1)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


def create_structure_filter(
    policies: Optional[dict[ContentBlockType, StructurePolicy]] = None,
) -> Callable[[ContentBlock], bool]:
    """
    Create a filter function for blocks that can be modified.

    Args:
        policies: Optional custom policies.

    Returns:
        Callable that returns True if block can be modified.
    """
    preserver = StructurePreserver(policies)
    return preserver.can_modify


def create_keyword_injection_filter(
    policies: Optional[dict[ContentBlockType, StructurePolicy]] = None,
) -> Callable[[ContentBlock], bool]:
    """
    Create a filter function for blocks that can receive keyword injection.

    Args:
        policies: Optional custom policies.

    Returns:
        Callable that returns True if keywords can be injected.
    """
    preserver = StructurePreserver(policies)
    return preserver.can_inject_keyword


def get_modifiable_blocks(
    blocks: list[ContentBlock],
    policies: Optional[dict[ContentBlockType, StructurePolicy]] = None,
) -> list[ContentBlock]:
    """
    Filter blocks to only those that can be modified.

    Args:
        blocks: List of ContentBlocks.
        policies: Optional custom policies.

    Returns:
        List of modifiable blocks.
    """
    preserver = StructurePreserver(policies)
    return [b for b in blocks if preserver.can_modify(b)]


def log_preservation_summary(
    blocks: list[ContentBlock],
    policies: Optional[dict[ContentBlockType, StructurePolicy]] = None,
) -> dict:
    """
    Generate summary of how blocks will be handled.

    Args:
        blocks: List of ContentBlocks.
        policies: Optional custom policies.

    Returns:
        Dictionary with summary statistics.
    """
    preserver = StructurePreserver(policies)

    summary = {
        "total_blocks": len(blocks),
        "modifiable": 0,
        "keyword_injectable": 0,
        "strict_preserve": 0,
        "by_type": {},
    }

    for block in blocks:
        block_type = block.block_type.value
        if block_type not in summary["by_type"]:
            summary["by_type"][block_type] = {
                "count": 0,
                "modifiable": 0,
                "injectable": 0,
            }

        summary["by_type"][block_type]["count"] += 1

        if preserver.can_modify(block):
            summary["modifiable"] += 1
            summary["by_type"][block_type]["modifiable"] += 1
        else:
            summary["strict_preserve"] += 1

        if preserver.can_inject_keyword(block):
            summary["keyword_injectable"] += 1
            summary["by_type"][block_type]["injectable"] += 1

    return summary
