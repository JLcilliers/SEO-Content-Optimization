"""
Micro-edit per block rewriting for SEO Content Optimizer V2 Architecture.

This module implements the block-level rewriting approach:
- Process blocks one at a time with constrained prompts
- Support "SKIP" output when no changes needed
- Keep changes minimal (max ~15% word change per block)
- Preserve structure (tables, lists, formatting)
- Token-level diff highlighting for precise change tracking

The goal is stability: minimal modifications that still achieve SEO benefit.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable

from .models import (
    Block,
    ContentDocument,
    Run,
    SemanticKeyword,
    SemanticKeywordPlan,
    OptimizationReport,
    FactualityClaim,
    BlockType,
)

logger = logging.getLogger(__name__)


# Block types that should NOT be rewritten
SKIP_BLOCK_TYPES: set[BlockType] = {
    "table", "tr", "td", "th", "caption",  # Tables preserved exactly
    "image",  # Images can't be rewritten
    "hr",  # Horizontal rules
}

# Block types with special handling
HEADING_TYPES: set[BlockType] = {"h1", "h2", "h3", "h4", "h5", "h6"}
LIST_TYPES: set[BlockType] = {"ul", "ol", "li"}
META_TYPES: set[BlockType] = {"title", "meta_title", "meta_desc"}

# Maximum word change ratio per block
MAX_WORD_CHANGE_RATIO = 0.15  # 15% max change

# Minimum block length for optimization (chars)
MIN_BLOCK_LENGTH = 30


@dataclass
class BlockRewriteConfig:
    """Configuration for block rewriting."""
    max_word_change_ratio: float = MAX_WORD_CHANGE_RATIO
    min_block_length: int = MIN_BLOCK_LENGTH
    skip_tables: bool = True
    skip_lists: bool = False  # Lists can be lightly optimized
    preserve_formatting: bool = True
    enable_factuality_check: bool = True


@dataclass
class BlockRewriteResult:
    """Result of rewriting a single block."""
    block: Block
    was_modified: bool = False
    skip_reason: Optional[str] = None
    original_text: Optional[str] = None
    word_change_ratio: float = 0.0
    keywords_added: list[str] = field(default_factory=list)
    claims_detected: list[FactualityClaim] = field(default_factory=list)
    claims_removed: list[FactualityClaim] = field(default_factory=list)


class BlockRewriter:
    """
    Micro-edit block rewriter for V2 architecture.

    Processes ContentDocument blocks one at a time with minimal changes.
    """

    def __init__(
        self,
        llm_client=None,
        config: Optional[BlockRewriteConfig] = None,
        factuality_checker: Optional[Callable] = None,
    ):
        """
        Initialize block rewriter.

        Args:
            llm_client: LLM client for rewriting prompts.
            config: Rewriting configuration.
            factuality_checker: Optional callback to validate factual claims.
        """
        self.llm_client = llm_client
        self.config = config or BlockRewriteConfig()
        self.factuality_checker = factuality_checker

    def rewrite_document(
        self,
        document: ContentDocument,
        keyword_plan: SemanticKeywordPlan,
        placement_strategy: Optional[dict] = None,
    ) -> tuple[ContentDocument, OptimizationReport]:
        """
        Rewrite all blocks in a document with micro-edits.

        Args:
            document: ContentDocument to optimize.
            keyword_plan: Keyword plan with primary/secondary/questions.
            placement_strategy: Optional dict mapping block indices to keywords.

        Returns:
            Tuple of (optimized_document, optimization_report).
        """
        report = OptimizationReport()
        optimized_blocks: list[Block] = []

        # Build placement strategy if not provided
        if placement_strategy is None:
            placement_strategy = self._build_placement_strategy(
                document, keyword_plan
            )

        # Process each block
        for i, block in enumerate(document.blocks):
            # Get keywords targeted for this block
            target_keywords = placement_strategy.get(i, [])

            # Rewrite the block
            result = self.rewrite_block(
                block=block,
                keyword_plan=keyword_plan,
                target_keywords=target_keywords,
                block_index=i,
                total_blocks=len(document.blocks),
            )

            optimized_blocks.append(result.block)

            # Update report
            report.blocks_analyzed += 1
            if result.was_modified:
                report.blocks_modified += 1
                report.block_changes.append({
                    "block_id": block.id,
                    "block_type": block.type,
                    "original": result.original_text,
                    "modified": result.block.text,
                    "keywords_added": result.keywords_added,
                    "word_change_ratio": result.word_change_ratio,
                })
            if result.skip_reason:
                report.blocks_skipped += 1

            # Track claims
            report.claims_detected += len(result.claims_detected)
            for claim in result.claims_removed:
                report.claims_removed += 1
                report.removed_claims.append(claim)

        # Build optimized document
        optimized_document = ContentDocument(
            blocks=optimized_blocks,
            source_url=document.source_url,
            source_docx_path=document.source_docx_path,
            extracted_title=document.extracted_title,
            extracted_meta_desc=document.extracted_meta_desc,
            language=document.language,
            extraction_timestamp=document.extraction_timestamp,
            original_word_count=document.original_word_count,
        )

        # Add keyword tracking to report
        report.keywords_selected = keyword_plan.all_phrases
        report.keywords_skipped = [kw.phrase for kw in keyword_plan.skipped_keywords]
        report.keyword_skip_reasons = keyword_plan.skip_reasons

        return optimized_document, report

    def rewrite_block(
        self,
        block: Block,
        keyword_plan: SemanticKeywordPlan,
        target_keywords: list[str],
        block_index: int = 0,
        total_blocks: int = 1,
    ) -> BlockRewriteResult:
        """
        Rewrite a single block with micro-edits.

        This is the core method - it processes one block with:
        - Type-specific handling (headings, paragraphs, lists)
        - SKIP logic when no optimization possible/needed
        - Word change ratio enforcement
        - Factuality checking

        Args:
            block: Block to rewrite.
            keyword_plan: Full keyword plan for context.
            target_keywords: Specific keywords to try adding to this block.
            block_index: Index of this block in document.
            total_blocks: Total number of blocks.

        Returns:
            BlockRewriteResult with optimized block and metadata.
        """
        # Store original for comparison
        original_text = block.text or ""

        # Check if block should be skipped entirely
        skip_reason = self._should_skip_block(block)
        if skip_reason:
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason=skip_reason,
                original_text=original_text,
            )

        # Check if block is too short
        if len(original_text) < self.config.min_block_length:
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason=f"Block too short ({len(original_text)} < {self.config.min_block_length} chars)",
                original_text=original_text,
            )

        # Check if keywords already present
        keywords_present = self._find_keywords_in_text(
            original_text, [keyword_plan.primary.phrase] + target_keywords
        )
        if keywords_present and len(keywords_present) >= len(target_keywords):
            # All target keywords already present
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason="All target keywords already present",
                original_text=original_text,
            )

        # Determine rewrite approach based on block type
        if block.type in HEADING_TYPES:
            result = self._rewrite_heading(
                block, keyword_plan, target_keywords
            )
        elif block.type in LIST_TYPES:
            result = self._rewrite_list_item(
                block, keyword_plan, target_keywords
            )
        elif block.type in META_TYPES:
            result = self._rewrite_meta(
                block, keyword_plan, target_keywords
            )
        else:
            # Standard paragraph rewrite
            result = self._rewrite_paragraph(
                block, keyword_plan, target_keywords,
                is_intro=(block_index < 2),
                is_conclusion=(block_index >= total_blocks - 2),
            )

        # Validate word change ratio
        if result.was_modified:
            ratio = self._compute_word_change_ratio(
                original_text, result.block.text or ""
            )
            result.word_change_ratio = ratio

            if ratio > self.config.max_word_change_ratio:
                # Change too large - revert to original
                logger.warning(
                    f"Block {block.id}: word change ratio {ratio:.2%} exceeds "
                    f"max {self.config.max_word_change_ratio:.2%}, reverting"
                )
                result.block = block
                result.was_modified = False
                result.skip_reason = f"Word change ratio too high ({ratio:.2%})"

        # Factuality check if enabled
        if result.was_modified and self.config.enable_factuality_check:
            result = self._apply_factuality_check(result, original_text)

        return result

    def _should_skip_block(self, block: Block) -> Optional[str]:
        """
        Determine if a block should be skipped entirely.

        Returns skip reason or None if block should be processed.
        """
        # Skip marked blocks
        if block.skip_optimization:
            return "Block marked for skip"

        # Skip certain block types
        if block.type in SKIP_BLOCK_TYPES:
            return f"Block type '{block.type}' preserved"

        # Skip tables if configured
        if self.config.skip_tables and block.type in {"table", "tr", "td", "th"}:
            return "Tables preserved"

        # Skip empty blocks
        if not block.text and not block.children:
            return "Empty block"

        return None

    def _rewrite_heading(
        self,
        block: Block,
        keyword_plan: SemanticKeywordPlan,
        target_keywords: list[str],
    ) -> BlockRewriteResult:
        """
        Rewrite a heading block (H1-H6) with minimal changes.

        Headings get special treatment:
        - Primary keyword should appear in H1
        - Secondary keywords can appear in H2-H3
        - Keep heading concise
        """
        original_text = block.text or ""
        primary = keyword_plan.primary.phrase

        # H1 gets primary keyword
        if block.type == "h1":
            target = primary
        else:
            target = target_keywords[0] if target_keywords else primary

        # Check if keyword already present
        if target.lower() in original_text.lower():
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason="Keyword already in heading",
                original_text=original_text,
            )

        # Try to add keyword naturally
        optimized_text = self._inject_keyword_in_heading(original_text, target)

        if optimized_text == original_text:
            # Couldn't inject naturally
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason="No natural keyword placement in heading",
                original_text=original_text,
            )

        # Create modified block
        modified_block = Block(
            id=block.id,
            type=block.type,
            text=optimized_text,
            runs=block.runs,
            children=block.children,
            attrs=block.attrs,
            original_text=original_text,
            was_modified=True,
        )

        return BlockRewriteResult(
            block=modified_block,
            was_modified=True,
            original_text=original_text,
            keywords_added=[target],
        )

    def _rewrite_paragraph(
        self,
        block: Block,
        keyword_plan: SemanticKeywordPlan,
        target_keywords: list[str],
        is_intro: bool = False,
        is_conclusion: bool = False,
    ) -> BlockRewriteResult:
        """
        Rewrite a paragraph block with micro-edits.

        Paragraphs can receive more substantial rewrites but still constrained:
        - Intro paragraphs get primary keyword emphasis
        - Conclusion paragraphs reinforce main topic
        - Body paragraphs get secondary keywords
        """
        original_text = block.text or ""
        primary = keyword_plan.primary.phrase

        # Determine which keywords to target
        if is_intro:
            keywords_to_add = [primary] + target_keywords[:1]
        elif is_conclusion:
            keywords_to_add = [primary] + target_keywords[:1]
        else:
            keywords_to_add = target_keywords if target_keywords else [primary]

        # Find which keywords need adding
        missing_keywords = [
            kw for kw in keywords_to_add
            if kw.lower() not in original_text.lower()
        ]

        if not missing_keywords:
            # All keywords present
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason="All keywords present",
                original_text=original_text,
            )

        # Use LLM for micro-edit if available
        if self.llm_client:
            optimized_text = self._llm_micro_edit(
                original_text,
                missing_keywords,
                is_intro=is_intro,
                is_conclusion=is_conclusion,
            )
        else:
            # Fallback: simple injection
            optimized_text = self._simple_keyword_injection(
                original_text, missing_keywords[0]
            )

        if optimized_text == original_text:
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason="LLM returned SKIP",
                original_text=original_text,
            )

        # Preserve runs if present (formatting)
        if self.config.preserve_formatting and block.runs:
            modified_runs = self._preserve_runs_formatting(
                block.runs, original_text, optimized_text
            )
        else:
            modified_runs = None

        # Create modified block
        modified_block = Block(
            id=block.id,
            type=block.type,
            text=optimized_text,
            runs=modified_runs,
            children=block.children,
            attrs=block.attrs,
            original_text=original_text,
            was_modified=True,
        )

        # Identify which keywords were actually added
        keywords_added = [
            kw for kw in missing_keywords
            if kw.lower() in optimized_text.lower()
        ]

        return BlockRewriteResult(
            block=modified_block,
            was_modified=True,
            original_text=original_text,
            keywords_added=keywords_added,
        )

    def _rewrite_list_item(
        self,
        block: Block,
        keyword_plan: SemanticKeywordPlan,
        target_keywords: list[str],
    ) -> BlockRewriteResult:
        """
        Rewrite a list item with very minimal changes.

        Lists are lightly touched - only add keyword if very natural fit.
        """
        original_text = block.text or ""

        # Lists are mostly preserved
        if self.config.skip_lists:
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason="Lists preserved",
                original_text=original_text,
            )

        # Only try to add one keyword, and only if natural
        target = target_keywords[0] if target_keywords else keyword_plan.primary.phrase

        if target.lower() in original_text.lower():
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason="Keyword already in list item",
                original_text=original_text,
            )

        # Very light touch - only prepend if it's a natural list item
        # Don't modify complex list items
        if len(original_text) > 100 or ":" in original_text:
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason="List item too complex for modification",
                original_text=original_text,
            )

        # Simple prepend for short list items
        optimized_text = f"{target} - {original_text}"

        modified_block = Block(
            id=block.id,
            type=block.type,
            text=optimized_text,
            runs=block.runs,
            children=block.children,
            attrs=block.attrs,
            original_text=original_text,
            was_modified=True,
        )

        return BlockRewriteResult(
            block=modified_block,
            was_modified=True,
            original_text=original_text,
            keywords_added=[target],
        )

    def _rewrite_meta(
        self,
        block: Block,
        keyword_plan: SemanticKeywordPlan,
        target_keywords: list[str],
    ) -> BlockRewriteResult:
        """
        Rewrite meta elements (title, meta_desc) with constraints.

        Meta elements have strict length limits:
        - Title: max 60 chars
        - Meta description: max 160 chars
        """
        original_text = block.text or ""
        primary = keyword_plan.primary.phrase

        # Length limits by type
        max_length = 60 if block.type in {"title", "meta_title"} else 160

        # Check if primary keyword present
        if primary.lower() in original_text.lower():
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason="Primary keyword already in meta",
                original_text=original_text,
            )

        # Use LLM for meta rewriting if available
        if self.llm_client:
            optimized_text = self._llm_rewrite_meta(
                original_text, primary, max_length, block.type
            )
        else:
            # Fallback: prepend keyword
            optimized_text = f"{primary}: {original_text}"
            if len(optimized_text) > max_length:
                optimized_text = optimized_text[:max_length-3] + "..."

        if optimized_text == original_text:
            return BlockRewriteResult(
                block=block,
                was_modified=False,
                skip_reason="Meta unchanged",
                original_text=original_text,
            )

        modified_block = Block(
            id=block.id,
            type=block.type,
            text=optimized_text,
            runs=block.runs,
            children=block.children,
            attrs=block.attrs,
            original_text=original_text,
            was_modified=True,
        )

        return BlockRewriteResult(
            block=modified_block,
            was_modified=True,
            original_text=original_text,
            keywords_added=[primary],
        )

    def _llm_micro_edit(
        self,
        text: str,
        keywords: list[str],
        is_intro: bool = False,
        is_conclusion: bool = False,
    ) -> str:
        """
        Use LLM for minimal rewriting to add keywords.

        Sends a constrained prompt that:
        - Emphasizes MINIMAL changes
        - Allows SKIP response
        - Preserves original meaning and style
        """
        keyword_list = ", ".join(f'"{kw}"' for kw in keywords)

        context = ""
        if is_intro:
            context = "This is an introduction paragraph. Place primary keyword early."
        elif is_conclusion:
            context = "This is a conclusion paragraph. Reinforce the main topic."
        else:
            context = "This is a body paragraph."

        prompt = f"""MICRO-EDIT TASK: Add keyword(s) to this paragraph with MINIMAL changes.

ORIGINAL TEXT:
{text}

KEYWORDS TO ADD: {keyword_list}

{context}

RULES:
1. Make the SMALLEST possible change to include the keyword(s)
2. Keep the original meaning, tone, and style
3. Do NOT add new facts, claims, or statistics
4. Do NOT remove any existing content
5. If no natural placement exists, respond with only: SKIP
6. Maximum word change: ~15% of original

Return ONLY the edited text (or SKIP). No explanations."""

        try:
            response = self.llm_client.client.messages.create(
                model=self.llm_client.model,
                max_tokens=len(text) + 200,
                system="You are a micro-edit assistant. Make minimal changes to add keywords naturally. Respond with SKIP if no good placement exists.",
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text.strip()

            # Check for SKIP response
            if result.upper() == "SKIP":
                return text

            return result

        except Exception as e:
            logger.error(f"LLM micro-edit failed: {e}")
            return text  # Return original on error

    def _llm_rewrite_meta(
        self,
        text: str,
        keyword: str,
        max_length: int,
        meta_type: str,
    ) -> str:
        """
        Use LLM to rewrite meta element with keyword.
        """
        element_name = "title" if "title" in meta_type else "meta description"

        prompt = f"""Rewrite this {element_name} to include the keyword "{keyword}" naturally.

ORIGINAL: {text}

REQUIREMENTS:
1. Include "{keyword}" naturally, preferably near the beginning
2. Keep under {max_length} characters
3. Maintain the original meaning
4. Make it compelling for search results
5. If no good rewrite possible, respond with: SKIP

Return ONLY the rewritten text (or SKIP)."""

        try:
            response = self.llm_client.client.messages.create(
                model=self.llm_client.model,
                max_tokens=200,
                system="You are an SEO meta element optimizer. Create compelling, keyword-rich meta content.",
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text.strip()

            if result.upper() == "SKIP":
                return text

            # Enforce length limit
            if len(result) > max_length:
                result = result[:max_length-3] + "..."

            return result

        except Exception as e:
            logger.error(f"LLM meta rewrite failed: {e}")
            return text

    def _inject_keyword_in_heading(self, heading: str, keyword: str) -> str:
        """
        Inject keyword into heading if natural placement exists.

        Strategies:
        1. Prepend if heading is short
        2. Replace generic words with keyword
        3. Append with separator
        """
        heading_lower = heading.lower()
        keyword_lower = keyword.lower()

        # Already present
        if keyword_lower in heading_lower:
            return heading

        # Strategy 1: Short headings - prepend
        if len(heading) < 40:
            return f"{keyword}: {heading}"

        # Strategy 2: Replace generic phrases
        generic_phrases = [
            "everything you need to know",
            "a complete guide",
            "your guide to",
            "understanding",
            "introduction to",
        ]
        for phrase in generic_phrases:
            if phrase in heading_lower:
                return re.sub(
                    re.escape(phrase),
                    f"{keyword} - {phrase}",
                    heading,
                    flags=re.IGNORECASE,
                    count=1,
                )

        # Strategy 3: If still short enough, prepend
        if len(heading) + len(keyword) + 3 < 70:
            return f"{keyword}: {heading}"

        # No natural placement
        return heading

    def _simple_keyword_injection(self, text: str, keyword: str) -> str:
        """
        Simple keyword injection fallback when LLM unavailable.

        Finds a natural sentence boundary and inserts a keyword phrase.
        """
        # Find first sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return text

        # Add keyword to first sentence if not too long
        first = sentences[0]
        if len(first) < 100:
            # Prepend keyword phrase
            modified_first = f"When it comes to {keyword}, {first[0].lower()}{first[1:]}"
            sentences[0] = modified_first
            return " ".join(sentences)

        # Otherwise add as new sentence at start
        intro = f"This section covers {keyword}."
        return f"{intro} {text}"

    def _compute_word_change_ratio(self, original: str, modified: str) -> float:
        """
        Compute the ratio of words changed between original and modified.
        """
        orig_words = set(original.lower().split())
        mod_words = set(modified.lower().split())

        if not orig_words:
            return 1.0 if mod_words else 0.0

        # Count words added or removed
        added = mod_words - orig_words
        removed = orig_words - mod_words
        changed = len(added) + len(removed)

        return changed / len(orig_words)

    def _find_keywords_in_text(
        self,
        text: str,
        keywords: list[str],
    ) -> list[str]:
        """Find which keywords are present in text."""
        text_lower = text.lower()
        return [kw for kw in keywords if kw.lower() in text_lower]

    def _preserve_runs_formatting(
        self,
        original_runs: list[Run],
        original_text: str,
        modified_text: str,
    ) -> Optional[list[Run]]:
        """
        Attempt to preserve Run formatting when text is modified.

        This is a best-effort approach - if text changes significantly,
        formatting may be lost.
        """
        if not original_runs:
            return None

        # If text unchanged, keep original runs
        if original_text == modified_text:
            return original_runs

        # Simple case: if only addition at start/end, adjust runs
        if modified_text.startswith(original_text):
            # Added to end
            added = modified_text[len(original_text):]
            return original_runs + [Run(text=added)]

        if modified_text.endswith(original_text):
            # Added to start
            added = modified_text[:-len(original_text)]
            return [Run(text=added)] + original_runs

        # Complex change - return single run with modified text
        # Preserve formatting from first run if available
        if original_runs:
            first_run = original_runs[0]
            return [Run(
                text=modified_text,
                bold=first_run.bold,
                italic=first_run.italic,
                underline=first_run.underline,
            )]

        return [Run(text=modified_text)]

    def _apply_factuality_check(
        self,
        result: BlockRewriteResult,
        original_text: str,
    ) -> BlockRewriteResult:
        """
        Apply factuality checking to modified block.

        If new factual claims are detected that weren't in original,
        they are flagged and optionally removed.
        """
        if not self.factuality_checker:
            return result

        modified_text = result.block.text or ""

        # Get claims from both versions
        try:
            original_claims = self.factuality_checker(original_text)
            modified_claims = self.factuality_checker(modified_text)

            # Find NEW claims (in modified but not in original)
            original_claim_texts = {c.claim_text for c in original_claims}
            new_claims = [
                c for c in modified_claims
                if c.claim_text not in original_claim_texts
            ]

            if new_claims:
                # New claims detected - revert to original
                logger.warning(
                    f"Block {result.block.id}: {len(new_claims)} new claims detected, reverting"
                )
                result.claims_detected = modified_claims
                result.claims_removed = new_claims

                # Revert the block
                result.block = Block(
                    id=result.block.id,
                    type=result.block.type,
                    text=original_text,
                    runs=result.block.runs,
                    children=result.block.children,
                    attrs=result.block.attrs,
                    original_text=original_text,
                    was_modified=False,
                )
                result.was_modified = False
                result.skip_reason = "Reverted due to new factual claims"

        except Exception as e:
            logger.error(f"Factuality check failed: {e}")

        return result

    def _build_placement_strategy(
        self,
        document: ContentDocument,
        keyword_plan: SemanticKeywordPlan,
    ) -> dict[int, list[str]]:
        """
        Build a keyword placement strategy for the document.

        Distributes keywords evenly across blocks:
        - Primary keyword: H1, intro, conclusion, every ~5 paragraphs
        - Secondary keywords: distributed throughout body
        - Question keywords: reserved for FAQ blocks
        """
        strategy: dict[int, list[str]] = {}
        primary = keyword_plan.primary.phrase
        secondary = [kw.phrase for kw in keyword_plan.secondary]

        # Identify block types
        paragraph_indices = []
        heading_indices = []
        h1_index = None

        for i, block in enumerate(document.blocks):
            if block.type == "h1":
                h1_index = i
            elif block.type in HEADING_TYPES:
                heading_indices.append(i)
            elif block.type == "p":
                paragraph_indices.append(i)

        # H1 gets primary keyword
        if h1_index is not None:
            strategy[h1_index] = [primary]

        # First paragraph gets primary
        if paragraph_indices:
            strategy[paragraph_indices[0]] = [primary]

        # Distribute secondary keywords across body
        if secondary and paragraph_indices:
            # Skip first and last paragraphs (handled separately)
            body_paragraphs = paragraph_indices[1:-1] if len(paragraph_indices) > 2 else []

            if body_paragraphs:
                interval = max(1, len(body_paragraphs) // (len(secondary) + 1))
                for i, kw in enumerate(secondary):
                    target_idx = min(i * interval, len(body_paragraphs) - 1)
                    block_idx = body_paragraphs[target_idx]
                    if block_idx not in strategy:
                        strategy[block_idx] = []
                    strategy[block_idx].append(kw)

        # Last paragraph gets primary (conclusion)
        if len(paragraph_indices) > 1:
            last_idx = paragraph_indices[-1]
            if last_idx not in strategy:
                strategy[last_idx] = []
            strategy[last_idx].append(primary)

        # Headings get secondary keywords
        for i, heading_idx in enumerate(heading_indices[:len(secondary)]):
            if heading_idx not in strategy:
                strategy[heading_idx] = []
            if i < len(secondary):
                strategy[heading_idx].append(secondary[i])

        return strategy


def rewrite_document_blocks(
    document: ContentDocument,
    keyword_plan: SemanticKeywordPlan,
    llm_client=None,
    config: Optional[BlockRewriteConfig] = None,
    factuality_checker: Optional[Callable] = None,
) -> tuple[ContentDocument, OptimizationReport]:
    """
    Convenience function for block-level document rewriting.

    Args:
        document: ContentDocument to optimize.
        keyword_plan: Semantic keyword plan.
        llm_client: Optional LLM client.
        config: Optional rewrite configuration.
        factuality_checker: Optional factuality checker callback.

    Returns:
        Tuple of (optimized_document, optimization_report).
    """
    rewriter = BlockRewriter(
        llm_client=llm_client,
        config=config,
        factuality_checker=factuality_checker,
    )
    return rewriter.rewrite_document(document, keyword_plan)
