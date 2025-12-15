"""
SEO Content Optimizer V2 - Block-based optimization with V2 Architecture.

This module implements the V2 optimization architecture with:
- Block-based content processing (ContentDocument model)
- Structure preservation policies (tables, lists, code blocks protected)
- Factuality guardrails (prevent hallucinated facts)
- Semantic keyword selection (embedding-based relevance)
- Token-level diff highlighting with change tracking

The V2 optimizer maintains backward compatibility while providing better
content preservation and more accurate highlighting.
"""

import logging
from typing import Optional, Union
from dataclasses import dataclass

from .models import (
    ContentBlock,
    ContentBlockType,
    ContentDocument,
    DocxContent,
    FAQItem,
    Keyword,
    KeywordPlan,
    ManualKeywordsConfig,
    MetaElement,
    OptimizationResult,
    PageMeta,
    ParagraphBlock,
    SemanticKeywordPlan,
    HeadingLevel,
)
from .structure_preservation import (
    StructurePreserver,
    get_modifiable_blocks,
)
from .factuality_guardrails import (
    FactualityChecker,
    FactualityCheckResult,
    validate_no_new_facts,
)
from .diff_highlighter import (
    TokenDiffer,
    compute_diff,
    find_new_keywords_in_text,
)
from .change_summary import (
    ChangeSummaryBuilder,
    OptimizationSummary,
    format_summary_text,
    format_summary_dict,
)
from .content_sources import convert_page_meta_to_blocks
from .keyword_filter import filter_keywords_for_content, get_content_topics
from .llm_client import LLMClient, create_llm_client, strip_markers
from .diff_markers import (
    MARK_START as ADD_START,
    MARK_END as ADD_END,
    add_markers_by_diff,
    normalize_paragraph_spacing,
)

logger = logging.getLogger(__name__)


@dataclass
class V2OptimizationConfig:
    """Configuration for V2 optimization."""
    enable_structure_preservation: bool = True
    enable_factuality_checks: bool = True
    enable_change_tracking: bool = True
    max_change_percent_per_block: float = 30.0
    primary_keyword_target: int = 6
    secondary_keyword_target: int = 3


class ContentOptimizerV2:
    """
    V2 Content Optimizer with block-based processing.

    This optimizer uses the V2 architecture for improved content handling:
    - ContentDocument model with typed blocks
    - Structure preservation (tables, lists, code protected)
    - Factuality guardrails to prevent hallucinated facts
    - Comprehensive change tracking and reporting
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        api_key: Optional[str] = None,
        config: Optional[V2OptimizationConfig] = None,
    ):
        """
        Initialize the V2 optimizer.

        Args:
            llm_client: Pre-configured LLM client.
            api_key: API key for LLM (used if llm_client is None).
            config: V2 optimization configuration.
        """
        if llm_client:
            self.llm = llm_client
        else:
            self.llm = create_llm_client(api_key=api_key)

        self.config = config or V2OptimizationConfig()

        # V2 components
        self.structure_preserver = StructurePreserver()
        self.factuality_checker = FactualityChecker()
        self.summary_builder = ChangeSummaryBuilder()

    def optimize(
        self,
        content: Union[PageMeta, DocxContent],
        keywords: list[Keyword],
        manual_keywords: Optional[ManualKeywordsConfig] = None,
        generate_faq: bool = True,
        faq_count: int = 4,
        max_secondary: int = 5,
    ) -> tuple[OptimizationResult, OptimizationSummary]:
        """
        Perform V2 SEO optimization on content.

        Args:
            content: Content to optimize (from URL or DOCX).
            keywords: List of available keywords.
            manual_keywords: Manual keyword selection config.
            generate_faq: Whether to generate FAQ section.
            faq_count: Number of FAQ items.
            max_secondary: Maximum secondary keywords.

        Returns:
            Tuple of (OptimizationResult, OptimizationSummary).
        """
        # Initialize change tracking
        self.summary_builder = ChangeSummaryBuilder()

        # Step 1: Convert content to ContentDocument
        document = self._convert_to_document(content)
        self.summary_builder.set_document_info(
            document.title,
            document.source_url,
        )

        # Step 2: Filter keywords for topical relevance
        full_text = document.full_text
        if manual_keywords is not None:
            keyword_plan = self._build_manual_keyword_plan(manual_keywords)
            filtered_keywords = keyword_plan.all_keywords
        else:
            filtered_keywords, _ = filter_keywords_for_content(
                keywords=keywords,
                content_text=full_text,
                min_overlap=0.5,
            )
            keyword_plan = self._build_automatic_keyword_plan(
                filtered_keywords, content, max_secondary
            )

        # Store keyword plan for tracking
        self._keyword_plan = keyword_plan

        # Step 3: Build list of keywords for differ
        all_keywords = [keyword_plan.primary.phrase]
        all_keywords.extend(kw.phrase for kw in keyword_plan.secondary)

        # Step 4: Create token differ with keywords
        self.token_differ = TokenDiffer(keywords=all_keywords)

        # Step 5: Optimize meta elements
        meta_elements = self._optimize_meta_elements_v2(
            content=content,
            keyword_plan=keyword_plan,
            full_text=full_text,
        )

        # Step 6: Optimize body blocks with structure preservation
        optimized_blocks = self._optimize_blocks_v2(
            document=document,
            keyword_plan=keyword_plan,
            all_keywords=all_keywords,
        )

        # Step 7: Convert back to ParagraphBlock format
        paragraph_blocks = self._convert_to_paragraph_blocks(optimized_blocks)

        # Step 8: Generate FAQ if requested
        faq_items = []
        if generate_faq:
            faq_items = self._generate_faq_v2(
                document=document,
                keyword_plan=keyword_plan,
                num_items=faq_count,
            )

        # Step 9: Build change summary
        summary = self.summary_builder.build_summary()

        # Step 10: Create result
        result = OptimizationResult(
            meta_elements=meta_elements,
            optimized_blocks=paragraph_blocks,
            faq_items=faq_items,
            primary_keyword=keyword_plan.primary.phrase,
            secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
            keyword_usage_counts=self._count_keyword_usage(
                meta_elements, paragraph_blocks, faq_items, keyword_plan
            ),
        )

        return result, summary

    def _convert_to_document(
        self,
        content: Union[PageMeta, DocxContent],
    ) -> ContentDocument:
        """Convert legacy content to ContentDocument."""
        if isinstance(content, PageMeta):
            blocks = []
            block_id = 0

            # Add H1 as first block
            if content.h1:
                blocks.append(ContentBlock(
                    block_id=f"h1_{block_id}",
                    block_type=ContentBlockType.HEADING,
                    text=content.h1,
                    order=block_id,
                    heading_level=1,
                ))
                block_id += 1

            # Add body paragraphs
            for para in content.body_paragraphs:
                blocks.append(ContentBlock(
                    block_id=f"p_{block_id}",
                    block_type=ContentBlockType.PARAGRAPH,
                    text=para,
                    order=block_id,
                ))
                block_id += 1

            return ContentDocument(
                title=content.title or "Untitled",
                source_url=content.url,
                blocks=blocks,
            )
        else:
            # DocxContent
            blocks = []
            block_id = 0

            for para in content.paragraphs:
                if para.is_heading:
                    blocks.append(ContentBlock(
                        block_id=f"h_{block_id}",
                        block_type=ContentBlockType.HEADING,
                        text=para.text,
                        order=block_id,
                        heading_level=para.heading_level.value if para.heading_level else 2,
                    ))
                else:
                    blocks.append(ContentBlock(
                        block_id=f"p_{block_id}",
                        block_type=ContentBlockType.PARAGRAPH,
                        text=para.text,
                        order=block_id,
                    ))
                block_id += 1

            return ContentDocument(
                title=content.h1 or "Untitled",
                blocks=blocks,
            )

    def _build_manual_keyword_plan(
        self,
        manual_keywords: ManualKeywordsConfig,
    ) -> KeywordPlan:
        """Build keyword plan from manual config."""
        primary = Keyword(phrase=manual_keywords.primary.strip(), is_brand=False)
        secondary = [
            Keyword(phrase=phrase.strip(), is_brand=False)
            for phrase in manual_keywords.secondary
            if phrase and phrase.strip()
        ]
        return KeywordPlan(
            primary=primary,
            secondary=secondary,
            long_tail_questions=[],
        )

    def _build_automatic_keyword_plan(
        self,
        keywords: list[Keyword],
        content: Union[PageMeta, DocxContent],
        max_secondary: int,
    ) -> KeywordPlan:
        """Build keyword plan from automatic selection."""
        from .prioritizer import create_keyword_plan
        from .analysis import analyze_content

        analysis = analyze_content(content, keywords)
        return create_keyword_plan(
            keywords=keywords,
            content=content,
            analysis=analysis,
            max_secondary=max_secondary,
        )

    def _optimize_meta_elements_v2(
        self,
        content: Union[PageMeta, DocxContent],
        keyword_plan: KeywordPlan,
        full_text: str,
    ) -> list[MetaElement]:
        """Optimize meta elements with V2 architecture."""
        meta_elements = []
        primary = keyword_plan.primary.phrase

        # Get current values
        if isinstance(content, PageMeta):
            current_title = content.title
            current_desc = content.meta_description
            current_h1 = content.h1
        else:
            current_title = None
            current_desc = None
            current_h1 = content.h1

        # Optimize title
        optimized_title = self._optimize_title(current_title, primary)
        meta_elements.append(MetaElement(
            element_name="Title Tag",
            current=current_title,
            optimized=optimized_title,
            why_changed="Optimized for primary keyword",
        ))

        # Optimize meta description
        optimized_desc = self._optimize_meta_description(current_desc, primary)
        meta_elements.append(MetaElement(
            element_name="Meta Description",
            current=current_desc,
            optimized=optimized_desc,
            why_changed="Optimized for primary keyword and CTR",
        ))

        # Optimize H1
        optimized_h1 = self._optimize_h1(current_h1, primary)
        meta_elements.append(MetaElement(
            element_name="H1",
            current=current_h1,
            optimized=optimized_h1,
            why_changed="Optimized for primary keyword",
        ))

        return meta_elements

    def _optimize_title(self, current: Optional[str], primary: str) -> str:
        """Optimize title tag."""
        optimized = self.llm.optimize_meta_title(
            current_title=current,
            primary_keyword=primary,
            topic=primary,
            max_length=60,
        )
        # Ensure keyword is present
        if primary.lower() not in optimized.lower():
            optimized = f"{primary}: {optimized}"
        # Add markers
        if current and optimized.strip() != current.strip():
            return f"{ADD_START}{optimized}{ADD_END}"
        return optimized

    def _optimize_meta_description(self, current: Optional[str], primary: str) -> str:
        """Optimize meta description."""
        optimized = self.llm.optimize_meta_description(
            current_description=current,
            primary_keyword=primary,
            topic=primary,
            max_length=160,
        )
        # Ensure keyword is present
        if primary.lower() not in optimized.lower():
            optimized = f"{primary}: {optimized}"
        # Add markers
        if current and optimized.strip() != current.strip():
            return f"{ADD_START}{optimized}{ADD_END}"
        return optimized

    def _optimize_h1(self, current: Optional[str], primary: str) -> str:
        """Optimize H1 heading."""
        optimized = self.llm.optimize_h1(
            current_h1=current,
            primary_keyword=primary,
            title=primary,
            topic=primary,
        )
        # Ensure keyword is present
        if primary.lower() not in optimized.lower():
            optimized = f"{primary}: {optimized}"
        # Add markers
        if current and optimized.strip() != current.strip():
            return f"{ADD_START}{optimized}{ADD_END}"
        return optimized

    def _optimize_blocks_v2(
        self,
        document: ContentDocument,
        keyword_plan: KeywordPlan,
        all_keywords: list[str],
    ) -> list[ContentBlock]:
        """
        Optimize content blocks with structure preservation.

        Args:
            document: ContentDocument to optimize.
            keyword_plan: Keyword plan.
            all_keywords: List of all keywords for diff highlighting.

        Returns:
            List of optimized ContentBlock objects.
        """
        optimized_blocks = []
        primary = keyword_plan.primary.phrase
        secondary = [kw.phrase for kw in keyword_plan.secondary]

        for block in document.blocks:
            # Check if block can be modified
            if not self.structure_preserver.can_modify(block):
                # Record as blocked
                self.summary_builder.record_blocked_block(
                    block,
                    f"Structure preservation: {block.block_type.value}",
                )
                optimized_blocks.append(block)
                continue

            # Skip H1 (handled in meta elements)
            if block.block_type == ContentBlockType.HEADING and block.heading_level == 1:
                optimized_blocks.append(block)
                continue

            # Optimize the block
            if block.block_type == ContentBlockType.PARAGRAPH:
                optimized_text = self._optimize_paragraph_block(
                    block, primary, secondary, all_keywords
                )
            elif block.block_type == ContentBlockType.HEADING:
                optimized_text = self._optimize_heading_block(
                    block, primary, secondary, all_keywords
                )
            else:
                # Other block types - keep as is
                optimized_text = block.text

            # Create optimized block
            optimized_block = ContentBlock(
                block_id=block.block_id,
                block_type=block.block_type,
                text=optimized_text,
                order=block.order,
                heading_level=block.heading_level,
            )

            # Record change
            keywords_injected = find_new_keywords_in_text(
                block.text, strip_markers(optimized_text), all_keywords
            )

            # Check factuality
            fact_result = None
            if self.config.enable_factuality_checks:
                fact_result = self.factuality_checker.compare_claims(
                    block.text, strip_markers(optimized_text)
                )

            self.summary_builder.record_block_change(
                block,
                strip_markers(optimized_text),
                keywords_injected=keywords_injected,
                factuality_result=fact_result,
            )

            optimized_blocks.append(optimized_block)

        return optimized_blocks

    def _optimize_paragraph_block(
        self,
        block: ContentBlock,
        primary: str,
        secondary: list[str],
        all_keywords: list[str],
    ) -> str:
        """Optimize a paragraph block."""
        # Use LLM to rewrite with keywords
        rewritten = self.llm.rewrite_with_markers(
            content=block.text,
            primary_keyword=primary,
            secondary_keywords=secondary[:2],
            context=f"Optimize this paragraph to include relevant keywords naturally.",
        )

        # Strip any markers from LLM output
        clean_rewritten = strip_markers(rewritten)

        # Apply token-level diff markers
        marked_text = add_markers_by_diff(
            block.text,
            clean_rewritten,
            keywords=all_keywords,
        )

        return marked_text

    def _optimize_heading_block(
        self,
        block: ContentBlock,
        primary: str,
        secondary: list[str],
        all_keywords: list[str],
    ) -> str:
        """Optimize a heading block."""
        text_lower = block.text.lower()

        # Check if already contains keywords
        if primary.lower() in text_lower:
            return block.text

        for kw in secondary:
            if kw.lower() in text_lower:
                return block.text

        # Try to optimize with LLM
        try:
            prompt = f"Optimize this heading to include a keyword naturally: {block.text}\nKeywords: {primary}, {', '.join(secondary[:2])}"
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=100,
                system="Return only the optimized heading text.",
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = response.content[0].text.strip()
            clean_rewritten = strip_markers(rewritten)

            # Apply markers
            return add_markers_by_diff(block.text, clean_rewritten, keywords=all_keywords)
        except Exception:
            return block.text

    def _convert_to_paragraph_blocks(
        self,
        blocks: list[ContentBlock],
    ) -> list[ParagraphBlock]:
        """Convert ContentBlocks back to ParagraphBlocks for compatibility."""
        result = []
        for block in blocks:
            heading_level = HeadingLevel.BODY
            if block.block_type == ContentBlockType.HEADING:
                level = block.heading_level or 2
                if level == 1:
                    heading_level = HeadingLevel.H1
                elif level == 2:
                    heading_level = HeadingLevel.H2
                elif level == 3:
                    heading_level = HeadingLevel.H3
                elif level == 4:
                    heading_level = HeadingLevel.H4
                elif level == 5:
                    heading_level = HeadingLevel.H5
                else:
                    heading_level = HeadingLevel.H6

            result.append(ParagraphBlock(
                text=block.text,
                heading_level=heading_level,
            ))

        return result

    def _generate_faq_v2(
        self,
        document: ContentDocument,
        keyword_plan: KeywordPlan,
        num_items: int,
    ) -> list[FAQItem]:
        """Generate FAQ items with V2 architecture."""
        faq_data = self.llm.generate_faq_items(
            topic=document.title,
            primary_keyword=keyword_plan.primary.phrase,
            secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
            question_keywords=[],
            faq_keywords=[kw.phrase for kw in keyword_plan.secondary[:2]],
            planned_questions=[],
            num_items=num_items,
        )

        # Wrap in markers
        return [
            FAQItem(
                question=f"{ADD_START}{item['question']}{ADD_END}",
                answer=f"{ADD_START}{item['answer']}{ADD_END}",
            )
            for item in faq_data
        ]

    def _count_keyword_usage(
        self,
        meta_elements: list[MetaElement],
        blocks: list[ParagraphBlock],
        faq_items: list[FAQItem],
        keyword_plan: KeywordPlan,
    ) -> dict[str, int]:
        """Count keyword usage across all content."""
        # Build full text
        parts = []
        for meta in meta_elements:
            parts.append(strip_markers(meta.optimized))
        for block in blocks:
            parts.append(strip_markers(block.text))
        for faq in faq_items:
            parts.append(strip_markers(faq.question))
            parts.append(strip_markers(faq.answer))

        full_text = " ".join(parts).lower()

        # Count keywords
        counts = {}
        counts[keyword_plan.primary.phrase] = full_text.count(
            keyword_plan.primary.phrase.lower()
        )
        for kw in keyword_plan.secondary:
            counts[kw.phrase] = full_text.count(kw.phrase.lower())

        return counts


def optimize_content_v2(
    content: Union[PageMeta, DocxContent],
    keywords: list[Keyword],
    api_key: Optional[str] = None,
    generate_faq: bool = True,
    faq_count: int = 4,
    config: Optional[V2OptimizationConfig] = None,
) -> tuple[OptimizationResult, OptimizationSummary]:
    """
    Convenience function for V2 content optimization.

    Args:
        content: Content to optimize.
        keywords: Available keywords.
        api_key: LLM API key.
        generate_faq: Whether to generate FAQ.
        faq_count: Number of FAQ items.
        config: V2 optimization configuration.

    Returns:
        Tuple of (OptimizationResult, OptimizationSummary).
    """
    optimizer = ContentOptimizerV2(api_key=api_key, config=config)
    return optimizer.optimize(
        content=content,
        keywords=keywords,
        generate_faq=generate_faq,
        faq_count=faq_count,
    )
