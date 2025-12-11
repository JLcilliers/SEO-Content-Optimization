"""
SEO optimization orchestration module.

This module coordinates the optimization process:
- Applies SEO rules to meta elements and content
- Uses LLM for intelligent rewriting
- Generates FAQ sections
- Produces structured optimization results
"""

from typing import Optional, Union

from .analysis import ContentAnalysis, analyze_content
from .content_sources import convert_page_meta_to_blocks
from .llm_client import (
    ADD_END,
    ADD_START,
    LLMClient,
    create_llm_client,
    ensure_markers_present,
    strip_markers,
)
from .models import (
    DocxContent,
    FAQItem,
    HeadingLevel,
    Keyword,
    KeywordPlan,
    MetaElement,
    OptimizationResult,
    PageMeta,
    ParagraphBlock,
)
from .prioritizer import create_keyword_plan


class ContentOptimizer:
    """
    Orchestrates SEO content optimization.

    This class coordinates all optimization tasks:
    - Meta element optimization (title, description, H1)
    - Body content optimization
    - FAQ generation
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
        generate_faq: bool = True,
        faq_count: int = 4,
        max_secondary: int = 5,
    ) -> OptimizationResult:
        """
        Perform full SEO optimization on content.

        Args:
            content: Content to optimize (from URL or DOCX).
            keywords: List of available keywords.
            generate_faq: Whether to generate FAQ section.
            faq_count: Number of FAQ items to generate.
            max_secondary: Maximum secondary keywords.

        Returns:
            OptimizationResult with all optimized content.
        """
        # Step 1: Analyze content
        analysis = analyze_content(content, keywords)

        # Step 2: Create keyword plan
        keyword_plan = create_keyword_plan(
            keywords=keywords,
            content=content,
            analysis=analysis,
            max_secondary=max_secondary,
            max_questions=faq_count,
        )

        # Step 3: Get current meta elements
        if isinstance(content, PageMeta):
            current_title = content.title
            current_meta_desc = content.meta_description
            current_h1 = content.h1
            blocks = convert_page_meta_to_blocks(content)
        else:
            current_title = None
            current_meta_desc = None
            current_h1 = content.h1
            blocks = content.paragraphs

        # Step 4: Optimize meta elements
        meta_elements = self._optimize_meta_elements(
            current_title=current_title,
            current_meta_desc=current_meta_desc,
            current_h1=current_h1,
            keyword_plan=keyword_plan,
            topic=analysis.topic,
        )

        # Step 5: Optimize body content
        optimized_blocks = self._optimize_body_content(
            blocks=blocks,
            keyword_plan=keyword_plan,
            analysis=analysis,
        )

        # Step 6: Generate FAQ if requested
        faq_items = []
        if generate_faq:
            faq_items = self._generate_faq(
                topic=analysis.topic,
                keyword_plan=keyword_plan,
                num_items=faq_count,
            )

        return OptimizationResult(
            meta_elements=meta_elements,
            optimized_blocks=optimized_blocks,
            faq_items=faq_items,
            primary_keyword=keyword_plan.primary.phrase,
            secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
        )

    def _optimize_meta_elements(
        self,
        current_title: Optional[str],
        current_meta_desc: Optional[str],
        current_h1: Optional[str],
        keyword_plan: KeywordPlan,
        topic: str,
    ) -> list[MetaElement]:
        """Optimize title, meta description, and H1."""
        meta_elements = []
        primary = keyword_plan.primary.phrase

        # Optimize title
        optimized_title = self.llm.optimize_meta_title(
            current_title=current_title,
            primary_keyword=primary,
            topic=topic,
            max_length=60,
        )
        # Ensure markers are present if content changed
        optimized_title = ensure_markers_present(
            original=current_title or "",
            optimized=optimized_title,
            element_type="title",
        )
        meta_elements.append(
            MetaElement(
                element_name="Title Tag",
                current=current_title,
                optimized=optimized_title,
                why_changed=self._explain_title_change(current_title, optimized_title, primary),
            )
        )

        # Optimize meta description
        optimized_desc = self.llm.optimize_meta_description(
            current_description=current_meta_desc,
            primary_keyword=primary,
            topic=topic,
            max_length=160,
        )
        # Ensure markers are present if content changed
        optimized_desc = ensure_markers_present(
            original=current_meta_desc or "",
            optimized=optimized_desc,
            element_type="meta_description",
        )
        meta_elements.append(
            MetaElement(
                element_name="Meta Description",
                current=current_meta_desc,
                optimized=optimized_desc,
                why_changed=self._explain_meta_desc_change(current_meta_desc, optimized_desc, primary),
            )
        )

        # Optimize H1
        clean_title = strip_markers(optimized_title)
        optimized_h1 = self.llm.optimize_h1(
            current_h1=current_h1,
            primary_keyword=primary,
            title=clean_title,
            topic=topic,
        )
        # Ensure markers are present if content changed
        optimized_h1 = ensure_markers_present(
            original=current_h1 or "",
            optimized=optimized_h1,
            element_type="h1",
        )
        meta_elements.append(
            MetaElement(
                element_name="H1",
                current=current_h1,
                optimized=optimized_h1,
                why_changed=self._explain_h1_change(current_h1, optimized_h1, primary),
            )
        )

        return meta_elements

    def _optimize_body_content(
        self,
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        analysis: ContentAnalysis,
    ) -> list[ParagraphBlock]:
        """Optimize body content blocks."""
        if not blocks:
            return []

        optimized_blocks = []
        primary = keyword_plan.primary.phrase
        secondary = [kw.phrase for kw in keyword_plan.secondary]

        # Determine which blocks to optimize
        # Focus on: first paragraph, headings, and a few body paragraphs
        for i, block in enumerate(blocks):
            if block.heading_level == HeadingLevel.H1:
                # H1 is handled separately in meta elements
                optimized_blocks.append(block)
                continue

            # Optimize headings (H2-H6)
            if block.is_heading:
                optimized_text = self._optimize_heading(
                    block.text,
                    primary,
                    secondary,
                    block.heading_level,
                )
                optimized_blocks.append(
                    ParagraphBlock(
                        text=optimized_text,
                        heading_level=block.heading_level,
                        style_name=block.style_name,
                    )
                )
            elif i < 3:  # First few paragraphs get full optimization
                optimized_text = self.llm.rewrite_with_markers(
                    content=block.text,
                    primary_keyword=primary,
                    secondary_keywords=secondary[:3],
                    context=f"This is paragraph {i+1} of the content about {analysis.topic}.",
                )
                # Ensure markers are present if content changed
                optimized_text = ensure_markers_present(
                    original=block.text,
                    optimized=optimized_text,
                    element_type="paragraph",
                )
                optimized_blocks.append(
                    ParagraphBlock(
                        text=optimized_text,
                        heading_level=block.heading_level,
                        style_name=block.style_name,
                    )
                )
            else:
                # Later paragraphs: lighter optimization
                # Check if already contains keywords
                text_lower = block.text.lower()
                has_primary = primary.lower() in text_lower
                has_secondary = any(kw.lower() in text_lower for kw in secondary)

                if not has_primary and not has_secondary and len(block.text) > 100:
                    # Optimize to add keywords
                    optimized_text = self.llm.rewrite_with_markers(
                        content=block.text,
                        primary_keyword=primary,
                        secondary_keywords=secondary[:2],
                        context="Lightly optimize this paragraph to include relevant keywords.",
                    )
                    # Ensure markers are present if content changed
                    optimized_text = ensure_markers_present(
                        original=block.text,
                        optimized=optimized_text,
                        element_type="paragraph",
                    )
                    optimized_blocks.append(
                        ParagraphBlock(
                            text=optimized_text,
                            heading_level=block.heading_level,
                            style_name=block.style_name,
                        )
                    )
                else:
                    # Keep as-is
                    optimized_blocks.append(block)

        return optimized_blocks

    def _optimize_heading(
        self,
        text: str,
        primary: str,
        secondary: list[str],
        level: HeadingLevel,
    ) -> str:
        """Optimize a heading to include keywords where appropriate."""
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
- Mark changes with [[[ADD]]]...[[[ENDADD]]]
- If no natural fit, return the original heading unchanged

Return ONLY the optimized heading."""

        try:
            response = self.llm.client.messages.create(
                model=self.llm.model,
                max_tokens=100,
                system="You are an SEO heading optimizer. Return only the optimized heading.",
                messages=[{"role": "user", "content": prompt}],
            )
            optimized = response.content[0].text.strip()
            # Ensure markers are present if content changed
            return ensure_markers_present(
                original=text,
                optimized=optimized,
                element_type="heading",
            )
        except Exception:
            # On error, return original
            return text

    def _generate_faq(
        self,
        topic: str,
        keyword_plan: KeywordPlan,
        num_items: int,
    ) -> list[FAQItem]:
        """Generate FAQ items."""
        faq_data = self.llm.generate_faq_items(
            topic=topic,
            primary_keyword=keyword_plan.primary.phrase,
            secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
            question_keywords=[kw.phrase for kw in keyword_plan.long_tail_questions],
            num_items=num_items,
        )

        return [
            FAQItem(question=item["question"], answer=item["answer"])
            for item in faq_data
        ]

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
