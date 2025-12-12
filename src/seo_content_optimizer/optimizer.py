"""
SEO optimization orchestration module.

This module coordinates the optimization process:
- Applies SEO rules to meta elements and content
- Uses LLM for intelligent rewriting
- Generates FAQ sections
- Produces structured optimization results

IMPORTANT: All keywords are filtered for topical relevance BEFORE optimization
to prevent injection of off-topic industries or spammy content.
"""

from typing import Optional, Union

from .analysis import ContentAnalysis, analyze_content
from .content_sources import convert_page_meta_to_blocks
from .keyword_filter import (
    filter_keywords_for_content,
    get_content_topics,
    log_filter_results,
)
from .llm_client import (
    LLMClient,
    create_llm_client,
    strip_markers,
)
from .diff_markers import (
    MARK_END as ADD_END,
    MARK_START as ADD_START,
    compute_markers,
    compute_h1_markers,
    filter_markers_by_keywords,
    inject_phrase_with_markers,
    mark_block_as_new,
)
from .models import (
    DocxContent,
    FAQItem,
    HeadingLevel,
    Keyword,
    KeywordPlan,
    ManualKeywordsConfig,
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
        manual_keywords: Optional[ManualKeywordsConfig] = None,
        generate_faq: bool = True,
        faq_count: int = 4,
        max_secondary: int = 5,
    ) -> OptimizationResult:
        """
        Perform full SEO optimization on content.

        Args:
            content: Content to optimize (from URL or DOCX).
            keywords: List of available keywords (used only if manual_keywords is None).
            manual_keywords: Manual keyword selection config. If provided, bypasses
                            automatic keyword selection and uses user-specified keywords
                            directly without filtering or scoring.
            generate_faq: Whether to generate FAQ section.
            faq_count: Number of FAQ items to generate.
            max_secondary: Maximum secondary keywords.

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

        # Step 0.6: Brand name detection and control
        brand_name = self._detect_brand_name(content)
        brand_count = full_text.lower().count(brand_name.lower()) if brand_name else 0
        self._brand_context = {
            "name": brand_name,
            "original_count": brand_count,
            "max_extra_mentions": min(3, max(1, brand_count)),
        }

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

        # Store content topics for use in LLM calls
        self._content_topics = content_topics

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
            full_original_text=full_text,
        )

        # Step 5: Optimize body content
        optimized_blocks = self._optimize_body_content(
            blocks=blocks,
            keyword_plan=keyword_plan,
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

        if optimized_h1_text:
            optimized_blocks = self._replace_h1_in_blocks(
                blocks=optimized_blocks,
                optimized_h1=optimized_h1_text,
            )

        # Step 6: Generate FAQ if requested
        faq_items = []
        if generate_faq:
            faq_items = self._generate_faq(
                topic=analysis.topic,
                keyword_plan=keyword_plan,
                num_items=faq_count,
            )

        # Step 7: GUARANTEE all keywords appear in content (especially for manual mode)
        # Check which keywords are missing from the body content
        optimized_blocks = self._ensure_all_keywords_present(
            blocks=optimized_blocks,
            keyword_plan=keyword_plan,
            topic=analysis.topic,
        )

        return OptimizationResult(
            meta_elements=meta_elements,
            optimized_blocks=optimized_blocks,
            faq_items=faq_items,
            primary_keyword=keyword_plan.primary.phrase,
            secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
        )

    def _build_keyword_plan_from_manual(
        self,
        manual_keywords: ManualKeywordsConfig,
    ) -> KeywordPlan:
        """
        Build a KeywordPlan directly from manual keyword configuration.

        This bypasses all automatic keyword selection, filtering, and scoring.
        User-specified keywords are used exactly as provided.

        Args:
            manual_keywords: The manual keyword configuration from user input.

        Returns:
            KeywordPlan with user-specified keywords.
        """
        # Create primary keyword - never marked as brand, no filtering
        primary = Keyword(
            phrase=manual_keywords.primary.strip(),
            is_brand=False,  # Manual keywords are never treated as brand
        )

        # Create secondary keywords from user-provided list
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

    def _optimize_meta_elements(
        self,
        current_title: Optional[str],
        current_meta_desc: Optional[str],
        current_h1: Optional[str],
        keyword_plan: KeywordPlan,
        topic: str,
        full_original_text: Optional[str] = None,
    ) -> list[MetaElement]:
        """Optimize title, meta description, and H1."""
        meta_elements = []
        primary = keyword_plan.primary.phrase
        secondary = [kw.phrase for kw in keyword_plan.secondary]
        all_keywords = [primary] + secondary

        # Optimize title
        optimized_title_raw = self.llm.optimize_meta_title(
            current_title=current_title,
            primary_keyword=primary,
            topic=topic,
            max_length=60,
        )
        # Use diff-based markers and filter to only keep SEO-relevant changes
        optimized_title = self._compute_filtered_markers(
            current_title or "", optimized_title_raw, all_keywords, full_original_text=full_original_text
        )
        # PROGRAMMATIC GUARANTEE: Primary keyword MUST appear in title
        optimized_title = self._ensure_phrase_present(
            optimized_title, primary, current_title or "", position="start"
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
        optimized_desc_raw = self.llm.optimize_meta_description(
            current_description=current_meta_desc,
            primary_keyword=primary,
            topic=topic,
            max_length=160,
        )
        # Use diff-based markers and filter to only keep SEO-relevant changes
        optimized_desc = self._compute_filtered_markers(
            current_meta_desc or "", optimized_desc_raw, all_keywords, full_original_text=full_original_text
        )
        # PROGRAMMATIC GUARANTEE: Primary keyword MUST appear in meta description
        optimized_desc = self._ensure_phrase_present(
            optimized_desc, primary, current_meta_desc or "", position="start"
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
        optimized_h1_raw = self.llm.optimize_h1(
            current_h1=current_h1,
            primary_keyword=primary,
            title=clean_title,
            topic=topic,
        )
        # Use special H1 marker handling:
        # If H1 changed at all, wrap ENTIRE H1 in markers (not just changed phrases)
        optimized_h1 = compute_h1_markers(current_h1 or "", optimized_h1_raw)
        # PROGRAMMATIC GUARANTEE: Primary keyword MUST appear in H1
        optimized_h1 = self._ensure_phrase_present(
            optimized_h1, primary, current_h1 or "", position="start"
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

    def _ensure_phrase_present(
        self,
        marked_text: str,
        phrase: str,
        original_text: str,
        position: str = "start",
    ) -> str:
        """
        Ensure the phrase is present in marked text.

        If the phrase already exists in the original, it should be unmarked.
        If the phrase is added, it should be marked.

        Args:
            marked_text: Text with diff markers.
            phrase: Phrase to ensure is present.
            original_text: Original text before optimization.
            position: Where to inject if missing - "start" or "end".

        Returns:
            Text with phrase guaranteed to be present.
        """
        # Check if phrase is already present (case-insensitive)
        clean_text = strip_markers(marked_text)
        if phrase.lower() in clean_text.lower():
            return marked_text  # Already present

        # Check if phrase was in original (shouldn't mark if it was)
        if phrase.lower() in original_text.lower():
            # Phrase was in original but somehow removed - add it back unmarked
            if position == "start":
                return f"{phrase}: {marked_text}"
            else:
                if marked_text.endswith("."):
                    return f"{marked_text[:-1]} - {phrase}."
                return f"{marked_text} | {phrase}"

        # Phrase is new - add with markers
        return inject_phrase_with_markers(marked_text, phrase, position)

    def _compute_filtered_markers(
        self,
        original: str,
        optimized: str,
        keywords: list[str],
        full_original_text: Optional[str] = None,
    ) -> str:
        """
        Compute diff markers and filter to only keep SEO-relevant changes.

        This ensures only changes containing keywords are highlighted, preventing
        random punctuation or minor word changes from being marked.

        Args:
            original: Original text.
            optimized: Optimized text (without markers).
            keywords: List of SEO keywords to filter by.
            full_original_text: Full document text for context.

        Returns:
            Text with markers only around keyword-containing changes.
        """
        # First compute all markers
        marked_text = compute_markers(original, optimized, full_original_text=full_original_text)
        # Then filter to only keep markers that contain keywords
        return filter_markers_by_keywords(marked_text, keywords)

    def _optimize_body_content(
        self,
        blocks: list[ParagraphBlock],
        keyword_plan: KeywordPlan,
        analysis: ContentAnalysis,
        full_original_text: Optional[str] = None,
    ) -> list[ParagraphBlock]:
        """Optimize body content blocks."""
        if not blocks:
            return []

        optimized_blocks = []
        primary = keyword_plan.primary.phrase
        secondary = [kw.phrase for kw in keyword_plan.secondary]
        # Build list of keywords for filtering markers
        all_keywords = [primary] + secondary

        # Get content topics for constraints (set during optimize())
        content_topics = getattr(self, '_content_topics', None)

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
                    full_original_text=full_original_text,
                    all_keywords=all_keywords,
                )
                optimized_blocks.append(
                    ParagraphBlock(
                        text=optimized_text,
                        heading_level=block.heading_level,
                        style_name=block.style_name,
                    )
                )
            elif i < 3:  # First few paragraphs get full optimization
                rewritten_text = self.llm.rewrite_with_markers(
                    content=block.text,
                    primary_keyword=primary,
                    secondary_keywords=secondary[:3],
                    context=f"This is paragraph {i+1} of the content about {analysis.topic}.",
                    content_topics=content_topics,
                    brand_context=getattr(self, '_brand_context', None),
                )
                # Apply diff-based markers and filter to only keep SEO-relevant changes
                optimized_text = self._compute_filtered_markers(
                    block.text, strip_markers(rewritten_text), all_keywords, full_original_text=full_original_text
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
                    rewritten_text = self.llm.rewrite_with_markers(
                        content=block.text,
                        primary_keyword=primary,
                        secondary_keywords=secondary[:2],
                        context="Lightly optimize this paragraph to include relevant keywords.",
                        content_topics=content_topics,
                        brand_context=getattr(self, '_brand_context', None),
                    )
                    # Apply diff-based markers and filter to only keep SEO-relevant changes
                    optimized_text = self._compute_filtered_markers(
                        block.text, strip_markers(rewritten_text), all_keywords, full_original_text=full_original_text
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
        full_original_text: Optional[str] = None,
        all_keywords: Optional[list[str]] = None,
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
            # Apply diff-based markers and filter to only keep SEO-relevant changes
            keywords = all_keywords or [primary] + secondary
            return self._compute_filtered_markers(
                text, strip_markers(rewritten), keywords, full_original_text=full_original_text
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
        result = []
        h1_replaced = False

        for block in blocks:
            if block.heading_level == HeadingLevel.H1 and not h1_replaced:
                # Replace with optimized H1
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

            # Split into sentences and filter
            sentences = []
            for sentence in result.replace("\n", " ").split(". "):
                sentence = sentence.strip()
                if sentence and not sentence.startswith("-") and not sentence[0].isdigit():
                    if not sentence.endswith("."):
                        sentence += "."
                    sentences.append(sentence)

            return sentences
        except Exception:
            # Fallback: create simple sentences manually
            sentences = []
            for kw in missing_keywords:
                sentences.append(f"Learn more about {kw} and how it relates to {topic}.")
            return sentences

    def _generate_faq(
        self,
        topic: str,
        keyword_plan: KeywordPlan,
        num_items: int,
    ) -> list[FAQItem]:
        """Generate FAQ items."""
        # Get content topics for constraints (set during optimize())
        content_topics = getattr(self, '_content_topics', None)
        brand_context = getattr(self, '_brand_context', None)

        faq_data = self.llm.generate_faq_items(
            topic=topic,
            primary_keyword=keyword_plan.primary.phrase,
            secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
            question_keywords=[kw.phrase for kw in keyword_plan.long_tail_questions],
            num_items=num_items,
            content_topics=content_topics,
            brand_context=brand_context,
        )

        # Wrap FAQ items in markers - all FAQ content is new and should be highlighted
        return [
            FAQItem(
                question=mark_block_as_new(item["question"]),
                answer=mark_block_as_new(item["answer"])
            )
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

    def _detect_brand_name(
        self,
        content: Union[PageMeta, DocxContent],
    ) -> str:
        """
        Detect the brand/company name from the content.

        Uses multiple signals:
        - URL domain (most reliable)
        - H1 heading
        - Title tag

        Args:
            content: The content being optimized.

        Returns:
            Detected brand name or empty string.
        """
        import re

        brand_name = ""

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
                    # Capitalize as brand name
                    brand_name = domain.title()
                except Exception:
                    pass

            # Fallback: look at H1 or title
            if not brand_name:
                # Look for common brand patterns in H1/title
                h1 = content.h1 or ""
                title = content.title or ""

                # Look for capitalized words at start of title
                for text in [title, h1]:
                    words = text.split()
                    if words:
                        # First capitalized word is often the brand
                        first_word = words[0].strip(":|–-")
                        if first_word and first_word[0].isupper():
                            brand_name = first_word
                            break

        elif isinstance(content, DocxContent):
            # For DOCX, try H1
            h1 = content.h1 or ""
            words = h1.split()
            if words:
                first_word = words[0].strip(":|–-")
                if first_word and first_word[0].isupper():
                    brand_name = first_word

        return brand_name


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
