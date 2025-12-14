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
    mark_block_as_new,
    build_original_sentence_index,
    normalize_sentence,
)
from .models import (
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
from .prioritizer import create_keyword_plan


def ensure_keyword_in_text(text: str, keyword: str, position: str = "start") -> str:
    """
    DETERMINISTIC: Ensure exact keyword phrase appears in text.

    If keyword is already present (case-insensitive), return text unchanged.
    Otherwise, prepend/append the keyword with appropriate formatting.

    This is a PROGRAMMATIC GUARANTEE - if the LLM doesn't include the keyword,
    we inject it ourselves.

    Args:
        text: The text to check/modify.
        keyword: The EXACT keyword phrase that must appear.
        position: Where to inject if missing - "start" or "end".

    Returns:
        Text guaranteed to contain the exact keyword phrase.
    """
    if not text:
        return keyword

    # Check if keyword already present (case-insensitive)
    if keyword.lower() in text.lower():
        return text

    # Keyword missing - inject it
    if position == "start":
        # Prepend keyword with separator
        return f"{keyword}: {text}"
    else:
        # Append keyword
        if text.rstrip().endswith((".", "!", "?")):
            # Insert before final punctuation
            text_stripped = text.rstrip()
            punct = text_stripped[-1]
            return f"{text_stripped[:-1]} - {keyword}{punct}"
        return f"{text} - {keyword}"


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

        # Store content topics and full text for use in LLM calls and validation
        self._content_topics = content_topics
        self._full_original_text = full_text

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

        if optimized_h1_text:
            optimized_blocks = self._replace_h1_in_blocks(
                blocks=optimized_blocks,
                optimized_h1=optimized_h1_text,
            )

        # Step 6: Generate FAQ if requested (Part 10 of 10-Part Framework)
        # Uses optimization plan's faq_keywords for targeted keyword integration
        faq_items = []
        if generate_faq:
            faq_items = self._generate_faq(
                topic=analysis.topic,
                keyword_plan=keyword_plan,
                optimization_plan=optimization_plan,
                num_items=faq_count,
            )

        # Step 7: GUARANTEE all keywords appear in content (especially for manual mode)
        # Check which keywords are missing from the body content
        optimized_blocks = self._ensure_all_keywords_present(
            blocks=optimized_blocks,
            keyword_plan=keyword_plan,
            topic=analysis.topic,
        )

        # Step 8: Compute keyword usage counts in final output
        keyword_usage_counts = self._compute_keyword_usage_counts(
            meta_elements=meta_elements,
            optimized_blocks=optimized_blocks,
            faq_items=faq_items,
            keyword_plan=keyword_plan,
        )

        return OptimizationResult(
            meta_elements=meta_elements,
            optimized_blocks=optimized_blocks,
            faq_items=faq_items,
            primary_keyword=keyword_plan.primary.phrase,
            secondary_keywords=[kw.phrase for kw in keyword_plan.secondary],
            keyword_usage_counts=keyword_usage_counts,
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
        optimized_title_raw = ensure_keyword_in_text(optimized_title_raw, primary, position="start")

        # V2: Sentence-level diff - entire element is all-or-nothing
        # For short elements like titles, if changed at all, wrap entire thing
        if normalize_sentence(current_title or "") != normalize_sentence(optimized_title_raw):
            optimized_title = f"{ADD_START}{optimized_title_raw}{ADD_END}"
        else:
            optimized_title = optimized_title_raw

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
        optimized_desc_raw = ensure_keyword_in_text(optimized_desc_raw, primary, position="start")

        # V2: Sentence-level diff - entire element is all-or-nothing for meta desc
        if normalize_sentence(current_meta_desc or "") != normalize_sentence(optimized_desc_raw):
            optimized_desc = f"{ADD_START}{optimized_desc_raw}{ADD_END}"
        else:
            optimized_desc = optimized_desc_raw

        meta_elements.append(
            MetaElement(
                element_name="Meta Description",
                current=current_meta_desc,
                optimized=optimized_desc,
                why_changed=self._explain_meta_desc_change(current_meta_desc, optimized_desc, primary),
            )
        )

        # Optimize H1 (Tier 2) - use target as hint if available
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
        optimized_h1_raw = ensure_keyword_in_text(optimized_h1_raw, primary, position="start")

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

    def _compute_markers_v2(
        self,
        original: str,
        optimized: str,
        full_original_text: Optional[str] = None,
    ) -> str:
        """
        Token-Level Diff: highlight ONLY new/changed tokens.

        This implements precise token-level diff using SequenceMatcher:
        - Only tokens that are NEW or CHANGED get highlighted (green)
        - Original unchanged tokens remain unhighlighted (black)
        - Prevents the issue where adding a sentence before existing content
          causes the existing content to also be highlighted

        Algorithm:
        1. Tokenize both original and optimized text
        2. Use SequenceMatcher to find exact changes
        3. Only wrap "insert" and "replace" operations in markers

        Args:
            original: Original text block.
            optimized: Optimized text (without markers).
            full_original_text: Full document text for context (used as baseline).

        Returns:
            Text with markers around ONLY new/changed tokens.
        """
        # Compare original block directly against optimized block
        # This ensures only tokens that changed within THIS block get highlighted
        return add_markers_by_diff(original, optimized)

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
                optimized_text = self._optimize_heading(
                    block.text,
                    primary,
                    subheading_keywords,
                    block.heading_level,
                    full_original_text=full_original_text,
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
                )
                # SAFETY: Validate LLM output for blocked industry terms
                rewritten_text, _ = validate_and_fallback(
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
                )
                # SAFETY: Validate LLM output for blocked industry terms
                rewritten_text, _ = validate_and_fallback(
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
                # Check if already contains keywords
                text_lower = block.text.lower()
                has_primary = primary.lower() in text_lower
                has_secondary = any(kw.lower() in text_lower for kw in secondary)

                # Find keywords that still need placement
                unplaced_keywords = [
                    kw for kw in body_keywords
                    if kw.lower() not in keywords_placed_in_body
                ][:2]

                if (not has_primary and not has_secondary and len(block.text) > 100) or unplaced_keywords:
                    # Optimize to add keywords
                    keywords_to_add = unplaced_keywords if unplaced_keywords else secondary[:2]
                    rewritten_text = self.llm.rewrite_with_markers(
                        content=block.text,
                        primary_keyword=primary if primary.lower() not in keywords_placed_in_body else keywords_to_add[0] if keywords_to_add else primary,
                        secondary_keywords=keywords_to_add,
                        context="Optimize this paragraph to naturally include relevant keywords.",
                        content_topics=content_topics,
                        brand_context=getattr(self, '_brand_context', None),
                    )
                    # SAFETY: Validate LLM output for blocked industry terms
                    rewritten_text, _ = validate_and_fallback(
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
                    # Keep as-is but track any keywords present
                    for kw in [primary] + secondary:
                        if kw.lower() in text_lower:
                            keywords_placed_in_body.add(kw.lower())
                    optimized_blocks.append(block)

        # KEYWORD GUARANTEE (Tier 3): Ensure primary keyword is in first ~100 words
        optimized_blocks = self._ensure_primary_in_first_100_words(
            optimized_blocks, primary, analysis.topic
        )

        return optimized_blocks

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

        # Wrap FAQ items in markers - all FAQ content is new and should be highlighted
        return [
            FAQItem(
                question=mark_block_as_new(item["question"]),
                answer=mark_block_as_new(item["answer"])
            )
            for item in valid_faqs
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

    def _compute_keyword_usage_counts(
        self,
        meta_elements: list[MetaElement],
        optimized_blocks: list[ParagraphBlock],
        faq_items: list[FAQItem],
        keyword_plan: KeywordPlan,
    ) -> dict[str, int]:
        """
        Compute keyword usage counts across all final optimized content.

        Counts each keyword occurrence in:
        - Meta elements (title, description, H1)
        - Body content blocks
        - FAQ items

        Args:
            meta_elements: The optimized meta elements.
            optimized_blocks: The optimized content blocks.
            faq_items: The generated FAQ items.
            keyword_plan: The keyword plan with primary and secondary keywords.

        Returns:
            Dictionary mapping keyword phrase to occurrence count.
        """
        # Build full text from all optimized content (strip markers for counting)
        text_parts = []

        # Add meta elements
        for meta in meta_elements:
            text_parts.append(strip_markers(meta.optimized))

        # Add body content
        for block in optimized_blocks:
            text_parts.append(strip_markers(block.text))

        # Add FAQ content
        for faq in faq_items:
            text_parts.append(strip_markers(faq.question))
            text_parts.append(strip_markers(faq.answer))

        full_text = " ".join(text_parts)

        # Count each keyword
        counts = {}

        # Count primary keyword
        counts[keyword_plan.primary.phrase] = count_keyword_in_text(
            full_text, keyword_plan.primary.phrase
        )

        # Count secondary keywords
        for kw in keyword_plan.secondary:
            counts[kw.phrase] = count_keyword_in_text(full_text, kw.phrase)

        return counts

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
