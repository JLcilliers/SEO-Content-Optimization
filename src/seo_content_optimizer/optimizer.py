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
    mark_block_as_new,
    build_original_sentence_index,
    normalize_sentence,
    generate_brand_variations,  # Brand exclusion: generate all variations of brand name
    normalize_brand_in_text,  # Brand exclusion: normalize brand to original spelling
    normalize_paragraph_spacing,  # Paragraph structure: fix "word.Word" patterns
    preprocess_keywords_for_diff,  # Keyword atomic units: preprocess for diff
    postprocess_keywords_from_diff,  # Keyword atomic units: postprocess after diff
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


@dataclass
class KeywordCounts:
    """
    Scoped keyword counts across different document sections.

    This dataclass separates keyword occurrences by location to ensure
    body content is the primary target for keyword satisfaction, not
    meta elements or FAQ sections.
    """
    meta: int = 0       # Count in title + description + H1
    headings: int = 0   # Count in H2+ headings (excluding H1)
    body: int = 0       # Count in body paragraphs (non-heading blocks)
    faq: int = 0        # Count in FAQ questions + answers

    @property
    def total(self) -> int:
        """Total count across all sections."""
        return self.meta + self.headings + self.body + self.faq

    @property
    def body_and_headings(self) -> int:
        """Count in body content (paragraphs + headings, excluding meta/FAQ)."""
        return self.body + self.headings


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

        # Step 7.5: DISTRIBUTE keywords EVENLY across document sections
        # This replaces separate primary/secondary enforcement to prevent clustering.
        # Keywords are distributed across sections based on H2 headings.
        target_count = manual_keywords.target_count if manual_keywords else 6
        optimized_blocks = self._distribute_keywords_across_sections(
            meta_elements=meta_elements,
            blocks=optimized_blocks,
            keyword_plan=keyword_plan,
            topic=analysis.topic,
            primary_target=target_count,
            secondary_target=3,  # Each secondary keyword should appear at least 3 times
        )

        # Step 7.7: ENFORCE BODY KEYWORD INVARIANTS before FAQ generation
        # This is the FINAL SAFETY check to ensure body targets are satisfied.
        # If any keyword is below target in body text, inject deterministically.
        optimized_blocks = self._enforce_body_keyword_invariants(
            blocks=optimized_blocks,
            keyword_plan=keyword_plan,
            topic=analysis.topic,
            primary_target=target_count,
            secondary_target=3,
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
        else:
            print(f"DEBUG Step 7.9: Content check passed, keeping optimized blocks", file=sys.stderr)

        # Debug: Final H1 check before return
        h1_final = [b for b in optimized_blocks if b.heading_level == HeadingLevel.H1]
        print(f"DEBUG FINAL: {len(h1_final)} H1 blocks in final optimized_blocks", file=sys.stderr)
        for i, b in enumerate(h1_final):
            print(f"  FINAL H1 Block {i}: '{b.text[:100]}...'", file=sys.stderr)

        # Step 8: Compute SCOPED keyword usage counts in final output
        # This provides transparency on where keywords appear (body vs meta vs FAQ)
        scoped_keyword_counts = self._compute_keyword_usage_counts(
            meta_elements=meta_elements,
            optimized_blocks=optimized_blocks,
            faq_items=faq_items,
            keyword_plan=keyword_plan,
            primary_target=target_count,
            secondary_target=3,
        )

        # Log scoped counts for transparency
        self._log_scoped_keyword_counts(scoped_keyword_counts)

        # Convert to legacy format for backward compatibility
        keyword_usage_counts = self._get_legacy_keyword_counts(scoped_keyword_counts)

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

        # BRAND NAME NORMALIZATION: Ensure LLM output uses original brand spelling
        # This prevents "Cell-Gate" when original says "CellGate"
        if self._brand_context and self._brand_context.get("name"):
            optimized_title_raw = normalize_brand_in_text(
                optimized_title_raw,
                self._brand_context["name"],
                self._brand_context.get("variations", set()),
            )

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

        # BRAND NAME NORMALIZATION: Ensure LLM output uses original brand spelling
        if self._brand_context and self._brand_context.get("name"):
            optimized_desc_raw = normalize_brand_in_text(
                optimized_desc_raw,
                self._brand_context["name"],
                self._brand_context.get("variations", set()),
            )

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
                    # Keep as-is but track any keywords present
                    for kw in [primary] + secondary:
                        if kw.lower() in text_lower:
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
        primary_target: int = 8,
        secondary_target: int = 3,
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
        scoped_counts[primary] = ScopedKeywordCounts(
            keyword=primary,
            target=primary_target,
            counts=KeywordCounts(
                meta=count_keyword_in_text(meta_text, primary),
                headings=count_keyword_in_text(headings_text, primary),
                body=count_keyword_in_text(body_text, primary),
                faq=count_keyword_in_text(faq_text, primary),
            ),
            is_primary=True,
        )

        # Count secondary keywords
        for kw in keyword_plan.secondary:
            phrase = kw.phrase
            scoped_counts[phrase] = ScopedKeywordCounts(
                keyword=phrase,
                target=secondary_target,
                counts=KeywordCounts(
                    meta=count_keyword_in_text(meta_text, phrase),
                    headings=count_keyword_in_text(headings_text, phrase),
                    body=count_keyword_in_text(body_text, phrase),
                    faq=count_keyword_in_text(faq_text, phrase),
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
            print(f"  Meta:     {counts.counts.meta:3d} (title/desc/H1 - not counted toward target)", file=sys.stderr)
            print(f"  Headings: {counts.counts.headings:3d} (H2+ subheadings)", file=sys.stderr)
            print(f"  FAQ:      {counts.counts.faq:3d} (not counted toward target)", file=sys.stderr)
            print(f"  Total:    {counts.counts.total:3d}", file=sys.stderr)

            if not counts.body_satisfied:
                print(f"  ⚠ Need {counts.body_needed} more in BODY to satisfy target", file=sys.stderr)

        print("\n" + "=" * 60, file=sys.stderr)

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
