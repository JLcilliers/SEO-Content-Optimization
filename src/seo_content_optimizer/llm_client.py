"""
LLM client abstraction for content optimization.

This module provides an interface for calling LLMs (Claude/Anthropic)
to perform SEO-focused content rewriting with change markers.
"""

import os
from typing import Optional

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


class LLMClientError(Exception):
    """Raised when LLM operations fail."""
    pass


# Marker tags for identifying changed content
ADD_START = "[[[ADD]]]"
ADD_END = "[[[ENDADD]]]"


# System prompt for SEO optimization (10-Part SEO Framework)
SEO_SYSTEM_PROMPT = """You are an expert SEO content optimizer following the 10-Part SEO Framework.

10-PART SEO FRAMEWORK - BODY CONTENT OPTIMIZATION:
This follows Parts 5, 7, and 8 of the framework:
- Part 5: Tiered keyword placement (Tiers 3-5, 7 for body content)
- Part 7: Clean, scannable formatting with natural keyword integration
- Part 8: High-quality, user-focused content

TIERED KEYWORD PLACEMENT (Part 5):
- Tier 3: Primary keyword MUST appear in first ~100 words
- Tier 4: Keywords should appear in subheadings (H2/H3) where natural
- Tier 5: Keywords distributed throughout body content
- Tier 7: Keywords in conclusion/summary paragraphs

CRITICAL RULES - MUST FOLLOW:
1. Keep all original text unless modification is necessary for SEO
2. Never delete significant original content - only add or modify
3. Maintain the original tone and style
4. Do not invent facts or make claims not supported by the original content
5. Keep sentences natural and readable - avoid keyword stuffing

KEYWORD PHRASE INTEGRITY - ESSENTIAL:
6. Use keywords as COMPLETE PHRASES - never break them into separate words
7. "payment processing solutions" = use this exact 3-word phrase together
8. Long-tail keywords must appear as one unbroken phrase in a single location
9. NEVER scatter individual words from a keyword phrase across different sentences
10. Use the EXACT keyword phrases provided - no word reordering or substitution

STRICT TOPICAL CONSTRAINTS - VIOLATIONS WILL BE REJECTED:
11. NEVER introduce industries, verticals, or business types not already in the original content
12. NEVER add mentions of: cannabis, hemp, CBD, gambling, adult content, firearms, or other high-risk industries unless they are explicitly in the original
13. NEVER claim the company "specializes in" or "serves" industries not mentioned in the original
14. ONLY use keywords that directly relate to what the page is actually about
15. Do not stuff the company/brand name repeatedly - keep brand mentions reasonable and natural
16. If a keyword doesn't fit the page topic, DO NOT force it in - skip it entirely

OUTPUT FORMAT:
- Return ONLY the optimized text
- Do NOT include any markers, tags, or formatting annotations
- Do NOT include any explanation or commentary
- Just return the clean optimized text"""


class LLMClient:
    """
    Client for LLM-based content optimization.

    Supports Anthropic Claude API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the LLM provider. If None, reads from ANTHROPIC_API_KEY env var.
            model: Model identifier to use.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

        if not self.api_key:
            raise LLMClientError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        if anthropic is None:
            raise LLMClientError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        import httpx
        # Create custom httpx client with appropriate timeouts for serverless
        http_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=30.0),
            follow_redirects=True,
        )
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            http_client=http_client,
        )

    def rewrite_with_markers(
        self,
        content: str,
        primary_keyword: str,
        secondary_keywords: list[str],
        context: str = "",
        max_tokens: int = 4096,
        content_topics: list[str] = None,
        brand_context: dict = None,
        optimization_mode: str = "enhanced",
    ) -> str:
        """
        Rewrite content with SEO optimization, marking all changes.

        Args:
            content: Original content to optimize.
            primary_keyword: Primary keyword to emphasize.
            secondary_keywords: List of secondary keywords to include.
            context: Additional context about the content/business.
            max_tokens: Maximum tokens in response.
            content_topics: List of main topics from original content (for constraints).
            brand_context: Dict with brand name and mention limits.
            optimization_mode: "minimal" for insert-only or "enhanced" for density mode.

        Returns:
            Optimized content with [[[ADD]]]...[[[ENDADD]]] markers.
        """
        user_prompt = self._build_rewrite_prompt(
            content, primary_keyword, secondary_keywords, context, content_topics, brand_context,
            optimization_mode=optimization_mode
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=SEO_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            return response.content[0].text
        except Exception as e:
            raise LLMClientError(f"LLM API call failed: {e}")

    def optimize_meta_title(
        self,
        current_title: Optional[str],
        primary_keyword: str,
        topic: str,
        max_length: int = 60,
        target_hint: Optional[str] = None,
    ) -> str:
        """
        Generate an optimized meta title.

        Args:
            current_title: Current title if any.
            primary_keyword: Primary keyword to include.
            topic: Content topic for context.
            max_length: Maximum title length.
            target_hint: Pre-computed target from optimization plan (used as guidance).

        Returns:
            Optimized title with markers for changed portions.
        """
        # Build hint guidance if available
        hint_guidance = ""
        if target_hint:
            hint_guidance = f"""
OPTIMIZATION HINT (use as guidance, you may improve upon it):
{target_hint}
"""

        prompt = f"""Create an SEO-optimized title tag for a page about: {topic}

Primary keyword (MUST include near beginning): {primary_keyword}
Current title: {current_title or "None"}
Maximum length: {max_length} characters
{hint_guidance}
CRITICAL REQUIREMENTS (Part 5 of SEO Framework - Tier 1 Placement):
- The EXACT phrase "{primary_keyword}" MUST appear in the title
- Place the primary keyword within the first 30 characters if possible
- Make it compelling and click-worthy
- Stay under {max_length} characters
- If there's a current title, use it as inspiration but optimize for SEO
- Do NOT split or paraphrase the keyword - use the EXACT phrase

Return ONLY the optimized title as plain text, nothing else. No markers, no explanations."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                system="You are an SEO title optimization expert. Return only the optimized title as plain text. The primary keyword phrase MUST appear exactly as provided.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise LLMClientError(f"Title optimization failed: {e}")

    def optimize_meta_description(
        self,
        current_description: Optional[str],
        primary_keyword: str,
        topic: str,
        max_length: int = 160,
        target_hint: Optional[str] = None,
    ) -> str:
        """
        Generate an optimized meta description.

        Args:
            current_description: Current description if any.
            primary_keyword: Primary keyword to include.
            topic: Content topic for context.
            max_length: Maximum description length.
            target_hint: Pre-computed target from optimization plan (used as guidance).

        Returns:
            Optimized description with markers.
        """
        # Build hint guidance if available
        hint_guidance = ""
        if target_hint:
            hint_guidance = f"""
OPTIMIZATION HINT (use as guidance, you may improve upon it):
{target_hint}
"""

        prompt = f"""Create an SEO-optimized meta description for a page about: {topic}

Primary keyword (MUST include naturally): {primary_keyword}
Current description: {current_description or "None"}
Maximum length: {max_length} characters
{hint_guidance}
CRITICAL REQUIREMENTS (Part 5 of SEO Framework - CTR Optimization):
- The EXACT phrase "{primary_keyword}" MUST appear in the description
- Use active voice
- Include a clear call-to-action (e.g., "Learn more", "Get a quote", "Discover how")
- Be compelling and informative
- Stay under {max_length} characters
- Do NOT split or paraphrase the keyword - use the EXACT phrase

Return ONLY the optimized description as plain text, nothing else. No markers, no explanations."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system="You are an SEO meta description expert. Return only the optimized description as plain text. The primary keyword phrase MUST appear exactly as provided.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise LLMClientError(f"Meta description optimization failed: {e}")

    def optimize_h1(
        self,
        current_h1: Optional[str],
        primary_keyword: str,
        title: str,
        topic: str,
        target_hint: Optional[str] = None,
    ) -> str:
        """
        Generate an optimized H1 heading.

        Args:
            current_h1: Current H1 if any.
            primary_keyword: Primary keyword to include.
            title: The page title (H1 should complement, not duplicate).
            topic: Content topic.
            target_hint: Pre-computed target from optimization plan (used as guidance).

        Returns:
            Optimized H1 with markers.
        """
        # Build hint guidance if available
        hint_guidance = ""
        if target_hint:
            hint_guidance = f"""
OPTIMIZATION HINT (use as guidance, you may improve upon it):
{target_hint}
"""

        prompt = f"""Create an SEO-optimized H1 heading for a page about: {topic}

Primary keyword (MUST include): {primary_keyword}
Current H1: {current_h1 or "None"}
Page title (H1 should complement, not duplicate exactly): {title}
{hint_guidance}
CRITICAL REQUIREMENTS (Part 5 of SEO Framework - Tier 2 Placement):
- The EXACT phrase "{primary_keyword}" MUST appear in the H1
- More descriptive than the title
- Not an exact copy of the title
- Clear and engaging
- Do NOT split or paraphrase the keyword - use the EXACT phrase

Return ONLY the optimized H1 as plain text, nothing else. No markers, no explanations."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                system="You are an SEO heading optimization expert. Return only the optimized H1 as plain text. The primary keyword phrase MUST appear exactly as provided.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise LLMClientError(f"H1 optimization failed: {e}")

    def generate_faq_items(
        self,
        topic: str,
        primary_keyword: str,
        secondary_keywords: list[str],
        question_keywords: list[str],
        num_items: int = 4,
        content_topics: list[str] = None,
        brand_context: dict = None,
        faq_keywords: list[str] = None,
        planned_questions: list[str] = None,
    ) -> list[dict[str, str]]:
        """
        Generate FAQ items based on keywords.

        Part 10 of 10-Part SEO Framework: FAQ generation for featured snippets.

        Args:
            topic: Content topic.
            primary_keyword: Primary keyword.
            secondary_keywords: Secondary keywords to include.
            question_keywords: Question-form keywords to address.
            num_items: Number of FAQ items to generate.
            content_topics: List of main topics from the original content (for constraints).
            brand_context: Dict with brand name and mention limits.
            faq_keywords: Specific keywords to integrate into FAQ answers (from placement plan).
            planned_questions: Pre-planned FAQ questions from optimization plan.

        Returns:
            List of dicts with 'question' and 'answer' keys.
        """
        questions_list = "\n".join(f"- {q}" for q in question_keywords) if question_keywords else "None provided"
        keywords_list = ", ".join([primary_keyword] + secondary_keywords[:3])

        # Build topic constraint if available
        topic_constraint = ""
        if content_topics:
            topic_constraint = f"""
ALLOWED TOPICS (FAQs must ONLY address these topics from the original content):
{', '.join(content_topics[:15])}

"""

        # Build brand control instruction
        brand_instruction = ""
        if brand_context and brand_context.get("name"):
            brand_name = brand_context["name"]
            brand_instruction = f"""
BRAND NAME: "{brand_name}"
- Mention the brand sparingly and naturally
- Do NOT stuff the brand name into every answer
"""

        # Build FAQ keyword integration instruction (Part 10 of 10-Part Framework)
        faq_keyword_instruction = ""
        if faq_keywords:
            faq_keyword_instruction = f"""
KEYWORDS TO INTEGRATE INTO FAQ ANSWERS (Part 10 of 10-Part SEO Framework):
The following keywords should be naturally incorporated into FAQ answers:
{', '.join(faq_keywords)}

IMPORTANT: Use these EXACT phrases in your answers where they fit naturally.
Each keyword should appear at least once across all FAQ items.
"""

        # Build planned questions instruction
        planned_questions_instruction = ""
        if planned_questions:
            planned_questions_instruction = f"""
SUGGESTED QUESTIONS (based on content audit):
The following questions address gaps identified in the original content:
{chr(10).join(f'- {q}' for q in planned_questions[:num_items])}

Prioritize these questions if they fit the topic, or create similar questions addressing these themes.
"""

        prompt = f"""Generate {num_items} FAQ items for a page about: {topic}

ALLOWED KEYWORDS (use ONLY these): {keywords_list}
{brand_instruction}
{faq_keyword_instruction}
{planned_questions_instruction}
Question keywords to address (if relevant):
{questions_list}
{topic_constraint}
STRICT REQUIREMENTS - READ CAREFULLY:
1. Each question must be a common user question DIRECTLY related to the page topic: "{topic}"
2. Answers should be 2-4 sentences, helpful and informative
3. ONLY use keywords from the ALLOWED KEYWORDS list above - no other keywords
4. Do NOT make up specific facts, prices, or claims not verifiable
5. Integrate the FAQ keywords naturally into answers where appropriate

ABSOLUTE RESTRICTIONS - VIOLATIONS WILL BE REJECTED:
- ONLY use keywords from the EXACT list provided above - introducing other keywords is forbidden
- NEVER create FAQs about industries or verticals not mentioned in the page topic
- NEVER introduce new business types like "hair salon", "restaurant", "CBD", "hemp", "cannabis", "gambling", "spa", "merchant services", etc.
- NEVER mention "high-risk" industries, "merchant accounts", or industry-specific payment processing
- NEVER claim the company specializes in or serves markets not explicitly in "{topic}"
- FAQs must be DIRECTLY and ONLY about: "{topic}"
- If the page is about "X vs Y comparison", FAQs should ONLY be about that comparison
- Do NOT create generic industry FAQs - be SPECIFIC to this page's exact content

Return in this exact format (plain text, no markers):
Q: Question text here?
A: Answer text here.

Q: Next question?
A: Next answer.

(Continue for all {num_items} items)"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system="""You are an SEO FAQ content expert. Generate helpful FAQ items that are STRICTLY on-topic.

Part 10 of 10-Part SEO Framework: FAQ Generation for Featured Snippets
- FAQs help capture "People Also Ask" featured snippets
- Each answer should naturally incorporate assigned keywords
- Questions should address common user queries about the specific topic

CRITICAL RULES - MUST FOLLOW:
1. You must NEVER introduce industries, verticals, or business types not in the original topic
2. You must ONLY use keywords from the exact list provided - no other keywords
3. If the page is about "Payment Processing Comparison", FAQs should be about that comparison ONLY
4. NEVER mention: restaurants, salons, spas, CBD, hemp, cannabis, gambling, high-risk merchants, or ANY specific industry
5. Keep answers focused on the exact topic provided
6. Integrate the provided FAQ keywords naturally into answers

Only generate FAQs that someone reading the specific page would actually ask about THAT SPECIFIC topic.""",
                messages=[{"role": "user", "content": prompt}],
            )

            return self._parse_faq_response(response.content[0].text)
        except Exception as e:
            raise LLMClientError(f"FAQ generation failed: {e}")

    def analyze_content_for_optimization(
        self,
        content_summary: str,
        keywords_sample: list[str],
    ) -> dict:
        """
        Use LLM to analyze content and suggest optimization approach.

        Args:
            content_summary: Summary of the content.
            keywords_sample: Sample of available keywords.

        Returns:
            Dict with 'summary', 'intent', 'recommended_keywords' keys.
        """
        prompt = f"""Analyze this content for SEO optimization:

Content Summary:
{content_summary}

Available Keywords (sample):
{', '.join(keywords_sample[:15])}

Provide:
1. A 1-2 sentence summary of what this page is about
2. The primary intent: "informational" or "transactional"
3. Which 5-7 keywords from the list seem most aligned with this content

Respond in this exact format:
SUMMARY: <your summary>
INTENT: <informational or transactional>
RECOMMENDED: <comma-separated list of recommended keywords>"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system="You are an SEO analyst. Provide concise analysis in the exact format requested.",
                messages=[{"role": "user", "content": prompt}],
            )

            return self._parse_analysis_response(response.content[0].text)
        except Exception as e:
            raise LLMClientError(f"Content analysis failed: {e}")

    def _build_rewrite_prompt(
        self,
        content: str,
        primary_keyword: str,
        secondary_keywords: list[str],
        context: str,
        content_topics: list[str] = None,
        brand_context: dict = None,
        optimization_mode: str = "enhanced",
    ) -> str:
        """Build the rewrite prompt for content optimization.

        Args:
            optimization_mode: "minimal" for insert-only (each keyword max once)
                              "enhanced" for density-based distribution
        """
        secondary_list = ", ".join(secondary_keywords) if secondary_keywords else "None"

        # Build topic constraint
        topic_constraint = ""
        if content_topics:
            topic_constraint = f"""
CONTENT TOPICS (only optimize using concepts related to these):
{', '.join(content_topics[:10])}
"""

        # Build brand control instruction
        brand_instruction = ""
        if brand_context and brand_context.get("name"):
            brand_name = brand_context["name"]
            original_count = brand_context.get("original_count", 0)
            max_extra = brand_context.get("max_extra_mentions", 2)
            brand_instruction = f"""
BRAND NAME CONTROL (STRICT):
- The brand name is: "{brand_name}"
- It already appears {original_count} times in the original content
- You may add the brand name at most {max_extra} additional times
- Do NOT stuff the brand name repeatedly
- If adding the brand name would sound unnatural, DO NOT add it
"""

        # BUILD MODE-SPECIFIC INSTRUCTIONS
        if optimization_mode == "minimal":
            # INSERT-ONLY MODE: Minimal changes, each keyword at most once
            return f"""Perform MINIMAL SEO optimization on the following content.

PRIMARY KEYWORD: {primary_keyword}
SECONDARY KEYWORDS: {secondary_list}

ADDITIONAL CONTEXT: {context or "None provided"}
{topic_constraint}{brand_instruction}
ORIGINAL CONTENT:
{content}

INSERT-ONLY MODE INSTRUCTIONS (CRITICAL):
This is INSERT-ONLY mode. Make MINIMAL changes to the content.

STRICT RULES:
1. Add each keyword AT MOST ONCE in the entire paragraph
2. If a keyword already exists in the original content, DO NOT add it again
3. Make the smallest possible change to include missing keywords
4. Prefer adding a short phrase rather than rewriting sentences
5. Preserve 95%+ of the original text unchanged
6. DO NOT distribute keywords - one occurrence per keyword is the MAXIMUM
7. DO NOT expand or lengthen the content beyond what's needed for keyword insertion

KEYWORD INSERTION PRIORITY:
- Only insert keywords that are completely missing from the original
- Insert at the most natural location (near a related concept)
- Use the EXACT phrase - never split keywords

ABSOLUTE RESTRICTIONS:
- ONLY use keywords from the EXACT list provided above
- NEVER add multiple occurrences of the same keyword
- NEVER introduce industries or topics not in the original
- Keep the paragraph structure exactly the same
- If a keyword doesn't fit naturally, SKIP IT - do not force it

Return ONLY the optimized content as plain text. No markers, no explanations."""

        else:
            # ENHANCED MODE: Full density-based optimization (original behavior)
            return f"""Optimize the following content for SEO following the 10-Part SEO Framework.

PRIMARY KEYWORD (MUST appear in first ~100 words - Tier 3): {primary_keyword}

SECONDARY KEYWORDS (use ONLY if they fit naturally): {secondary_list}

ADDITIONAL CONTEXT: {context or "None provided"}
{topic_constraint}{brand_instruction}
ORIGINAL CONTENT:
{content}

10-PART SEO FRAMEWORK - TIERED KEYWORD PLACEMENT (Part 5):
1. PRIMARY KEYWORD (Tier 3): MUST appear in first ~100 words of body content
2. SUBHEADINGS (Tier 4): Place keywords in H2/H3 headings where natural
3. BODY CONTENT (Tier 5): Distribute keywords throughout body paragraphs
4. CONCLUSION (Tier 7): Include primary keyword in conclusion/summary

OPTIMIZATION INSTRUCTIONS:
1. Preserve all original content and meaning
2. GUARANTEE the primary keyword appears in the first paragraph (Tier 3 requirement)
3. Weave in secondary keywords ONLY where they fit naturally
4. Add clarifying phrases or sentences if they help include keywords naturally
5. Keep the same paragraph structure
6. Do not be repetitive or stuffed - maintain natural reading flow

CRITICAL PHRASE USAGE RULES:
7. Use each keyword as a COMPLETE PHRASE - never split it into individual words
8. "credit card processing" means the FULL phrase, not "credit" and "card" and "processing" separately
9. Long-tail keywords like "best payment processor for small business" must appear as one complete phrase
10. Do NOT scatter keyword words throughout unrelated sentences - use the full phrase in one place

ABSOLUTE RESTRICTIONS - VIOLATIONS WILL BE REJECTED:
- ONLY use keywords from the EXACT list provided above - no other keywords allowed
- NEVER introduce industries, verticals, or business types not in the original content
- NEVER add references to: cannabis, hemp, CBD, THC, gambling, casinos, salons, spas, restaurants, firearms, adult content, or ANY other industries unless they are ALREADY in the original content
- NEVER claim the company "specializes in" or "serves" markets not mentioned in the original
- NEVER invent or add industry-specific terms like "merchant services," "high-risk processing," "POS systems for [industry]" unless these EXACT phrases are in the original
- If a keyword doesn't fit naturally or introduces off-topic concepts, DO NOT use it - skip it entirely
- Keep brand/company name mentions within the limits specified above

Return ONLY the optimized content as plain text. No markers, no explanations, no formatting annotations."""

    def _parse_faq_response(self, response: str) -> list[dict[str, str]]:
        """Parse FAQ response into structured format."""
        faqs = []
        lines = response.strip().split("\n")

        current_q = None
        current_a = None

        for line in lines:
            line = line.strip()
            if line.startswith("Q:"):
                if current_q and current_a:
                    faqs.append({"question": current_q, "answer": current_a})
                current_q = line[2:].strip()
                current_a = None
            elif line.startswith("A:"):
                current_a = line[2:].strip()

        # Add last item
        if current_q and current_a:
            faqs.append({"question": current_q, "answer": current_a})

        return faqs

    def _parse_analysis_response(self, response: str) -> dict:
        """Parse content analysis response."""
        result = {
            "summary": "",
            "intent": "informational",
            "recommended_keywords": [],
        }

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("SUMMARY:"):
                result["summary"] = line[8:].strip()
            elif line.startswith("INTENT:"):
                intent = line[7:].strip().lower()
                if intent in ("informational", "transactional"):
                    result["intent"] = intent
            elif line.startswith("RECOMMENDED:"):
                keywords = line[12:].strip()
                result["recommended_keywords"] = [k.strip() for k in keywords.split(",")]

        return result


def create_llm_client(
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        api_key: Optional API key. If None, uses environment variable.
        model: Model to use.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(api_key=api_key, model=model)


def strip_markers(text: str) -> str:
    """
    Remove all ADD markers from text, returning clean text.

    Args:
        text: Text with markers.

    Returns:
        Clean text without markers.
    """
    return text.replace(ADD_START, "").replace(ADD_END, "")


def ensure_markers_present(
    original: str,
    optimized: str,
    element_type: str = "content",
) -> str:
    """
    Ensure that optimized content has markers when it differs from original.

    If the LLM made changes but didn't include markers, this function wraps
    the entire optimized content in markers to ensure visibility.

    Args:
        original: Original content before optimization.
        optimized: LLM-optimized content (may or may not have markers).
        element_type: Type of element being optimized (for logging).

    Returns:
        Content with markers guaranteed (either from LLM or added by fallback).
    """
    # If markers are already present, return as-is
    if has_markers(optimized):
        return optimized

    # If content is identical to original, no need for markers
    clean_optimized = optimized.strip()
    clean_original = original.strip() if original else ""

    if clean_optimized == clean_original:
        return optimized

    # LLM made changes but didn't add markers - wrap the entire output
    # This is a fallback to ensure changes are visible
    return f"{ADD_START}{optimized}{ADD_END}"


def has_markers(text: str) -> bool:
    """Check if text contains any ADD markers."""
    return ADD_START in text or ADD_END in text


def ensure_contains_phrase(
    text: str,
    phrase: str,
    fallback_position: str = "start",
) -> str:
    """
    Ensure the text contains the exact keyword phrase.

    If the phrase is not present (case-insensitive), inject it at the specified
    position with ADD markers. This is a programmatic guarantee for critical
    keywords like the primary keyword in Title, Meta Description, and H1.

    Args:
        text: The text to check/modify.
        phrase: The keyword phrase that must be present.
        fallback_position: Where to inject if missing - "start" or "end".

    Returns:
        Text guaranteed to contain the phrase.
    """
    if not text or not phrase:
        return text

    # Clean the text of markers for checking
    clean_text = strip_markers(text)

    # Case-insensitive check for phrase presence
    if phrase.lower() in clean_text.lower():
        return text  # Phrase already present, no modification needed

    # Phrase is missing - inject it with markers
    injection = f"{ADD_START}{phrase}{ADD_END}"

    if fallback_position == "start":
        # Prepend phrase followed by separator
        if has_markers(text):
            # Already has markers, just prepend
            return f"{injection}: {text}"
        else:
            # No markers, wrap the modified result
            return f"{injection}: {text}"
    else:
        # Append phrase with separator
        if text.endswith("."):
            return f"{text[:-1]} - {injection}."
        elif text.endswith(":") or text.endswith("|"):
            return f"{text} {injection}"
        else:
            return f"{text} | {injection}"
