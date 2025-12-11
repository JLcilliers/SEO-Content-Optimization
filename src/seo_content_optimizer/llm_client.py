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


# System prompt for SEO optimization
SEO_SYSTEM_PROMPT = """You are an expert SEO content optimizer. Your task is to enhance content for search engine optimization while maintaining readability and the original message.

CRITICAL RULES:
1. Keep all original text unless modification is necessary for SEO
2. Mark ALL new or changed text with [[[ADD]]] and [[[ENDADD]]] markers
3. Never delete significant original content - only add or modify
4. Maintain the original tone and style
5. Do not invent facts or make claims not supported by the original content
6. Keep sentences natural and readable - avoid keyword stuffing

MARKER FORMAT:
- Wrap only the NEW or CHANGED portions with markers
- Example: "We offer [[[ADD]]]comprehensive PTO insurance[[[ENDADD]]] coverage."
- For new sentences: "[[[ADD]]]This is a completely new sentence.[[[ENDADD]]]"
- Multiple changes in one text block are fine: "Our [[[ADD]]]professional[[[ENDADD]]] team provides [[[ADD]]]expert insurance solutions[[[ENDADD]]] for your needs."

Do NOT include any explanation or commentary - return ONLY the optimized text with markers."""


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
    ) -> str:
        """
        Rewrite content with SEO optimization, marking all changes.

        Args:
            content: Original content to optimize.
            primary_keyword: Primary keyword to emphasize.
            secondary_keywords: List of secondary keywords to include.
            context: Additional context about the content/business.
            max_tokens: Maximum tokens in response.

        Returns:
            Optimized content with [[[ADD]]]...[[[ENDADD]]] markers.
        """
        user_prompt = self._build_rewrite_prompt(
            content, primary_keyword, secondary_keywords, context
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
    ) -> str:
        """
        Generate an optimized meta title.

        Args:
            current_title: Current title if any.
            primary_keyword: Primary keyword to include.
            topic: Content topic for context.
            max_length: Maximum title length.

        Returns:
            Optimized title with markers for changed portions.
        """
        prompt = f"""Create an SEO-optimized title tag for a page about: {topic}

Primary keyword (must include near beginning): {primary_keyword}
Current title: {current_title or "None"}
Maximum length: {max_length} characters

Requirements:
- Include the primary keyword within the first 30 characters if possible
- Make it compelling and click-worthy
- Stay under {max_length} characters
- Mark all new/changed text with [[[ADD]]]...[[[ENDADD]]] markers
- If there's a current title, preserve what works and mark only changes

Return ONLY the optimized title with markers, nothing else."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                system="You are an SEO title optimization expert. Return only the optimized title with change markers.",
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
    ) -> str:
        """
        Generate an optimized meta description.

        Args:
            current_description: Current description if any.
            primary_keyword: Primary keyword to include.
            topic: Content topic for context.
            max_length: Maximum description length.

        Returns:
            Optimized description with markers.
        """
        prompt = f"""Create an SEO-optimized meta description for a page about: {topic}

Primary keyword (must include naturally): {primary_keyword}
Current description: {current_description or "None"}
Maximum length: {max_length} characters

Requirements:
- Include the primary keyword naturally
- Use active voice
- Include a clear call-to-action (e.g., "Learn more", "Get a quote", "Discover how")
- Be compelling and informative
- Stay under {max_length} characters
- Mark all new/changed text with [[[ADD]]]...[[[ENDADD]]] markers

Return ONLY the optimized description with markers, nothing else."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system="You are an SEO meta description expert. Return only the optimized description with change markers.",
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
    ) -> str:
        """
        Generate an optimized H1 heading.

        Args:
            current_h1: Current H1 if any.
            primary_keyword: Primary keyword to include.
            title: The page title (H1 should complement, not duplicate).
            topic: Content topic.

        Returns:
            Optimized H1 with markers.
        """
        prompt = f"""Create an SEO-optimized H1 heading for a page about: {topic}

Primary keyword (must include): {primary_keyword}
Current H1: {current_h1 or "None"}
Page title (H1 should complement, not duplicate exactly): {title}

Requirements:
- Include the primary keyword
- More descriptive than the title
- Not an exact copy of the title
- Clear and engaging
- Mark all new/changed text with [[[ADD]]]...[[[ENDADD]]] markers

Return ONLY the optimized H1 with markers, nothing else."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                system="You are an SEO heading optimization expert. Return only the optimized H1 with change markers.",
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
    ) -> list[dict[str, str]]:
        """
        Generate FAQ items based on keywords.

        Args:
            topic: Content topic.
            primary_keyword: Primary keyword.
            secondary_keywords: Secondary keywords to include.
            question_keywords: Question-form keywords to address.
            num_items: Number of FAQ items to generate.

        Returns:
            List of dicts with 'question' and 'answer' keys, with markers.
        """
        questions_list = "\n".join(f"- {q}" for q in question_keywords) if question_keywords else "None provided"
        keywords_list = ", ".join([primary_keyword] + secondary_keywords[:3])

        prompt = f"""Generate {num_items} FAQ items for a page about: {topic}

Keywords to incorporate naturally: {keywords_list}

Question keywords to address (if relevant):
{questions_list}

Requirements:
- Each question should be a common user question about the topic
- Answers should be 2-4 sentences, helpful and informative
- Naturally include relevant keywords in answers
- Mark all text with [[[ADD]]]...[[[ENDADD]]] since these are all new
- Do NOT make up specific facts, prices, or claims not verifiable

Return in this exact format:
Q: [[[ADD]]]Question text here?[[[ENDADD]]]
A: [[[ADD]]]Answer text here.[[[ENDADD]]]

Q: [[[ADD]]]Next question?[[[ENDADD]]]
A: [[[ADD]]]Next answer.[[[ENDADD]]]

(Continue for all {num_items} items)"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system="You are an SEO FAQ content expert. Generate helpful, keyword-rich FAQ items.",
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
    ) -> str:
        """Build the rewrite prompt for content optimization."""
        secondary_list = ", ".join(secondary_keywords) if secondary_keywords else "None"

        return f"""Optimize the following content for SEO.

PRIMARY KEYWORD (must appear in first 100 words and several times throughout): {primary_keyword}

SECONDARY KEYWORDS (sprinkle naturally where appropriate): {secondary_list}

ADDITIONAL CONTEXT: {context or "None provided"}

ORIGINAL CONTENT:
{content}

INSTRUCTIONS:
1. Preserve all original content and meaning
2. Add the primary keyword naturally in the first paragraph if not present
3. Weave in secondary keywords where they fit naturally
4. Add clarifying phrases or sentences if they help include keywords naturally
5. Mark ALL additions and changes with [[[ADD]]]...[[[ENDADD]]] markers
6. Keep the same paragraph structure
7. Do not be repetitive or stuffed - maintain natural reading flow

Return the optimized content with markers. Do not include any explanation."""

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
