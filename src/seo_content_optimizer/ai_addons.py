# -*- coding: utf-8 -*-
"""
AI Optimization Add-ons module.

Generates AI-friendly content sections:
- Key Takeaways (3-6 concise bullets)
- Chunk Map (structured content chunks for AI retrieval)
- FAQ fallback generation

These sections are designed to be:
1. Highly extractable by AI systems
2. Useful for RAG/retrieval applications
3. Consistent across all page types
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Chunk:
    """A content chunk for AI retrieval."""
    chunk_id: str
    heading_path: str  # e.g., "H1 > H2 > H3"
    content: str
    summary: str
    best_question: str  # Natural language query this chunk answers
    keywords_present: List[str] = field(default_factory=list)
    word_count: int = 0
    token_estimate: int = 0  # Approximate tokens (words * 1.3)


@dataclass
class ChunkMap:
    """Collection of chunks with metadata."""
    chunks: List[Chunk] = field(default_factory=list)
    total_chunks: int = 0
    total_words: int = 0
    total_tokens: int = 0


@dataclass
class AIAddons:
    """Container for all AI optimization add-ons."""
    key_takeaways: List[str] = field(default_factory=list)
    chunk_map: Optional[ChunkMap] = None
    faqs: List[dict] = field(default_factory=list)  # [{"question": ..., "answer": ...}]


# Default chunking parameters (based on Microsoft recommendations)
DEFAULT_CHUNK_TARGET_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 128


def generate_key_takeaways(
    content_blocks: List[str],
    primary_keyword: str,
    secondary_keywords: List[str] = None,
    brand_name: str = None,
    max_takeaways: int = 6,
    min_takeaways: int = 3,
) -> List[str]:
    """
    Generate key takeaways from content blocks.

    Uses deterministic extraction where possible, falling back to
    summarization patterns when needed.

    Rules enforced:
    - 3-6 bullets (configurable)
    - Direct, factual tone
    - No promotional language
    - Based only on source content

    Args:
        content_blocks: List of content paragraphs/blocks.
        primary_keyword: The main keyword for context.
        secondary_keywords: Additional keywords.
        brand_name: Brand name if detected.
        max_takeaways: Maximum number of takeaways.
        min_takeaways: Minimum number of takeaways.

    Returns:
        List of takeaway strings.
    """
    if not content_blocks:
        return []

    secondary_keywords = secondary_keywords or []
    takeaways = []

    # Strategy 1: Extract key sentences from content
    # Look for sentences with strong signals (numbers, "key", "important", etc.)
    key_patterns = [
        r'\b\d+%',  # Percentages
        r'\b\d+\s*(years?|months?|days?)',  # Time references
        r'\b(key|important|essential|critical|main|primary)\b',
        r'\b(benefit|advantage|feature|solution)\b',
        r'\b(helps?|provides?|offers?|enables?|allows?)\b',
    ]

    full_text = " ".join(content_blocks)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    # Score sentences by relevance
    scored_sentences = []
    for sentence in sentences:
        if len(sentence) < 20 or len(sentence) > 200:
            continue

        score = 0
        sentence_lower = sentence.lower()

        # Boost for keyword presence
        if primary_keyword.lower() in sentence_lower:
            score += 3
        for kw in secondary_keywords[:3]:  # Top 3 secondaries
            if kw.lower() in sentence_lower:
                score += 1

        # Boost for key patterns
        for pattern in key_patterns:
            if re.search(pattern, sentence_lower):
                score += 2

        # Penalize promotional/hyperbolic language
        if re.search(r'\b(best|amazing|incredible|revolutionary|game-?changing)\b', sentence_lower):
            score -= 2

        if score > 0:
            scored_sentences.append((score, sentence))

    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    for score, sentence in scored_sentences[:max_takeaways]:
        # Clean up the sentence
        takeaway = sentence.strip()
        # Remove leading conjunctions
        takeaway = re.sub(r'^(And|But|So|However|Therefore|Thus|Hence)\s+', '', takeaway)
        # Ensure it ends with punctuation
        if not takeaway.endswith(('.', '!', '?')):
            takeaway += '.'
        takeaways.append(takeaway)

    # Strategy 2: If not enough takeaways, generate from headings + first sentences
    if len(takeaways) < min_takeaways:
        # Look for heading-like content (short, capitalized)
        for block in content_blocks:
            if len(block) < 100 and len(block.split()) > 3:
                # Could be a heading or key point
                if block not in takeaways:
                    takeaways.append(block if block.endswith('.') else block + '.')
                if len(takeaways) >= min_takeaways:
                    break

    # Strategy 3: If still not enough, create generic but safe takeaways
    if len(takeaways) < min_takeaways:
        if primary_keyword:
            takeaways.append(f"This page provides information about {primary_keyword}.")
        if brand_name:
            takeaways.append(f"{brand_name} offers solutions in this area.")

    # Deduplicate and limit
    seen = set()
    unique_takeaways = []
    for t in takeaways:
        normalized = t.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_takeaways.append(t)

    return unique_takeaways[:max_takeaways]


def build_chunks(
    content_blocks: List[str],
    headings: List[Tuple[str, int]] = None,  # [(heading_text, level), ...]
    chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> List[Chunk]:
    """
    Build content chunks for AI retrieval.

    Uses structure-aware chunking:
    1. First pass: Group by heading boundaries
    2. Second pass: Split oversized sections
    3. Apply overlap between chunks

    Args:
        content_blocks: List of content paragraphs.
        headings: List of (heading_text, level) tuples.
        chunk_target_tokens: Target tokens per chunk (~512 default).
        chunk_overlap_tokens: Overlap tokens between chunks (~128 default).

    Returns:
        List of Chunk objects.
    """
    if not content_blocks:
        return []

    headings = headings or []
    chunks = []
    chunk_counter = 1

    def estimate_tokens(text: str) -> int:
        """Estimate tokens (roughly 1 token per 4 characters or 1.3 per word)."""
        return max(len(text) // 4, int(len(text.split()) * 1.3))

    def create_chunk(content: str, heading_path: str) -> Chunk:
        nonlocal chunk_counter
        word_count = len(content.split())
        token_est = estimate_tokens(content)

        chunk = Chunk(
            chunk_id=f"C{chunk_counter:02d}",
            heading_path=heading_path,
            content=content,
            summary="",  # Will be filled later
            best_question="",  # Will be filled later
            word_count=word_count,
            token_estimate=token_est,
        )
        chunk_counter += 1
        return chunk

    # If no headings, chunk by token count
    if not headings:
        current_content = []
        current_tokens = 0

        for block in content_blocks:
            block_tokens = estimate_tokens(block)

            if current_tokens + block_tokens > chunk_target_tokens and current_content:
                # Create chunk from current content
                chunks.append(create_chunk(
                    " ".join(current_content),
                    "Content"
                ))
                # Start new chunk with overlap
                overlap_text = " ".join(current_content[-2:]) if len(current_content) > 1 else ""
                current_content = [overlap_text, block] if overlap_text else [block]
                current_tokens = estimate_tokens(" ".join(current_content))
            else:
                current_content.append(block)
                current_tokens += block_tokens

        # Don't forget the last chunk
        if current_content:
            chunks.append(create_chunk(" ".join(current_content), "Content"))

    else:
        # Structure-aware chunking based on headings
        current_section = []
        current_heading_path = "Introduction"

        for block in content_blocks:
            # Check if this block is a heading
            is_heading = False
            for heading_text, level in headings:
                if heading_text.lower() in block.lower() and len(block) < 150:
                    # This is a heading - save current section
                    if current_section:
                        chunks.append(create_chunk(
                            " ".join(current_section),
                            current_heading_path
                        ))
                    current_section = []
                    current_heading_path = f"H{level}: {heading_text}"
                    is_heading = True
                    break

            if not is_heading:
                current_section.append(block)

                # Check if section is too large
                section_tokens = estimate_tokens(" ".join(current_section))
                if section_tokens > chunk_target_tokens * 1.5:
                    # Split the section
                    chunks.append(create_chunk(
                        " ".join(current_section[:-1]),
                        current_heading_path
                    ))
                    current_section = [current_section[-1]]

        # Save final section
        if current_section:
            chunks.append(create_chunk(" ".join(current_section), current_heading_path))

    return chunks


def generate_chunk_map(
    chunks: List[Chunk],
    primary_keyword: str,
    secondary_keywords: List[str] = None,
) -> ChunkMap:
    """
    Generate a chunk map with summaries and metadata.

    For each chunk, generates:
    - Summary (1-2 sentences)
    - Best question answered
    - Keywords present

    Args:
        chunks: List of Chunk objects from build_chunks().
        primary_keyword: Primary keyword for analysis.
        secondary_keywords: Secondary keywords for analysis.

    Returns:
        ChunkMap with enhanced chunk metadata.
    """
    if not chunks:
        return ChunkMap()

    secondary_keywords = secondary_keywords or []
    all_keywords = [primary_keyword] + secondary_keywords

    total_words = 0
    total_tokens = 0

    for chunk in chunks:
        total_words += chunk.word_count
        total_tokens += chunk.token_estimate

        # Find keywords present in this chunk
        chunk_lower = chunk.content.lower()
        chunk.keywords_present = [
            kw for kw in all_keywords
            if kw.lower() in chunk_lower
        ]

        # Generate summary (first 1-2 sentences or truncated)
        sentences = re.split(r'(?<=[.!?])\s+', chunk.content)
        if sentences:
            chunk.summary = " ".join(sentences[:2])
            if len(chunk.summary) > 200:
                chunk.summary = chunk.summary[:197] + "..."

        # Generate "best question answered"
        chunk.best_question = _generate_question_for_chunk(
            chunk.content,
            chunk.heading_path,
            primary_keyword
        )

    return ChunkMap(
        chunks=chunks,
        total_chunks=len(chunks),
        total_words=total_words,
        total_tokens=total_tokens,
    )


def _generate_question_for_chunk(content: str, heading_path: str, keyword: str) -> str:
    """
    Generate a natural language question that this chunk answers.

    Uses pattern matching to identify the chunk's purpose and generate
    an appropriate question. Patterns are prioritized for better relevance.

    Args:
        content: The chunk content text.
        heading_path: The heading path (e.g., "H1 > H2 > H3").
        keyword: The primary keyword for context.

    Returns:
        A natural language question string.
    """
    # Extract key terms from heading
    heading_clean = re.sub(r'^H\d:\s*', '', heading_path)
    content_lower = content.lower()

    # Extended question patterns with priority (order matters)
    question_patterns = [
        # How-to patterns (high priority)
        (r'\b(?:how|step|process|method|procedure|guide|tutorial)\b',
         f"How do I {keyword}?"),
        # Benefit patterns
        (r'\b(?:benefit|advantage|why|reason|value|help|improve)\b',
         f"What are the benefits of {keyword}?"),
        # Cost/pricing patterns
        (r'\b(?:cost|price|fee|pricing|\$|\d+\s*(?:per|\/|a)\s*(?:month|year|user))\b',
         f"How much does {keyword} cost?"),
        # Feature patterns
        (r'\b(?:feature|include|offer|provide|capability|function)\b',
         f"What features does {keyword} include?"),
        # Comparison patterns
        (r'\b(?:vs|versus|compare|differ|better|worse|alternative)\b',
         f"How does {keyword} compare to alternatives?"),
        # Time/duration patterns
        (r'\b(?:when|timeline|duration|long|fast|quick|time)\b',
         f"How long does {keyword} take?"),
        # Eligibility/requirements patterns
        (r'\b(?:who|eligible|qualify|require|need|must|should)\b',
         f"Who is eligible for {keyword}?"),
        # What is patterns
        (r'\b(?:definition|defined?|meaning|what\s+is|what\s+are)\b',
         f"What is {keyword}?"),
        # Getting started patterns
        (r'\b(?:start|begin|get\s+started|first\s+step|setup|install)\b',
         f"How do I get started with {keyword}?"),
        # Types/categories patterns
        (r'\b(?:type|kind|category|option|choice|variety)\b',
         f"What types of {keyword} are available?"),
    ]

    for pattern, question in question_patterns:
        if re.search(pattern, content_lower):
            return question

    # Fallback to heading-based question
    if heading_clean and heading_clean not in ("Content", "Introduction", "Overview"):
        # Try to make a sensible question from the heading
        heading_lower = heading_clean.lower()
        if any(word in heading_lower for word in ['how', 'what', 'why', 'when', 'where']):
            # Heading is already a question-like phrase
            return heading_clean + ("?" if not heading_clean.endswith("?") else "")
        return f"What is {heading_clean}?"

    # Ultimate fallback
    return f"What should I know about {keyword}?"


def generate_fallback_faqs(
    content_blocks: List[str],
    primary_keyword: str,
    secondary_keywords: List[str] = None,
    brand_name: str = None,
    min_faqs: int = 3,
    max_faqs: int = 6,
) -> List[dict]:
    """
    Generate fallback FAQ items when LLM generation fails or is empty.

    Uses deterministic patterns based on keyword and content.
    Answers are extracted/composed from content blocks only.

    Args:
        content_blocks: Source content for answers.
        primary_keyword: Primary keyword for questions.
        secondary_keywords: Secondary keywords.
        brand_name: Brand name if available.
        min_faqs: Minimum FAQs to generate.
        max_faqs: Maximum FAQs to generate.

    Returns:
        List of {"question": ..., "answer": ...} dicts.
    """
    if not content_blocks:
        return []

    secondary_keywords = secondary_keywords or []
    faqs = []
    full_text = " ".join(content_blocks)

    # Question templates with extraction patterns
    question_templates = [
        {
            "q": f"What is {primary_keyword}?",
            "pattern": r'(?:is|are|refers?\s+to|means?)[^.]*' + re.escape(primary_keyword.lower()),
        },
        {
            "q": f"How does {primary_keyword} work?",
            "pattern": r'(?:works?|functions?|operates?)[^.]*',
        },
        {
            "q": f"What are the benefits of {primary_keyword}?",
            "pattern": r'(?:benefits?|advantages?|helps?)[^.]*',
        },
        {
            "q": f"Who should consider {primary_keyword}?",
            "pattern": r'(?:for|ideal|suited|designed)[^.]*(?:people|users?|customers?|anyone)',
        },
    ]

    # Add brand-specific question if available
    if brand_name:
        question_templates.insert(1, {
            "q": f"What makes {brand_name} different?",
            "pattern": r'(?:unique|different|unlike|sets?\s+apart)[^.]*',
        })

    # Add secondary keyword questions
    for kw in secondary_keywords[:2]:
        question_templates.append({
            "q": f"What about {kw}?",
            "pattern": re.escape(kw.lower()),
        })

    # Generate FAQs
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    for template in question_templates[:max_faqs]:
        # Try to find a relevant sentence for the answer
        answer_sentences = []

        for sentence in sentences:
            if len(sentence) < 30 or len(sentence) > 300:
                continue

            # Check if sentence matches the pattern
            if re.search(template["pattern"], sentence.lower()):
                answer_sentences.append(sentence)

            # Also check for keyword presence
            if primary_keyword.lower() in sentence.lower():
                if sentence not in answer_sentences:
                    answer_sentences.append(sentence)

        # Build answer from found sentences
        if answer_sentences:
            answer = " ".join(answer_sentences[:2])
            if len(answer) > 300:
                answer = answer[:297] + "..."
        else:
            # Fallback generic answer
            answer = f"This page provides detailed information about {primary_keyword}. Please review the content above for specific details."

        faqs.append({
            "question": template["q"],
            "answer": answer,
        })

        if len(faqs) >= max_faqs:
            break

    # Ensure minimum FAQs
    while len(faqs) < min_faqs:
        faqs.append({
            "question": f"Where can I learn more about {primary_keyword}?",
            "answer": "Contact us or visit our website for more information and personalized assistance.",
        })

    return faqs[:max_faqs]


def generate_ai_addons(
    content_blocks: List[str],
    primary_keyword: str,
    secondary_keywords: List[str] = None,
    brand_name: str = None,
    headings: List[Tuple[str, int]] = None,
    existing_faqs: List[dict] = None,
    generate_takeaways: bool = True,
    generate_chunks: bool = True,
    generate_faqs: bool = True,
    chunk_target_tokens: int = DEFAULT_CHUNK_TARGET_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> AIAddons:
    """
    Generate all AI optimization add-ons in one call.

    This is the main entry point for generating:
    - Key Takeaways
    - Chunk Map
    - FAQ fallbacks (if existing FAQs are empty)

    Args:
        content_blocks: Source content.
        primary_keyword: Primary keyword.
        secondary_keywords: Secondary keywords.
        brand_name: Brand name if detected.
        headings: List of (heading, level) tuples.
        existing_faqs: Already generated FAQs (if any).
        generate_takeaways: Whether to generate key takeaways.
        generate_chunks: Whether to generate chunk map.
        generate_faqs: Whether to generate fallback FAQs.
        chunk_target_tokens: Target tokens per chunk (default 512).
        chunk_overlap_tokens: Overlap between chunks (default 128).

    Returns:
        AIAddons object with all generated content.
    """
    secondary_keywords = secondary_keywords or []
    addons = AIAddons()

    # Generate Key Takeaways
    if generate_takeaways:
        addons.key_takeaways = generate_key_takeaways(
            content_blocks,
            primary_keyword,
            secondary_keywords,
            brand_name,
        )

    # Generate Chunk Map
    if generate_chunks:
        chunks = build_chunks(
            content_blocks,
            headings,
            chunk_target_tokens=chunk_target_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        addons.chunk_map = generate_chunk_map(
            chunks,
            primary_keyword,
            secondary_keywords,
        )

    # Handle FAQs - use existing or generate fallbacks
    if generate_faqs:
        if existing_faqs and len(existing_faqs) >= 2:
            addons.faqs = existing_faqs
        else:
            # Generate fallback FAQs
            addons.faqs = generate_fallback_faqs(
                content_blocks,
                primary_keyword,
                secondary_keywords,
                brand_name,
            )

    return addons
