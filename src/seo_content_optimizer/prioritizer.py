"""
Keyword prioritization module.

This module selects the optimal keywords for SEO optimization:
- Filters keywords by relevance and intent
- Chooses primary and secondary keywords
- Identifies long-tail question keywords for FAQs
"""

import re
from typing import Optional, Union

from .analysis import ContentAnalysis, get_keyword_usage_stats
from .models import (
    ContentIntent,
    DocxContent,
    Keyword,
    KeywordPlan,
    PageMeta,
)


class KeywordPrioritizer:
    """
    Prioritizes and selects keywords for SEO optimization.

    This class filters and ranks keywords based on:
    - Relevance to the content topic
    - Content intent alignment
    - Search volume and difficulty
    - Current usage in content
    """

    def __init__(
        self,
        keywords: list[Keyword],
        content: Union[PageMeta, DocxContent],
        analysis: ContentAnalysis,
    ):
        """
        Initialize the prioritizer.

        Args:
            keywords: Full list of available keywords.
            content: The content being optimized.
            analysis: Analysis results for the content.
        """
        self.keywords = keywords
        self.content = content
        self.analysis = analysis
        self.full_text = content.full_text.lower()
        self.topic = analysis.topic.lower()

    def create_keyword_plan(
        self,
        max_secondary: int = 5,
        max_questions: int = 4,
    ) -> KeywordPlan:
        """
        Create a keyword optimization plan.

        Args:
            max_secondary: Maximum number of secondary keywords.
            max_questions: Maximum number of long-tail question keywords.

        Returns:
            KeywordPlan with primary, secondary, and question keywords.
        """
        # Step 1: Filter keywords by relevance and intent
        relevant_keywords = self._filter_relevant_keywords()

        if not relevant_keywords:
            # If no relevant keywords, use all keywords sorted by volume
            relevant_keywords = sorted(
                self.keywords,
                key=lambda k: (k.search_volume or 0),
                reverse=True,
            )

        # Step 2: Select primary keyword
        primary = self._select_primary_keyword(relevant_keywords)

        # Remove primary from candidates
        remaining = [kw for kw in relevant_keywords if kw.phrase != primary.phrase]

        # Step 3: Select secondary keywords
        secondary = self._select_secondary_keywords(remaining, primary, max_secondary)

        # Step 4: Select long-tail question keywords
        questions = self._select_question_keywords(remaining, primary, max_questions)

        # Remove questions from secondary to avoid duplication
        question_phrases = {q.phrase.lower() for q in questions}
        secondary = [s for s in secondary if s.phrase.lower() not in question_phrases]

        return KeywordPlan(
            primary=primary,
            secondary=secondary[:max_secondary],
            long_tail_questions=questions[:max_questions],
        )

    def _filter_relevant_keywords(self) -> list[Keyword]:
        """Filter keywords that are relevant to the content topic and intent."""
        relevant = []

        for kw in self.keywords:
            # Calculate relevance score
            score = self._calculate_relevance_score(kw)

            if score > 0:
                relevant.append((kw, score))

        # Sort by score descending
        relevant.sort(key=lambda x: x[1], reverse=True)

        return [kw for kw, _ in relevant]

    def _calculate_relevance_score(self, keyword: Keyword) -> float:
        """
        Calculate relevance score for a keyword.

        Higher scores indicate better relevance.
        """
        score = 0.0
        phrase_lower = keyword.phrase.lower()
        words = set(phrase_lower.split())

        # Check topic overlap
        topic_words = set(self.topic.split())
        word_overlap = len(words & topic_words)
        score += word_overlap * 10

        # Check if keyword appears in topic or H1
        if phrase_lower in self.topic:
            score += 20

        # Check if keyword appears in content
        if phrase_lower in self.full_text:
            score += 5

        # Check individual word overlap with content
        content_words = set(self.full_text.split())
        content_overlap = len(words & content_words)
        score += content_overlap * 2

        # Intent alignment bonus
        if keyword.intent:
            kw_intent = keyword.intent.lower()
            content_intent = self.analysis.intent.value.lower()

            if kw_intent == content_intent:
                score += 15
            elif content_intent == "mixed":
                score += 10
            elif kw_intent in ("informational", "transactional") and content_intent in ("informational", "transactional"):
                score += 5  # Partial match

        # Volume bonus (logarithmic scaling)
        if keyword.search_volume:
            import math
            volume_bonus = math.log10(keyword.search_volume + 1) * 2
            score += volume_bonus

        # Difficulty penalty (prefer lower difficulty)
        if keyword.difficulty is not None:
            difficulty_penalty = keyword.difficulty / 10
            score -= difficulty_penalty

        return max(0, score)

    def _select_primary_keyword(self, candidates: list[Keyword]) -> Keyword:
        """
        Select the best primary keyword.

        Prioritizes:
        1. Relevance (already filtered)
        2. Search volume
        3. Lower difficulty
        4. Already present in content (continuity)
        """
        if not candidates:
            raise ValueError("No candidate keywords available")

        scored = []

        for kw in candidates:
            score = 0.0

            # Volume score
            if kw.search_volume:
                score += min(kw.search_volume / 100, 100)

            # Difficulty bonus (lower is better)
            if kw.difficulty is not None:
                score += (100 - kw.difficulty)

            # Presence bonus - prefer keywords already in content
            if kw.phrase.lower() in self.full_text:
                score += 50

            # Check if in H1 or title
            stats = get_keyword_usage_stats(
                kw.phrase,
                self.content.full_text,
                h1=self.analysis.topic,
            )
            if stats.in_h1:
                score += 30

            scored.append((kw, score))

        # Sort by score and return best
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _select_secondary_keywords(
        self,
        candidates: list[Keyword],
        primary: Keyword,
        max_count: int,
    ) -> list[Keyword]:
        """
        Select secondary keywords.

        Aims for a mix of:
        - Variants/synonyms of primary
        - Related supporting topics
        - Long-tail variations
        """
        if not candidates:
            return []

        primary_words = set(primary.phrase.lower().split())
        selected = []

        # Score each candidate
        scored = []
        for kw in candidates:
            if kw.phrase == primary.phrase:
                continue

            score = 0.0
            kw_words = set(kw.phrase.lower().split())

            # Partial overlap with primary (but not identical)
            overlap = len(primary_words & kw_words)
            if 0 < overlap < len(primary_words):
                score += overlap * 10

            # Volume consideration
            if kw.search_volume:
                score += min(kw.search_volume / 200, 50)

            # Prefer underused keywords
            stats = self.analysis.existing_keywords.get(kw.phrase, {})
            if not stats.get("in_h1") and not stats.get("in_headings"):
                score += 20

            # Length bonus for long-tail
            if len(kw.phrase.split()) >= 3:
                score += 15

            scored.append((kw, score))

        # Sort and select top
        scored.sort(key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in scored[:max_count]]

    def _select_question_keywords(
        self,
        candidates: list[Keyword],
        primary: Keyword,
        max_count: int,
    ) -> list[Keyword]:
        """
        Select long-tail question keywords for FAQ section.

        Prioritizes keywords that:
        - Are in question format
        - Relate to the primary keyword
        - Are not already well-covered in content
        """
        questions = []
        primary_words = set(primary.phrase.lower().split())

        for kw in candidates:
            # Check if it's a question
            if kw.is_question:
                questions.append(kw)
            elif any(kw.phrase.lower().startswith(q) for q in ("how", "what", "why", "when", "where", "can", "does", "is")):
                questions.append(kw)

        if not questions:
            # No explicit questions, generate potential ones from long-tail keywords
            for kw in candidates:
                if len(kw.phrase.split()) >= 4:
                    questions.append(kw)

        # Score questions by relevance to primary
        scored = []
        for q in questions:
            q_words = set(q.phrase.lower().split())
            overlap = len(primary_words & q_words)
            score = overlap * 10

            # Volume bonus
            if q.search_volume:
                score += min(q.search_volume / 100, 30)

            scored.append((q, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [q for q, _ in scored[:max_count]]


def create_keyword_plan(
    keywords: list[Keyword],
    content: Union[PageMeta, DocxContent],
    analysis: ContentAnalysis,
    max_secondary: int = 5,
    max_questions: int = 4,
) -> KeywordPlan:
    """
    Convenience function to create a keyword plan.

    Args:
        keywords: Full list of available keywords.
        content: The content being optimized.
        analysis: Analysis results for the content.
        max_secondary: Maximum number of secondary keywords.
        max_questions: Maximum number of question keywords.

    Returns:
        KeywordPlan with selected keywords.
    """
    prioritizer = KeywordPrioritizer(keywords, content, analysis)
    return prioritizer.create_keyword_plan(max_secondary, max_questions)


def score_keyword_fit(
    keyword: Keyword,
    topic: str,
    intent: ContentIntent,
) -> float:
    """
    Score how well a keyword fits the content topic and intent.

    Args:
        keyword: Keyword to score.
        topic: Content topic.
        intent: Content intent.

    Returns:
        Fit score from 0 to 100.
    """
    score = 0.0

    # Topic word overlap
    topic_words = set(topic.lower().split())
    kw_words = set(keyword.phrase.lower().split())
    overlap = len(topic_words & kw_words)
    score += overlap * 15

    # Intent alignment
    if keyword.intent:
        if keyword.intent.lower() == intent.value:
            score += 30
        elif intent == ContentIntent.MIXED:
            score += 20

    # Length bonus (moderate length preferred)
    word_count = len(keyword.phrase.split())
    if 2 <= word_count <= 4:
        score += 10
    elif word_count > 4:
        score += 5

    return min(100, score)
