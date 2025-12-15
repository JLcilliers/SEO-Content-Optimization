"""
Semantic analysis module for SEO Content Optimizer V2 Architecture.

This module provides:
- Embedding-based keyword relevance scoring (replaces lexical token matching)
- Topic fingerprint extraction (keyphrases, entities, summary, page type)
- Intent classification with confidence scores

Uses sentence-transformers for embeddings and Claude LLM for complex analysis.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import (
    ContentDocument,
    TopicFingerprint,
    IntentClassification,
    SemanticKeyword,
    KeywordCluster,
    KeywordVariant,
    PageType,
    IntentType,
    ConversionGoal,
)

logger = logging.getLogger(__name__)

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. "
        "Semantic relevance scoring will fall back to lexical matching. "
        "Install with: pip install sentence-transformers"
    )


# Default embedding model - small and fast
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Relevance threshold for keyword acceptance
DEFAULT_RELEVANCE_THRESHOLD = 0.35

# Clustering threshold for grouping similar keywords
DEFAULT_CLUSTER_THRESHOLD = 0.75


class SemanticAnalyzer:
    """
    Semantic analysis engine for content and keyword relevance.

    Uses embedding models to compute semantic similarity between
    content and keywords, replacing simple lexical token matching.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
        llm_client=None,
    ):
        """
        Initialize the semantic analyzer.

        Args:
            embedding_model: Name of sentence-transformers model to use.
            relevance_threshold: Minimum similarity score for keyword acceptance.
            cluster_threshold: Similarity threshold for grouping keywords.
            llm_client: Optional LLMClient for complex analysis tasks.
        """
        self.relevance_threshold = relevance_threshold
        self.cluster_threshold = cluster_threshold
        self.llm_client = llm_client
        self._model = None
        self._model_name = embedding_model

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None and EMBEDDINGS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"Loaded embedding model: {self._model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
        return self._model

    def compute_content_embedding(self, document: ContentDocument) -> Optional[np.ndarray]:
        """
        Compute embedding for document content.

        Args:
            document: ContentDocument to embed.

        Returns:
            Numpy array of embeddings or None if unavailable.
        """
        if not self.model:
            return None

        # Use full text for content embedding
        text = document.full_text
        if not text:
            return None

        # Truncate if too long (most models have limits)
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]

        return self.model.encode(text, convert_to_numpy=True)

    def compute_keyword_embedding(self, keyword: str) -> Optional[np.ndarray]:
        """
        Compute embedding for a keyword phrase.

        Args:
            keyword: Keyword phrase to embed.

        Returns:
            Numpy array of embeddings or None if unavailable.
        """
        if not self.model:
            return None
        return self.model.encode(keyword, convert_to_numpy=True)

    def compute_relevance_score(
        self,
        keyword: str,
        document: ContentDocument,
        content_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute semantic relevance score between keyword and document.

        Args:
            keyword: Keyword phrase to score.
            document: Document to compare against.
            content_embedding: Pre-computed document embedding (optional).

        Returns:
            Relevance score from 0.0 to 1.0.
        """
        if not self.model:
            # Fall back to lexical matching
            return self._lexical_relevance_score(keyword, document)

        # Get embeddings
        if content_embedding is None:
            content_embedding = self.compute_content_embedding(document)
        if content_embedding is None:
            return self._lexical_relevance_score(keyword, document)

        keyword_embedding = self.compute_keyword_embedding(keyword)
        if keyword_embedding is None:
            return self._lexical_relevance_score(keyword, document)

        # Compute cosine similarity
        similarity = self._cosine_similarity(keyword_embedding, content_embedding)

        # Normalize to 0-1 range (cosine similarity can be -1 to 1)
        score = (similarity + 1) / 2

        return float(score)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _lexical_relevance_score(self, keyword: str, document: ContentDocument) -> float:
        """
        Fallback lexical relevance scoring when embeddings unavailable.

        Uses improved token matching with partial credit.
        """
        content = document.full_text.lower()
        keyword_lower = keyword.lower()

        # Check for exact phrase match
        if keyword_lower in content:
            return 0.9  # High score for exact match

        # Check for word overlap
        keyword_words = set(keyword_lower.split())
        content_words = set(content.split())

        if not keyword_words:
            return 0.0

        overlap = len(keyword_words & content_words)
        score = overlap / len(keyword_words)

        # Reduce score for partial matches
        return score * 0.7

    def score_keywords(
        self,
        keywords: list[str],
        document: ContentDocument,
    ) -> list[SemanticKeyword]:
        """
        Score multiple keywords against a document.

        Args:
            keywords: List of keyword phrases to score.
            document: Document to compare against.

        Returns:
            List of SemanticKeyword objects with relevance scores.
        """
        # Pre-compute document embedding for efficiency
        content_embedding = self.compute_content_embedding(document)

        scored_keywords = []
        for keyword in keywords:
            score = self.compute_relevance_score(keyword, document, content_embedding)

            semantic_kw = SemanticKeyword(
                phrase=keyword,
                relevance_score=score,
                selected=score >= self.relevance_threshold,
                skip_reason=None if score >= self.relevance_threshold else f"Low relevance ({score:.2f} < {self.relevance_threshold})",
            )
            scored_keywords.append(semantic_kw)

        return scored_keywords

    def cluster_keywords(
        self,
        keywords: list[SemanticKeyword],
    ) -> list[KeywordCluster]:
        """
        Cluster semantically similar keywords.

        Prevents selecting multiple keywords that are essentially
        the same concept (e.g., "SEO tools" and "SEO software").

        Args:
            keywords: List of scored keywords to cluster.

        Returns:
            List of KeywordCluster objects.
        """
        if not self.model or len(keywords) < 2:
            # No clustering without embeddings or with single keyword
            return [
                KeywordCluster(
                    id=f"cluster_{i}",
                    keywords=[kw],
                    centroid_phrase=kw.phrase,
                )
                for i, kw in enumerate(keywords)
            ]

        # Compute embeddings for all keywords
        phrases = [kw.phrase for kw in keywords]
        embeddings = self.model.encode(phrases, convert_to_numpy=True)

        # Simple agglomerative clustering
        clusters: list[KeywordCluster] = []
        used_indices: set[int] = set()

        for i, (kw, emb) in enumerate(zip(keywords, embeddings)):
            if i in used_indices:
                continue

            # Start new cluster
            cluster_keywords = [kw]
            cluster_indices = {i}

            # Find similar keywords
            for j, (other_kw, other_emb) in enumerate(zip(keywords, embeddings)):
                if j <= i or j in used_indices:
                    continue

                similarity = self._cosine_similarity(emb, other_emb)
                if similarity >= self.cluster_threshold:
                    cluster_keywords.append(other_kw)
                    cluster_indices.add(j)

            used_indices.update(cluster_indices)

            # Create cluster
            cluster = KeywordCluster(
                id=f"cluster_{len(clusters)}",
                keywords=cluster_keywords,
                centroid_phrase=kw.phrase,  # Use first keyword as centroid
            )
            clusters.append(cluster)

        return clusters

    def generate_keyword_variants(self, keyword: str) -> KeywordVariant:
        """
        Generate variants for a keyword (plurals, synonyms, etc.).

        Args:
            keyword: Base keyword phrase.

        Returns:
            KeywordVariant with all forms.
        """
        variants = KeywordVariant(canonical=keyword)

        # Simple pluralization rules
        words = keyword.split()
        if words:
            last_word = words[-1]

            # Generate plural form
            if not last_word.endswith('s'):
                if last_word.endswith('y') and len(last_word) > 1 and last_word[-2] not in 'aeiou':
                    plural = last_word[:-1] + 'ies'
                elif last_word.endswith(('s', 'x', 'z', 'ch', 'sh')):
                    plural = last_word + 'es'
                else:
                    plural = last_word + 's'

                plural_phrase = ' '.join(words[:-1] + [plural])
                variants.plurals.append(plural_phrase)

            # Check for singular form if already plural
            if last_word.endswith('s') and len(last_word) > 1:
                singular = last_word[:-1]
                singular_phrase = ' '.join(words[:-1] + [singular])
                variants.plurals.append(singular_phrase)

        # Common acronym expansions
        acronym_map = {
            'seo': 'search engine optimization',
            'sem': 'search engine marketing',
            'ppc': 'pay per click',
            'cro': 'conversion rate optimization',
            'cta': 'call to action',
            'roi': 'return on investment',
            'kpi': 'key performance indicator',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
        }

        keyword_lower = keyword.lower()
        for acronym, expansion in acronym_map.items():
            if acronym in keyword_lower:
                # Add expansion version
                expanded = keyword_lower.replace(acronym, expansion)
                variants.acronyms.append(expanded)
            elif expansion in keyword_lower:
                # Add acronym version
                acronymed = keyword_lower.replace(expansion, acronym)
                variants.acronyms.append(acronymed)

        return variants


class TopicExtractor:
    """
    Extracts topic fingerprint from content using LLM.

    Identifies keyphrases, entities, page type, and content summary.
    """

    def __init__(self, llm_client):
        """
        Initialize topic extractor.

        Args:
            llm_client: LLMClient for calling Claude API.
        """
        self.llm_client = llm_client

    def extract_fingerprint(self, document: ContentDocument) -> TopicFingerprint:
        """
        Extract topic fingerprint from document.

        Args:
            document: ContentDocument to analyze.

        Returns:
            TopicFingerprint with extracted information.
        """
        if not self.llm_client:
            return self._fallback_extraction(document)

        # Prepare content for analysis (truncate if needed)
        content = document.full_text[:6000]

        prompt = f"""Analyze this content and extract a topic fingerprint. Return ONLY valid JSON.

Content:
{content}

Return JSON in this exact format:
{{
    "keyphrases": ["phrase1", "phrase2", "phrase3"],
    "entities": ["Brand1", "Product1", "Person1"],
    "summary": "1-2 sentence summary of what this content is about",
    "page_type": "one of: guide, service, pricing, comparison, faq, policy, blog, landing, category, other",
    "industry": "industry name or null",
    "target_audience": "target audience description or null",
    "content_depth": "one of: shallow, medium, deep"
}}

Rules:
- keyphrases: 3-7 core topic phrases that capture what this content is about
- entities: Named entities like brands, products, people, companies mentioned
- summary: Brief, factual summary (no marketing language)
- page_type: Best match from the list
- content_depth: shallow (<500 words, surface level), medium (500-1500 words, moderate detail), deep (>1500 words, comprehensive)"""

        try:
            response = self.llm_client.call_claude(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,
            )

            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return self._parse_fingerprint_response(data, document)
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")

        return self._fallback_extraction(document)

    def _parse_fingerprint_response(self, data: dict, document: ContentDocument) -> TopicFingerprint:
        """Parse LLM response into TopicFingerprint."""
        page_type = data.get("page_type", "other")
        if page_type not in ("guide", "service", "pricing", "comparison", "faq", "policy", "blog", "landing", "category", "other"):
            page_type = "other"

        content_depth = data.get("content_depth", "medium")
        if content_depth not in ("shallow", "medium", "deep"):
            content_depth = "medium"

        return TopicFingerprint(
            keyphrases=data.get("keyphrases", [])[:7],
            entities=data.get("entities", [])[:10],
            summary=data.get("summary", "")[:500],
            page_type=page_type,
            industry=data.get("industry"),
            target_audience=data.get("target_audience"),
            content_depth=content_depth,
        )

    def _fallback_extraction(self, document: ContentDocument) -> TopicFingerprint:
        """Fallback topic extraction without LLM."""
        # Extract simple keyphrases from headings
        keyphrases = []
        for block in document.headings[:5]:
            if block.plain_text:
                keyphrases.append(block.plain_text.lower())

        # Use title as summary if available
        summary = document.extracted_title or ""

        # Estimate content depth
        word_count = document.word_count
        if word_count < 500:
            depth = "shallow"
        elif word_count < 1500:
            depth = "medium"
        else:
            depth = "deep"

        return TopicFingerprint(
            keyphrases=keyphrases,
            entities=[],
            summary=summary,
            page_type="other",
            content_depth=depth,
        )


class IntentClassifier:
    """
    Classifies content intent using LLM analysis.

    Provides primary/secondary intent, confidence scores,
    and conversion goal identification.
    """

    def __init__(self, llm_client):
        """
        Initialize intent classifier.

        Args:
            llm_client: LLMClient for calling Claude API.
        """
        self.llm_client = llm_client

    def classify(self, document: ContentDocument) -> IntentClassification:
        """
        Classify document intent.

        Args:
            document: ContentDocument to classify.

        Returns:
            IntentClassification with intent details.
        """
        if not self.llm_client:
            return self._heuristic_classification(document)

        # Prepare content for analysis
        content = document.full_text[:4000]
        title = document.extracted_title or ""

        prompt = f"""Analyze this content's search intent. Return ONLY valid JSON.

Title: {title}

Content:
{content}

Return JSON in this exact format:
{{
    "primary_intent": "one of: informational, transactional, navigational, commercial",
    "secondary_intent": "one of: informational, transactional, navigational, commercial, or null",
    "confidence": 0.0 to 1.0,
    "conversion_goal": "one of: get_quote, book_call, download, subscribe, contact, buy, none",
    "transactional_signals": ["signal1", "signal2"],
    "informational_signals": ["signal1", "signal2"],
    "notes": "Brief explanation of intent classification"
}}

Intent definitions:
- informational: User wants to learn/understand something (how-to, guides, explanations)
- transactional: User wants to complete a transaction (buy, sign up, download)
- navigational: User wants to find a specific page/brand
- commercial: User is researching before a purchase (comparisons, reviews, "best X")

Conversion goals:
- get_quote: Content promotes getting a quote/estimate
- book_call: Content promotes booking a call/demo/consultation
- download: Content promotes downloading something
- subscribe: Content promotes newsletter/subscription signup
- contact: Content promotes contacting the company
- buy: Content promotes direct purchase
- none: No clear conversion goal"""

        try:
            response = self.llm_client.call_claude(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3,
            )

            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return self._parse_intent_response(data)
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")

        return self._heuristic_classification(document)

    def _parse_intent_response(self, data: dict) -> IntentClassification:
        """Parse LLM response into IntentClassification."""
        primary = data.get("primary_intent", "informational")
        if primary not in ("informational", "transactional", "navigational", "commercial"):
            primary = "informational"

        secondary = data.get("secondary_intent")
        if secondary and secondary not in ("informational", "transactional", "navigational", "commercial"):
            secondary = None

        conversion = data.get("conversion_goal", "none")
        if conversion not in ("get_quote", "book_call", "download", "subscribe", "contact", "buy", "none"):
            conversion = "none"

        return IntentClassification(
            primary_intent=primary,
            secondary_intent=secondary,
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
            conversion_goal=conversion,
            notes=data.get("notes", ""),
            transactional_signals=data.get("transactional_signals", [])[:5],
            informational_signals=data.get("informational_signals", [])[:5],
        )

    def _heuristic_classification(self, document: ContentDocument) -> IntentClassification:
        """Fallback heuristic classification without LLM."""
        content = document.full_text.lower()
        title = (document.extracted_title or "").lower()

        # Transactional signals
        transactional_patterns = [
            r'\b(buy|purchase|order|shop|cart|checkout)\b',
            r'\b(pricing|price|cost|free trial|sign up)\b',
            r'\b(get started|start now|try free)\b',
            r'\$\d+',
        ]
        transactional_signals = []
        for pattern in transactional_patterns:
            if re.search(pattern, content):
                transactional_signals.append(pattern.replace(r'\b', '').replace('(', '').replace(')', '').split('|')[0])

        # Informational signals
        informational_patterns = [
            r'\b(how to|what is|why|guide|tutorial|learn)\b',
            r'\b(explained|introduction|overview|basics)\b',
            r'\b(tips|tricks|best practices|steps)\b',
        ]
        informational_signals = []
        for pattern in informational_patterns:
            if re.search(pattern, content):
                informational_signals.append(pattern.replace(r'\b', '').replace('(', '').replace(')', '').split('|')[0])

        # Commercial signals
        commercial_patterns = [
            r'\b(best|top|review|comparison|vs|versus|alternative)\b',
            r'\b(pros and cons|features|benefits)\b',
        ]
        commercial_score = sum(1 for p in commercial_patterns if re.search(p, content))

        # Determine primary intent
        trans_score = len(transactional_signals)
        info_score = len(informational_signals)

        if trans_score > info_score and trans_score > commercial_score:
            primary = "transactional"
        elif commercial_score > info_score:
            primary = "commercial"
        else:
            primary = "informational"

        # Determine conversion goal
        conversion = "none"
        if re.search(r'\b(get.{0,20}quote|request.{0,20}quote|free.{0,20}quote)\b', content):
            conversion = "get_quote"
        elif re.search(r'\b(book.{0,20}(call|demo|consultation)|schedule.{0,20}(call|demo))\b', content):
            conversion = "book_call"
        elif re.search(r'\b(download|get.{0,20}(free|ebook|guide|whitepaper))\b', content):
            conversion = "download"
        elif re.search(r'\b(subscribe|newsletter|sign.{0,20}up)\b', content):
            conversion = "subscribe"
        elif re.search(r'\b(contact.{0,20}us|get.{0,20}in.{0,20}touch)\b', content):
            conversion = "contact"
        elif re.search(r'\b(buy.{0,20}now|add.{0,20}to.{0,20}cart|purchase)\b', content):
            conversion = "buy"

        return IntentClassification(
            primary_intent=primary,
            secondary_intent=None,
            confidence=0.6,  # Lower confidence for heuristic
            conversion_goal=conversion,
            notes="Classified using heuristic patterns (LLM unavailable)",
            transactional_signals=transactional_signals[:5],
            informational_signals=informational_signals[:5],
        )


def analyze_content(
    document: ContentDocument,
    keywords: list[str],
    llm_client=None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[TopicFingerprint, IntentClassification, list[SemanticKeyword], list[KeywordCluster]]:
    """
    Perform complete semantic analysis of content.

    Convenience function that runs all analysis in one call.

    Args:
        document: ContentDocument to analyze.
        keywords: List of candidate keyword phrases.
        llm_client: Optional LLMClient for complex analysis.
        embedding_model: Name of embedding model to use.

    Returns:
        Tuple of (TopicFingerprint, IntentClassification, scored_keywords, clusters).
    """
    # Initialize analyzers
    semantic = SemanticAnalyzer(
        embedding_model=embedding_model,
        llm_client=llm_client,
    )
    topic_extractor = TopicExtractor(llm_client)
    intent_classifier = IntentClassifier(llm_client)

    # Extract topic fingerprint
    fingerprint = topic_extractor.extract_fingerprint(document)

    # Classify intent
    intent = intent_classifier.classify(document)

    # Score keywords
    scored_keywords = semantic.score_keywords(keywords, document)

    # Generate variants for scored keywords
    for kw in scored_keywords:
        kw.variants = semantic.generate_keyword_variants(kw.phrase)

    # Cluster similar keywords
    clusters = semantic.cluster_keywords(scored_keywords)

    return fingerprint, intent, scored_keywords, clusters
