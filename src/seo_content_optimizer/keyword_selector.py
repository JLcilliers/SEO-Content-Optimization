"""
Semantic keyword selection for SEO Content Optimizer V2 Architecture.

This module replaces the old lexical token-based keyword_filter with
intelligent semantic selection using:
- Embedding-based relevance scoring
- Keyword clustering to avoid redundancy
- Skip logic for brand/off-topic keywords
- Smart primary/secondary selection

The goal is to select keywords that are genuinely relevant to the content
topic, not just keywords that share some tokens.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    ContentDocument,
    SemanticKeyword,
    SemanticKeywordPlan,
    KeywordCluster,
    TopicFingerprint,
    IntentClassification,
)
from .semantic_analysis import (
    SemanticAnalyzer,
    TopicExtractor,
    IntentClassifier,
    DEFAULT_RELEVANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# Brand/navigational keyword patterns - these should usually be skipped
BRAND_PATTERNS = [
    r'\b(login|sign\s*in|sign\s*up|account|my\s*account)\b',
    r'\b(contact\s*us|about\s*us|careers|jobs)\b',
    r'\b(facebook|twitter|linkedin|instagram|youtube)\b',
    r'\.com\b|\.org\b|\.net\b',
]

# High-risk industry terms - block unless explicitly in content
HIGH_RISK_TERMS = {
    # Adult/vice
    "adult", "gambling", "casino", "betting", "poker",
    "cannabis", "marijuana", "hemp", "cbd", "thc",
    "firearms", "guns", "weapons", "ammunition",
    "escort", "porn", "xxx",

    # Regulated industries
    "forex", "cryptocurrency", "crypto", "bitcoin",
    "payday loan", "debt collection",
}

# Maximum keywords per cluster to select
MAX_PER_CLUSTER = 2

# Maximum secondary keywords to select
MAX_SECONDARY_KEYWORDS = 5

# Maximum question keywords for FAQ
MAX_QUESTION_KEYWORDS = 3


@dataclass
class KeywordSelectionConfig:
    """Configuration for keyword selection."""
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    max_secondary: int = MAX_SECONDARY_KEYWORDS
    max_questions: int = MAX_QUESTION_KEYWORDS
    max_per_cluster: int = MAX_PER_CLUSTER
    allow_brand_keywords: bool = False
    allow_high_risk: bool = False


@dataclass
class KeywordSelectionResult:
    """Complete result of keyword selection process."""
    plan: SemanticKeywordPlan
    fingerprint: Optional[TopicFingerprint] = None
    intent: Optional[IntentClassification] = None

    # Statistics
    total_candidates: int = 0
    total_relevant: int = 0
    total_skipped: int = 0
    total_clustered: int = 0

    # Detailed skip reasons
    skip_details: dict[str, list[str]] = field(default_factory=dict)


class KeywordSelector:
    """
    Intelligent keyword selector using semantic analysis.

    Replaces the old token-overlap based filtering with embedding-based
    relevance scoring and smart clustering.
    """

    def __init__(
        self,
        llm_client=None,
        config: Optional[KeywordSelectionConfig] = None,
    ):
        """
        Initialize keyword selector.

        Args:
            llm_client: Optional LLMClient for complex analysis.
            config: Selection configuration.
        """
        self.llm_client = llm_client
        self.config = config or KeywordSelectionConfig()

        self.analyzer = SemanticAnalyzer(
            relevance_threshold=self.config.relevance_threshold,
            llm_client=llm_client,
        )
        self.topic_extractor = TopicExtractor(llm_client)
        self.intent_classifier = IntentClassifier(llm_client)

    def select_keywords(
        self,
        candidates: list[str],
        document: ContentDocument,
        primary_override: Optional[str] = None,
    ) -> KeywordSelectionResult:
        """
        Select optimal keywords from candidates for the document.

        Args:
            candidates: List of candidate keyword phrases.
            document: ContentDocument to optimize for.
            primary_override: If provided, use this as primary keyword.

        Returns:
            KeywordSelectionResult with plan and statistics.
        """
        # Initialize tracking
        skip_details: dict[str, list[str]] = {
            "low_relevance": [],
            "brand_navigational": [],
            "high_risk": [],
            "cluster_limit": [],
            "duplicate": [],
        }

        # Step 1: Extract topic fingerprint and classify intent
        fingerprint = self.topic_extractor.extract_fingerprint(document)
        intent = self.intent_classifier.classify(document)

        # Step 2: Score all candidates for relevance
        scored_keywords = self.analyzer.score_keywords(candidates, document)

        # Step 3: Apply filters (brand, high-risk, relevance)
        filtered_keywords = []
        for kw in scored_keywords:
            # Check brand/navigational patterns
            if not self.config.allow_brand_keywords and self._is_brand_keyword(kw.phrase):
                kw.selected = False
                kw.skip_reason = "Brand/navigational keyword"
                skip_details["brand_navigational"].append(kw.phrase)
                continue

            # Check high-risk terms
            if not self.config.allow_high_risk and self._contains_high_risk(kw.phrase, document):
                kw.selected = False
                kw.skip_reason = "Contains high-risk term not in content"
                skip_details["high_risk"].append(kw.phrase)
                continue

            # Check relevance threshold
            if kw.relevance_score < self.config.relevance_threshold:
                kw.selected = False
                kw.skip_reason = f"Low relevance ({kw.relevance_score:.2f} < {self.config.relevance_threshold})"
                skip_details["low_relevance"].append(kw.phrase)
                continue

            filtered_keywords.append(kw)

        # Step 4: Cluster similar keywords
        clusters = self.analyzer.cluster_keywords(filtered_keywords)

        # Step 5: Select from clusters
        selected_keywords = self._select_from_clusters(
            clusters, skip_details, fingerprint
        )

        # Step 6: Determine primary keyword
        primary = self._select_primary(
            selected_keywords,
            primary_override,
            fingerprint,
        )

        if not primary:
            # No valid primary - create dummy
            logger.warning("No valid primary keyword selected")
            primary = SemanticKeyword(
                phrase=fingerprint.keyphrases[0] if fingerprint.keyphrases else "content",
                relevance_score=0.5,
                selected=True,
            )

        # Step 7: Select secondary keywords
        secondary = self._select_secondary(
            selected_keywords,
            primary,
        )

        # Step 8: Select question keywords for FAQ
        questions = self._select_questions(
            selected_keywords,
            primary,
            secondary,
        )

        # Build the plan
        plan = SemanticKeywordPlan(
            primary=primary,
            secondary=secondary,
            questions=questions,
            clusters=clusters,
            skipped_keywords=[kw for kw in scored_keywords if not kw.selected],
            skip_reasons={kw.phrase: kw.skip_reason for kw in scored_keywords if kw.skip_reason},
        )

        return KeywordSelectionResult(
            plan=plan,
            fingerprint=fingerprint,
            intent=intent,
            total_candidates=len(candidates),
            total_relevant=len(filtered_keywords),
            total_skipped=len(candidates) - len(filtered_keywords),
            total_clustered=len(clusters),
            skip_details=skip_details,
        )

    def _is_brand_keyword(self, keyword: str) -> bool:
        """Check if keyword is likely a brand/navigational query."""
        keyword_lower = keyword.lower()

        for pattern in BRAND_PATTERNS:
            if re.search(pattern, keyword_lower, re.IGNORECASE):
                return True

        return False

    def _contains_high_risk(self, keyword: str, document: ContentDocument) -> bool:
        """
        Check if keyword contains high-risk terms not present in content.

        High-risk terms are allowed if they're already in the original content.
        """
        keyword_lower = keyword.lower()
        content_lower = document.full_text.lower()

        for term in HIGH_RISK_TERMS:
            pattern = r'\b' + re.escape(term) + r'\b'
            # Block if term is in keyword but NOT in content
            if re.search(pattern, keyword_lower) and not re.search(pattern, content_lower):
                return True

        return False

    def _select_from_clusters(
        self,
        clusters: list[KeywordCluster],
        skip_details: dict[str, list[str]],
        fingerprint: TopicFingerprint,
    ) -> list[SemanticKeyword]:
        """Select best keywords from each cluster."""
        selected = []
        seen_phrases = set()

        for cluster in clusters:
            # Sort cluster keywords by relevance, then search volume
            sorted_keywords = sorted(
                cluster.keywords,
                key=lambda k: (-k.relevance_score, -(k.search_volume or 0)),
            )

            # Select up to max_per_cluster from each cluster
            cluster_selected = 0
            for kw in sorted_keywords:
                if kw.phrase.lower() in seen_phrases:
                    skip_details["duplicate"].append(kw.phrase)
                    continue

                if cluster_selected >= self.config.max_per_cluster:
                    kw.selected = False
                    kw.skip_reason = f"Cluster limit reached ({self.config.max_per_cluster} per cluster)"
                    skip_details["cluster_limit"].append(kw.phrase)
                    continue

                kw.selected = True
                kw.cluster_id = cluster.id
                selected.append(kw)
                seen_phrases.add(kw.phrase.lower())
                cluster_selected += 1

        return selected

    def _select_primary(
        self,
        keywords: list[SemanticKeyword],
        override: Optional[str],
        fingerprint: TopicFingerprint,
    ) -> Optional[SemanticKeyword]:
        """Select the primary keyword."""
        # If override provided, try to find it in selected keywords
        if override:
            override_lower = override.lower()
            for kw in keywords:
                if kw.phrase.lower() == override_lower:
                    return kw
            # Override not in selected - create new SemanticKeyword
            return SemanticKeyword(
                phrase=override,
                relevance_score=1.0,  # Assume user knows what they want
                selected=True,
            )

        # Otherwise, select highest relevance non-question keyword
        for kw in sorted(keywords, key=lambda k: -k.relevance_score):
            if not kw.is_question:
                return kw

        # Fallback to any keyword
        return keywords[0] if keywords else None

    def _select_secondary(
        self,
        keywords: list[SemanticKeyword],
        primary: SemanticKeyword,
    ) -> list[SemanticKeyword]:
        """Select secondary keywords (excluding primary and questions)."""
        secondary = []
        primary_lower = primary.phrase.lower()

        for kw in sorted(keywords, key=lambda k: -k.relevance_score):
            if kw.phrase.lower() == primary_lower:
                continue
            if kw.is_question:
                continue
            if len(secondary) >= self.config.max_secondary:
                break

            secondary.append(kw)

        return secondary

    def _select_questions(
        self,
        keywords: list[SemanticKeyword],
        primary: SemanticKeyword,
        secondary: list[SemanticKeyword],
    ) -> list[SemanticKeyword]:
        """Select question keywords for FAQ section."""
        questions = []
        used_phrases = {primary.phrase.lower()}
        used_phrases.update(kw.phrase.lower() for kw in secondary)

        for kw in sorted(keywords, key=lambda k: -k.relevance_score):
            if kw.phrase.lower() in used_phrases:
                continue
            if not kw.is_question:
                continue
            if len(questions) >= self.config.max_questions:
                break

            questions.append(kw)

        return questions


def select_keywords_for_content(
    candidates: list[str],
    document: ContentDocument,
    llm_client=None,
    primary_override: Optional[str] = None,
    config: Optional[KeywordSelectionConfig] = None,
) -> KeywordSelectionResult:
    """
    Convenience function for keyword selection.

    Args:
        candidates: List of candidate keyword phrases.
        document: ContentDocument to optimize for.
        llm_client: Optional LLMClient for semantic analysis.
        primary_override: If provided, use this as primary keyword.
        config: Optional selection configuration.

    Returns:
        KeywordSelectionResult with plan and statistics.
    """
    selector = KeywordSelector(llm_client=llm_client, config=config)
    return selector.select_keywords(candidates, document, primary_override)


def convert_keywords_to_candidates(
    keywords_data: list[dict],
) -> list[str]:
    """
    Convert keyword data from API/CSV to simple phrase list.

    Args:
        keywords_data: List of keyword dicts with 'phrase' or 'keyword' key.

    Returns:
        List of keyword phrase strings.
    """
    phrases = []
    for kw in keywords_data:
        phrase = kw.get("phrase") or kw.get("keyword") or kw.get("term")
        if phrase:
            phrases.append(str(phrase).strip())
    return phrases


def validate_keyword_plan(plan: SemanticKeywordPlan) -> list[str]:
    """
    Validate a keyword plan for common issues.

    Args:
        plan: SemanticKeywordPlan to validate.

    Returns:
        List of warning messages (empty if valid).
    """
    warnings = []

    # Check primary keyword
    if not plan.primary:
        warnings.append("No primary keyword selected")
    elif plan.primary.relevance_score < 0.3:
        warnings.append(f"Primary keyword has low relevance: {plan.primary.relevance_score:.2f}")

    # Check for keyword variety
    all_phrases = plan.all_phrases
    if len(all_phrases) < 2:
        warnings.append("Only one keyword selected - consider adding secondary keywords")

    # Check for very similar keywords
    for i, p1 in enumerate(all_phrases):
        for p2 in all_phrases[i+1:]:
            # Simple similarity check
            p1_words = set(p1.lower().split())
            p2_words = set(p2.lower().split())
            if p1_words and p2_words:
                overlap = len(p1_words & p2_words) / min(len(p1_words), len(p2_words))
                if overlap > 0.8:
                    warnings.append(f"Very similar keywords: '{p1}' and '{p2}'")

    # Check for questions
    has_questions = any(kw.is_question for kw in plan.all_selected)
    if not has_questions and plan.questions:
        warnings.append("Question keywords available but none selected")

    return warnings


def log_selection_results(result: KeywordSelectionResult) -> dict:
    """
    Generate a summary of keyword selection for logging/reporting.

    Args:
        result: KeywordSelectionResult from selection.

    Returns:
        Dict with summary information.
    """
    plan = result.plan

    summary = {
        "primary_keyword": plan.primary.phrase,
        "primary_relevance": plan.primary.relevance_score,
        "secondary_keywords": [kw.phrase for kw in plan.secondary],
        "question_keywords": [kw.phrase for kw in plan.questions],
        "total_candidates": result.total_candidates,
        "total_relevant": result.total_relevant,
        "total_skipped": result.total_skipped,
        "clusters_formed": result.total_clustered,
    }

    if result.fingerprint:
        summary["topic_summary"] = result.fingerprint.summary
        summary["page_type"] = result.fingerprint.page_type

    if result.intent:
        summary["intent"] = result.intent.primary_intent
        summary["conversion_goal"] = result.intent.conversion_goal

    # Add skip summary
    skip_summary = {}
    for reason, keywords in result.skip_details.items():
        if keywords:
            skip_summary[reason] = len(keywords)
    if skip_summary:
        summary["skip_summary"] = skip_summary

    return summary
