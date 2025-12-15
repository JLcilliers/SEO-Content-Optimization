"""
Factuality guardrail system for SEO Content Optimizer V2 Architecture.

This module prevents hallucinated facts from being introduced during optimization:
- Detects factual claims (statistics, dates, numbers, specific names)
- Compares claims between original and modified text
- Flags new claims not present in the original
- Provides callbacks for block rewriter to validate changes

The goal is: NO NEW FACTS - only add keywords to existing content.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Callable

from .models import FactualityClaim

logger = logging.getLogger(__name__)


# Patterns for detecting factual claims
CLAIM_PATTERNS = {
    # Statistics and percentages
    "percentage": r'\b(\d+(?:\.\d+)?)\s*%',
    "statistic": r'\b(statistic(?:s|ally)?|data\s+shows?|research\s+(?:shows?|indicates?|suggests?))',

    # Numbers with units
    "number_with_unit": r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|trillion|thousand|hundred|k|m|b|dollars?|\$|€|£|years?|months?|days?|hours?|minutes?|people|users?|customers?|employees?)\b',

    # Dates and years
    "specific_year": r'\b(19|20)\d{2}\b',
    "date_pattern": r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b',
    "founded_date": r'\b(founded|established|started|launched|created)\s+(?:in\s+)?(\d{4})\b',

    # Specific quantities
    "quantity": r'\b(first|second|third|largest|smallest|oldest|newest|#\d+|number\s+\d+|top\s+\d+)\b',
    "ranking": r'\b(ranked?\s+#?\d+|#\d+\s+in|top\s+\d+)\b',

    # Certifications and awards
    "certification": r'\b(certified|accredited|licensed|award[- ]winning|ISO\s*\d+|FDA|EPA|certified\s+by)\b',
    "award": r'\b(won|received|awarded|winner|finalist)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Award|Prize|Medal))\b',

    # Named entities (potential fact sources)
    "study_reference": r'\b(according\s+to|study\s+(?:by|from)|research\s+(?:by|from)|report\s+(?:by|from)|survey\s+(?:by|from))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
    "quote_attribution": r'"[^"]+"\s*[-–—]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',

    # Comparisons with specifics
    "comparison": r'\b(\d+(?:\.\d+)?\s*(?:x|times|percent|%)\s+(?:more|less|better|worse|faster|slower))\b',

    # Contact information (should be preserved, not added)
    "phone": r'\b(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b',
    "email": r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
    "address": r'\b(\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way))?)\b',
}

# Claim severity levels
SEVERITY_HIGH = "high"  # Statistics, dates, numbers - very likely to be wrong if hallucinated
SEVERITY_MEDIUM = "medium"  # Certifications, awards - could be problematic
SEVERITY_LOW = "low"  # General claims that might be opinion


@dataclass
class ClaimDetectionConfig:
    """Configuration for claim detection."""
    detect_statistics: bool = True
    detect_dates: bool = True
    detect_numbers: bool = True
    detect_certifications: bool = True
    detect_contact_info: bool = True
    detect_quotes: bool = True
    min_number_to_flag: int = 10  # Numbers below this aren't flagged
    allow_common_numbers: bool = True  # Allow 100%, 24/7, etc.


@dataclass
class FactualityCheckResult:
    """Result of factuality check between original and modified text."""
    is_valid: bool
    original_claims: list[FactualityClaim]
    modified_claims: list[FactualityClaim]
    new_claims: list[FactualityClaim]  # Claims in modified but not original
    removed_claims: list[FactualityClaim]  # Claims in original but not modified
    warnings: list[str] = field(default_factory=list)


class FactualityChecker:
    """
    Detects and validates factual claims in text.

    Used to prevent LLM from hallucinating facts during rewriting.
    """

    def __init__(self, config: Optional[ClaimDetectionConfig] = None):
        """
        Initialize factuality checker.

        Args:
            config: Detection configuration.
        """
        self.config = config or ClaimDetectionConfig()
        self._common_numbers = self._build_common_numbers_set()

    def detect_claims(self, text: str) -> list[FactualityClaim]:
        """
        Detect all factual claims in text.

        Args:
            text: Text to analyze.

        Returns:
            List of detected FactualityClaim objects.
        """
        claims = []

        # Process each pattern type
        if self.config.detect_statistics:
            claims.extend(self._detect_statistics(text))

        if self.config.detect_dates:
            claims.extend(self._detect_dates(text))

        if self.config.detect_numbers:
            claims.extend(self._detect_numbers(text))

        if self.config.detect_certifications:
            claims.extend(self._detect_certifications(text))

        if self.config.detect_contact_info:
            claims.extend(self._detect_contact_info(text))

        if self.config.detect_quotes:
            claims.extend(self._detect_quotes(text))

        # Deduplicate claims
        seen = set()
        unique_claims = []
        for claim in claims:
            key = (claim.claim_text, claim.claim_type)
            if key not in seen:
                seen.add(key)
                unique_claims.append(claim)

        return unique_claims

    def compare_claims(
        self,
        original_text: str,
        modified_text: str,
    ) -> FactualityCheckResult:
        """
        Compare factual claims between original and modified text.

        Args:
            original_text: Original text before modification.
            modified_text: Text after modification.

        Returns:
            FactualityCheckResult with detailed comparison.
        """
        original_claims = self.detect_claims(original_text)
        modified_claims = self.detect_claims(modified_text)

        # Normalize for comparison
        original_normalized = {
            self._normalize_claim(c.claim_text): c
            for c in original_claims
        }
        modified_normalized = {
            self._normalize_claim(c.claim_text): c
            for c in modified_claims
        }

        # Find new claims (in modified but not original)
        new_claims = []
        for norm_text, claim in modified_normalized.items():
            if norm_text not in original_normalized:
                new_claims.append(claim)

        # Find removed claims (in original but not modified)
        removed_claims = []
        for norm_text, claim in original_normalized.items():
            if norm_text not in modified_normalized:
                removed_claims.append(claim)

        # Generate warnings
        warnings = []
        for claim in new_claims:
            if claim.severity == SEVERITY_HIGH:
                warnings.append(
                    f"HIGH RISK: New {claim.claim_type} claim added: '{claim.claim_text}'"
                )
            elif claim.severity == SEVERITY_MEDIUM:
                warnings.append(
                    f"MEDIUM RISK: New {claim.claim_type} claim: '{claim.claim_text}'"
                )

        # Determine validity - invalid if any high-severity new claims
        is_valid = not any(c.severity == SEVERITY_HIGH for c in new_claims)

        return FactualityCheckResult(
            is_valid=is_valid,
            original_claims=original_claims,
            modified_claims=modified_claims,
            new_claims=new_claims,
            removed_claims=removed_claims,
            warnings=warnings,
        )

    def _detect_statistics(self, text: str) -> list[FactualityClaim]:
        """Detect statistical claims."""
        claims = []

        # Percentages
        for match in re.finditer(CLAIM_PATTERNS["percentage"], text, re.IGNORECASE):
            value = match.group(1)
            # Skip common percentages
            if self.config.allow_common_numbers and value in {"100", "0", "50"}:
                continue

            # Get context (surrounding sentence)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="percentage",
                source_block_id="",
                source_sentence=context,
                severity=SEVERITY_HIGH,
                is_new=False,
            ))

        # Research/study mentions
        for match in re.finditer(CLAIM_PATTERNS["statistic"], text, re.IGNORECASE):
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 100)
            context = text[start:end]

            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="research_reference",
                source_block_id="",
                source_sentence=context,
                severity=SEVERITY_HIGH,
                is_new=False,
            ))

        return claims

    def _detect_dates(self, text: str) -> list[FactualityClaim]:
        """Detect date claims."""
        claims = []

        # Specific years
        for match in re.finditer(CLAIM_PATTERNS["specific_year"], text):
            year = match.group(0)
            # Skip very recent years (might be current context)
            if int(year) >= 2023:
                continue

            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]

            claims.append(FactualityClaim(
                claim_text=year,
                claim_type="year",
                source_block_id="",
                source_sentence=context,
                severity=SEVERITY_HIGH,
                is_new=False,
            ))

        # Founded dates
        for match in re.finditer(CLAIM_PATTERNS["founded_date"], text, re.IGNORECASE):
            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="founding_date",
                source_block_id="",
                source_sentence=match.group(0),
                severity=SEVERITY_HIGH,
                is_new=False,
            ))

        return claims

    def _detect_numbers(self, text: str) -> list[FactualityClaim]:
        """Detect numeric claims with units."""
        claims = []

        # Numbers with units
        for match in re.finditer(CLAIM_PATTERNS["number_with_unit"], text, re.IGNORECASE):
            number_str = match.group(1).replace(",", "")
            try:
                number = float(number_str)
                if number < self.config.min_number_to_flag:
                    continue
            except ValueError:
                continue

            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="number_with_unit",
                source_block_id="",
                source_sentence=match.group(0),
                severity=SEVERITY_HIGH,
                is_new=False,
            ))

        # Rankings
        for match in re.finditer(CLAIM_PATTERNS["ranking"], text, re.IGNORECASE):
            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="ranking",
                source_block_id="",
                source_sentence=match.group(0),
                severity=SEVERITY_HIGH,
                is_new=False,
            ))

        return claims

    def _detect_certifications(self, text: str) -> list[FactualityClaim]:
        """Detect certification and award claims."""
        claims = []

        # Certifications
        for match in re.finditer(CLAIM_PATTERNS["certification"], text, re.IGNORECASE):
            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="certification",
                source_block_id="",
                source_sentence=match.group(0),
                severity=SEVERITY_MEDIUM,
                is_new=False,
            ))

        # Awards
        for match in re.finditer(CLAIM_PATTERNS["award"], text, re.IGNORECASE):
            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="award",
                source_block_id="",
                source_sentence=match.group(0),
                severity=SEVERITY_MEDIUM,
                is_new=False,
            ))

        return claims

    def _detect_contact_info(self, text: str) -> list[FactualityClaim]:
        """Detect contact information (should be preserved, never added)."""
        claims = []

        # Phone numbers
        for match in re.finditer(CLAIM_PATTERNS["phone"], text):
            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="phone",
                source_block_id="",
                source_sentence=match.group(0),
                severity=SEVERITY_HIGH,  # High because adding wrong contact is serious
                is_new=False,
            ))

        # Email addresses
        for match in re.finditer(CLAIM_PATTERNS["email"], text):
            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="email",
                source_block_id="",
                source_sentence=match.group(0),
                severity=SEVERITY_HIGH,
                is_new=False,
            ))

        return claims

    def _detect_quotes(self, text: str) -> list[FactualityClaim]:
        """Detect quoted text with attribution."""
        claims = []

        # Study/research references
        for match in re.finditer(CLAIM_PATTERNS["study_reference"], text, re.IGNORECASE):
            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="research_citation",
                source_block_id="",
                source_sentence=match.group(0),
                severity=SEVERITY_HIGH,
                is_new=False,
            ))

        # Quote attributions
        for match in re.finditer(CLAIM_PATTERNS["quote_attribution"], text):
            claims.append(FactualityClaim(
                claim_text=match.group(0),
                claim_type="quote",
                source_block_id="",
                source_sentence=match.group(0),
                severity=SEVERITY_MEDIUM,
                is_new=False,
            ))

        return claims

    def _normalize_claim(self, claim_text: str) -> str:
        """Normalize claim text for comparison."""
        # Lowercase, remove extra whitespace
        normalized = re.sub(r'\s+', ' ', claim_text.lower().strip())
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized

    def _build_common_numbers_set(self) -> set[str]:
        """Build set of common numbers that shouldn't be flagged."""
        return {
            "100", "0", "50",  # Common percentages
            "24", "7",  # 24/7
            "1", "2", "3", "4", "5", "10",  # Single digits, 10
            "12", "365",  # Time-related
        }


def create_factuality_callback(
    config: Optional[ClaimDetectionConfig] = None,
) -> Callable[[str], list[FactualityClaim]]:
    """
    Create a factuality check callback for use with BlockRewriter.

    Args:
        config: Optional detection configuration.

    Returns:
        Callable that takes text and returns list of claims.
    """
    checker = FactualityChecker(config)
    return checker.detect_claims


def validate_no_new_facts(
    original_text: str,
    modified_text: str,
    config: Optional[ClaimDetectionConfig] = None,
) -> tuple[bool, list[str]]:
    """
    Convenience function to validate no new facts were added.

    Args:
        original_text: Original text.
        modified_text: Modified text.
        config: Optional detection configuration.

    Returns:
        Tuple of (is_valid, list_of_warnings).
    """
    checker = FactualityChecker(config)
    result = checker.compare_claims(original_text, modified_text)
    return result.is_valid, result.warnings


def get_claim_summary(claims: list[FactualityClaim]) -> dict:
    """
    Generate a summary of detected claims.

    Args:
        claims: List of claims to summarize.

    Returns:
        Dictionary with claim type counts and severity breakdown.
    """
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {"high": 0, "medium": 0, "low": 0}

    for claim in claims:
        by_type[claim.claim_type] = by_type.get(claim.claim_type, 0) + 1
        by_severity[claim.severity] = by_severity.get(claim.severity, 0) + 1

    return {
        "total_claims": len(claims),
        "by_type": by_type,
        "by_severity": by_severity,
        "high_risk_count": by_severity["high"],
    }


def log_factuality_result(result: FactualityCheckResult) -> None:
    """
    Log factuality check result for debugging.

    Args:
        result: FactualityCheckResult to log.
    """
    logger.info(f"Factuality check: valid={result.is_valid}")
    logger.info(f"  Original claims: {len(result.original_claims)}")
    logger.info(f"  Modified claims: {len(result.modified_claims)}")
    logger.info(f"  New claims: {len(result.new_claims)}")
    logger.info(f"  Removed claims: {len(result.removed_claims)}")

    for warning in result.warnings:
        logger.warning(f"  {warning}")
