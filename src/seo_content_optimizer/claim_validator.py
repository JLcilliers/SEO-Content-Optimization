"""
Claim validation module for preventing hallucination.

Extracts verifiable claims from source content and validates that
LLM-generated content does not introduce unsupported claims.

Types of claims tracked:
- Numbers/statistics/percentages
- Dates/years
- Proper nouns (company names, products, certifications)
- Comparative claims ("better than", "fastest", "only")
- Medical/legal/financial claims
- Location/contact information
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Literal

# Claim types
ClaimType = Literal[
    "number",       # Numbers, stats, percentages
    "date",         # Years, dates
    "entity",       # Proper nouns, brands, products
    "comparative",  # "best", "fastest", "only"
    "medical",      # Health/medical claims
    "legal",        # Legal/compliance claims
    "financial",    # Pricing, costs, financial claims
    "location",     # Addresses, phone numbers
    "certification",# Awards, certifications, accreditations
]


@dataclass
class ExtractedClaim:
    """A verifiable claim extracted from source content."""
    claim_type: ClaimType
    value: str
    context: str  # Surrounding text for context
    source_position: int = 0  # Character position in source


@dataclass
class ClaimViolation:
    """A claim in generated content that wasn't in source."""
    claim_type: ClaimType
    claim_text: str
    context: str
    severity: str  # "high", "medium", "low"
    recommendation: str


@dataclass
class FactsLedger:
    """Collection of verifiable facts from source content."""
    numbers: set[str] = field(default_factory=set)
    dates: set[str] = field(default_factory=set)
    entities: set[str] = field(default_factory=set)
    comparatives: set[str] = field(default_factory=set)
    locations: set[str] = field(default_factory=set)
    certifications: set[str] = field(default_factory=set)
    raw_claims: list[ExtractedClaim] = field(default_factory=list)

    def contains_number(self, num: str) -> bool:
        """Check if a number/stat is in the ledger."""
        # Normalize number for comparison
        normalized = num.lower().replace(",", "").replace(" ", "")
        return any(
            normalized in n.lower().replace(",", "").replace(" ", "")
            for n in self.numbers
        )

    def contains_entity(self, entity: str) -> bool:
        """Check if an entity is in the ledger."""
        entity_lower = entity.lower()
        return any(entity_lower in e.lower() for e in self.entities)

    def contains_year(self, year: str) -> bool:
        """Check if a year is in the ledger."""
        return year in self.dates


# Patterns for extracting claims
NUMBER_PATTERNS = [
    r"\d+(?:,\d{3})*(?:\.\d+)?%",  # Percentages
    r"\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|B|K))?",  # Currency
    r"\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|times|x|users|customers|years|months|days)",  # Stats
    r"#\d+\s+(?:in|ranked)",  # Rankings
    r"\d+(?:\.\d+)?\s*(?:stars?|rating)",  # Ratings
]

DATE_PATTERNS = [
    r"\b(?:19|20)\d{2}\b",  # Years
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
    r"\d{1,2}/\d{1,2}/\d{2,4}",  # Date formats
]

COMPARATIVE_PATTERNS = [
    r"\b(?:only|first|best|leading|fastest|most|largest|top|#1|number one|revolutionary)\b",
    r"\bbetter than\b",
    r"\bsuperior to\b",
    r"\bunlike (?:other|any)\b",
    r"\bthe (?:only|first)\b",
]

LOCATION_PATTERNS = [
    r"\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)\b",
    r"\(\d{3}\)\s*\d{3}[-.]?\d{4}",  # Phone numbers
    r"\d{3}[-.]?\d{3}[-.]?\d{4}",  # Phone numbers alt
    r"\b[A-Z][a-z]+(?:,\s*[A-Z]{2})?\s+\d{5}(?:-\d{4})?\b",  # ZIP codes
]

CERTIFICATION_PATTERNS = [
    r"\b(?:certified|accredited|licensed|registered|approved)\b",
    r"\bISO\s*\d+",
    r"\bFDA\s+(?:approved|cleared|registered)",
    r"\b(?:award|prize|recognition)\b",
]

MEDICAL_PATTERNS = [
    r"\b(?:clinical|trial|study|research|proven|FDA|medical|health|treatment|therapy|cure|heal)\b",
    r"\b(?:doctor|physician|nurse|patient)\s+(?:recommended|approved)",
]

FINANCIAL_PATTERNS = [
    r"\b(?:guarantee|refund|free|discount|save|cost|price|fee)\b",
    r"\b(?:ROI|return on investment|profit|revenue|growth)\b",
]


def extract_facts_ledger(source_text: str) -> FactsLedger:
    """
    Extract all verifiable claims from source text.

    Creates a "facts ledger" that can be used to validate
    LLM output for hallucinated claims.

    Args:
        source_text: Original source content.

    Returns:
        FactsLedger with all extracted claims.
    """
    ledger = FactsLedger()

    # Extract numbers and statistics
    for pattern in NUMBER_PATTERNS:
        for match in re.finditer(pattern, source_text, re.IGNORECASE):
            value = match.group()
            ledger.numbers.add(value)
            ledger.raw_claims.append(ExtractedClaim(
                claim_type="number",
                value=value,
                context=_get_context(source_text, match.start(), match.end()),
                source_position=match.start(),
            ))

    # Extract dates/years
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, source_text):
            value = match.group()
            ledger.dates.add(value)
            ledger.raw_claims.append(ExtractedClaim(
                claim_type="date",
                value=value,
                context=_get_context(source_text, match.start(), match.end()),
                source_position=match.start(),
            ))

    # Extract comparatives (these are claims that need to be preserved exactly)
    for pattern in COMPARATIVE_PATTERNS:
        for match in re.finditer(pattern, source_text, re.IGNORECASE):
            value = match.group()
            ledger.comparatives.add(value.lower())
            ledger.raw_claims.append(ExtractedClaim(
                claim_type="comparative",
                value=value,
                context=_get_context(source_text, match.start(), match.end()),
                source_position=match.start(),
            ))

    # Extract locations
    for pattern in LOCATION_PATTERNS:
        for match in re.finditer(pattern, source_text):
            value = match.group()
            ledger.locations.add(value)
            ledger.raw_claims.append(ExtractedClaim(
                claim_type="location",
                value=value,
                context=_get_context(source_text, match.start(), match.end()),
                source_position=match.start(),
            ))

    # Extract certifications
    for pattern in CERTIFICATION_PATTERNS:
        for match in re.finditer(pattern, source_text, re.IGNORECASE):
            value = match.group()
            ledger.certifications.add(value.lower())
            ledger.raw_claims.append(ExtractedClaim(
                claim_type="certification",
                value=value,
                context=_get_context(source_text, match.start(), match.end()),
                source_position=match.start(),
            ))

    # Extract proper nouns/entities (capitalized words that could be brands/products)
    # This is a simple heuristic - can be improved with NER
    entity_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b"
    for match in re.finditer(entity_pattern, source_text):
        value = match.group()
        # Filter out common words that start sentences
        if value.lower() not in {"the", "this", "that", "these", "those", "it", "we", "our", "your"}:
            ledger.entities.add(value)

    return ledger


def validate_generated_content(
    generated_text: str,
    facts_ledger: FactsLedger,
    strict: bool = False,
) -> tuple[list[ClaimViolation], str]:
    """
    Validate generated content against the facts ledger.

    Identifies claims in generated text that weren't in source.

    Args:
        generated_text: LLM-generated content to validate.
        facts_ledger: Facts extracted from source.
        strict: If True, flag all new claims. If False, only flag high-severity.

    Returns:
        Tuple of (list of violations, cleaned text with violations removed/flagged).
    """
    violations = []
    cleaned_text = generated_text

    # Check for new numbers/stats not in source
    for pattern in NUMBER_PATTERNS:
        for match in re.finditer(pattern, generated_text, re.IGNORECASE):
            value = match.group()
            if not facts_ledger.contains_number(value):
                violations.append(ClaimViolation(
                    claim_type="number",
                    claim_text=value,
                    context=_get_context(generated_text, match.start(), match.end()),
                    severity="high",
                    recommendation="Remove or verify this statistic - not found in source",
                ))

    # Check for new years not in source
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, generated_text):
            value = match.group()
            if not facts_ledger.contains_year(value):
                # Years that are clearly current year or close are usually OK
                import datetime
                current_year = datetime.datetime.now().year
                try:
                    year = int(re.search(r"\d{4}", value).group())
                    if abs(year - current_year) <= 1:
                        continue  # Allow current year references
                except:
                    pass

                violations.append(ClaimViolation(
                    claim_type="date",
                    claim_text=value,
                    context=_get_context(generated_text, match.start(), match.end()),
                    severity="medium",
                    recommendation="Verify this date/year reference",
                ))

    # Check for NEW comparative claims (not in source)
    for pattern in COMPARATIVE_PATTERNS:
        for match in re.finditer(pattern, generated_text, re.IGNORECASE):
            value = match.group().lower()
            if value not in facts_ledger.comparatives:
                violations.append(ClaimViolation(
                    claim_type="comparative",
                    claim_text=match.group(),
                    context=_get_context(generated_text, match.start(), match.end()),
                    severity="medium",
                    recommendation="Remove superlative/comparative claim - not in source",
                ))

    # Remove/flag violations in cleaned text if strict mode
    if strict and violations:
        for v in violations:
            if v.severity == "high":
                # Remove high-severity violations
                cleaned_text = cleaned_text.replace(v.claim_text, "[REMOVED]")

    return violations, cleaned_text


def remove_hallucinated_claims(
    generated_text: str,
    facts_ledger: FactsLedger,
) -> str:
    """
    Remove hallucinated claims from generated text.

    This is a more aggressive cleanup that removes sentences
    containing unsupported claims.

    Args:
        generated_text: LLM-generated content.
        facts_ledger: Facts from source.

    Returns:
        Cleaned text with hallucinated claims removed.
    """
    violations, _ = validate_generated_content(generated_text, facts_ledger, strict=True)

    if not violations:
        return generated_text

    result = generated_text

    # For high-severity violations, remove the containing sentence
    high_severity = [v for v in violations if v.severity == "high"]

    for v in high_severity:
        # Find the sentence containing this claim
        # Simple sentence boundary detection
        claim_pos = result.find(v.claim_text)
        if claim_pos == -1:
            continue

        # Find sentence boundaries
        start = result.rfind(".", 0, claim_pos)
        end = result.find(".", claim_pos)

        if start == -1:
            start = 0
        else:
            start += 1  # Skip the period

        if end == -1:
            end = len(result)
        else:
            end += 1  # Include the period

        sentence = result[start:end].strip()

        # Remove the sentence
        result = result[:start] + result[end:]

    # Clean up double spaces and empty lines
    result = re.sub(r" +", " ", result)
    result = re.sub(r"\n\s*\n", "\n\n", result)

    return result.strip()


def _get_context(text: str, start: int, end: int, window: int = 50) -> str:
    """Get surrounding context for a match."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    return text[ctx_start:ctx_end]


def get_allowed_claims(facts_ledger: FactsLedger) -> str:
    """
    Generate a summary of allowed claims for LLM prompt.

    This can be included in the LLM prompt to constrain
    what claims it can make.

    Args:
        facts_ledger: Extracted facts.

    Returns:
        Formatted string of allowed claims.
    """
    parts = []

    if facts_ledger.numbers:
        parts.append(f"ALLOWED NUMBERS/STATS: {', '.join(list(facts_ledger.numbers)[:10])}")

    if facts_ledger.dates:
        parts.append(f"ALLOWED YEARS/DATES: {', '.join(list(facts_ledger.dates)[:5])}")

    if facts_ledger.comparatives:
        parts.append(f"ALLOWED SUPERLATIVES: {', '.join(list(facts_ledger.comparatives)[:5])}")

    if facts_ledger.certifications:
        parts.append(f"ALLOWED CERTIFICATIONS: {', '.join(list(facts_ledger.certifications)[:5])}")

    if not parts:
        return "No specific claims found in source. Do not add new statistics, dates, or superlatives."

    return "\n".join(parts) + "\n\nDO NOT introduce new numbers, dates, or comparative claims not listed above."
