"""
Diff-based marker insertion for accurate highlighting.

V2 SENTENCE-LEVEL SEMANTICS (Simple, Predictable):
============================================

This module computes differences between original and rewritten text
and inserts [[[ADD]]]/[[[ENDADD]]] markers around changed content.

KEY PRINCIPLE: Sentence is the unit of change.
KEY RULE: STRICT EXACT MATCH ONLY - NO similarity thresholds.

For each sentence S in rewritten text:
1. Normalize S (NFKC, collapse whitespace, lowercase, normalize punctuation)
2. If normalized S exists in original_sentence_index → UNCHANGED (no markers)
3. Else → CHANGED/NEW → wrap ENTIRE sentence in markers

NO token-level diff inside sentences. A sentence is either:
- Fully unchanged (black text) - ONLY if EXACT match after normalization
- Fully changed (green highlight) - if ANY difference at all

This eliminates confusing partial highlights and makes the output predictable.

Key guarantees:
- Green = new/changed sentences (ANY change = full green)
- Black = sentences IDENTICAL to original (after normalization)
- FAQ content is always fully green (it's all new)
- Meta elements are all-or-nothing (changed = full green)
- Punctuation-only changes (curly quotes, smart apostrophes) don't trigger highlighting
  (handled by normalization, not similarity thresholds)

IMPORTANT: There are NO similarity thresholds. A sentence must be an EXACT match
(after normalization) to remain black. Any difference = full green highlight.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

# Marker constants - must match llm_client.py
MARK_START = "[[[ADD]]]"
MARK_END = "[[[ENDADD]]]"

# Tokenization regex: split into whitespace and non-whitespace tokens
TOKEN_RE = re.compile(r"\s+|[^\s]+")

# Sentence boundary pattern - splits on .!? followed by space(s)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])(\s+)")

# DEPRECATED: Similarity thresholds are no longer used in V2 sentence-level diff
# V2 uses STRICT EXACT MATCH only (after normalization)
# These constants are kept for backward compatibility but are NOT used
DEFAULT_SIMILARITY_THRESHOLD = 0.80  # DEPRECATED - not used in V2
MIN_WORDS_FOR_PHRASE_DIFF = 6  # DEPRECATED - not used in V2

# Punctuation normalization map for diff comparison
# Maps typographic/smart punctuation to ASCII equivalents
# Using explicit Unicode code points to avoid encoding issues
PUNCT_NORMALIZE_MAP = str.maketrans({
    "\u2019": "'",  # Right single quote → straight quote
    "\u2018": "'",  # Left single quote → straight quote
    "\u201B": "'",  # Single high-reversed-9 quotation mark → straight quote
    "\u201C": '"',  # Left double quote → straight double quote
    "\u201D": '"',  # Right double quote → straight double quote
    "\u201F": '"',  # Double high-reversed-9 quotation mark → straight double quote
    "\u2013": "-",  # En dash → hyphen
    "\u2014": "-",  # Em dash → hyphen
})


def generate_brand_variations(brand_name: str) -> set[str]:
    """
    Generate all common variations of a brand name for exclusion matching.

    Given a brand name like "CellGate", generates:
    - CellGate, cellgate, CELLGATE (case variations)
    - Cell-Gate, cell-gate, CELL-GATE (hyphenated)
    - Cell Gate, cell gate, CELL GATE (spaced)

    Args:
        brand_name: The original brand name.

    Returns:
        Set of all brand name variations (lowercase for comparison).
    """
    if not brand_name:
        return set()

    variations = set()

    # Clean the input brand name
    clean_brand = brand_name.strip()
    if not clean_brand:
        return variations

    # Add the original as-is
    variations.add(clean_brand.lower())

    # Split on common separators (hyphen, space) and CamelCase
    import re as _re

    # Split CamelCase: "CellGate" -> ["Cell", "Gate"]
    parts = _re.findall(r'[A-Z][a-z]*|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', clean_brand)

    # If no CamelCase parts found, try splitting on hyphens/spaces
    if len(parts) <= 1:
        parts = _re.split(r'[-\s]+', clean_brand)

    if len(parts) >= 2:
        # Generate variations from parts
        # CamelCase: CellGate
        camel = ''.join(p.capitalize() for p in parts)
        variations.add(camel.lower())

        # Hyphenated: Cell-Gate
        hyphenated = '-'.join(p.capitalize() for p in parts)
        variations.add(hyphenated.lower())

        # Spaced: Cell Gate
        spaced = ' '.join(p.capitalize() for p in parts)
        variations.add(spaced.lower())

        # All lowercase joined: cellgate
        variations.add(''.join(parts).lower())

        # All uppercase: CELLGATE
        variations.add(''.join(parts).upper().lower())  # Store lowercase for comparison

    return variations


def normalize_brand_in_text(text: str, original_brand: str, brand_variations: set[str]) -> str:
    """
    Normalize all brand name variations in text to match the original spelling.

    This ensures the LLM output uses the EXACT original brand spelling,
    preventing diff from detecting brand name changes.

    Args:
        text: Text that may contain brand variations.
        original_brand: The original brand name spelling to use.
        brand_variations: Set of lowercase brand variations to match.

    Returns:
        Text with all brand variations normalized to original spelling.
    """
    if not text or not original_brand or not brand_variations:
        return text

    import re as _re

    result = text

    # Sort variations by length (longest first) to avoid partial replacements
    sorted_variations = sorted(brand_variations, key=len, reverse=True)

    for variation in sorted_variations:
        # Create case-insensitive pattern for this variation
        # Escape special regex chars in the variation
        pattern = _re.escape(variation)
        # Find all occurrences (case-insensitive)
        matches = list(_re.finditer(pattern, result, flags=_re.IGNORECASE))

        # Replace from end to start to preserve indices
        for match in reversed(matches):
            result = result[:match.start()] + original_brand + result[match.end():]

    return result


def is_brand_token(token: str, brand_variations: set[str]) -> bool:
    """
    Check if a token is a brand name variation.

    Args:
        token: Token to check.
        brand_variations: Set of lowercase brand variations.

    Returns:
        True if token is a brand name variation.
    """
    if not token or not brand_variations:
        return False

    # Normalize token for comparison (strip punctuation, lowercase)
    clean_token = token.strip('.,;:!?()[]{}"\'-').lower()
    return clean_token in brand_variations


def normalize_token_for_diff(token: str) -> str:
    """
    Normalize a token for diff comparison purposes.

    Applies Unicode NFKC normalization and maps typographic punctuation
    to ASCII equivalents. This prevents false positives from curly quotes,
    smart apostrophes, and other typographic variations.

    Args:
        token: The token to normalize.

    Returns:
        Normalized token for comparison (not for output).
    """
    if not token:
        return token
    # Apply Unicode NFKC normalization (compatibility decomposition + composition)
    normalized = unicodedata.normalize("NFKC", token)
    # Apply punctuation mapping
    normalized = normalized.translate(PUNCT_NORMALIZE_MAP)
    # Handle ellipsis separately (maps to multiple chars)
    normalized = normalized.replace("…", "...")
    return normalized


def preprocess_keywords_for_diff(
    text: str,
    keywords: list[str],
) -> tuple[str, dict[str, str]]:
    """
    Replace multi-word keywords with single atomic tokens before diff.

    This ensures multi-word keyword phrases like "gate security camera"
    are treated as single units during diff comparison, preventing
    partial highlighting like "[gate security]{.mark} camera".

    Args:
        text: Text to preprocess.
        keywords: List of keyword phrases to treat as atomic units.

    Returns:
        Tuple of (preprocessed_text, token_map) where token_map
        maps tokens back to original keywords.
    """
    if not text or not keywords:
        return text, {}

    token_map: dict[str, str] = {}
    result = text

    # Sort keywords by length (longest first) to avoid partial replacements
    sorted_keywords = sorted(
        [kw for kw in keywords if kw and " " in kw],  # Only multi-word keywords
        key=len,
        reverse=True,
    )

    for i, keyword in enumerate(sorted_keywords):
        if not keyword.strip():
            continue

        # Create unique token for this keyword
        token = f"__KWPHRASE_{i}__"
        token_map[token] = keyword

        # Replace all occurrences (case-insensitive, preserve original case)
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        def preserve_case_replace(match: re.Match) -> str:
            """Replace keeping track for postprocessing."""
            return token

        result = pattern.sub(preserve_case_replace, result)

    return result, token_map


def postprocess_keywords_from_diff(
    text: str,
    token_map: dict[str, str],
) -> str:
    """
    Restore original keyword phrases from atomic tokens after diff.

    Reverses the preprocessing done by preprocess_keywords_for_diff.

    Args:
        text: Text with atomic tokens and markers.
        token_map: Map from tokens to original keyword phrases.

    Returns:
        Text with keywords restored.
    """
    if not text or not token_map:
        return text

    result = text
    for token, keyword in token_map.items():
        result = result.replace(token, keyword)

    return result


def normalize_sentence(s: str) -> str:
    """
    Normalize a sentence for equality comparison.

    Lowercased, single-spaced, strip leading/trailing whitespace.
    Also normalizes punctuation (curly quotes → straight quotes, etc.).

    Args:
        s: The sentence to normalize.

    Returns:
        Normalized sentence string.
    """
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # Normalize punctuation before lowercasing
    s = normalize_token_for_diff(s)
    return s.lower()


def build_original_sentence_index(full_original_text: str) -> set[str]:
    """
    Build a set of normalized sentences from the full original text.

    Used to quickly check if a sentence existed anywhere in the original.

    Args:
        full_original_text: The complete original document text.

    Returns:
        Set of normalized sentence strings.
    """
    if not full_original_text:
        return set()
    sentences = split_into_sentences(full_original_text)
    return {normalize_sentence(s) for s in sentences if s.strip()}


def find_most_similar_sentence(
    sentence: str,
    candidates: list[str]
) -> tuple[Optional[str], float]:
    """
    Find the most similar sentence from a list of candidates.

    Args:
        sentence: The sentence to match.
        candidates: List of candidate sentences to compare against.

    Returns:
        Tuple of (most_similar_sentence, similarity_ratio).
        Returns (None, 0.0) if no candidates.
    """
    if not candidates or not sentence:
        return None, 0.0

    best_match = None
    best_ratio = 0.0

    norm_sentence = normalize_sentence(sentence)

    for candidate in candidates:
        norm_candidate = normalize_sentence(candidate)
        if not norm_candidate:
            continue
        ratio = SequenceMatcher(None, norm_sentence, norm_candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    return best_match, best_ratio


def tokenize(text: str) -> list[str]:
    """
    Split text into a list of tokens where:
    - whitespace (spaces/newlines) is kept as separate tokens
    - non-whitespace chunks are separate tokens

    This preserves positions and punctuation for accurate diffs.

    Args:
        text: Input text to tokenize.

    Returns:
        List of tokens (whitespace and words/punctuation).
    """
    if not text:
        return []
    return TOKEN_RE.findall(text)


def add_markers_by_diff(
    original: str,
    rewritten: str,
    brand_variations: Optional[set[str]] = None,
    keywords: Optional[list[str]] = None,
) -> str:
    """
    Compare original and rewritten text and return rewritten text with
    [[[ADD]]]/[[[ENDADD]]] markers around inserted/replaced segments.

    This guarantees:
    - Only real inserted/replaced tokens get wrapped
    - Any token that existed unchanged in original remains unwrapped
    - No more "highlighting existing keywords" problem
    - Brand name variations are NEVER highlighted (excluded from diff)
    - Multi-word keywords are treated as atomic units (not partially highlighted)

    Args:
        original: The original text before optimization.
        rewritten: The rewritten/optimized text (plain, no markers).
        brand_variations: Optional set of lowercase brand name variations to exclude
                         from highlighting. E.g., {"cellgate", "cell-gate", "cell gate"}
        keywords: Optional list of keyword phrases to treat as atomic units.
                  Multi-word keywords will not be partially highlighted.

    Returns:
        The rewritten text with markers around changed portions.
    """
    if not original:
        # If no original, everything is new
        if rewritten:
            return f"{MARK_START}{rewritten}{MARK_END}"
        return ""

    if not rewritten:
        return ""

    # If texts are identical, no markers needed
    if original.strip() == rewritten.strip():
        return rewritten

    # Preprocess: Replace multi-word keywords with atomic tokens
    # This prevents partial highlighting like "[gate security]{.mark} camera"
    orig_preprocessed = original
    new_preprocessed = rewritten
    token_map: dict[str, str] = {}

    if keywords:
        orig_preprocessed, token_map_orig = preprocess_keywords_for_diff(original, keywords)
        new_preprocessed, token_map_new = preprocess_keywords_for_diff(rewritten, keywords)
        # Combine token maps (they should be the same)
        token_map = {**token_map_orig, **token_map_new}

    orig_tokens = tokenize(orig_preprocessed)
    new_tokens = tokenize(new_preprocessed)

    # Normalize tokens for comparison (handles curly quotes, smart apostrophes, etc.)
    # We compare normalized versions but output the original tokens
    orig_tokens_normalized = [normalize_token_for_diff(t) for t in orig_tokens]
    new_tokens_normalized = [normalize_token_for_diff(t) for t in new_tokens]

    sm = SequenceMatcher(None, orig_tokens_normalized, new_tokens_normalized, autojunk=False)
    out: list[str] = []
    in_add = False

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # Close any open add block
            if in_add:
                out.append(MARK_END)
                in_add = False
            out.extend(new_tokens[j1:j2])
        elif tag in {"insert", "replace"}:
            # Check if this is a brand name variation change (should NOT be highlighted)
            if brand_variations and tag == "replace":
                # Check if ALL replaced tokens are brand variations on both sides
                orig_segment = orig_tokens[i1:i2]
                new_segment = new_tokens[j1:j2]

                # Check if both segments are brand name variations
                orig_is_brand = all(
                    is_brand_token(t, brand_variations) or t.isspace()
                    for t in orig_segment
                )
                new_is_brand = all(
                    is_brand_token(t, brand_variations) or t.isspace()
                    for t in new_segment
                )

                if orig_is_brand and new_is_brand:
                    # This is a brand name change - treat as equal (no highlighting)
                    if in_add:
                        out.append(MARK_END)
                        in_add = False
                    out.extend(new_tokens[j1:j2])
                    continue

            # Regular insert/replace - highlight it
            if not in_add:
                out.append(MARK_START)
                in_add = True
            out.extend(new_tokens[j1:j2])
        elif tag == "delete":
            # Deletions exist only in original; nothing to emit
            # But close any open add block first
            if in_add:
                out.append(MARK_END)
                in_add = False

    if in_add:
        out.append(MARK_END)

    result = "".join(out)

    # Clean up any empty marker pairs (including those with only whitespace)
    # Use regex to handle all whitespace variations (spaces, tabs, newlines)
    import re as _re
    result = _re.sub(
        rf"{_re.escape(MARK_START)}\s*{_re.escape(MARK_END)}",
        "",
        result
    )

    # Clean up markers containing only whitespace (no actual content)
    # These can occur when diff only detects whitespace changes
    result = _re.sub(
        rf"{_re.escape(MARK_START)}(\s+){_re.escape(MARK_END)}",
        r"\1",  # Keep the whitespace but remove markers
        result
    )

    # Postprocess: Restore multi-word keywords from atomic tokens
    if token_map:
        result = postprocess_keywords_from_diff(result, token_map)

    return result


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences, preserving sentence boundaries.

    Args:
        text: Text to split.

    Returns:
        List of sentences.
    """
    if not text:
        return []

    # Split on sentence-ending punctuation followed by whitespace
    parts = SENT_SPLIT_RE.split(text.strip())

    # Recombine parts into sentences (odd indices are separators)
    sentences = []
    current = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Sentence content
            current += part
        else:  # Separator (whitespace after punctuation)
            if current:
                sentences.append(current)
            current = ""

    if current:
        sentences.append(current)

    return sentences


def strip_markers(text: str) -> str:
    """
    Remove all [[[ADD]]]/[[[ENDADD]]] markers from text.

    Args:
        text: Text potentially containing markers.

    Returns:
        Text with markers removed.
    """
    return text.replace(MARK_START, "").replace(MARK_END, "")


def normalize_paragraph_spacing(text: str) -> str:
    """
    Normalize text to ensure proper spacing and paragraph structure.

    This function fixes common issues that cause text to run together:
    - Missing space after sentence-ending punctuation (e.g., "word.Word" → "word. Word")
    - Multiple spaces collapsed to single space (except newlines)
    - Proper spacing around markers

    Args:
        text: Text that may have spacing issues.

    Returns:
        Text with proper spacing between sentences and paragraphs.
    """
    if not text:
        return text

    # First, handle spacing after sentence-ending punctuation
    # Fix "word.Word" patterns - add space after .!? when followed by capital letter
    # But don't break abbreviations like "U.S." or "Dr."
    result = re.sub(
        r'([.!?])([A-Z])',
        r'\1 \2',
        text
    )

    # Fix missing space after sentence punctuation when followed by lowercase
    # (rare but can happen with typos)
    result = re.sub(
        r'([.!?])([a-z])',
        r'\1 \2',
        result
    )

    # Collapse multiple spaces (but preserve newlines)
    result = re.sub(r'[ \t]+', ' ', result)

    # Ensure proper spacing around markers
    # Fix "[[[ADD]]]word" → "[[[ADD]]]word" (markers should not have leading/trailing spaces inside)
    # Fix "word[[[ADD]]]" → "word [[[ADD]]]" (ensure space before marker if needed)
    result = re.sub(
        rf'([^\s\[])({re.escape(MARK_START)})',
        r'\1 \2',
        result
    )

    # Fix marker followed immediately by word without space (if no space after marker end)
    result = re.sub(
        rf'({re.escape(MARK_END)})([^\s\]\.\,\!\?\;\:])',
        r'\1 \2',
        result
    )

    return result.strip()


def mark_block_as_new(text: str) -> str:
    """
    Wrap an entire block of text in markers to mark it as new content.

    Use this for content that is entirely new (like FAQ items) and should
    be fully highlighted without any diff comparison.

    Args:
        text: Plain text to mark as new.

    Returns:
        Text wrapped in [[[ADD]]]...[[[ENDADD]]] markers.
        Returns empty string if text is empty.
    """
    if not text or not text.strip():
        return ""
    # Strip any existing markers first to avoid nesting
    clean_text = strip_markers(text)
    return f"{MARK_START}{clean_text}{MARK_END}"


def expand_markers_to_full_sentence(original: str, marked: str) -> str:
    """
    Expand markers so that entirely new sentences are fully highlighted.

    If a sentence contains markers and no plain-text version of that
    sentence exists in the original, wrap the ENTIRE sentence in markers.

    Args:
        original: The original text (for comparison).
        marked: The text with diff-based markers.

    Returns:
        Text with markers expanded to cover full new sentences.
    """
    if not marked or MARK_START not in marked:
        return marked

    if not original:
        # If no original, everything is new - just wrap the whole thing
        plain = strip_markers(marked)
        return f"{MARK_START}{plain}{MARK_END}"

    # Split both into sentences for comparison
    orig_sentences = split_into_sentences(original)
    orig_sentences_lower = [s.lower().strip() for s in orig_sentences]

    marked_sentences = split_into_sentences(marked)

    processed = []
    for sent in marked_sentences:
        # Check if this sentence contains any markers
        if MARK_START in sent or MARK_END in sent:
            # Get the plain version of this sentence
            plain = strip_markers(sent).strip()
            plain_lower = plain.lower()

            # Check if this sentence (or something very similar) existed in original
            sentence_is_new = True
            for orig_sent in orig_sentences_lower:
                # Use fuzzy matching - if >70% of the sentence is the same, it's not "new"
                if orig_sent and plain_lower:
                    sm = SequenceMatcher(None, orig_sent, plain_lower)
                    if sm.ratio() > 0.7:
                        sentence_is_new = False
                        break

            if sentence_is_new:
                # This is a mostly/entirely new sentence - wrap the whole thing
                processed.append(f"{MARK_START}{plain}{MARK_END}")
            else:
                # Sentence existed but was modified - keep fine-grained markers
                processed.append(sent)
        else:
            # No markers in this sentence - keep as-is
            processed.append(sent)

    return " ".join(processed)


def compute_markers(
    original_block: str,
    rewritten: str,
    full_original_text: Optional[str] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> str:
    """
    DEPRECATED: Use compute_markers_v2() instead.

    This function uses similarity thresholds and phrase-level diff which can
    cause confusing partial highlights. The V2 function uses strict exact
    matching only.

    Compute diff-based markers using sentence-level semantics.

    SENTENCE-LEVEL ALGORITHM:
    For each sentence S in rewritten text:
    1. If S (normalized) is IDENTICAL to any sentence in full_original_text → NO markers
    2. Else, find most similar sentence S0 in original_block:
       - If ratio >= similarity_threshold: phrase-level diff (small edits highlighted)
       - Else: entirely NEW sentence → wrap ENTIRE sentence in markers

    GUARDRAILS:
    - If entire block is unchanged (normalized), return as-is with no markers
    - If a new sentence has < 6 words, always wrap entire sentence (no phrase-level diff)

    Args:
        original_block: Original text block being compared (e.g., a paragraph).
        rewritten: Rewritten text (plain, no markers from LLM).
        full_original_text: The complete original document text. If provided,
            sentences that appear ANYWHERE in the full document won't be marked.
            If None, falls back to original_block.
        similarity_threshold: DEPRECATED - not used in V2.
            Default is 0.80 (80% similar = phrase diff, <80% = full wrap).

    Returns:
        Rewritten text with accurate [[[ADD]]]/[[[ENDADD]]] markers.

    .. deprecated::
        Use :func:`compute_markers_v2` instead for strict exact matching.
    """
    if not rewritten:
        return ""

    if not original_block:
        # Everything is new
        return f"{MARK_START}{rewritten}{MARK_END}"

    # GUARDRAIL 1: If entire block is unchanged, skip markers entirely
    if normalize_sentence(original_block) == normalize_sentence(rewritten):
        return rewritten

    # Use full_original_text for identity checks, fall back to original_block
    full_text = full_original_text if full_original_text else original_block

    # Build index of all sentences in the full original document
    original_sentence_index = build_original_sentence_index(full_text)

    # Get original_block sentences for similarity matching
    original_block_sentences = split_into_sentences(original_block)

    # Process rewritten text sentence by sentence
    rewritten_sentences = split_into_sentences(rewritten)
    result_parts = []

    for sentence in rewritten_sentences:
        if not sentence.strip():
            result_parts.append(sentence)
            continue

        norm_sentence = normalize_sentence(sentence)

        # Step 1: Check if sentence is IDENTICAL to any in full original
        if norm_sentence in original_sentence_index:
            # Sentence exists unchanged - NO markers
            result_parts.append(sentence)
            continue

        # Step 2: Find most similar sentence in original_block
        best_match, similarity = find_most_similar_sentence(
            sentence, original_block_sentences
        )

        # GUARDRAIL 2: Short sentences (< 6 words) always wrap entire sentence
        word_count = len(sentence.split())
        if word_count < MIN_WORDS_FOR_PHRASE_DIFF:
            # Short sentence - always wrap entire thing
            result_parts.append(f"{MARK_START}{sentence}{MARK_END}")
        elif similarity >= similarity_threshold and best_match:
            # Similar enough - do phrase-level diff
            marked_sentence = add_markers_by_diff(best_match, sentence)
            result_parts.append(marked_sentence)
        else:
            # Entirely new sentence - wrap the whole thing
            result_parts.append(f"{MARK_START}{sentence}{MARK_END}")

    # Join sentences with space
    result = " ".join(result_parts)

    # Clean up any marker issues
    result = cleanup_markers(result)

    return result


def cleanup_markers(text: str) -> str:
    """
    Clean up marker issues like nested markers, empty markers, etc.

    Args:
        text: Text with markers.

    Returns:
        Cleaned text.
    """
    if not text:
        return text

    # Remove empty marker pairs
    text = re.sub(rf"{re.escape(MARK_START)}\s*{re.escape(MARK_END)}", "", text)

    # Merge adjacent marker blocks (end followed by start with only whitespace)
    text = re.sub(
        rf"{re.escape(MARK_END)}(\s*){re.escape(MARK_START)}",
        r"\1",  # Keep just the whitespace
        text
    )

    # Ensure markers are balanced
    start_count = text.count(MARK_START)
    end_count = text.count(MARK_END)

    if start_count > end_count:
        # Add missing end markers
        text += MARK_END * (start_count - end_count)
    elif end_count > start_count:
        # Remove orphaned end markers
        for _ in range(end_count - start_count):
            # Find first unmatched end marker
            pos = 0
            depth = 0
            for i, c in enumerate(text):
                if text[i:].startswith(MARK_START):
                    depth += 1
                elif text[i:].startswith(MARK_END):
                    if depth == 0:
                        # This is an orphaned end marker
                        text = text[:i] + text[i + len(MARK_END):]
                        break
                    depth -= 1

    return text


def compute_h1_markers(original_h1: str, optimized_h1: str) -> str:
    """
    Token-level marker handling for H1 headings.

    Uses precise token-level diff to highlight ONLY the changed/added tokens.

    Example:
        Original: "Fortell AI Hearing Aids"
        Optimized: "Fortell AI Hearing Aids: Advanced Technology for Clear Sound"
        Result: "Fortell AI Hearing Aids[[[ADD]]]: Advanced Technology for Clear Sound[[[ENDADD]]]"

    This ensures:
    - Unchanged text stays unhighlighted (black)
    - Only new/changed tokens are highlighted (green)
    - Appended subtitles/suffixes are properly highlighted

    Args:
        original_h1: The original H1 text.
        optimized_h1: The optimized H1 text (plain, no markers).

    Returns:
        Optimized H1 with token-level markers around changed portions only.
    """
    if not optimized_h1:
        return ""

    if not original_h1:
        # No original H1 - everything is new
        return f"{MARK_START}{optimized_h1}{MARK_END}"

    # Normalize both for comparison (to detect if actually different)
    norm_original = normalize_sentence(original_h1)
    norm_optimized = normalize_sentence(optimized_h1)

    if norm_original == norm_optimized:
        # H1 is unchanged - no markers
        return optimized_h1

    # H1 changed - use TOKEN-LEVEL diff (not all-or-nothing)
    # This ensures "Fortell AI Hearing Aids" stays black even when
    # ": Advanced Technology" is appended (which should be green)
    return add_markers_by_diff(original_h1, optimized_h1)


def inject_phrase_with_markers(
    text: str,
    phrase: str,
    position: str = "start",
) -> str:
    """
    Inject a phrase into text with markers, for cases like ensuring
    the primary keyword appears in title/meta/H1.

    Args:
        text: The text to inject into.
        phrase: The phrase to inject.
        position: Where to inject ("start" or "end").

    Returns:
        Text with the phrase injected and marked.
    """
    if not phrase:
        return text

    # Check if phrase already exists (case-insensitive)
    if phrase.lower() in text.lower():
        return text

    marked_phrase = f"{MARK_START}{phrase}{MARK_END}"

    if position == "start":
        # Add at start with appropriate punctuation
        if text and text[0].isupper():
            # Capitalize the phrase and add separator
            marked_phrase = f"{MARK_START}{phrase.capitalize()}{MARK_END}: "
        return marked_phrase + text
    else:
        # Add at end
        if text and text[-1] in ".!?":
            # Insert before final punctuation
            return text[:-1] + f" - {marked_phrase}" + text[-1]
        return text + f" - {marked_phrase}"


def inject_keyword_naturally(
    text: str,
    keyword: str,
    strategy: str = "auto",
) -> str:
    """
    Inject a keyword into body text using natural insertion patterns.

    Unlike inject_phrase_with_markers (for titles/H1), this function
    finds natural insertion points within paragraphs for body content.

    Strategies:
        - "auto": Pick best strategy based on text structure
        - "parenthetical": Add "(including [keyword])" after a noun phrase
        - "appositive": Add ", [keyword]," after a comma or conjunction
        - "sentence_end": Add "for [keyword]" or "with [keyword]" before period

    Args:
        text: The paragraph text to inject into.
        keyword: The keyword phrase to inject.
        strategy: Injection strategy to use.

    Returns:
        Text with keyword injected and marked, or original if keyword exists.
    """
    if not keyword or not text:
        return text

    # Check if keyword already exists (case-insensitive)
    if keyword.lower() in text.lower():
        return text

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text

    # Find the best sentence to inject into (prefer longer ones with more context)
    best_idx = 0
    best_len = 0
    for i, sent in enumerate(sentences):
        # Skip very short sentences
        if len(sent) > best_len and len(sent) > 30:
            best_len = len(sent)
            best_idx = i

    target_sentence = sentences[best_idx]
    modified_sentence = target_sentence

    # Strategy: Find a natural insertion point
    # Look for patterns like "services", "solutions", "options", "needs" etc.
    # and add the keyword as a clarifying phrase

    # Pattern 1: After words like "services", "solutions", "needs", "options"
    service_words = r'\b(services?|solutions?|needs?|options?|products?|offerings?|work)\b'
    match = re.search(service_words, target_sentence, re.IGNORECASE)
    if match:
        insert_pos = match.end()
        marked = f"{MARK_START}including {keyword}{MARK_END}"
        modified_sentence = (
            target_sentence[:insert_pos] +
            f" {marked}" +
            target_sentence[insert_pos:]
        )
    # Pattern 2: Before the final period - add "for [keyword]"
    elif target_sentence.rstrip().endswith('.'):
        stripped = target_sentence.rstrip()
        marked = f"{MARK_START}for {keyword}{MARK_END}"
        modified_sentence = stripped[:-1] + f" {marked}."
    # Pattern 3: After a comma - add keyword as appositive
    elif ',' in target_sentence:
        # Find last comma before the end
        comma_pos = target_sentence.rfind(',')
        if comma_pos > len(target_sentence) // 2:
            marked = f"{MARK_START}{keyword}{MARK_END}"
            modified_sentence = (
                target_sentence[:comma_pos + 1] +
                f" particularly {marked}," +
                target_sentence[comma_pos + 1:]
            )
        else:
            # Fallback: add at end
            marked = f"{MARK_START}for {keyword}{MARK_END}"
            if target_sentence.rstrip().endswith('.'):
                stripped = target_sentence.rstrip()
                modified_sentence = stripped[:-1] + f" {marked}."
            else:
                modified_sentence = target_sentence + f" {marked}"
    else:
        # Fallback: add at end of sentence
        marked = f"{MARK_START}for {keyword}{MARK_END}"
        if target_sentence.rstrip().endswith('.'):
            stripped = target_sentence.rstrip()
            modified_sentence = stripped[:-1] + f" {marked}."
        else:
            modified_sentence = target_sentence + f" {marked}"

    # Reconstruct paragraph with modified sentence
    sentences[best_idx] = modified_sentence
    return ' '.join(sentences)


def filter_markers_by_keywords(
    text_with_markers: str,
    keywords: list[str],
) -> str:
    """
    Filter out markers that don't contain any SEO-relevant keywords.

    For each [[[ADD]]]...[[[ENDADD]]] block, checks if the content inside
    contains any of the specified keywords. If it does, keep the markers.
    If not, remove the markers but keep the text content.

    This prevents highlighting of non-SEO changes like punctuation tweaks
    or minor rewording that doesn't involve target keywords.

    Args:
        text_with_markers: Text containing [[[ADD]]]...[[[ENDADD]]] markers.
        keywords: List of keyword phrases to check for (case-insensitive).

    Returns:
        Text with markers removed from blocks that don't contain keywords.
    """
    if not text_with_markers or MARK_START not in text_with_markers:
        return text_with_markers

    if not keywords:
        # No keywords to filter by - keep all markers
        return text_with_markers

    # Normalize keywords for case-insensitive matching
    keywords_lower = [kw.lower().strip() for kw in keywords if kw.strip()]
    if not keywords_lower:
        return text_with_markers

    # Pattern to find marker blocks
    marker_pattern = re.compile(
        rf"{re.escape(MARK_START)}(.*?){re.escape(MARK_END)}",
        re.DOTALL
    )

    def replace_if_no_keyword(match: re.Match) -> str:
        """Replace marker block - keep markers only if contains keyword."""
        content = match.group(1)
        content_lower = content.lower()

        # Check if any keyword is in the content
        for keyword in keywords_lower:
            if keyword in content_lower:
                # Keyword found - keep the markers
                return match.group(0)

        # No keyword found - remove markers but keep content
        return content

    # Apply filter to all marker blocks
    result = marker_pattern.sub(replace_if_no_keyword, text_with_markers)

    return result


# =============================================================================
# V2 SENTENCE-LEVEL DIFF API
# =============================================================================
# The functions below implement the simpler V2 sentence-level semantics:
# - A sentence is either FULLY unchanged (black) or FULLY changed (green)
# - NO token-level diff inside sentences
# - Eliminates confusing partial highlights
# =============================================================================


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences for V2 sentence-level diff.

    Splits on sentence-ending punctuation (.!?) followed by whitespace.

    Args:
        text: Text to split into sentences.

    Returns:
        List of sentence strings.
    """
    text = text.strip()
    if not text:
        return []
    # Split on .!? followed by whitespace, keeping the punctuation with the sentence
    parts = SENT_SPLIT_RE.split(text)
    # SENT_SPLIT_RE captures the whitespace separator; recombine properly
    sentences = []
    current = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Content part (sentence)
            current += part
        else:  # Separator (whitespace after punctuation)
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def compute_markers_sentence_level(
    original_block: str,
    rewritten_block: str,
    *,
    original_sentence_index: set[str],
) -> str:
    """
    V2 Sentence-Level Diff: The core function for simple, predictable highlighting.

    For each sentence in rewritten_block:
    - If normalized sentence exists in original_sentence_index → UNCHANGED (no markers)
    - Else → sentence is new/changed → wrap ENTIRE sentence with markers

    NO token-level diff. A sentence is either fully black or fully green.

    Args:
        original_block: The original block text (used for logging/debugging only in V2).
        rewritten_block: The rewritten text to process.
        original_sentence_index: Set of normalized sentences from full original document.

    Returns:
        Rewritten text with [[[ADD]]]/[[[ENDADD]]] markers around changed sentences.
    """
    if not rewritten_block:
        return ""

    # If no original sentence index provided, everything is new
    if not original_sentence_index:
        return f"{MARK_START}{rewritten_block}{MARK_END}"

    # Split rewritten block into sentences
    sentences = split_sentences(rewritten_block)

    if not sentences:
        return rewritten_block

    result_parts = []
    for sentence in sentences:
        if not sentence.strip():
            result_parts.append(sentence)
            continue

        # Normalize the sentence for comparison
        norm = normalize_sentence(sentence)

        if norm in original_sentence_index:
            # Sentence is UNCHANGED - no markers
            result_parts.append(sentence)
        else:
            # Sentence is NEW or CHANGED - wrap ENTIRE sentence
            result_parts.append(f"{MARK_START}{sentence}{MARK_END}")

    # Join with single space
    result = " ".join(result_parts)

    # Merge adjacent marker blocks for cleaner output
    result = _merge_adjacent_markers(result)

    return result


def _merge_adjacent_markers(text: str) -> str:
    """
    Merge adjacent marker blocks separated only by whitespace.

    For example: "[[[ADD]]]foo[[[ENDADD]]] [[[ADD]]]bar[[[ENDADD]]]"
    becomes:     "[[[ADD]]]foo bar[[[ENDADD]]]"

    Args:
        text: Text with markers.

    Returns:
        Text with adjacent markers merged.
    """
    if not text or MARK_START not in text:
        return text

    # Pattern: ENDADD followed by whitespace followed by ADD
    pattern = rf"{re.escape(MARK_END)}(\s+){re.escape(MARK_START)}"
    # Replace with just the whitespace (keeps ADD block open)
    result = re.sub(pattern, r"\1", text)

    return result


def compute_markers_v2(
    original_block: str,
    rewritten: str,
    full_original_text: str | None = None,
) -> str:
    """
    V2 Entry Point: Compute sentence-level markers with NO token-level diff.

    This is the V2 replacement for compute_markers().

    Algorithm:
    1. Build original_sentence_index from full_original_text (or original_block)
    2. For each sentence in rewritten:
       - If normalized sentence exists in index → UNCHANGED (black)
       - Else → CHANGED/NEW → wrap ENTIRE sentence (green)

    Args:
        original_block: Original text block being compared.
        rewritten: Rewritten text (plain, no markers).
        full_original_text: Complete original document text.
            If provided, sentences matching ANYWHERE in full doc stay black.
            If None, falls back to original_block.

    Returns:
        Rewritten text with sentence-level markers.
    """
    if not rewritten:
        return ""

    if not original_block and not full_original_text:
        # Everything is new
        return f"{MARK_START}{rewritten}{MARK_END}"

    # Use full_original_text if provided, else original_block
    full_text = full_original_text if full_original_text else original_block

    # Build the sentence index from the full original text
    sentence_index = build_original_sentence_index(full_text)

    # Quick check: if entire block is unchanged, skip markers
    if normalize_sentence(original_block or "") == normalize_sentence(rewritten):
        return rewritten

    return compute_markers_sentence_level(
        original_block or "",
        rewritten,
        original_sentence_index=sentence_index,
    )


def compute_markers_token_level(
    original_block: str,
    rewritten: str,
    full_original_text: str | None = None,
    brand_variations: set[str] | None = None,
    keywords: list[str] | None = None,
) -> str:
    """
    Token-level marker computation for body content.

    Unlike sentence-level diff, this ONLY highlights the specific tokens
    that were inserted or changed. Unchanged text remains unhighlighted.

    This provides precise, minimal highlighting that:
    - Only highlights actual changes (not entire sentences)
    - Prevents "highlighting unchanged words" problem
    - Makes it clear exactly what was modified

    Args:
        original_block: Original text block being compared.
        rewritten: Rewritten text (plain, no markers).
        full_original_text: Complete original document text.
            If provided, uses it for global context matching.
        brand_variations: Optional set of brand name variations to exclude.
        keywords: Optional list of keywords to treat as atomic units.

    Returns:
        Rewritten text with token-level markers around changed portions only.
    """
    if not rewritten:
        return ""

    if not original_block and not full_original_text:
        # Everything is new
        return f"{MARK_START}{rewritten}{MARK_END}"

    # Use full_original_text if provided for better context matching
    comparison_base = full_original_text if full_original_text else original_block

    # Quick check: if entire block is unchanged, skip markers
    if normalize_sentence(original_block or "") == normalize_sentence(rewritten):
        return rewritten

    # Use token-level diff (same as meta elements)
    return add_markers_by_diff(
        original=comparison_base or original_block,
        rewritten=rewritten,
        brand_variations=brand_variations,
        keywords=keywords,
    )


# =============================================================================
# UNIFIED MARKER COMPUTATION ENTRY POINT
# =============================================================================


def compute_markers_unified(
    original_block: str,
    rewritten: str,
    full_original_text: str | None = None,
    diff_mode: str = "token",
    brand_variations: set[str] | None = None,
    keywords: list[str] | None = None,
) -> str:
    """
    Unified marker computation with selectable diff mode.

    This is the recommended entry point for computing markers.

    Args:
        original_block: Original text block being compared.
        rewritten: Rewritten text (plain, no markers).
        full_original_text: Complete original document text.
        diff_mode: Diff mode - "token" (default) or "sentence".
            - "token": Only highlight actual inserted/changed tokens (recommended)
            - "sentence": Highlight entire sentences if any change detected
        brand_variations: Optional set of brand name variations to exclude.
        keywords: Optional list of keywords to treat as atomic units.

    Returns:
        Rewritten text with markers around changed portions.
    """
    if diff_mode == "sentence":
        return compute_markers_v2(
            original_block=original_block,
            rewritten=rewritten,
            full_original_text=full_original_text,
        )
    else:
        # Default: token-level diff
        return compute_markers_token_level(
            original_block=original_block,
            rewritten=rewritten,
            full_original_text=full_original_text,
            brand_variations=brand_variations,
            keywords=keywords,
        )


# =============================================================================
# TOKEN-LEVEL DIFF FOR META ELEMENTS (Title, Meta Description)
# =============================================================================
# These functions provide precise token-level diff for short text elements
# where the all-or-nothing approach would incorrectly highlight unchanged text.
# =============================================================================


def compute_title_markers(original_title: str, optimized_title: str) -> str:
    """
    Token-level marker handling for Title tags.

    Uses precise token-level diff to highlight ONLY changed/added tokens.

    Example:
        Original: "Best Hearing Aids 2024"
        Optimized: "Best AI Hearing Aids 2024 | Crystal Clear Sound"
        Result: "Best [[[ADD]]]AI [[[ENDADD]]]Hearing Aids 2024[[[ADD]]] | Crystal Clear Sound[[[ENDADD]]]"

    Args:
        original_title: The original title text.
        optimized_title: The optimized title text (plain, no markers).

    Returns:
        Optimized title with token-level markers around changed portions only.
    """
    if not optimized_title:
        return ""

    if not original_title:
        # No original - everything is new
        return f"{MARK_START}{optimized_title}{MARK_END}"

    # Normalize for comparison
    norm_original = normalize_sentence(original_title)
    norm_optimized = normalize_sentence(optimized_title)

    if norm_original == norm_optimized:
        # Title is unchanged - no markers
        return optimized_title

    # Use token-level diff
    return add_markers_by_diff(original_title, optimized_title)


def compute_meta_desc_markers(original_desc: str, optimized_desc: str) -> str:
    """
    Token-level marker handling for Meta Descriptions.

    Uses precise token-level diff to highlight ONLY changed/added tokens.

    Args:
        original_desc: The original meta description.
        optimized_desc: The optimized meta description (plain, no markers).

    Returns:
        Optimized description with token-level markers around changed portions only.
    """
    if not optimized_desc:
        return ""

    if not original_desc:
        # No original - everything is new
        return f"{MARK_START}{optimized_desc}{MARK_END}"

    # Normalize for comparison
    norm_original = normalize_sentence(original_desc)
    norm_optimized = normalize_sentence(optimized_desc)

    if norm_original == norm_optimized:
        # Description is unchanged - no markers
        return optimized_desc

    # Use token-level diff
    return add_markers_by_diff(original_desc, optimized_desc)


# =============================================================================
# URL-PROTECTED DIFF FUNCTIONS
# =============================================================================
# These functions wrap the diff operations with URL protection to prevent
# URL corruption during processing.
# =============================================================================


def add_markers_with_url_protection(
    original: str,
    rewritten: str,
    brand_variations: Optional[set[str]] = None,
    keywords: Optional[list[str]] = None,
) -> str:
    """
    Add markers using token-level diff WITH URL protection.

    This function:
    1. Protects URLs/emails/phones with placeholders
    2. Runs token-level diff on protected text
    3. Restores original URLs/emails/phones

    This prevents URL corruption like "fortell.com" → "Fortell. com"

    Args:
        original: Original text.
        rewritten: Rewritten text (plain, no markers).
        brand_variations: Optional set of brand variations to exclude from highlighting.
        keywords: Optional list of keywords to treat as atomic units.

    Returns:
        Text with markers and original URLs preserved.
    """
    # Import here to avoid circular imports
    from .locked_tokens import protect_locked_tokens, restore_locked_tokens

    if not original:
        if rewritten:
            return f"{MARK_START}{rewritten}{MARK_END}"
        return ""

    if not rewritten:
        return ""

    # Protect URLs in both texts
    protected_original, orig_map = protect_locked_tokens(original)
    protected_rewritten, rewr_map = protect_locked_tokens(rewritten)

    # Merge token maps (prefer original's URLs for restoration)
    combined_map = {**rewr_map, **orig_map}

    # Run token-level diff on protected text
    marked = add_markers_by_diff(
        protected_original,
        protected_rewritten,
        brand_variations=brand_variations,
        keywords=keywords,
    )

    # Restore original URLs
    return restore_locked_tokens(marked, combined_map)


# =============================================================================
# UNHIGHLIGHTED SPAN VALIDATION
# =============================================================================
# These functions validate that unhighlighted (black) text actually matches
# the original source text. This catches cases where formatting changes
# sneak into "unchanged" content.
# =============================================================================


def extract_unhighlighted_spans(marked_text: str) -> list[str]:
    """
    Extract all unhighlighted (non-marked) text spans from marked text.

    Args:
        marked_text: Text with [[[ADD]]]/[[[ENDADD]]] markers.

    Returns:
        List of text spans that are NOT inside markers.
    """
    if not marked_text:
        return []

    # Remove marked content and extract remaining spans
    result = []
    current_pos = 0
    text = marked_text

    while True:
        # Find next marker start
        start_pos = text.find(MARK_START, current_pos)

        if start_pos == -1:
            # No more markers - rest is unhighlighted
            remaining = text[current_pos:].strip()
            if remaining:
                result.append(remaining)
            break

        # Extract unhighlighted span before marker
        span = text[current_pos:start_pos].strip()
        if span:
            result.append(span)

        # Find marker end
        end_pos = text.find(MARK_END, start_pos)
        if end_pos == -1:
            # Malformed markers - stop
            break

        current_pos = end_pos + len(MARK_END)

    return result


def validate_unhighlighted_matches_source(
    original: str,
    marked_output: str,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validate that unhighlighted spans in output exist in original text.

    This catches cases where:
    - Formatting changes sneak into "unchanged" text
    - Words are reordered without being highlighted
    - Punctuation is changed without being highlighted

    Args:
        original: The original source text.
        marked_output: The output text with markers.
        strict: If True, require exact substring match.
                If False, allow normalized (whitespace/punctuation) matching.

    Returns:
        Tuple of (is_valid, mismatched_spans).
        is_valid is True if all unhighlighted spans exist in original.
        mismatched_spans contains any spans that don't match.
    """
    if not original or not marked_output:
        return True, []

    unhighlighted = extract_unhighlighted_spans(marked_output)
    mismatches = []

    for span in unhighlighted:
        if strict:
            # Exact substring match
            if span not in original:
                mismatches.append(span)
        else:
            # Normalized match (collapse whitespace, normalize punctuation)
            norm_span = normalize_sentence(span)
            norm_original = normalize_sentence(original)

            # Check if normalized span appears in normalized original
            # Allow some flexibility for word boundaries
            words = norm_span.split()
            if len(words) >= 3:
                # For longer spans, check if the core words appear
                found = all(word in norm_original for word in words)
                if not found:
                    mismatches.append(span)

    return len(mismatches) == 0, mismatches


def get_highlight_integrity_report(
    original: str,
    marked_output: str,
) -> dict:
    """
    Generate a comprehensive report on highlight integrity.

    Checks:
    1. All unhighlighted text exists in original
    2. No URL corruption
    3. Marker balance (no leaked markers)

    Args:
        original: Original source text.
        marked_output: Output text with markers.

    Returns:
        Dict with 'is_valid', 'issues' list, and diagnostic details.
    """
    from .locked_tokens import validate_urls_preserved, detect_url_corruption

    report = {
        "is_valid": True,
        "issues": [],
        "unhighlighted_valid": True,
        "urls_preserved": True,
        "markers_balanced": True,
        "details": {},
    }

    # Check unhighlighted spans
    valid, mismatches = validate_unhighlighted_matches_source(original, marked_output)
    if not valid:
        report["is_valid"] = False
        report["unhighlighted_valid"] = False
        report["issues"].append(f"Unhighlighted text not in source: {mismatches[:3]}")
        report["details"]["mismatched_spans"] = mismatches

    # Check URL preservation
    stripped_output = strip_markers(marked_output)
    lost_urls = validate_urls_preserved(original, stripped_output)
    if lost_urls:
        report["is_valid"] = False
        report["urls_preserved"] = False
        report["issues"].append(f"URLs lost or corrupted: {lost_urls[:3]}")
        report["details"]["lost_urls"] = lost_urls

    # Check for URL corruption patterns
    corruptions = detect_url_corruption(stripped_output)
    if corruptions:
        report["is_valid"] = False
        report["urls_preserved"] = False
        report["issues"].append(f"URL corruption detected: {corruptions[:3]}")
        report["details"]["url_corruptions"] = corruptions

    # Check marker balance
    start_count = marked_output.count(MARK_START)
    end_count = marked_output.count(MARK_END)
    if start_count != end_count:
        report["is_valid"] = False
        report["markers_balanced"] = False
        report["issues"].append(f"Unbalanced markers: {start_count} starts, {end_count} ends")
        report["details"]["marker_counts"] = {"start": start_count, "end": end_count}

    return report
