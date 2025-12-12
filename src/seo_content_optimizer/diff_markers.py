"""
Diff-based marker insertion for accurate highlighting.

This module computes differences between original and rewritten text
and inserts [[[ADD]]]/[[[ENDADD]]] markers around ONLY the changed parts.

SENTENCE-LEVEL SEMANTICS:
For each sentence S in rewritten text:
1. If S (normalized) is IDENTICAL to any sentence in full_original_text → NO markers
2. Else, find most similar sentence S0 in original_block:
   - If ratio >= SIMILARITY_THRESHOLD (0.80): phrase-level diff (small edits highlighted)
   - Else: entirely NEW sentence → wrap ENTIRE sentence in markers

Key guarantees:
- Only ACTUALLY new/changed text gets markers
- Existing text (including keywords) stays unhighlighted
- Entire new sentences get fully highlighted (not just keywords within them)
- Modified sentences show only the changed phrases highlighted
"""

import re
from difflib import SequenceMatcher
from typing import Optional

# Marker constants - must match llm_client.py
MARK_START = "[[[ADD]]]"
MARK_END = "[[[ENDADD]]]"

# Tokenization regex: split into whitespace and non-whitespace tokens
TOKEN_RE = re.compile(r"\s+|[^\s]+")

# Sentence boundary pattern - splits on .!? followed by space(s)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])(\s+)")

# Default similarity threshold for sentence comparison
DEFAULT_SIMILARITY_THRESHOLD = 0.80

# Minimum word count for phrase-level diff (short sentences always wrap entire)
MIN_WORDS_FOR_PHRASE_DIFF = 6


def normalize_sentence(s: str) -> str:
    """
    Normalize a sentence for equality comparison.

    Lowercased, single-spaced, strip leading/trailing whitespace.

    Args:
        s: The sentence to normalize.

    Returns:
        Normalized sentence string.
    """
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
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


def add_markers_by_diff(original: str, rewritten: str) -> str:
    """
    Compare original and rewritten text and return rewritten text with
    [[[ADD]]]/[[[ENDADD]]] markers around inserted/replaced segments.

    This guarantees:
    - Only real inserted/replaced tokens get wrapped
    - Any token that existed unchanged in original remains unwrapped
    - No more "highlighting existing keywords" problem

    Args:
        original: The original text before optimization.
        rewritten: The rewritten/optimized text (plain, no markers).

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

    orig_tokens = tokenize(original)
    new_tokens = tokenize(rewritten)

    sm = SequenceMatcher(None, orig_tokens, new_tokens, autojunk=False)
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
            # Open add block if not already in one
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

    # Clean up any empty marker pairs
    result = result.replace(f"{MARK_START}{MARK_END}", "")
    result = result.replace(f"{MARK_START} {MARK_END}", " ")

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
    Compute diff-based markers using sentence-level semantics.

    This is the main entry point for the diff-based highlighting system.

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
        similarity_threshold: Threshold for phrase-level diff vs full wrap.
            Default is 0.80 (80% similar = phrase diff, <80% = full wrap).

    Returns:
        Rewritten text with accurate [[[ADD]]]/[[[ENDADD]]] markers.
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
    Special marker handling for H1 headings.

    H1s are treated specially: if the H1 changed at all, wrap the ENTIRE
    optimized H1 in markers. This ensures the full new heading is green,
    not just individual changed words.

    Args:
        original_h1: The original H1 text.
        optimized_h1: The optimized H1 text (plain, no markers).

    Returns:
        Optimized H1 with markers if changed, or unchanged text if identical.
    """
    if not optimized_h1:
        return ""

    if not original_h1:
        # No original H1 - everything is new
        return f"{MARK_START}{optimized_h1}{MARK_END}"

    # Normalize both for comparison
    norm_original = normalize_sentence(original_h1)
    norm_optimized = normalize_sentence(optimized_h1)

    if norm_original == norm_optimized:
        # H1 is unchanged - no markers
        return optimized_h1
    else:
        # H1 changed - wrap ENTIRE optimized H1
        return f"{MARK_START}{optimized_h1}{MARK_END}"


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
