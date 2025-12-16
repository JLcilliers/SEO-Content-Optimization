# CLAUDE.md - Repository Guide for Claude Code

This file provides Claude Code with essential context about the SEO Content Optimizer repository.

## Project Overview

SEO Content Optimizer is a fully automated SEO content optimization tool that:
- Accepts content from URLs or Word documents
- Analyzes keywords from CSV/Excel files
- Produces optimized Word documents with green-highlighted changes

## Quick Commands

```bash
# Run tests
pytest -q

# Run tests with coverage
pytest --cov=src/seo_content_optimizer

# Lint code
ruff check .

# Type check
mypy src/

# Run the API server
uvicorn api.main:app --reload

# Run optimization from CLI
python -m seo_content_optimizer optimize --url "https://example.com" --keywords keywords.csv
```

## Key Pipeline Files

The optimization pipeline flows through these modules in order:

```
content_sources.py    → Fetch and extract content from URL/DOCX
     ↓
analyzer.py          → Analyze content structure and existing keywords
     ↓
prioritizer.py       → Select and prioritize keywords
     ↓
optimizer.py         → Main optimization logic (LLM calls)
     ↓
diff_markers.py      → Compute changes and add markers
     ↓
output_validator.py  → Validate output quality
     ↓
docx_writer.py       → Generate final Word document
```

### Supporting Modules (Quality Gates)

- `text_repair.py` - Mojibake detection and repair, whitespace normalization
- `claim_validator.py` - Hallucination prevention via facts ledger
- `repetition_guard.py` - Duplicate sentence detection, keyword density
- `page_archetype.py` - Page type detection (homepage vs blog vs guide)
- `keyword_filter.py` - Block inappropriate industry terms

## Non-Negotiable Quality Rules

These rules MUST be enforced. Violations should fail the build:

1. **No Marker Leakage**: `[[[ADD]]]` and `[[[ENDADD]]]` tokens must NEVER appear in final output
2. **No Mojibake**: Characters like `â€™`, `Ã`, `Â`, `ï¿½` must be repaired or rejected
3. **No Hallucinated Claims**: LLM output cannot introduce new numbers, dates, or factual claims not in source
4. **Preserve Whitespace**: Spaces around marker boundaries must be maintained
5. **No Duplicate Sentences**: Exact duplicate sentences must be removed
6. **Natural Keyword Integration**: No crude "keyword:" prefix injection in titles/headings
7. **Content Preservation**: Output must preserve all original content blocks (ratio >= 0.70)

## Extraction Ladder

Content extraction uses a multi-stage fallback approach:

1. **FireCrawl** (if API key available) - Best for structured content
2. **Trafilatura favor_recall** - Balanced extraction
3. **Trafilatura max_recall** - Higher recall settings
4. **Trafilatura bare_extraction** - Raw baseline extraction
5. **BeautifulSoup fallback** - DOM-based, captures everything

If extraction produces < 100 words or < 3 blocks, automatically escalate to next method.

## Adding Test Fixtures

To add a new regression test case:

1. Place the input HTML/DOCX in `tests/fixtures/`
2. Name it descriptively: `test_<issue>_input.<ext>`
3. Create expected output: `test_<issue>_expected.<ext>`
4. Add test in `tests/test_<module>.py`

Example:
```python
def test_no_marker_leakage():
    """Markers must never appear in final output."""
    result = optimize_content(fixture_content)
    assert "[[[ADD]]]" not in result.full_text
    assert "[[[ENDADD]]]" not in result.full_text
```

## Common Gotchas

### Encoding Issues
- Always use `text_repair.repair_text()` on extracted content
- Check for mojibake markers before processing
- Use UTF-8 encoding explicitly when reading/writing files

### LLM Output Validation
- LLM can hallucinate statistics and claims - always validate
- LLM may repeat sentences - run repetition guard
- LLM may stuff keywords - check density, not just counts

### Marker Handling
- Markers are for internal diff tracking only
- Strip markers before any user-visible output
- Handle malformed markers gracefully (strip, don't output)

### Page Types
- Homepages/landing pages: NO FAQ generation, NO "guide" framing
- Blog posts: FAQ and guide framing ARE appropriate
- Always detect page archetype before optimization

## Configuration

Key environment variables:

```bash
OPENAI_API_KEY=<your-key>       # Required for LLM optimization
FIRECRAWL_API_KEY=<your-key>    # Optional, for better extraction
```

Key config options in code:

```python
# content_sources.py
MIN_WORD_COUNT = 100    # Minimum words for good extraction
MIN_BLOCK_COUNT = 3     # Minimum content blocks

# output_validator.py
MIN_PRESERVATION_RATIO = 0.70  # Content preservation threshold

# repetition_guard.py
KEYWORD_MIN_DISTANCE = 50      # Min words between same keyword
```

## Testing Checklist

Before committing, verify:

- [ ] `pytest` passes
- [ ] No marker tokens in any output fixtures
- [ ] No mojibake characters in output
- [ ] Content preservation ratio meets threshold
- [ ] No duplicate sentences in optimized content
- [ ] Keywords integrated naturally (no prefix injection)

## Architecture Decisions

### Why multiple extraction methods?
Different extractors excel at different page types. Homepage/landing pages need DOM-based extraction; articles work well with trafilatura.

### Why facts ledger validation?
LLMs hallucinate. The facts ledger approach prevents introducing unsupported claims by tracking verifiable facts from the source.

### Why page archetype detection?
FAQ sections and "guide" framing are inappropriate for landing pages but appropriate for blog posts. Archetype detection enables context-aware optimization.

### Why density-based keyword targets?
Fixed keyword counts lead to stuffing on short content and under-optimization on long content. Density (0.5-1.5%) scales naturally.
