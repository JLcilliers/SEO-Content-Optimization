# SEO Content Optimizer

A fully automated SEO content optimization tool that analyzes content from URLs or Word documents, applies keyword optimization using LLM intelligence, and produces optimized Word documents with all changes highlighted in green.

## Features

- **Multiple Content Sources**: Accept content from URLs (web pages) or Word `.docx` files
- **Flexible Keyword Input**: Load keywords from CSV or Excel files with volume, difficulty, and intent data
- **Intelligent Optimization**: Uses Claude LLM to rewrite content naturally while incorporating target keywords
- **Visual Change Tracking**: All additions are highlighted in green in the output document
- **SEO Best Practices**: Follows proven SEO rules for title tags, meta descriptions, and content structure
- **FAQ Generation**: Automatically generates FAQ sections from question-based keywords
- **Detailed Analysis**: Includes content analysis, keyword prioritization, and optimization explanations

## Installation

### Requirements

- Python 3.10 or higher
- Anthropic API key for LLM optimization

### Install from Source

```bash
# Clone or navigate to the project directory
cd "Content Optimizing  Tool"

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev]"
```

### Dependencies

Core dependencies (automatically installed):
- `requests` - HTTP requests for URL fetching
- `trafilatura` - Main content extraction from web pages
- `beautifulsoup4` - HTML parsing for meta elements
- `pandas` - CSV/Excel keyword file parsing
- `python-docx` - Word document reading and writing
- `anthropic` - Claude LLM API client
- `click` - Command-line interface
- `pydantic` - Data validation and models
- `rich` - Formatted console output

## Usage

### Basic Command

```bash
# Optimize content from a URL
seo-optimize --source-url "https://example.com/page" --keywords keywords.csv --output optimized.docx

# Optimize content from a Word document
seo-optimize --source-docx content.docx --keywords keywords.xlsx --output optimized.docx
```

### CLI Options

```
Usage: seo-optimize [OPTIONS]

Options:
  --source-url TEXT       URL to fetch content from
  --source-docx PATH      Path to Word document to optimize
  --keywords PATH         Path to keyword file (CSV or Excel)  [required]
  --output PATH           Output path for optimized document  [required]
  --api-key TEXT          Anthropic API key (or set ANTHROPIC_API_KEY env var)
  --no-faq                Skip FAQ generation
  --faq-count INTEGER     Number of FAQ items to generate (default: 5)
  --max-secondary INTEGER Maximum secondary keywords to use (default: 5)
  --verbose               Enable verbose output
  --help                  Show this message and exit
```

### Examples

**Optimize a web page with verbose output:**
```bash
seo-optimize \
  --source-url "https://example.com/insurance-guide" \
  --keywords keywords.csv \
  --output insurance-optimized.docx \
  --verbose
```

**Optimize a Word document without FAQ:**
```bash
seo-optimize \
  --source-docx draft-article.docx \
  --keywords seo-keywords.xlsx \
  --output final-article.docx \
  --no-faq
```

**Using environment variable for API key:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
seo-optimize --source-docx content.docx --keywords kw.csv --output out.docx
```

## Keyword File Format

### CSV Format

```csv
keyword,volume,difficulty,intent
PTO insurance,1200,45,informational
professional liability coverage,800,52,transactional
what is PTO insurance,400,30,informational
buy liability insurance,200,60,transactional
```

### Supported Column Names

The tool recognizes various column name formats:
- **Keyword**: `keyword`, `keywords`, `term`, `phrase`, `search_term`
- **Volume**: `volume`, `search_volume`, `searches`, `monthly_searches`
- **Difficulty**: `difficulty`, `kd`, `keyword_difficulty`, `competition`
- **Intent**: `intent`, `search_intent`, `type`

## Output Document Structure

The generated Word document includes:

### 1. Reading Guide
A highlighted box explaining that all green-highlighted text represents new additions or optimizations made to the original content.

### 2. Current vs Optimized Meta Elements Table

| Element | Current | Optimized | Why Changed |
|---------|---------|-----------|-------------|
| Title Tag | Old title | New optimized title | Explanation |
| Meta Description | Old description | New description | Explanation |
| H1 | Old H1 | New H1 | Explanation |

### 3. Optimized Content
The full optimized content with all additions highlighted in green using the marker system.

### 4. FAQ Section (Optional)
Generated FAQ items based on question-type keywords, with both questions and answers fully highlighted as new content.

## SEO Rules Applied

The optimizer follows these SEO best practices:

- **Title Tag**: 50-60 characters, includes primary keyword
- **Meta Description**: 150-160 characters, includes primary keyword, has call-to-action
- **H1 Tag**: Includes primary keyword, matches search intent
- **Primary Keyword Placement**: In first 100 words of content
- **Keyword Density**: Natural usage without stuffing (target 1-2%)
- **Heading Structure**: Logical H1-H6 hierarchy
- **Content Enhancement**: Natural incorporation of secondary keywords

## How It Works

1. **Content Extraction**: Fetches and parses content from URL or DOCX file
2. **Keyword Loading**: Parses keyword file and extracts metadata
3. **Content Analysis**: Analyzes existing content for topic, intent, and current keyword usage
4. **Keyword Prioritization**: Selects primary, secondary, and question keywords based on:
   - Relevance to content topic
   - Search volume
   - Keyword difficulty
   - Intent alignment
5. **LLM Optimization**: Uses Claude to intelligently rewrite content with markers
6. **Document Generation**: Creates Word document with green highlights for all changes

## Marker System

The tool uses a marker system to track changes:
- `[[[ADD]]]content here[[[ENDADD]]]` marks new additions
- These markers are converted to green highlights in the output document
- The marker system ensures precise tracking of all optimizations

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=seo_content_optimizer
```

### Project Structure

```
Content Optimizing  Tool/
├── src/
│   └── seo_content_optimizer/
│       ├── __init__.py
│       ├── models.py          # Data models (Pydantic)
│       ├── content_sources.py # URL and DOCX extraction
│       ├── keyword_loader.py  # CSV/Excel parsing
│       ├── analysis.py        # Content analysis
│       ├── prioritizer.py     # Keyword selection
│       ├── llm_client.py      # LLM integration
│       ├── optimizer.py       # Main orchestration
│       ├── docx_writer.py     # Document output
│       └── cli.py             # Command-line interface
├── tests/
│   ├── conftest.py            # Test fixtures
│   ├── test_analysis.py
│   ├── test_content_sources.py
│   ├── test_docx_writer.py
│   └── test_keyword_loader.py
├── docs/
│   └── mcp_usage.md           # MCP server documentation
├── pyproject.toml
└── README.md
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude LLM | Yes (or pass via --api-key) |

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request
