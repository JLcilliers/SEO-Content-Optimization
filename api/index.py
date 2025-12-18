"""
FastAPI wrapper for SEO Content Optimizer - Vercel Serverless Function.

This module exposes the SEO optimization functionality as a REST API
for deployment on Vercel.
"""

import base64
import io
import os
import tempfile
import zipfile
from datetime import datetime
from enum import Enum
from typing import Optional, NamedTuple

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seo_content_optimizer.content_sources import fetch_url_content, load_docx_content
from seo_content_optimizer.keyword_loader import load_keywords
from seo_content_optimizer.optimizer import ContentOptimizer
from seo_content_optimizer.docx_writer import DocxWriter
from seo_content_optimizer.filename_generator import suggest_filename_for_download
from seo_content_optimizer.models import Keyword, ManualKeywordsConfig
from seo_content_optimizer.config import OptimizationConfig
from seo_content_optimizer.diff_markers import strip_markers

app = FastAPI(
    title="SEO Content Optimizer API",
    description="Automated SEO content optimization that analyzes content and keywords to produce optimized documents",
    version="1.0.0",
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class KeywordInput(BaseModel):
    """Single keyword input model."""
    phrase: str
    volume: Optional[int] = None
    difficulty: Optional[int] = None
    intent: Optional[str] = None


class ManualKeywordsInput(BaseModel):
    """Manual keyword selection input.

    When provided, bypasses automatic keyword selection and uses
    user-specified keywords directly without filtering or scoring.
    """
    primary: str = Field(..., description="Required primary keyword phrase")
    secondary: list[str] = Field(default_factory=list, description="Up to 3 secondary keyword phrases")


class OptimizationModeEnum(str, Enum):
    """Optimization mode selection."""
    insert_only = "insert_only"  # Strictest: no LLM rewrites, deterministic injection only
    minimal = "minimal"  # Insert-only with minimal LLM assistance
    enhanced = "enhanced"  # Full optimization: density targeting, distribution


class OptimizeURLRequest(BaseModel):
    """Request model for URL-based optimization."""
    source_url: str = Field(..., description="URL to fetch content from")
    keywords: list[KeywordInput] = Field(default_factory=list, description="List of keywords to optimize for")
    manual_keywords: Optional[ManualKeywordsInput] = Field(None, description="Manual keyword selection (bypasses auto-selection)")
    optimization_mode: Optional[OptimizationModeEnum] = Field(
        None,
        description="Optimization mode: 'minimal' (insert-only, each keyword once) or 'enhanced' (full density targeting). Defaults to 'minimal' when manual_keywords provided, 'enhanced' otherwise."
    )
    generate_faq: bool = Field(True, description="Whether to generate FAQ section")
    faq_count: int = Field(4, description="Number of FAQ items to generate")
    max_secondary: int = Field(5, description="Maximum secondary keywords to use")
    include_debug: bool = Field(False, description="Include debug bundle with config, keyword plan, and enforcement details")


class OptimizeResponse(BaseModel):
    """Response model for optimization results."""
    success: bool
    message: str
    primary_keyword: Optional[str] = None
    secondary_keywords: Optional[list[str]] = None
    meta_elements: Optional[list[dict]] = None
    faq_items: Optional[list[dict]] = None
    document_base64: Optional[str] = None
    suggested_filename: Optional[str] = None
    # Debug bundle (only included when include_debug=true)
    debug_bundle: Optional[dict] = Field(
        None,
        description="Debug bundle with config, keyword plan, and enforcement details. Only included when include_debug=true."
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


# ============================================================================
# Bulk Optimization Models and Helpers
# ============================================================================

class BulkOptimizationStatus(str, Enum):
    """Status of individual optimization in bulk job."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


class BulkItemResult(BaseModel):
    """Result for a single item in bulk optimization."""
    row_number: int
    url: str
    primary_keyword: str
    status: BulkOptimizationStatus
    filename: Optional[str] = None
    error_message: Optional[str] = None
    secondary_keywords_used: list[str] = Field(default_factory=list)


class BulkOptimizationRow(NamedTuple):
    """Parsed row from bulk optimization Excel file."""
    row_number: int  # 1-indexed for user clarity (Excel row number)
    url: str
    primary_keyword: str
    secondary_keywords: list[str]


class BulkExcelParseError(Exception):
    """Raised when bulk Excel parsing fails."""
    pass


def _find_column_variant(df: pd.DataFrame, variants: list[str]) -> Optional[str]:
    """Find a column matching one of the variants (case-insensitive)."""
    normalized = {
        col.lower().strip().replace(" ", "_").replace("-", "_"): col
        for col in df.columns
    }
    for variant in variants:
        key = variant.lower().replace(" ", "_").replace("-", "_")
        if key in normalized:
            return normalized[key]
    return None


def parse_bulk_optimization_excel(file_path: Path) -> list[BulkOptimizationRow]:
    """
    Parse bulk optimization Excel file.

    Expected columns:
    - URL (required): The URL to optimize
    - Primary Keyword (required): Main keyword for optimization
    - Secondary Keywords (optional): Comma-separated secondary keywords

    Args:
        file_path: Path to the Excel file.

    Returns:
        List of parsed rows ready for optimization.

    Raises:
        BulkExcelParseError: If parsing fails or validation fails.
    """
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise BulkExcelParseError(f"Failed to read Excel file: {e}")

    if df.empty:
        raise BulkExcelParseError("Excel file contains no data rows")

    # Find columns using variant matching
    url_col = _find_column_variant(df, ["url", "source_url", "page_url", "target_url"])
    primary_col = _find_column_variant(df, ["primary_keyword", "primary", "main_keyword", "keyword"])
    secondary_col = _find_column_variant(df, ["secondary_keywords", "secondary", "other_keywords", "additional_keywords"])

    if not url_col:
        raise BulkExcelParseError(
            "Missing required column: URL. Expected one of: URL, Source URL, Page URL, Target URL"
        )
    if not primary_col:
        raise BulkExcelParseError(
            "Missing required column: Primary Keyword. Expected one of: Primary Keyword, Primary, Main Keyword, Keyword"
        )

    # Validate row count (max 10)
    if len(df) > 10:
        raise BulkExcelParseError(
            f"Batch limit exceeded: {len(df)} URLs found, maximum is 10"
        )

    rows: list[BulkOptimizationRow] = []

    for idx, row in df.iterrows():
        row_num = idx + 2  # Excel row number (1-indexed header + data)

        url = str(row[url_col]).strip() if pd.notna(row[url_col]) else ""
        primary = str(row[primary_col]).strip() if pd.notna(row[primary_col]) else ""

        # Validate URL
        if not url:
            raise BulkExcelParseError(f"Row {row_num}: URL is required")
        if not url.startswith(("http://", "https://")):
            raise BulkExcelParseError(
                f"Row {row_num}: Invalid URL format '{url}'. Must start with http:// or https://"
            )

        # Validate primary keyword
        if not primary:
            raise BulkExcelParseError(f"Row {row_num}: Primary keyword is required")

        # Parse secondary keywords (comma-separated)
        secondary: list[str] = []
        if secondary_col and pd.notna(row.get(secondary_col)):
            raw_secondary = str(row[secondary_col])
            secondary = [kw.strip() for kw in raw_secondary.split(",") if kw.strip()]

        rows.append(BulkOptimizationRow(
            row_number=row_num,
            url=url,
            primary_keyword=primary,
            secondary_keywords=secondary[:5],  # Limit to 5 secondary keywords
        ))

    return rows


def _ensure_unique_filename(filename: str, existing: list[str]) -> str:
    """Ensure filename is unique by adding numeric suffix if needed."""
    if filename not in existing:
        return filename

    base = filename.rsplit(".", 1)[0]
    ext = ".docx"
    counter = 1

    while f"{base}-{counter}{ext}" in existing:
        counter += 1

    return f"{base}-{counter}{ext}"


def _create_manifest(results: list[BulkItemResult]) -> str:
    """Create a manifest text file summarizing the bulk operation."""
    lines = [
        "SEO Content Optimizer - Bulk Optimization Results",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total URLs processed: {len(results)}",
        f"Successful: {sum(1 for r in results if r.status == BulkOptimizationStatus.SUCCESS)}",
        f"Failed: {sum(1 for r in results if r.status == BulkOptimizationStatus.FAILED)}",
        "",
        "Details:",
        "-" * 50,
    ]

    for r in results:
        status_icon = "[OK]" if r.status == BulkOptimizationStatus.SUCCESS else "[FAIL]"
        lines.append(f"\nRow {r.row_number}: {status_icon}")
        lines.append(f"  URL: {r.url}")
        lines.append(f"  Primary Keyword: {r.primary_keyword}")
        if r.secondary_keywords_used:
            lines.append(f"  Secondary Keywords: {', '.join(r.secondary_keywords_used)}")
        if r.filename:
            lines.append(f"  Output File: {r.filename}")
        if r.error_message:
            lines.append(f"  Error: {r.error_message}")

    return "\n".join(lines)


async def process_bulk_optimization(
    rows: list[BulkOptimizationRow],
    api_key: str,
    generate_faq: bool = True,
    faq_count: int = 4,
) -> tuple[io.BytesIO, list[BulkItemResult]]:
    """
    Process multiple URLs for optimization.

    Stops on first error and returns partial results.

    Args:
        rows: Parsed Excel rows to process.
        api_key: Anthropic API key.
        generate_faq: Whether to generate FAQ sections.
        faq_count: Number of FAQ items per document.

    Returns:
        Tuple of (ZIP file buffer, list of results).
    """
    results: list[BulkItemResult] = []
    docx_files: list[tuple[str, bytes]] = []  # (filename, bytes)

    optimizer = ContentOptimizer(api_key=api_key)

    for row in rows:
        try:
            # Step 1: Fetch content from URL
            content = fetch_url_content(row.url)

            # Step 2: Build manual keywords config
            manual_keywords_config = ManualKeywordsConfig(
                primary=row.primary_keyword,
                secondary=row.secondary_keywords,
            )

            # Step 3: Build optimization config - minimal mode for bulk (manual keywords)
            faq_policy = "always" if generate_faq else "never"
            opt_config = OptimizationConfig.minimal(
                faq_policy=faq_policy,
                faq_count=faq_count,
                max_secondary=len(row.secondary_keywords) if row.secondary_keywords else 5,
            )

            # Step 4: Run optimization
            result = optimizer.optimize(
                content=content,
                keywords=[],  # Empty - using manual_keywords
                manual_keywords=manual_keywords_config,
                generate_faq=generate_faq,
                faq_count=faq_count,
                max_secondary=len(row.secondary_keywords) if row.secondary_keywords else 5,
                config=opt_config,
            )

            # Step 4: Generate DOCX
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                output_path = Path(tmp.name)

            # Extract H1 for filename
            h1_heading = None
            for me in result.meta_elements:
                if me.element_name == "H1":
                    # Strip markers to prevent [[[ADD]]] from appearing in filename
                    h1_heading = strip_markers(me.optimized or me.current or "")
                    break

            # Generate filename
            filename = suggest_filename_for_download(
                source=row.url,
                h1_heading=h1_heading,
            )

            # Ensure unique filename in ZIP
            filename = _ensure_unique_filename(filename, [f[0] for f in docx_files])

            # Create document title
            document_title = filename.replace(".docx", "").replace("-optimized-content", "")

            writer = DocxWriter()
            writer.write(
                result,
                output_path,
                source_url=row.url,
                document_title=f"{document_title} | Content Improvement",
                ai_addons=result.ai_addons,
            )

            # Read DOCX bytes
            with open(output_path, "rb") as f:
                docx_bytes = f.read()
            output_path.unlink()

            docx_files.append((filename, docx_bytes))

            results.append(BulkItemResult(
                row_number=row.row_number,
                url=row.url,
                primary_keyword=row.primary_keyword,
                status=BulkOptimizationStatus.SUCCESS,
                filename=filename,
                secondary_keywords_used=result.secondary_keywords or [],
            ))

        except Exception as e:
            # Stop on first error - include partial results
            results.append(BulkItemResult(
                row_number=row.row_number,
                url=row.url,
                primary_keyword=row.primary_keyword,
                status=BulkOptimizationStatus.FAILED,
                error_message=str(e),
            ))
            # Stop processing further rows
            break

    # Create ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add all DOCX files
        for filename, docx_bytes in docx_files:
            zf.writestr(filename, docx_bytes)

        # Add manifest/summary file
        manifest = _create_manifest(results)
        zf.writestr("_manifest.txt", manifest)

    zip_buffer.seek(0)
    return zip_buffer, results


# Path to public directory for static files
PUBLIC_DIR = Path(__file__).parent.parent / "public"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    index_path = PUBLIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    # Fallback to health check if no frontend
    return HTMLResponse(content="<h1>SEO Content Optimizer API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.get("/api/test-llm")
async def test_llm():
    """Test LLM connection endpoint."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"status": "error", "message": "ANTHROPIC_API_KEY not set"}

    try:
        import anthropic
        import httpx

        http_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=30.0),
            follow_redirects=True,
        )
        client = anthropic.Anthropic(
            api_key=api_key,
            http_client=http_client,
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hello in 5 words or less."}],
        )
        return {
            "status": "success",
            "response": response.content[0].text,
            "model": response.model,
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


@app.post("/api/optimize/url", response_model=OptimizeResponse)
async def optimize_from_url(request: OptimizeURLRequest):
    """
    Optimize content from a URL.

    Fetches content from the provided URL, applies SEO optimization
    using the provided keywords, and returns the optimized document.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable not set"
        )

    try:
        # Fetch content from URL
        content = fetch_url_content(request.source_url)

        # Convert keywords
        keywords = [
            Keyword(
                phrase=kw.phrase,
                search_volume=kw.volume,
                difficulty=kw.difficulty,
                intent=kw.intent,
            )
            for kw in request.keywords
        ]

        # Convert manual keywords if provided
        manual_keywords_config = None
        if request.manual_keywords:
            manual_keywords_config = ManualKeywordsConfig(
                primary=request.manual_keywords.primary,
                secondary=request.manual_keywords.secondary,
            )

        # Determine optimization mode:
        # - If explicitly specified, use that
        # - If manual keywords provided, default to "minimal" (insert-only)
        # - Otherwise, default to "enhanced"
        mode = request.optimization_mode
        if mode is None:
            if request.manual_keywords:
                mode = OptimizationModeEnum.minimal
            else:
                mode = OptimizationModeEnum.enhanced

        # Build OptimizationConfig based on mode
        if mode == OptimizationModeEnum.insert_only:
            # Strictest insert-only mode: NO LLM rewrites, deterministic injection only
            faq_policy = "always" if request.generate_faq else "never"
            opt_config = OptimizationConfig.insert_only(
                faq_policy=faq_policy,
                faq_count=request.faq_count,
                max_secondary=request.max_secondary,
            )
        elif mode == OptimizationModeEnum.minimal:
            # Insert-only mode with minimal LLM: caps at 1, no FAQ by default
            faq_policy = "always" if request.generate_faq else "never"
            opt_config = OptimizationConfig.minimal(
                faq_policy=faq_policy,
                faq_count=request.faq_count,
                max_secondary=request.max_secondary,
            )
        else:
            # Enhanced mode: density targeting, FAQ auto
            faq_policy = "auto" if request.generate_faq else "never"
            opt_config = OptimizationConfig.enhanced(
                faq_policy=faq_policy,
                faq_count=request.faq_count,
                max_secondary=request.max_secondary,
            )

        # Run optimization with config
        optimizer = ContentOptimizer(api_key=api_key)
        result = optimizer.optimize(
            content=content,
            keywords=keywords,
            manual_keywords=manual_keywords_config,
            generate_faq=request.generate_faq,
            faq_count=request.faq_count,
            max_secondary=request.max_secondary,
            config=opt_config,
            include_debug=request.include_debug,
        )

        # Extract H1 from meta elements for filename generation
        h1_heading = None
        for me in result.meta_elements:
            if me.element_name == "H1":
                # Strip markers to prevent [[[ADD]]] from appearing in filename
                h1_heading = strip_markers(me.optimized or me.current or "")
                break

        # Generate suggested filename from URL (prioritizes H1 if available)
        suggested_filename = suggest_filename_for_download(
            source=request.source_url,
            h1_heading=h1_heading,
        )

        # Generate DOCX with source URL and document title
        writer = DocxWriter()
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            output_path = Path(tmp.name)

        # Create document title from filename (without extension and suffix)
        document_title = suggested_filename.replace(".docx", "").replace("-optimized-content", "")

        writer.write(
            result,
            output_path,
            source_url=request.source_url,
            document_title=f"{document_title} | Content Improvement",
            ai_addons=result.ai_addons,
        )

        # Read and encode the document
        with open(output_path, "rb") as f:
            doc_bytes = f.read()
        doc_base64 = base64.b64encode(doc_bytes).decode("utf-8")

        # Clean up temp file
        output_path.unlink()

        return OptimizeResponse(
            success=True,
            message="Content optimized successfully",
            primary_keyword=result.primary_keyword,
            secondary_keywords=result.secondary_keywords,
            meta_elements=[
                {
                    "element_name": me.element_name,
                    "current": me.current,
                    "optimized": me.optimized,
                    "why_changed": me.why_changed,
                }
                for me in result.meta_elements
            ],
            faq_items=[
                {"question": faq.question, "answer": faq.answer}
                for faq in result.faq_items
            ],
            document_base64=doc_base64,
            suggested_filename=suggested_filename,
            debug_bundle=result.debug_bundle,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/file")
async def optimize_from_file(
    file: UploadFile = File(..., description="Word document (.docx) to optimize"),
    keywords_file: UploadFile = File(..., description="Keywords file (CSV or Excel)"),
    optimization_mode: str = Form("enhanced", description="Optimization mode: 'minimal' (insert-only) or 'enhanced' (full density targeting)"),
    generate_faq: bool = Form(True),
    faq_count: int = Form(4),
    max_secondary: int = Form(5),
    include_debug: bool = Form(False, description="Include debug bundle in response headers"),
):
    """
    Optimize content from an uploaded Word document.

    Upload a .docx file and keywords file (CSV/Excel), returns
    the optimized document.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable not set"
        )

    try:
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            content_bytes = await file.read()
            tmp.write(content_bytes)
            docx_path = Path(tmp.name)

        # Determine keywords file extension
        kw_suffix = Path(keywords_file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=kw_suffix, delete=False) as tmp:
            kw_bytes = await keywords_file.read()
            tmp.write(kw_bytes)
            keywords_path = Path(tmp.name)

        # Load content and keywords
        content = load_docx_content(docx_path)
        keywords = load_keywords(keywords_path)

        # Build optimization config based on mode
        mode = optimization_mode.lower().strip()
        if mode == "insert_only":
            # Strictest insert-only mode: NO LLM rewrites, deterministic injection only
            faq_policy = "always" if generate_faq else "never"
            opt_config = OptimizationConfig.insert_only(
                faq_policy=faq_policy,
                faq_count=faq_count,
                max_secondary=max_secondary,
            )
        elif mode == "minimal":
            # Insert-only mode with minimal LLM assistance
            faq_policy = "always" if generate_faq else "never"
            opt_config = OptimizationConfig.minimal(
                faq_policy=faq_policy,
                faq_count=faq_count,
                max_secondary=max_secondary,
            )
        else:
            # Enhanced mode: density targeting, FAQ auto
            faq_policy = "auto" if generate_faq else "never"
            opt_config = OptimizationConfig.enhanced(
                faq_policy=faq_policy,
                faq_count=faq_count,
                max_secondary=max_secondary,
            )

        # Run optimization
        optimizer = ContentOptimizer(api_key=api_key)
        result = optimizer.optimize(
            content=content,
            keywords=keywords,
            generate_faq=generate_faq,
            faq_count=faq_count,
            max_secondary=max_secondary,
            config=opt_config,
            include_debug=include_debug,
        )

        # Extract H1 from meta elements for filename generation
        h1_heading = None
        for me in result.meta_elements:
            if me.element_name == "H1":
                # Strip markers to prevent [[[ADD]]] from appearing in filename
                h1_heading = strip_markers(me.optimized or me.current or "")
                break

        # Generate better filename for download (prioritizes H1 if available)
        download_filename = suggest_filename_for_download(
            source="file-upload",
            original_filename=file.filename,
            h1_heading=h1_heading,
        )

        # Create document title from original filename
        original_name = Path(file.filename).stem if file.filename else "document"
        document_title = f"{original_name} | Content Improvement"

        # Generate output DOCX
        writer = DocxWriter()
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            output_path = Path(tmp.name)

        writer.write(
            result,
            output_path,
            source_url=None,  # No URL for file uploads
            document_title=document_title,
            ai_addons=result.ai_addons,
        )

        # Clean up input temp files
        docx_path.unlink()
        keywords_path.unlink()

        # Return the optimized document as a download
        with open(output_path, "rb") as f:
            doc_bytes = f.read()

        output_path.unlink()

        # Build response headers
        response_headers = {
            "Content-Disposition": f'attachment; filename="{download_filename}"'
        }

        # Add debug bundle as base64 encoded header if requested
        if include_debug and result.debug_bundle:
            import json
            debug_json = json.dumps(result.debug_bundle)
            debug_b64 = base64.b64encode(debug_json.encode()).decode()
            response_headers["X-Debug-Bundle"] = debug_b64

        return StreamingResponse(
            io.BytesIO(doc_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers=response_headers,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/url-with-keywords-file", response_model=OptimizeResponse)
async def optimize_url_with_keywords_file(
    source_url: str = Form(..., description="URL to fetch content from"),
    keywords_file: UploadFile = File(..., description="Keywords file (CSV or Excel)"),
    optimization_mode: str = Form("enhanced", description="Optimization mode: 'minimal' (insert-only) or 'enhanced' (full density targeting)"),
    generate_faq: bool = Form(True),
    faq_count: int = Form(4),
    include_debug: bool = Form(False, description="Include debug bundle with config, keyword plan, and enforcement details"),
    max_secondary: int = Form(5),
):
    """
    Optimize content from a URL using keywords from an uploaded file.

    Fetches content from the provided URL, loads keywords from the uploaded
    CSV/Excel file, and returns the optimized document.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable not set"
        )

    try:
        # Fetch content from URL
        content = fetch_url_content(source_url)

        # Save keywords file temporarily
        kw_suffix = Path(keywords_file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=kw_suffix, delete=False) as tmp:
            kw_bytes = await keywords_file.read()
            tmp.write(kw_bytes)
            keywords_path = Path(tmp.name)

        # Load keywords from file
        keywords = load_keywords(keywords_path)

        # Clean up temp file
        keywords_path.unlink()

        # Build optimization config based on mode
        mode = optimization_mode.lower().strip()
        if mode == "insert_only":
            # Strictest insert-only mode: NO LLM rewrites, deterministic injection only
            faq_policy = "always" if generate_faq else "never"
            opt_config = OptimizationConfig.insert_only(
                faq_policy=faq_policy,
                faq_count=faq_count,
                max_secondary=max_secondary,
            )
        elif mode == "minimal":
            # Insert-only mode with minimal LLM assistance
            faq_policy = "always" if generate_faq else "never"
            opt_config = OptimizationConfig.minimal(
                faq_policy=faq_policy,
                faq_count=faq_count,
                max_secondary=max_secondary,
            )
        else:
            # Enhanced mode: density targeting, FAQ auto
            faq_policy = "auto" if generate_faq else "never"
            opt_config = OptimizationConfig.enhanced(
                faq_policy=faq_policy,
                faq_count=faq_count,
                max_secondary=max_secondary,
            )

        # Run optimization
        optimizer = ContentOptimizer(api_key=api_key)
        result = optimizer.optimize(
            content=content,
            keywords=keywords,
            generate_faq=generate_faq,
            faq_count=faq_count,
            max_secondary=max_secondary,
            config=opt_config,
            include_debug=include_debug,
        )

        # Extract H1 from meta elements for filename generation
        h1_heading = None
        for me in result.meta_elements:
            if me.element_name == "H1":
                # Strip markers to prevent [[[ADD]]] from appearing in filename
                h1_heading = strip_markers(me.optimized or me.current or "")
                break

        # Generate suggested filename from URL (prioritizes H1 if available)
        suggested_filename = suggest_filename_for_download(
            source=source_url,
            h1_heading=h1_heading,
        )

        # Create document title from filename (without extension and suffix)
        document_title = suggested_filename.replace(".docx", "").replace("-optimized-content", "")

        # Generate DOCX with source URL and document title
        writer = DocxWriter()
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            output_path = Path(tmp.name)

        writer.write(
            result,
            output_path,
            source_url=source_url,
            document_title=f"{document_title} | Content Improvement",
            ai_addons=result.ai_addons,
        )

        # Read and encode the document
        with open(output_path, "rb") as f:
            doc_bytes = f.read()
        doc_base64 = base64.b64encode(doc_bytes).decode("utf-8")

        # Clean up temp file
        output_path.unlink()

        return OptimizeResponse(
            success=True,
            message="Content optimized successfully",
            primary_keyword=result.primary_keyword,
            secondary_keywords=result.secondary_keywords,
            meta_elements=[
                {
                    "element_name": me.element_name,
                    "current": me.current,
                    "optimized": me.optimized,
                    "why_changed": me.why_changed,
                }
                for me in result.meta_elements
            ],
            faq_items=[
                {"question": faq.question, "answer": faq.answer}
                for faq in result.faq_items
            ],
            document_base64=doc_base64,
            suggested_filename=suggested_filename,
            debug_bundle=result.debug_bundle,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/bulk")
async def optimize_bulk(
    file: UploadFile = File(..., description="Excel file with URLs and keywords"),
    generate_faq: bool = Form(True),
    faq_count: int = Form(4),
):
    """
    Bulk optimize multiple URLs from an Excel file.

    Excel format:
    - Column 1: URL (required) - The URL to fetch and optimize
    - Column 2: Primary Keyword (required) - Main keyword for optimization
    - Column 3: Secondary Keywords (optional) - Comma-separated list

    Maximum 10 URLs per batch. Processing stops on first error.

    Returns a ZIP file containing:
    - One optimized DOCX per successful URL
    - _manifest.txt with processing summary
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable not set"
        )

    # Validate file was provided
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".xlsx", ".xls"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {suffix}. Expected .xlsx or .xls"
        )

    excel_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content_bytes = await file.read()
            tmp.write(content_bytes)
            excel_path = Path(tmp.name)

        # Parse Excel file
        try:
            rows = parse_bulk_optimization_excel(excel_path)
        except BulkExcelParseError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Process all rows
        zip_buffer, results = await process_bulk_optimization(
            rows=rows,
            api_key=api_key,
            generate_faq=generate_faq,
            faq_count=faq_count,
        )

        # Calculate statistics
        successful = sum(1 for r in results if r.status == BulkOptimizationStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == BulkOptimizationStatus.FAILED)

        # Generate ZIP filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        zip_filename = f"optimized-content-batch-{timestamp}.zip"

        # Return ZIP as streaming response
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{zip_filename}"',
                "X-Bulk-Total": str(len(rows)),
                "X-Bulk-Successful": str(successful),
                "X-Bulk-Failed": str(failed),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk optimization failed: {str(e)}")
    finally:
        # Clean up temp file
        if excel_path and excel_path.exists():
            excel_path.unlink()


@app.get("/api/info")
async def api_info():
    """Get API information and usage instructions."""
    return {
        "name": "SEO Content Optimizer API",
        "version": "1.0.0",
        "description": "Automated SEO content optimization tool",
        "endpoints": {
            "GET /": "Health check",
            "GET /api/health": "Health check",
            "POST /api/optimize/url": "Optimize content from URL with JSON keywords",
            "POST /api/optimize/url-with-keywords-file": "Optimize content from URL with keywords file",
            "POST /api/optimize/file": "Optimize uploaded Word document with keywords file",
            "POST /api/optimize/bulk": "Bulk optimize multiple URLs from Excel file (max 10, returns ZIP)",
            "GET /api/info": "This endpoint",
        },
        "documentation": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/styles.css")
async def serve_css():
    """Serve CSS file."""
    css_path = PUBLIC_DIR / "styles.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")


@app.get("/app.js")
async def serve_js():
    """Serve JavaScript file."""
    js_path = PUBLIC_DIR / "app.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JavaScript file not found")
