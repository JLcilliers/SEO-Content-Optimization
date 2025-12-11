"""
FastAPI wrapper for SEO Content Optimizer - Vercel Serverless Function.

This module exposes the SEO optimization functionality as a REST API
for deployment on Vercel.
"""

import base64
import io
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seo_content_optimizer.content_sources import fetch_url_content, load_docx_content
from seo_content_optimizer.keyword_loader import load_keywords
from seo_content_optimizer.optimizer import ContentOptimizer
from seo_content_optimizer.docx_writer import DocxWriter
from seo_content_optimizer.models import Keyword

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


class OptimizeURLRequest(BaseModel):
    """Request model for URL-based optimization."""
    source_url: str = Field(..., description="URL to fetch content from")
    keywords: list[KeywordInput] = Field(..., description="List of keywords to optimize for")
    generate_faq: bool = Field(True, description="Whether to generate FAQ section")
    faq_count: int = Field(4, description="Number of FAQ items to generate")
    max_secondary: int = Field(5, description="Maximum secondary keywords to use")


class OptimizeResponse(BaseModel):
    """Response model for optimization results."""
    success: bool
    message: str
    primary_keyword: Optional[str] = None
    secondary_keywords: Optional[list[str]] = None
    meta_elements: Optional[list[dict]] = None
    faq_items: Optional[list[dict]] = None
    document_base64: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


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
                volume=kw.volume,
                difficulty=kw.difficulty,
                intent=kw.intent,
            )
            for kw in request.keywords
        ]

        # Run optimization
        optimizer = ContentOptimizer(api_key=api_key)
        result = optimizer.optimize(
            content=content,
            keywords=keywords,
            generate_faq=request.generate_faq,
            faq_count=request.faq_count,
            max_secondary=request.max_secondary,
        )

        # Generate DOCX
        writer = DocxWriter()
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            output_path = Path(tmp.name)

        writer.write(result, output_path)

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
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize/file")
async def optimize_from_file(
    file: UploadFile = File(..., description="Word document (.docx) to optimize"),
    keywords_file: UploadFile = File(..., description="Keywords file (CSV or Excel)"),
    generate_faq: bool = Form(True),
    faq_count: int = Form(4),
    max_secondary: int = Form(5),
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

        # Run optimization
        optimizer = ContentOptimizer(api_key=api_key)
        result = optimizer.optimize(
            content=content,
            keywords=keywords,
            generate_faq=generate_faq,
            faq_count=faq_count,
            max_secondary=max_secondary,
        )

        # Generate output DOCX
        writer = DocxWriter()
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            output_path = Path(tmp.name)

        writer.write(result, output_path)

        # Clean up input temp files
        docx_path.unlink()
        keywords_path.unlink()

        # Return the optimized document as a download
        with open(output_path, "rb") as f:
            doc_bytes = f.read()

        output_path.unlink()

        return StreamingResponse(
            io.BytesIO(doc_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="optimized_{file.filename}"'
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            "POST /api/optimize/url": "Optimize content from URL",
            "POST /api/optimize/file": "Optimize uploaded Word document",
            "GET /api/info": "This endpoint",
        },
        "documentation": "/docs",
        "openapi": "/openapi.json",
    }
