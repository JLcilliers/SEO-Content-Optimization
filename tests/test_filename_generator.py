"""Tests for intuitive filename generation."""

import pytest
from pathlib import Path

from seo_content_optimizer.filename_generator import (
    generate_output_filename,
    suggest_filename_for_download,
    _generate_from_url,
    _generate_from_file,
    _slugify,
)


class TestSlugify:
    """Tests for the _slugify helper function."""

    def test_basic_slugify(self):
        """Test basic text slugification."""
        assert _slugify("Hello World") == "hello-world"

    def test_removes_special_characters(self):
        """Test that special characters are removed."""
        assert _slugify("Hello! @World#") == "hello-world"

    def test_converts_underscores_to_hyphens(self):
        """Test that underscores become hyphens."""
        assert _slugify("hello_world_test") == "hello-world-test"

    def test_handles_multiple_spaces(self):
        """Test multiple spaces become single hyphen."""
        assert _slugify("hello    world") == "hello-world"

    def test_removes_leading_trailing_hyphens(self):
        """Test leading/trailing hyphens are removed."""
        assert _slugify("-hello-world-") == "hello-world"

    def test_empty_string_returns_content(self):
        """Test empty string returns 'content' fallback."""
        assert _slugify("") == "content"
        assert _slugify("!!!") == "content"


class TestGenerateFromUrl:
    """Tests for URL-based filename generation."""

    def test_extracts_path_segment(self):
        """Test extraction of last path segment."""
        assert _generate_from_url("https://example.com/about-us") == "about-us"
        assert _generate_from_url("https://example.com/services/plumbing") == "plumbing"

    def test_removes_html_extension(self):
        """Test that .html extension is removed."""
        assert _generate_from_url("https://example.com/about.html") == "about"
        assert _generate_from_url("https://example.com/page.htm") == "page"

    def test_removes_php_extension(self):
        """Test that .php extension is removed."""
        assert _generate_from_url("https://example.com/contact.php") == "contact"

    def test_skips_index_page(self):
        """Test that 'index' is skipped in favor of parent segment."""
        # When index is last, should try to use parent directory
        assert _generate_from_url("https://example.com/services/index.html") == "services"

    def test_falls_back_to_domain(self):
        """Test fallback to domain name when path is empty/index."""
        assert _generate_from_url("https://acmecorp.com/") == "acmecorp"
        assert _generate_from_url("https://www.acmecorp.com/") == "acmecorp"

    def test_handles_complex_urls(self):
        """Test handling of complex URLs with query strings."""
        result = _generate_from_url("https://example.com/blog/my-post?ref=home")
        assert result == "my-post"

    def test_handles_root_url(self):
        """Test handling of root URL."""
        result = _generate_from_url("https://example.com")
        assert result == "example"

    def test_handles_www_prefix(self):
        """Test that www prefix is removed from domain fallback."""
        result = _generate_from_url("https://www.mysite.org/")
        assert result == "mysite"


class TestGenerateFromFile:
    """Tests for file-based filename generation."""

    def test_extracts_stem(self):
        """Test basic stem extraction."""
        assert _generate_from_file(Path("my-document.docx")) == "my-document"

    def test_removes_optimized_prefix(self):
        """Test that 'optimized' prefix is stripped to avoid duplication."""
        assert _generate_from_file(Path("optimized-document.docx")) == "document"
        assert _generate_from_file(Path("optimized_document.docx")) == "document"
        assert _generate_from_file(Path("Optimized-Report.docx")) == "report"

    def test_handles_complex_path(self):
        """Test handling of complex paths."""
        assert _generate_from_file(Path("/home/user/docs/my-report.docx")) == "my-report"

    def test_handles_spaces_in_filename(self):
        """Test handling of spaces in filename."""
        assert _generate_from_file(Path("My Document.docx")) == "my-document"


class TestGenerateOutputFilename:
    """Tests for the main generate_output_filename function."""

    def test_url_generates_optimized_prefix(self):
        """Test that URLs generate files with 'optimized-' prefix."""
        result = generate_output_filename("https://example.com/about-us")
        assert result.name == "optimized-about-us.docx"

    def test_file_generates_optimized_prefix(self):
        """Test that file paths generate files with 'optimized-' prefix."""
        result = generate_output_filename(Path("my-doc.docx"))
        assert result.name == "optimized-my-doc.docx"

    def test_includes_primary_keyword(self):
        """Test that primary keyword can be included."""
        result = generate_output_filename(
            "https://example.com/services",
            primary_keyword="plumbing repair"
        )
        assert "plumbing-repair" in result.name
        assert result.name == "optimized-services-plumbing-repair.docx"

    def test_keyword_not_duplicated(self):
        """Test that keyword isn't added if already in name."""
        result = generate_output_filename(
            "https://example.com/plumbing-services",
            primary_keyword="plumbing"
        )
        # Should not have 'plumbing' twice
        assert result.name == "optimized-plumbing-services.docx"

    def test_respects_output_dir(self):
        """Test that output directory is respected."""
        result = generate_output_filename(
            "https://example.com/page",
            output_dir=Path("/output/dir")
        )
        assert result.parent == Path("/output/dir")

    def test_truncates_long_names(self):
        """Test that very long names are truncated."""
        long_url = "https://example.com/" + "a" * 200
        result = generate_output_filename(long_url)
        # Name should be truncated (100 chars max for base + prefix + extension)
        assert len(result.stem) <= 110  # optimized- (10) + base (100)

    def test_ensures_docx_extension(self):
        """Test that .docx extension is always present."""
        result = generate_output_filename("https://example.com/page")
        assert result.suffix == ".docx"


class TestSuggestFilenameForDownload:
    """Tests for the download filename suggestion function."""

    def test_uses_original_filename(self):
        """Test that original filename is used when provided."""
        result = suggest_filename_for_download(
            "https://example.com/upload",
            original_filename="my-report.docx"
        )
        assert result == "optimized-my-report.docx"

    def test_handles_optimized_original(self):
        """Test handling when original already has 'optimized' prefix."""
        result = suggest_filename_for_download(
            "https://example.com/upload",
            original_filename="optimized-report.docx"
        )
        assert result == "optimized-report.docx"

    def test_generates_from_url_without_original(self):
        """Test URL-based generation when no original filename."""
        result = suggest_filename_for_download("https://example.com/about-page")
        assert result == "optimized-about-page.docx"

    def test_fallback_filename(self):
        """Test fallback when nothing else works."""
        result = suggest_filename_for_download("not-a-url")
        assert result == "optimized-content.docx"
