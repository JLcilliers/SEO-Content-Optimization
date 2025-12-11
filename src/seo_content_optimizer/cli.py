"""
Command-line interface for SEO Content Optimizer.

Provides a CLI for running content optimization from URLs or DOCX files.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .content_sources import ContentExtractionError, load_content
from .docx_writer import write_optimization_result
from .keyword_loader import KeywordLoadError, load_keywords
from .llm_client import LLMClientError
from .optimizer import ContentOptimizer

console = Console()


@click.command()
@click.option(
    "--source-url",
    type=str,
    help="URL to fetch and optimize content from.",
)
@click.option(
    "--source-docx",
    type=click.Path(exists=True, path_type=Path),
    help="Path to Word document to optimize.",
)
@click.option(
    "--keywords",
    "-k",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to keyword file (CSV or Excel).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for optimized Word document.",
)
@click.option(
    "--api-key",
    type=str,
    envvar="ANTHROPIC_API_KEY",
    help="Anthropic API key. Can also be set via ANTHROPIC_API_KEY env var.",
)
@click.option(
    "--no-faq",
    is_flag=True,
    default=False,
    help="Skip FAQ generation.",
)
@click.option(
    "--faq-count",
    type=int,
    default=4,
    help="Number of FAQ items to generate (default: 4).",
)
@click.option(
    "--max-secondary",
    type=int,
    default=5,
    help="Maximum secondary keywords (default: 5).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
def main(
    source_url: Optional[str],
    source_docx: Optional[Path],
    keywords: Path,
    output: Path,
    api_key: Optional[str],
    no_faq: bool,
    faq_count: int,
    max_secondary: int,
    verbose: bool,
) -> None:
    """
    SEO Content Optimizer - Optimize content for search engines.

    Accepts content from a URL or Word document, optimizes it using
    keywords from a CSV/Excel file, and outputs an optimized Word
    document with changes highlighted in green.

    Examples:

        seo-optimize --source-url https://example.com/page --keywords kw.csv -o out.docx

        seo-optimize --source-docx input.docx --keywords keywords.xlsx -o optimized.docx
    """
    # Validate source arguments
    if not source_url and not source_docx:
        console.print("[red]Error:[/red] Must provide either --source-url or --source-docx")
        sys.exit(1)

    if source_url and source_docx:
        console.print("[red]Error:[/red] Provide only one of --source-url or --source-docx")
        sys.exit(1)

    source = source_url or str(source_docx)

    # Display startup info
    console.print(Panel.fit(
        "[bold blue]SEO Content Optimizer[/bold blue]\n"
        "Optimizing content with keyword-focused improvements",
        border_style="blue",
    ))

    try:
        # Step 1: Load content
        with console.status("[bold green]Loading content..."):
            content = load_content(source)
            if verbose:
                console.print(f"  Loaded content from: {source}")
                console.print(f"  Word count: ~{len(content.full_text.split())}")

        # Step 2: Load keywords
        with console.status("[bold green]Loading keywords..."):
            keyword_list = load_keywords(keywords)
            if verbose:
                console.print(f"  Loaded {len(keyword_list)} keywords from: {keywords}")

        # Step 3: Run optimization
        console.print("\n[bold]Running optimization...[/bold]")

        optimizer = ContentOptimizer(api_key=api_key)
        result = optimizer.optimize(
            content=content,
            keywords=keyword_list,
            generate_faq=not no_faq,
            faq_count=faq_count,
            max_secondary=max_secondary,
        )

        # Step 4: Write output
        with console.status("[bold green]Writing output document..."):
            output_path = write_optimization_result(result, output)

        # Display summary
        _display_summary(result, output_path, verbose)

        console.print(f"\n[bold green]Success![/bold green] Output saved to: {output_path}")

    except ContentExtractionError as e:
        console.print(f"[red]Content extraction error:[/red] {e}")
        sys.exit(1)
    except KeywordLoadError as e:
        console.print(f"[red]Keyword loading error:[/red] {e}")
        sys.exit(1)
    except LLMClientError as e:
        console.print(f"[red]LLM error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


def _display_summary(result, output_path: Path, verbose: bool) -> None:
    """Display optimization summary."""
    console.print("\n[bold]Optimization Summary[/bold]")

    # Keywords table
    kw_table = Table(title="Selected Keywords", show_header=True)
    kw_table.add_column("Type", style="cyan")
    kw_table.add_column("Keyword", style="green")

    kw_table.add_row("Primary", result.primary_keyword)
    for kw in result.secondary_keywords[:5]:
        kw_table.add_row("Secondary", kw)

    console.print(kw_table)

    # Meta changes table
    if result.meta_elements:
        meta_table = Table(title="Meta Element Changes", show_header=True)
        meta_table.add_column("Element", style="cyan")
        meta_table.add_column("Changed", style="yellow")

        for meta in result.meta_elements:
            changed = "Yes" if meta.was_changed else "No"
            meta_table.add_row(meta.element_name, changed)

        console.print(meta_table)

    # FAQ info
    if result.faq_items:
        console.print(f"\n[cyan]FAQ items generated:[/cyan] {len(result.faq_items)}")

    if verbose:
        console.print(f"\n[dim]Output file: {output_path}[/dim]")


def run_cli() -> None:
    """Entry point for the CLI."""
    main()


if __name__ == "__main__":
    run_cli()
