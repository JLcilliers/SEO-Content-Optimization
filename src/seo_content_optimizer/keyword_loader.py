"""
Keyword list loading and parsing from CSV and Excel files.

This module handles ingestion of keyword data from:
- CSV files
- Excel files (.xlsx, .xls)
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .models import Keyword


class KeywordLoadError(Exception):
    """Raised when keyword loading fails."""
    pass


# Common column name variations for keyword data
KEYWORD_COLUMN_VARIANTS = ["keyword", "keywords", "term", "terms", "query", "queries", "phrase"]
VOLUME_COLUMN_VARIANTS = ["search_volume", "volume", "searchvolume", "sv", "avg_monthly_searches"]
DIFFICULTY_COLUMN_VARIANTS = ["difficulty", "kd", "keyword_difficulty", "seo_difficulty", "competition"]
INTENT_COLUMN_VARIANTS = ["intent", "search_intent", "keyword_intent"]
BRAND_COLUMN_VARIANTS = ["is_brand", "brand", "branded", "type"]  # "type" can be "brand" or "topic"


def _normalize_column_name(name: str) -> str:
    """Normalize column name for matching."""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def _find_column(df: pd.DataFrame, variants: list[str]) -> Optional[str]:
    """
    Find a column in the DataFrame matching one of the variant names.

    Args:
        df: The DataFrame to search.
        variants: List of possible column name variants.

    Returns:
        The actual column name if found, None otherwise.
    """
    normalized_columns = {_normalize_column_name(col): col for col in df.columns}

    for variant in variants:
        normalized = _normalize_column_name(variant)
        if normalized in normalized_columns:
            return normalized_columns[normalized]

    return None


def load_keywords_from_csv(file_path: Union[str, Path]) -> list[Keyword]:
    """
    Load keywords from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of Keyword objects.

    Raises:
        KeywordLoadError: If the file cannot be read or parsed.
    """
    path = Path(file_path)

    if not path.exists():
        raise KeywordLoadError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        # Try alternative encoding
        try:
            df = pd.read_csv(path, encoding="latin-1")
        except Exception as e:
            raise KeywordLoadError(f"Failed to read CSV file: {e}")
    except Exception as e:
        raise KeywordLoadError(f"Failed to read CSV file: {e}")

    return _parse_keyword_dataframe(df)


def load_keywords_from_excel(file_path: Union[str, Path], sheet_name: Optional[str] = None) -> list[Keyword]:
    """
    Load keywords from an Excel file.

    Args:
        file_path: Path to the Excel file (.xlsx or .xls).
        sheet_name: Optional sheet name to read from. Defaults to first sheet.

    Returns:
        List of Keyword objects.

    Raises:
        KeywordLoadError: If the file cannot be read or parsed.
    """
    path = Path(file_path)

    if not path.exists():
        raise KeywordLoadError(f"File not found: {file_path}")

    try:
        if sheet_name:
            df = pd.read_excel(path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(path)
    except Exception as e:
        raise KeywordLoadError(f"Failed to read Excel file: {e}")

    return _parse_keyword_dataframe(df)


def _parse_keyword_dataframe(df: pd.DataFrame) -> list[Keyword]:
    """
    Parse a DataFrame into a list of Keyword objects.

    Args:
        df: DataFrame containing keyword data.

    Returns:
        List of Keyword objects.

    Raises:
        KeywordLoadError: If required columns are missing.
    """
    if df.empty:
        raise KeywordLoadError("Keyword file is empty")

    # Find the keyword column (required)
    keyword_col = _find_column(df, KEYWORD_COLUMN_VARIANTS)
    if keyword_col is None:
        raise KeywordLoadError(
            f"No keyword column found. Expected one of: {', '.join(KEYWORD_COLUMN_VARIANTS)}. "
            f"Found columns: {', '.join(df.columns)}"
        )

    # Find optional columns
    volume_col = _find_column(df, VOLUME_COLUMN_VARIANTS)
    difficulty_col = _find_column(df, DIFFICULTY_COLUMN_VARIANTS)
    intent_col = _find_column(df, INTENT_COLUMN_VARIANTS)
    brand_col = _find_column(df, BRAND_COLUMN_VARIANTS)

    keywords: list[Keyword] = []

    for _, row in df.iterrows():
        # Get keyword phrase
        phrase = row[keyword_col]
        if pd.isna(phrase) or not str(phrase).strip():
            continue

        phrase = str(phrase).strip()

        # Get optional fields
        search_volume: Optional[int] = None
        if volume_col and not pd.isna(row[volume_col]):
            try:
                search_volume = int(float(row[volume_col]))
            except (ValueError, TypeError):
                pass

        difficulty: Optional[float] = None
        if difficulty_col and not pd.isna(row[difficulty_col]):
            try:
                difficulty = float(row[difficulty_col])
                # Normalize to 0-100 range if it's a decimal
                if 0 < difficulty < 1:
                    difficulty *= 100
            except (ValueError, TypeError):
                pass

        intent: Optional[str] = None
        if intent_col and not pd.isna(row[intent_col]):
            intent_value = str(row[intent_col]).strip().lower()
            # Normalize common intent values
            if intent_value in ("info", "informational", "i"):
                intent = "informational"
            elif intent_value in ("transactional", "trans", "t", "commercial"):
                intent = "transactional"
            elif intent_value in ("nav", "navigational", "n"):
                intent = "navigational"
            else:
                intent = intent_value

        # Parse is_brand flag from column
        is_brand: bool = False
        if brand_col and not pd.isna(row[brand_col]):
            brand_value = str(row[brand_col]).strip().lower()
            # Support various formats: true/false, yes/no, 1/0, brand/topic
            if brand_value in ("true", "yes", "1", "brand", "branded", "y"):
                is_brand = True
            elif brand_value in ("false", "no", "0", "topic", "n"):
                is_brand = False

        keywords.append(
            Keyword(
                phrase=phrase,
                search_volume=search_volume,
                difficulty=difficulty,
                intent=intent,
                is_brand=is_brand,
            )
        )

    if not keywords:
        raise KeywordLoadError("No valid keywords found in file")

    return keywords


def load_keywords(file_path: Union[str, Path], sheet_name: Optional[str] = None) -> list[Keyword]:
    """
    Load keywords from a CSV or Excel file.

    Automatically detects file type based on extension.

    Args:
        file_path: Path to the keyword file.
        sheet_name: Optional sheet name for Excel files.

    Returns:
        List of Keyword objects.

    Raises:
        KeywordLoadError: If the file cannot be read or is invalid.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_keywords_from_csv(path)
    elif suffix in (".xlsx", ".xls"):
        return load_keywords_from_excel(path, sheet_name)
    else:
        raise KeywordLoadError(
            f"Unsupported file format: {suffix}. Supported formats: .csv, .xlsx, .xls"
        )


def deduplicate_keywords(keywords: list[Keyword]) -> list[Keyword]:
    """
    Remove duplicate keywords based on phrase (case-insensitive).

    Keeps the first occurrence of each keyword.

    Args:
        keywords: List of keywords to deduplicate.

    Returns:
        Deduplicated list of keywords.
    """
    seen: set[str] = set()
    unique: list[Keyword] = []

    for kw in keywords:
        key = kw.phrase.lower()
        if key not in seen:
            seen.add(key)
            unique.append(kw)

    return unique


def filter_keywords_by_intent(
    keywords: list[Keyword], intent: str, include_none: bool = True
) -> list[Keyword]:
    """
    Filter keywords by intent.

    Args:
        keywords: List of keywords to filter.
        intent: Intent to filter by (e.g., "informational", "transactional").
        include_none: Whether to include keywords with no intent specified.

    Returns:
        Filtered list of keywords.
    """
    intent_lower = intent.lower()
    return [
        kw
        for kw in keywords
        if (kw.intent and kw.intent.lower() == intent_lower)
        or (include_none and kw.intent is None)
    ]


def sort_keywords_by_priority(keywords: list[Keyword]) -> list[Keyword]:
    """
    Sort keywords by priority (search volume descending, difficulty ascending).

    Keywords with higher volume and lower difficulty come first.

    Args:
        keywords: List of keywords to sort.

    Returns:
        Sorted list of keywords.
    """
    def priority_key(kw: Keyword) -> tuple[int, float]:
        # Higher volume = higher priority (negative for descending)
        volume_score = -(kw.search_volume or 0)
        # Lower difficulty = higher priority
        difficulty_score = kw.difficulty or 50.0  # Default to medium
        return (volume_score, difficulty_score)

    return sorted(keywords, key=priority_key)
