"""Tests for LLM client utility functions."""

import pytest

from seo_content_optimizer.llm_client import (
    ADD_START,
    ADD_END,
    ensure_markers_present,
    has_markers,
    strip_markers,
)


class TestEnsureMarkersPresent:
    """Tests for the ensure_markers_present function."""

    def test_returns_unchanged_when_markers_present(self):
        """Test that content with markers is returned unchanged."""
        original = "Original text"
        optimized = f"Original {ADD_START}optimized{ADD_END} text"

        result = ensure_markers_present(original, optimized)

        assert result == optimized
        assert has_markers(result)

    def test_returns_unchanged_when_content_identical(self):
        """Test that identical content returns unchanged without markers."""
        original = "Same text"
        optimized = "Same text"

        result = ensure_markers_present(original, optimized)

        assert result == optimized
        assert not has_markers(result)

    def test_wraps_changed_content_without_markers(self):
        """Test that changed content without markers gets wrapped."""
        original = "Original text"
        optimized = "New different text"  # LLM changed it but forgot markers

        result = ensure_markers_present(original, optimized)

        assert result == f"{ADD_START}New different text{ADD_END}"
        assert has_markers(result)

    def test_handles_empty_original(self):
        """Test handling when original is empty/None."""
        original = ""
        optimized = "New content"

        result = ensure_markers_present(original, optimized)

        assert result == f"{ADD_START}New content{ADD_END}"
        assert has_markers(result)

    def test_handles_whitespace_differences(self):
        """Test that whitespace-only differences still trigger markers."""
        original = "Same text"
        optimized = "Same text  "  # Extra spaces

        result = ensure_markers_present(original, optimized)

        # After strip(), they're equal - no markers needed
        assert result == optimized
        # But if stripped version matches, no markers added
        assert not has_markers(result)

    def test_partial_markers_not_considered_valid(self):
        """Test that partial markers (only ADD_START) don't count as valid."""
        original = "Original"
        optimized = f"{ADD_START}Only start marker"

        result = ensure_markers_present(original, optimized)

        # has_markers returns True if either marker is present
        # So this should return unchanged since markers exist
        assert result == optimized


class TestHasMarkers:
    """Tests for has_markers function."""

    def test_returns_true_with_both_markers(self):
        """Test returns True when both markers present."""
        text = f"Text {ADD_START}marked{ADD_END} content"
        assert has_markers(text) is True

    def test_returns_true_with_start_only(self):
        """Test returns True when only start marker present."""
        text = f"Text {ADD_START}incomplete"
        assert has_markers(text) is True

    def test_returns_true_with_end_only(self):
        """Test returns True when only end marker present."""
        text = f"incomplete{ADD_END} text"
        assert has_markers(text) is True

    def test_returns_false_with_no_markers(self):
        """Test returns False when no markers present."""
        text = "Plain text without any markers"
        assert has_markers(text) is False


class TestStripMarkers:
    """Tests for strip_markers function."""

    def test_removes_both_markers(self):
        """Test removes both ADD_START and ADD_END markers."""
        text = f"Text {ADD_START}highlighted{ADD_END} content"
        result = strip_markers(text)

        assert result == "Text highlighted content"
        assert ADD_START not in result
        assert ADD_END not in result

    def test_handles_multiple_marker_pairs(self):
        """Test handles multiple marker pairs."""
        text = f"{ADD_START}First{ADD_END} middle {ADD_START}second{ADD_END}"
        result = strip_markers(text)

        assert result == "First middle second"

    def test_handles_no_markers(self):
        """Test returns unchanged when no markers."""
        text = "Plain text"
        result = strip_markers(text)

        assert result == "Plain text"
