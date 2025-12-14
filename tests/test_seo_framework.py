"""
Tests for the 10-Part SEO Framework implementation.

This module tests:
- ContentAudit creation and keyword placement tracking
- OptimizationPlan generation with tiered keyword placement
- KeywordPlacementStatus tracking across tiers
- Keyword enforcement in optimization flow
"""

import pytest
from src.seo_content_optimizer.models import (
    ContentAudit,
    ContentIntent,
    DocxContent,
    HeadingLevel,
    Keyword,
    KeywordPlan,
    KeywordPlacementPlan,
    KeywordPlacementStatus,
    OptimizationPlan,
    PageMeta,
    ParagraphBlock,
)
from src.seo_content_optimizer.analysis import (
    audit_content,
    build_optimization_plan,
)


class TestKeywordPlacementStatus:
    """Tests for KeywordPlacementStatus model."""

    def test_placement_score_full_coverage(self):
        """Test placement score with all tiers covered."""
        status = KeywordPlacementStatus(
            keyword="external cameras",
            in_title=True,
            in_meta_description=True,
            in_h1=True,
            in_first_100_words=True,
            in_subheadings=True,
            in_body=True,
            in_conclusion=True,
            body_count=5,
        )

        # Full coverage should have high score
        assert status.placement_score > 40  # All tiers covered
        assert status.missing_placements == []

    def test_placement_score_partial_coverage(self):
        """Test placement score with partial tier coverage."""
        status = KeywordPlacementStatus(
            keyword="property security",
            in_title=False,
            in_meta_description=True,
            in_h1=True,
            in_first_100_words=False,
            in_subheadings=True,
            in_body=True,
            in_conclusion=False,
            body_count=2,
        )

        # Missing Tier 1 (title) and Tier 3 (first 100 words)
        assert "title" in status.missing_placements
        assert "first_100_words" in status.missing_placements
        assert "conclusion" in status.missing_placements
        assert status.placement_score > 0

    def test_placement_score_minimal_coverage(self):
        """Test placement score with minimal coverage."""
        status = KeywordPlacementStatus(
            keyword="access control",
            in_title=False,
            in_meta_description=False,
            in_h1=False,
            in_first_100_words=False,
            in_subheadings=False,
            in_body=True,
            in_conclusion=False,
            body_count=1,
        )

        # Only Tier 5 (body) covered
        assert status.placement_score == 4  # Only body score
        assert len(status.missing_placements) == 6  # Missing 6 placements

    def test_missing_placements_list(self):
        """Test that missing_placements correctly identifies gaps."""
        status = KeywordPlacementStatus(
            keyword="test keyword",
            in_title=True,
            in_meta_description=True,
            in_h1=True,
            in_first_100_words=True,
            in_subheadings=False,
            in_body=False,
            in_conclusion=False,
        )

        missing = status.missing_placements
        assert "subheadings" in missing
        assert "body" in missing
        assert "conclusion" in missing
        assert "title" not in missing
        assert "h1" not in missing


class TestContentAudit:
    """Tests for ContentAudit model and audit_content function."""

    def test_audit_page_meta_basic(self):
        """Test auditing PageMeta content."""
        content = PageMeta(
            title="External Cameras for Property Security",
            meta_description="Learn about external cameras for your property.",
            h1="Guide to External Cameras",
            content_blocks=[
                "External cameras are essential for property security.",
                "Modern external cameras offer high-resolution footage.",
                "Consider access control integration.",
            ],
        )

        keyword_plan = KeywordPlan(
            primary=Keyword(phrase="external cameras"),
            secondary=[
                Keyword(phrase="property security"),
                Keyword(phrase="access control"),
            ],
        )

        audit = audit_content(
            content=content,
            keyword_plan=keyword_plan,
            meta_title="External Cameras for Property Security",
            meta_description="Learn about external cameras for your property.",
        )

        assert isinstance(audit, ContentAudit)
        assert audit.word_count > 0
        assert audit.current_meta_title == "External Cameras for Property Security"
        assert len(audit.keyword_status) >= 1

    def test_audit_detects_primary_keyword_placement(self):
        """Test that audit detects primary keyword placement correctly."""
        content = PageMeta(
            title="External Cameras Guide",
            meta_description="Learn about external cameras.",
            h1="External Cameras for Security",
            content_blocks=[
                "External cameras provide excellent surveillance.",
                "These cameras are essential for property protection.",
            ],
        )

        keyword_plan = KeywordPlan(
            primary=Keyword(phrase="external cameras"),
            secondary=[],
        )

        audit = audit_content(
            content=content,
            keyword_plan=keyword_plan,
            meta_title="External Cameras Guide",
            meta_description="Learn about external cameras.",
        )

        # Check primary keyword status
        primary_status = audit.primary_keyword_status
        assert primary_status is not None
        assert primary_status.keyword == "external cameras"
        assert primary_status.in_title is True
        assert primary_status.in_meta_description is True
        assert primary_status.in_h1 is True
        assert primary_status.in_first_100_words is True

    def test_audit_detects_missing_placements(self):
        """Test that audit detects missing keyword placements."""
        content = PageMeta(
            title="Security Systems Overview",  # No primary keyword
            meta_description="Learn about security.",  # No primary keyword
            h1="Security Guide",  # No primary keyword
            content_blocks=[
                "Security is important for homes.",  # No primary keyword
            ],
        )

        keyword_plan = KeywordPlan(
            primary=Keyword(phrase="external cameras"),
            secondary=[],
        )

        audit = audit_content(
            content=content,
            keyword_plan=keyword_plan,
            meta_title="Security Systems Overview",
            meta_description="Learn about security.",
        )

        primary_status = audit.primary_keyword_status
        assert primary_status is not None
        assert primary_status.in_title is False
        assert primary_status.in_meta_description is False
        assert primary_status.in_h1 is False
        assert audit.has_critical_gaps is True

    def test_audit_identifies_depth_gaps(self):
        """Test that audit identifies content depth gaps."""
        content = PageMeta(
            title="External Cameras",
            h1="External Cameras Guide",
            content_blocks=[
                "External cameras are great.",
            ],
        )

        keyword_plan = KeywordPlan(
            primary=Keyword(phrase="external cameras"),
            secondary=[
                Keyword(phrase="installation"),
                Keyword(phrase="maintenance"),
            ],
        )

        audit = audit_content(
            content=content,
            keyword_plan=keyword_plan,
        )

        # Should identify missing topics based on keywords
        assert audit.depth_gaps is not None

    def test_audit_docx_content(self):
        """Test auditing DocxContent."""
        content = DocxContent(
            paragraphs=[
                ParagraphBlock(
                    text="External Cameras for Property Security",
                    heading_level=HeadingLevel.H1,
                ),
                ParagraphBlock(
                    text="External cameras are essential for modern security.",
                    heading_level=HeadingLevel.BODY,
                ),
                ParagraphBlock(
                    text="Types of Cameras",
                    heading_level=HeadingLevel.H2,
                ),
            ]
        )

        keyword_plan = KeywordPlan(
            primary=Keyword(phrase="external cameras"),
            secondary=[],
        )

        audit = audit_content(
            content=content,
            keyword_plan=keyword_plan,
        )

        assert isinstance(audit, ContentAudit)
        assert len(audit.heading_outline) >= 1


class TestOptimizationPlan:
    """Tests for OptimizationPlan and build_optimization_plan function."""

    def test_build_optimization_plan_basic(self):
        """Test building a basic optimization plan."""
        audit = ContentAudit(
            topic_summary="Guide to external cameras for property security",
            intent="informational",
            word_count=500,
            current_meta_title="External Cameras Guide",
            current_meta_description="Learn about cameras.",
            current_h1="Camera Guide",
            keyword_status=[
                KeywordPlacementStatus(
                    keyword="external cameras",
                    in_title=True,
                    in_meta_description=False,
                    in_h1=False,
                    in_first_100_words=True,
                    in_subheadings=True,
                    in_body=True,
                    in_conclusion=False,
                    body_count=3,
                ),
            ],
        )

        keyword_plan = KeywordPlan(
            primary=Keyword(phrase="external cameras"),
            secondary=[
                Keyword(phrase="property security"),
                Keyword(phrase="access control"),
            ],
        )

        plan = build_optimization_plan(audit=audit, keyword_plan=keyword_plan)

        assert isinstance(plan, OptimizationPlan)
        assert plan.primary_keyword == "external cameras"
        assert "property security" in plan.secondary_keywords
        assert plan.audit == audit

    def test_optimization_plan_has_placement_plan(self):
        """Test that optimization plan includes keyword placement plan."""
        audit = ContentAudit(
            topic_summary="Security cameras guide",
            intent="informational",
            word_count=500,
        )

        keyword_plan = KeywordPlan(
            primary=Keyword(phrase="external cameras"),
            secondary=[Keyword(phrase="property security")],
        )

        plan = build_optimization_plan(audit=audit, keyword_plan=keyword_plan)

        assert plan.placement_plan is not None
        assert isinstance(plan.placement_plan, KeywordPlacementPlan)
        assert plan.placement_plan.title == "external cameras"
        assert plan.placement_plan.h1 == "external cameras"

    def test_optimization_plan_generates_targets(self):
        """Test that optimization plan generates target meta elements."""
        audit = ContentAudit(
            topic_summary="External cameras for property security",
            intent="informational",
            word_count=500,
            current_meta_title="Old Title",
            current_meta_description="Old description.",
            current_h1="Old H1",
        )

        keyword_plan = KeywordPlan(
            primary=Keyword(phrase="external cameras"),
            secondary=[],
        )

        plan = build_optimization_plan(audit=audit, keyword_plan=keyword_plan)

        # Targets should contain primary keyword
        assert "external cameras" in plan.target_meta_title.lower()
        assert "external cameras" in plan.target_meta_description.lower()
        assert "external cameras" in plan.target_h1.lower()

    def test_optimization_plan_all_keywords_property(self):
        """Test the all_keywords property."""
        plan = OptimizationPlan(
            primary_keyword="external cameras",
            secondary_keywords=["property security", "access control"],
        )

        all_kws = plan.all_keywords
        assert len(all_kws) == 3
        assert "external cameras" in all_kws
        assert "property security" in all_kws
        assert "access control" in all_kws


class TestKeywordPlacementPlan:
    """Tests for KeywordPlacementPlan model."""

    def test_placement_plan_structure(self):
        """Test KeywordPlacementPlan structure."""
        plan = KeywordPlacementPlan(
            title="external cameras",
            meta_description="external cameras",
            h1="external cameras",
            first_100_words="external cameras",
            subheadings=["property security", "access control"],
            body_priority=["external cameras", "property security", "access control"],
            faq_keywords=["external cameras", "property security"],
            conclusion=["external cameras"],
        )

        assert plan.title == "external cameras"
        assert len(plan.subheadings) == 2
        assert len(plan.body_priority) == 3
        assert len(plan.faq_keywords) == 2

    def test_placement_plan_defaults(self):
        """Test KeywordPlacementPlan default values."""
        plan = KeywordPlacementPlan(
            title="keyword",
            meta_description="keyword",
            h1="keyword",
            first_100_words="keyword",
        )

        assert plan.subheadings == []
        assert plan.body_priority == []
        assert plan.faq_keywords == []
        assert plan.conclusion == []


class TestKeywordEnforcementHelpers:
    """Tests for keyword enforcement helper functions."""

    def test_ensure_keyword_in_text_when_present(self):
        """If keyword already present, text unchanged."""
        from src.seo_content_optimizer.optimizer import ensure_keyword_in_text

        text = "External cameras are great for security."
        result = ensure_keyword_in_text(text, "external cameras")
        assert result == text  # No change needed

    def test_ensure_keyword_in_text_case_insensitive(self):
        """Keyword check is case-insensitive."""
        from src.seo_content_optimizer.optimizer import ensure_keyword_in_text

        text = "EXTERNAL CAMERAS are essential."
        result = ensure_keyword_in_text(text, "external cameras")
        assert result == text  # No change needed - already present

    def test_ensure_keyword_in_text_inject_at_start(self):
        """If keyword missing, inject at start with colon separator."""
        from src.seo_content_optimizer.optimizer import ensure_keyword_in_text

        text = "Security systems help protect property."
        result = ensure_keyword_in_text(text, "external cameras", position="start")
        assert result.startswith("external cameras:")
        assert "external cameras" in result.lower()

    def test_ensure_keyword_in_text_inject_at_end(self):
        """If keyword missing, inject at end."""
        from src.seo_content_optimizer.optimizer import ensure_keyword_in_text

        text = "Security systems help protect property."
        result = ensure_keyword_in_text(text, "external cameras", position="end")
        assert "external cameras" in result.lower()

    def test_count_keyword_in_text_basic(self):
        """Basic keyword counting works."""
        from src.seo_content_optimizer.optimizer import count_keyword_in_text

        text = "External cameras for property. External cameras are great."
        count = count_keyword_in_text(text, "external cameras")
        assert count == 2

    def test_count_keyword_in_text_case_insensitive(self):
        """Keyword counting is case-insensitive."""
        from src.seo_content_optimizer.optimizer import count_keyword_in_text

        text = "EXTERNAL CAMERAS and external cameras and External Cameras."
        count = count_keyword_in_text(text, "external cameras")
        assert count == 3


class TestTieredPlacement:
    """Tests for tiered keyword placement hierarchy."""

    def test_tier_1_title_priority(self):
        """Test that Tier 1 (title) has highest priority score."""
        tier1 = KeywordPlacementStatus(
            keyword="test",
            in_title=True,
            in_meta_description=False,
            in_h1=False,
            in_first_100_words=False,
            in_subheadings=False,
            in_body=False,
            in_conclusion=False,
        )

        tier5 = KeywordPlacementStatus(
            keyword="test",
            in_title=False,
            in_meta_description=False,
            in_h1=False,
            in_first_100_words=False,
            in_subheadings=False,
            in_body=True,
            in_conclusion=False,
        )

        assert tier1.placement_score > tier5.placement_score

    def test_tier_2_h1_priority(self):
        """Test that Tier 2 (H1) has higher priority than lower tiers."""
        tier2 = KeywordPlacementStatus(
            keyword="test",
            in_title=False,
            in_meta_description=False,
            in_h1=True,
            in_first_100_words=False,
            in_subheadings=False,
            in_body=False,
            in_conclusion=False,
        )

        tier4 = KeywordPlacementStatus(
            keyword="test",
            in_title=False,
            in_meta_description=False,
            in_h1=False,
            in_first_100_words=False,
            in_subheadings=True,
            in_body=False,
            in_conclusion=False,
        )

        assert tier2.placement_score > tier4.placement_score

    def test_tier_3_first_100_words_priority(self):
        """Test that Tier 3 (first 100 words) has proper priority."""
        tier3 = KeywordPlacementStatus(
            keyword="test",
            in_title=False,
            in_meta_description=False,
            in_h1=False,
            in_first_100_words=True,
            in_subheadings=False,
            in_body=False,
            in_conclusion=False,
        )

        tier5 = KeywordPlacementStatus(
            keyword="test",
            in_title=False,
            in_meta_description=False,
            in_h1=False,
            in_first_100_words=False,
            in_subheadings=False,
            in_body=True,
            in_conclusion=False,
        )

        assert tier3.placement_score > tier5.placement_score


class TestContentAuditIntegration:
    """Integration tests for ContentAudit with real content patterns."""

    def test_cellgate_style_content_audit(self):
        """Test audit of CellGate-style security camera content."""
        content = PageMeta(
            title="External Cameras for Enhanced Property Security: A Guide",
            meta_description="Learn how external cameras can improve your property security with CellGate solutions.",
            h1="External Cameras for Enhanced Property Security",
            content_blocks=[
                "External cameras are essential for modern property security systems.",
                "CellGate offers advanced external camera solutions for residential and commercial properties.",
                "When choosing external cameras, consider factors like resolution, night vision, and integration.",
                "Access control systems can be enhanced with external cameras for comprehensive security.",
                "In conclusion, external cameras provide peace of mind and enhanced property security.",
            ],
        )

        keyword_plan = KeywordPlan(
            primary=Keyword(phrase="external cameras"),
            secondary=[
                Keyword(phrase="property security"),
                Keyword(phrase="access control"),
            ],
        )

        audit = audit_content(
            content=content,
            keyword_plan=keyword_plan,
            meta_title=content.title,
            meta_description=content.meta_description,
        )

        # Primary keyword should be in all critical placements
        primary_status = audit.primary_keyword_status
        assert primary_status.in_title is True
        assert primary_status.in_meta_description is True
        assert primary_status.in_h1 is True
        assert primary_status.in_first_100_words is True
        assert primary_status.in_conclusion is True
        assert audit.has_critical_gaps is False

    def test_audit_with_poor_keyword_coverage(self):
        """Test audit detection of poor keyword coverage."""
        content = PageMeta(
            title="Security Guide",  # Missing primary keyword
            meta_description="A complete guide to security.",  # Missing primary keyword
            h1="Security Solutions",  # Missing primary keyword
            content_blocks=[
                "Security is important for every property.",  # No primary keyword
                "Consider installing cameras for better protection.",  # Not the exact phrase
            ],
        )

        keyword_plan = KeywordPlan(
            primary=Keyword(phrase="external cameras"),
            secondary=[Keyword(phrase="property security")],
        )

        audit = audit_content(
            content=content,
            keyword_plan=keyword_plan,
            meta_title=content.title,
            meta_description=content.meta_description,
        )

        # Primary keyword should be missing from critical placements
        primary_status = audit.primary_keyword_status
        assert primary_status.in_title is False
        assert primary_status.in_h1 is False
        assert audit.has_critical_gaps is True
        assert len(audit.high_priority_issues) > 0
