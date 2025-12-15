"""
Tests for SEO Content Optimizer V2 Architecture components.

Tests cover:
- ContentDocument and ContentBlock models
- Structure preservation policies
- Factuality guardrails
- Token-level diff highlighting
- Change summary reporting
"""

import pytest
from datetime import datetime

from seo_content_optimizer.models import (
    ContentBlock,
    ContentBlockType,
    ContentDocument,
    SemanticKeyword,
    SemanticKeywordPlan,
    FactualityClaim,
)
from seo_content_optimizer.structure_preservation import (
    StructurePreserver,
    StructurePolicy,
    PreservationPolicy,
    get_modifiable_blocks,
    DEFAULT_POLICIES,
)
from seo_content_optimizer.factuality_guardrails import (
    FactualityChecker,
    ClaimDetectionConfig,
    validate_no_new_facts,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
)
from seo_content_optimizer.diff_highlighter import (
    TokenDiffer,
    DiffType,
    compute_diff,
    get_changes_summary,
    find_new_keywords_in_text,
)
from seo_content_optimizer.change_summary import (
    ChangeSummaryBuilder,
    ChangeType,
    format_summary_text,
    format_summary_dict,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_content_blocks() -> list[ContentBlock]:
    """Create sample content blocks for testing."""
    return [
        ContentBlock(
            block_id="h1_0",
            block_type=ContentBlockType.HEADING,
            text="Professional Liability Insurance Guide",
            order=0,
            heading_level=1,
        ),
        ContentBlock(
            block_id="p_1",
            block_type=ContentBlockType.PARAGRAPH,
            text="This guide covers everything you need to know about insurance coverage.",
            order=1,
        ),
        ContentBlock(
            block_id="h2_2",
            block_type=ContentBlockType.HEADING,
            text="What is Professional Liability?",
            order=2,
            heading_level=2,
        ),
        ContentBlock(
            block_id="p_3",
            block_type=ContentBlockType.PARAGRAPH,
            text="Professional liability protects against claims of negligence.",
            order=3,
        ),
        ContentBlock(
            block_id="list_4",
            block_type=ContentBlockType.LIST,
            text="- Coverage for legal fees\n- Protection against claims\n- Peace of mind",
            order=4,
        ),
        ContentBlock(
            block_id="table_5",
            block_type=ContentBlockType.TABLE,
            text="| Plan | Price |\n| Basic | $100 |\n| Premium | $200 |",
            order=5,
        ),
        ContentBlock(
            block_id="code_6",
            block_type=ContentBlockType.CODE,
            text="function calculatePremium() { return base * factor; }",
            order=6,
        ),
    ]


@pytest.fixture
def sample_content_document(sample_content_blocks) -> ContentDocument:
    """Create a sample content document."""
    return ContentDocument(
        title="Insurance Guide",
        source_url="https://example.com/insurance",
        blocks=sample_content_blocks,
    )


@pytest.fixture
def sample_keyword_plan() -> SemanticKeywordPlan:
    """Create a sample keyword plan."""
    primary = SemanticKeyword(
        phrase="professional liability insurance",
        relevance_score=0.95,
        selected=True,
    )
    secondary = [
        SemanticKeyword(phrase="liability coverage", relevance_score=0.85, selected=True),
        SemanticKeyword(phrase="business insurance", relevance_score=0.80, selected=True),
    ]
    return SemanticKeywordPlan(
        primary=primary,
        secondary=secondary,
        questions=[],
        clusters=[],
    )


# =============================================================================
# CONTENT MODEL TESTS
# =============================================================================

class TestContentBlock:
    """Tests for ContentBlock model."""

    def test_create_paragraph_block(self):
        """Test creating a paragraph block."""
        block = ContentBlock(
            block_id="p_1",
            block_type=ContentBlockType.PARAGRAPH,
            text="Sample paragraph text.",
            order=0,
        )
        assert block.block_id == "p_1"
        assert block.block_type == ContentBlockType.PARAGRAPH
        assert block.text == "Sample paragraph text."

    def test_create_heading_block(self):
        """Test creating a heading block."""
        block = ContentBlock(
            block_id="h1_0",
            block_type=ContentBlockType.HEADING,
            text="Main Title",
            order=0,
            heading_level=1,
        )
        assert block.heading_level == 1

    def test_word_count(self, sample_content_blocks):
        """Test word count calculation."""
        para_block = sample_content_blocks[1]
        word_count = len(para_block.text.split())
        assert word_count > 5


class TestContentDocument:
    """Tests for ContentDocument model."""

    def test_document_creation(self, sample_content_document):
        """Test document creation."""
        doc = sample_content_document
        assert doc.title == "Insurance Guide"
        assert len(doc.blocks) == 7

    def test_full_text_property(self, sample_content_document):
        """Test full text extraction."""
        full_text = sample_content_document.full_text
        assert "Professional Liability Insurance Guide" in full_text
        assert "insurance coverage" in full_text

    def test_paragraphs_filter(self, sample_content_document):
        """Test paragraph filtering."""
        paragraphs = sample_content_document.paragraphs
        assert len(paragraphs) == 2
        assert all(p.block_type == ContentBlockType.PARAGRAPH for p in paragraphs)


# =============================================================================
# STRUCTURE PRESERVATION TESTS
# =============================================================================

class TestStructurePreserver:
    """Tests for structure preservation policies."""

    def test_default_policies_exist(self):
        """Test that default policies are defined."""
        assert ContentBlockType.TABLE in DEFAULT_POLICIES
        assert ContentBlockType.LIST in DEFAULT_POLICIES
        assert ContentBlockType.PARAGRAPH in DEFAULT_POLICIES

    def test_table_is_strict(self):
        """Test that tables have strict policy."""
        policy = DEFAULT_POLICIES[ContentBlockType.TABLE]
        assert policy.preservation == PreservationPolicy.STRICT
        assert policy.allow_keyword_injection is False

    def test_paragraph_is_relaxed(self):
        """Test that paragraphs have relaxed policy."""
        policy = DEFAULT_POLICIES[ContentBlockType.PARAGRAPH]
        assert policy.preservation == PreservationPolicy.RELAXED
        assert policy.allow_keyword_injection is True

    def test_can_modify_paragraph(self, sample_content_blocks):
        """Test that paragraphs can be modified."""
        preserver = StructurePreserver()
        para_block = sample_content_blocks[1]
        assert preserver.can_modify(para_block) is True

    def test_cannot_modify_table(self, sample_content_blocks):
        """Test that tables cannot be modified."""
        preserver = StructurePreserver()
        table_block = sample_content_blocks[5]
        assert preserver.can_modify(table_block) is False

    def test_cannot_modify_code(self, sample_content_blocks):
        """Test that code blocks cannot be modified."""
        preserver = StructurePreserver()
        code_block = sample_content_blocks[6]
        assert preserver.can_modify(code_block) is False

    def test_get_modifiable_blocks(self, sample_content_blocks):
        """Test filtering modifiable blocks."""
        modifiable = get_modifiable_blocks(sample_content_blocks)
        # Headings, paragraphs, lists (intro) should be modifiable
        # Tables and code should not
        assert len(modifiable) < len(sample_content_blocks)

    def test_validate_modification_strict_violation(self, sample_content_blocks):
        """Test that strict policy blocks modifications."""
        preserver = StructurePreserver()
        table_block = sample_content_blocks[5]

        result = preserver.validate_modification(
            table_block,
            "Modified table content"
        )
        assert result.is_valid is False
        assert len(result.violations) > 0


# =============================================================================
# FACTUALITY GUARDRAILS TESTS
# =============================================================================

class TestFactualityChecker:
    """Tests for factuality detection."""

    def test_detect_percentage(self):
        """Test percentage detection."""
        checker = FactualityChecker()
        claims = checker.detect_claims("Our service has a 95% success rate.")
        assert len(claims) >= 1
        assert any(c.claim_type == "percentage" for c in claims)

    def test_detect_year(self):
        """Test year detection."""
        checker = FactualityChecker()
        claims = checker.detect_claims("Founded in 1995, the company has grown.")
        assert any(c.claim_type in ("year", "founding_date") for c in claims)

    def test_detect_number_with_unit(self):
        """Test number with unit detection."""
        checker = FactualityChecker()
        claims = checker.detect_claims("We serve over 10,000 customers worldwide.")
        assert len(claims) >= 1

    def test_detect_certification(self):
        """Test certification detection."""
        checker = FactualityChecker()
        claims = checker.detect_claims("We are ISO 9001 certified.")
        assert any(c.claim_type == "certification" for c in claims)

    def test_compare_claims_no_new(self):
        """Test that no new claims are detected for unchanged text."""
        checker = FactualityChecker()
        original = "We have a 95% success rate since 2010."
        modified = "We have a 95% success rate since 2010."

        result = checker.compare_claims(original, modified)
        assert result.is_valid is True
        assert len(result.new_claims) == 0

    def test_compare_claims_new_stat(self):
        """Test detection of new statistical claim."""
        checker = FactualityChecker()
        original = "Our service is reliable."
        modified = "Our service is reliable with 99% uptime."

        result = checker.compare_claims(original, modified)
        assert len(result.new_claims) >= 1
        assert result.is_valid is False  # High severity claim added

    def test_validate_no_new_facts_function(self):
        """Test convenience validation function."""
        # Note: 50% is in allowed common numbers set, use 95% which is not
        is_valid, warnings = validate_no_new_facts(
            "Simple text.",
            "Simple text with 95% improvement."
        )
        assert is_valid is False
        assert len(warnings) > 0


# =============================================================================
# DIFF HIGHLIGHTER TESTS
# =============================================================================

class TestTokenDiffer:
    """Tests for token-level diffing."""

    def test_identical_texts(self):
        """Test diff of identical texts."""
        differ = TokenDiffer()
        result = differ.diff("Hello world", "Hello world")
        assert all(s.diff_type == DiffType.EQUAL for s in result.spans)

    def test_insertion(self):
        """Test detection of insertion."""
        differ = TokenDiffer()
        result = differ.diff(
            "The quick fox",
            "The quick brown fox"
        )
        assert result.total_insertions > 0
        insertions = [s for s in result.spans if s.diff_type == DiffType.INSERT]
        assert len(insertions) >= 1

    def test_keyword_detection(self):
        """Test keyword detection in diff."""
        differ = TokenDiffer(keywords=["insurance coverage"])
        result = differ.diff(
            "Get protection today.",
            "Get insurance coverage today."
        )
        keyword_spans = [s for s in result.spans if s.is_keyword]
        assert len(keyword_spans) >= 1

    def test_find_keyword_positions(self):
        """Test finding keyword positions in text."""
        differ = TokenDiffer(keywords=["liability insurance", "coverage"])
        positions = differ.find_keyword_positions(
            "Get liability insurance and coverage today."
        )
        assert len(positions) >= 2

    def test_compute_diff_convenience(self):
        """Test compute_diff convenience function."""
        result = compute_diff(
            "Original text.",
            "Modified text.",
            keywords=["modified"]
        )
        assert result.original == "Original text."
        assert result.modified == "Modified text."

    def test_get_changes_summary(self):
        """Test changes summary generation."""
        result = compute_diff(
            "The original",
            "The modified version",
            keywords=["modified"]
        )
        summary = get_changes_summary(result)
        assert "total_changes" in summary
        assert "keywords_injected" in summary


class TestFindNewKeywords:
    """Tests for keyword injection detection."""

    def test_find_new_keywords(self):
        """Test finding newly injected keywords."""
        new_kws = find_new_keywords_in_text(
            "Original content about services.",
            "Original content about insurance services.",
            ["insurance", "coverage"]
        )
        assert "insurance" in new_kws
        assert "coverage" not in new_kws  # Not in modified

    def test_no_new_keywords(self):
        """Test when no new keywords are found."""
        new_kws = find_new_keywords_in_text(
            "Insurance coverage options.",
            "Insurance coverage options available.",
            ["insurance", "coverage"]
        )
        assert len(new_kws) == 0  # Both already in original


# =============================================================================
# CHANGE SUMMARY TESTS
# =============================================================================

class TestChangeSummaryBuilder:
    """Tests for change summary building."""

    def test_build_empty_summary(self):
        """Test building summary with no changes."""
        builder = ChangeSummaryBuilder()
        builder.set_document_info("Test Doc")
        summary = builder.build_summary()

        assert summary.document_title == "Test Doc"
        assert summary.total_blocks == 0
        assert summary.modified_blocks == 0

    def test_record_block_change(self, sample_content_blocks):
        """Test recording a block change."""
        builder = ChangeSummaryBuilder()
        builder.set_document_info("Test Doc")

        block = sample_content_blocks[1]  # Paragraph
        change = builder.record_block_change(
            block,
            "Modified paragraph with insurance keywords.",
            keywords_injected=["insurance"]
        )

        assert change.change_type == ChangeType.KEYWORD_INJECTION
        assert "insurance" in change.keywords_injected

    def test_record_blocked_block(self, sample_content_blocks):
        """Test recording a blocked block."""
        builder = ChangeSummaryBuilder()
        builder.set_document_info("Test Doc")

        block = sample_content_blocks[5]  # Table
        change = builder.record_blocked_block(block, "Tables cannot be modified")

        assert change.change_type == ChangeType.BLOCKED
        assert "Tables cannot be modified" in change.warnings

    def test_full_summary_build(self, sample_content_blocks, sample_keyword_plan):
        """Test building complete summary."""
        builder = ChangeSummaryBuilder()
        builder.set_document_info("Insurance Guide", "https://example.com")
        builder.set_keyword_plan(sample_keyword_plan)

        # Record changes
        for block in sample_content_blocks:
            if block.block_type == ContentBlockType.PARAGRAPH:
                builder.record_block_change(
                    block,
                    block.text + " with professional liability insurance.",
                    keywords_injected=["professional liability insurance"]
                )
            elif block.block_type in (ContentBlockType.TABLE, ContentBlockType.CODE):
                builder.record_blocked_block(block, "Protected structure")
            else:
                builder.record_block_change(block, block.text)

        summary = builder.build_summary()

        assert summary.total_blocks == 7
        assert summary.modified_blocks > 0
        assert summary.total_keyword_injections > 0

    def test_format_summary_text(self, sample_content_blocks):
        """Test text formatting of summary."""
        builder = ChangeSummaryBuilder()
        builder.set_document_info("Test Doc")
        builder.record_block_change(
            sample_content_blocks[1],
            "Modified text",
            keywords_injected=["keyword"]
        )

        summary = builder.build_summary()
        text_output = format_summary_text(summary)

        assert "SEO CONTENT OPTIMIZATION SUMMARY" in text_output
        assert "Test Doc" in text_output

    def test_format_summary_dict(self, sample_content_blocks):
        """Test dict formatting of summary."""
        builder = ChangeSummaryBuilder()
        builder.set_document_info("Test Doc", "https://example.com")
        builder.record_block_change(
            sample_content_blocks[1],
            "Modified text",
            keywords_injected=["keyword"]
        )

        summary = builder.build_summary()
        dict_output = format_summary_dict(summary)

        assert dict_output["document"]["title"] == "Test Doc"
        assert "keywords" in dict_output
        assert "blocks" in dict_output


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestV2Integration:
    """Integration tests for V2 architecture."""

    def test_end_to_end_optimization_flow(self, sample_content_document, sample_keyword_plan):
        """Test simulated optimization flow."""
        doc = sample_content_document
        preserver = StructurePreserver()
        checker = FactualityChecker()
        builder = ChangeSummaryBuilder()

        builder.set_document_info(doc.title, doc.source_url)
        builder.set_keyword_plan(sample_keyword_plan)

        # Simulate processing each block
        for block in doc.blocks:
            if not preserver.can_modify(block):
                builder.record_blocked_block(block, f"Protected: {block.block_type.value}")
                continue

            # Simulate modification
            if block.block_type == ContentBlockType.PARAGRAPH:
                modified = block.text + " Professional liability insurance provides protection."
                keywords = ["professional liability insurance"]
            else:
                modified = block.text
                keywords = []

            # Check factuality
            fact_result = checker.compare_claims(block.text, modified)

            builder.record_block_change(
                block,
                modified,
                keywords_injected=keywords,
                factuality_result=fact_result,
            )

        summary = builder.build_summary()

        # Verify summary
        assert summary.total_blocks == len(doc.blocks)
        assert summary.modified_blocks > 0
        assert summary.blocked_blocks > 0  # Tables and code

    def test_factuality_blocks_bad_modification(self, sample_content_blocks):
        """Test that factuality checker catches problematic modifications."""
        checker = FactualityChecker()

        original = sample_content_blocks[1].text  # "This guide covers..."
        # Add fake statistic
        modified = original + " Studies show 95% of clients save money."

        result = checker.compare_claims(original, modified)

        # Should detect new percentage claim
        assert result.is_valid is False
        assert len(result.new_claims) >= 1
