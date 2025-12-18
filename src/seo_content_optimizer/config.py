# -*- coding: utf-8 -*-
"""
Centralized configuration for SEO Content Optimizer.

This module provides a unified configuration dataclass that controls
optimization behavior, including FAQ generation policy, AI add-ons,
and chunking parameters.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


# Type alias for optimization mode
# - "insert_only": Strictest mode. No LLM rewrites, deterministic keyword injection only.
# - "minimal": Insert-only with minimal LLM assistance for natural integration.
# - "enhanced": Full optimization with density targeting and content expansion.
OptimizationMode = Literal["insert_only", "minimal", "enhanced"]


@dataclass
class OptimizationConfig:
    """
    Central configuration for content optimization behavior.

    Attributes:
        optimization_mode: Controls overall optimization behavior:
            - "insert_only": Strictest mode. No LLM rewrites at all. Uses
              deterministic, rule-based keyword injection only. Original text
              is preserved exactly. Best for when you want minimal changes.
            - "minimal": Insert-only with minimal changes. Each keyword appears
              once in an appropriate place. No density targets, no keyword
              distribution, no content expansion. FAQ/AI sections disabled.
              This is the default when manual keywords are provided.
            - "enhanced": Full optimization. Targets keyword density (0.5-1.5%),
              distributes keywords across paragraphs, may expand content.
              FAQ/AI sections enabled by default.

        manual_keywords_only: If True, ONLY the provided keywords can be used.
            No synonyms, LSI phrases, or additional keywords will be added.
            Automatically set True when manual keywords are provided.

        faq_policy: Controls FAQ generation behavior:
            - "auto": Generate FAQ only if archetype recommends (default)
            - "always": Always generate FAQ, warn if archetype is inappropriate
            - "never": Never generate FAQ
        faq_count: Target number of FAQ items to generate.
        faq_min_valid: Minimum valid FAQs required before fallback triggers.
        faq_retry_on_insufficient: Whether to retry LLM if < faq_min_valid.

        generate_ai_sections: Master switch for AI add-ons sections.
        generate_key_takeaways: Whether to generate Key Takeaways bullets.
        generate_chunk_map: Whether to generate Chunk Data table.

        chunk_target_tokens: Target tokens per chunk (default 512 per Microsoft).
        chunk_overlap_tokens: Overlap between chunks (default 128 for continuity).

        max_secondary: Maximum secondary keywords to use.

        keyword_allowlist: Optional explicit list of allowed keywords. When set,
            ONLY these keywords can be inserted - the LLM cannot add variations
            or synonyms. None means no restriction.

        Insertion Budget Controls (anti-stuffing):
            max_new_sentences_total: Maximum new sentences that can be added
                across the entire document. Default 2 in insert_only mode.
            max_new_words_total: Maximum new words that can be added across
                the entire document. Default 40 in insert_only mode.
    """

    # Optimization Mode Control
    optimization_mode: OptimizationMode = "minimal"

    # Manual Keywords Control
    manual_keywords_only: bool = False  # If True, only allowlisted keywords can be used

    # Insertion Budget Controls (anti-stuffing)
    max_new_sentences_total: Optional[int] = None  # None = no limit, set in insert_only
    max_new_words_total: Optional[int] = None  # None = no limit, set in insert_only

    # FAQ Control
    faq_policy: Literal["auto", "always", "never"] = "auto"
    faq_count: int = 4
    faq_min_valid: int = 2
    faq_retry_on_insufficient: bool = True

    # AI Sections Control
    generate_ai_sections: bool = True
    generate_key_takeaways: bool = True
    generate_chunk_map: bool = True

    # Chunk Parameters (based on Microsoft recommendations)
    chunk_target_tokens: int = 512
    chunk_overlap_tokens: int = 128

    # Keyword limits
    max_secondary: int = 5

    # Keyword Allowlist (for hard enforcement)
    keyword_allowlist: Optional[set[str]] = field(default=None)

    # Keyword Caps (for insert-only/minimal mode)
    # These are MAXIMUM occurrences, not targets
    primary_keyword_body_cap: int = 1  # Max primary keyword uses in body
    secondary_keyword_body_cap: int = 1  # Max each secondary keyword in body
    enforce_keyword_caps: bool = True  # Enable cap enforcement in minimal mode

    # Delta Budget Controls (preferred over caps for insert-only mode)
    # These limit NEW additions, not total count. Original content can have any count.
    # Formula: new_additions = final_count - source_count <= allowed_new
    allowed_new_primary: int = 1  # Max NEW additions for primary keyword
    allowed_new_secondary: int = 1  # Max NEW additions for each secondary keyword

    @property
    def is_insert_only_mode(self) -> bool:
        """Check if running in strict insert-only mode (no LLM rewrites)."""
        return self.optimization_mode == "insert_only"

    @property
    def is_minimal_mode(self) -> bool:
        """Check if running in minimal/insert-only mode (includes insert_only)."""
        return self.optimization_mode in ("minimal", "insert_only")

    @property
    def is_enhanced_mode(self) -> bool:
        """Check if running in enhanced/full optimization mode."""
        return self.optimization_mode == "enhanced"

    @property
    def should_use_llm_for_body(self) -> bool:
        """Check if LLM should be used for body content optimization.

        In insert_only mode, NO LLM is used - purely deterministic injection.
        In minimal mode, LLM may be used with strict prompts.
        In enhanced mode, full LLM optimization.
        """
        return not self.is_insert_only_mode

    @property
    def should_target_density(self) -> bool:
        """Check if keyword density targeting is enabled.

        In minimal mode, we only insert each keyword once.
        In enhanced mode, we target 0.5-1.5% density.
        """
        return self.is_enhanced_mode

    @property
    def should_distribute_keywords(self) -> bool:
        """Check if keywords should be distributed across paragraphs.

        In minimal mode, keywords go in the most appropriate single location.
        In enhanced mode, keywords are distributed throughout content.
        """
        return self.is_enhanced_mode

    @property
    def should_expand_content(self) -> bool:
        """Check if content expansion is allowed.

        In minimal mode, no new paragraphs or sections are added.
        In enhanced mode, content may be expanded with new sections.
        """
        return self.is_enhanced_mode

    @property
    def should_lock_headings(self) -> bool:
        """Check if headings (H1, H2, etc.) should be locked from modification.

        In insert_only mode, headings are NEVER modified - they are preserved exactly.
        In minimal mode, minor keyword injection may be allowed.
        In enhanced mode, headings can be optimized via LLM.
        """
        return self.is_insert_only_mode

    @property
    def should_lock_existing_faq(self) -> bool:
        """Check if existing FAQ blocks should be locked from modification.

        In insert_only mode, existing FAQ sections are preserved exactly.
        No new FAQ generation happens if an FAQ already exists.
        In enhanced mode, FAQ may be augmented or regenerated.
        """
        return self.is_insert_only_mode

    @property
    def should_enforce_keyword_caps(self) -> bool:
        """Check if keyword occurrence caps should be enforced.

        In minimal mode with enforce_keyword_caps=True, the optimizer will
        REMOVE excess keyword occurrences to ensure each keyword appears
        at most N times (where N is the cap value).

        This is the key difference from enhanced mode which only enforces
        MINIMUMS (targets), not MAXIMUMS (caps).
        """
        return self.is_minimal_mode and self.enforce_keyword_caps

    @property
    def has_keyword_allowlist(self) -> bool:
        """Check if a keyword allowlist is enforced."""
        return self.keyword_allowlist is not None and len(self.keyword_allowlist) > 0

    def is_keyword_allowed(self, keyword: str) -> bool:
        """Check if a keyword is in the allowlist (if enforced).

        Args:
            keyword: The keyword to check

        Returns:
            True if no allowlist exists, or if keyword is in the allowlist.
            False if allowlist exists and keyword is not in it.
        """
        if not self.has_keyword_allowlist:
            return True
        return keyword.lower() in {k.lower() for k in self.keyword_allowlist}

    @property
    def should_generate_faq(self) -> bool:
        """Check if FAQ generation is enabled at config level.

        In minimal mode, FAQ is disabled unless faq_policy='always'.
        In enhanced mode, FAQ follows the faq_policy setting.
        """
        if self.is_minimal_mode:
            # In minimal mode, only generate FAQ if explicitly forced
            return self.faq_policy == "always"
        return self.faq_policy != "never"

    @property
    def force_faq(self) -> bool:
        """Check if FAQ should be forced regardless of archetype."""
        return self.faq_policy == "always"

    @property
    def should_generate_ai_addons(self) -> bool:
        """Check if any AI add-ons should be generated.

        In minimal mode, AI add-ons are disabled unless explicitly enabled.
        In enhanced mode, follows the generate_ai_sections flag.
        """
        if self.is_minimal_mode:
            # In minimal mode, AI add-ons must be explicitly requested
            # They're only generated if the master switch is on AND
            # we're not using default values (user explicitly enabled them)
            return (
                self.generate_ai_sections and
                (self.generate_key_takeaways or self.generate_chunk_map)
            )
        return (
            self.generate_ai_sections and
            (self.generate_key_takeaways or self.generate_chunk_map)
        )

    def __post_init__(self):
        """Validate configuration values."""
        if self.optimization_mode not in ("insert_only", "minimal", "enhanced"):
            raise ValueError(
                f"optimization_mode must be 'insert_only', 'minimal', or 'enhanced', "
                f"got '{self.optimization_mode}'"
            )
        if self.faq_policy not in ("auto", "always", "never"):
            raise ValueError(
                f"faq_policy must be 'auto', 'always', or 'never', "
                f"got '{self.faq_policy}'"
            )
        if self.faq_count < 1:
            raise ValueError(f"faq_count must be >= 1, got {self.faq_count}")
        if self.faq_min_valid < 1:
            raise ValueError(f"faq_min_valid must be >= 1, got {self.faq_min_valid}")
        if self.chunk_target_tokens < 100:
            raise ValueError(
                f"chunk_target_tokens must be >= 100, got {self.chunk_target_tokens}"
            )
        if self.chunk_overlap_tokens < 0:
            raise ValueError(
                f"chunk_overlap_tokens must be >= 0, got {self.chunk_overlap_tokens}"
            )
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be < "
                f"chunk_target_tokens ({self.chunk_target_tokens})"
            )

    @classmethod
    def minimal(cls, **overrides) -> "OptimizationConfig":
        """Create config with minimal/insert-only mode defaults.

        Minimal mode:
        - Inserts each keyword once in an appropriate place
        - No density targeting
        - No keyword distribution across paragraphs
        - No content expansion
        - FAQ and AI add-ons disabled by default

        Args:
            **overrides: Override any config values (e.g., faq_policy='always')

        Returns:
            OptimizationConfig with minimal mode defaults
        """
        defaults = {
            "optimization_mode": "minimal",
            "faq_policy": "never",
            "generate_ai_sections": False,
            "generate_key_takeaways": False,
            "generate_chunk_map": False,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def insert_only(cls, **overrides) -> "OptimizationConfig":
        """Create config with strict insert-only mode defaults.

        Insert-only mode (strictest):
        - NO LLM rewrites of any kind - purely deterministic
        - Only injects keywords where missing (once each)
        - Preserves original text exactly
        - Strict insertion budgets enforced
        - FAQ and AI add-ons disabled
        - Best for minimal, predictable changes

        Args:
            **overrides: Override any config values

        Returns:
            OptimizationConfig with insert-only mode defaults
        """
        defaults = {
            "optimization_mode": "insert_only",
            "faq_policy": "never",
            "generate_ai_sections": False,
            "generate_key_takeaways": False,
            "generate_chunk_map": False,
            "manual_keywords_only": True,
            "enforce_keyword_caps": True,
            "max_new_sentences_total": 2,
            "max_new_words_total": 40,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def enhanced(cls, **overrides) -> "OptimizationConfig":
        """Create config with enhanced/full optimization mode defaults.

        Enhanced mode:
        - Targets keyword density (0.5-1.5%)
        - Distributes keywords across paragraphs
        - May expand content with new sections
        - FAQ and AI add-ons enabled by default

        Args:
            **overrides: Override any config values

        Returns:
            OptimizationConfig with enhanced mode defaults
        """
        defaults = {
            "optimization_mode": "enhanced",
            "faq_policy": "auto",
            "generate_ai_sections": True,
            "generate_key_takeaways": True,
            "generate_chunk_map": True,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_manual_keywords(
        cls,
        keywords: list[str],
        mode: OptimizationMode = "minimal",
        **overrides,
    ) -> "OptimizationConfig":
        """Create config for manually specified keywords.

        When manual keywords are provided, defaults to minimal mode and
        enforces a keyword allowlist.

        Args:
            keywords: List of manually specified keywords
            mode: Optimization mode (defaults to 'minimal' for manual keywords)
            **overrides: Override any config values

        Returns:
            OptimizationConfig with keyword allowlist enforced
        """
        # Build allowlist from provided keywords
        allowlist = {k.strip().lower() for k in keywords if k.strip()}

        if mode == "minimal":
            return cls.minimal(keyword_allowlist=allowlist, **overrides)
        else:
            return cls.enhanced(keyword_allowlist=allowlist, **overrides)
