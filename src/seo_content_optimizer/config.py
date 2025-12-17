# -*- coding: utf-8 -*-
"""
Centralized configuration for SEO Content Optimizer.

This module provides a unified configuration dataclass that controls
optimization behavior, including FAQ generation policy, AI add-ons,
and chunking parameters.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class OptimizationConfig:
    """
    Central configuration for content optimization behavior.

    Attributes:
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
    """

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

    @property
    def should_generate_faq(self) -> bool:
        """Check if FAQ generation is enabled at config level."""
        return self.faq_policy != "never"

    @property
    def force_faq(self) -> bool:
        """Check if FAQ should be forced regardless of archetype."""
        return self.faq_policy == "always"

    @property
    def should_generate_ai_addons(self) -> bool:
        """Check if any AI add-ons should be generated."""
        return (
            self.generate_ai_sections and
            (self.generate_key_takeaways or self.generate_chunk_map)
        )

    def __post_init__(self):
        """Validate configuration values."""
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
