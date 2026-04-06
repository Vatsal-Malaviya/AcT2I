"""Tests for act2i.prompt module."""

from unittest.mock import MagicMock

import pytest

from act2i.prompt.templates import DIMENSIONS
from act2i.prompt.templates import SYSTEM_PROMPTS


class TestTemplates:
    """Verify system prompt templates are well-formed."""

    def test_dimensions_tuple(self):
        assert isinstance(DIMENSIONS, tuple)
        assert set(DIMENSIONS) == {"emotional", "spatial", "temporal"}

    def test_all_dimensions_have_prompts(self):
        for dim in DIMENSIONS:
            assert dim in SYSTEM_PROMPTS
            assert isinstance(SYSTEM_PROMPTS[dim], str)
            assert len(SYSTEM_PROMPTS[dim]) > 100

    def test_prompts_contain_key_instructions(self):
        assert "Facial Expressions" in SYSTEM_PROMPTS["emotional"]
        assert "Body Language" in SYSTEM_PROMPTS["emotional"]
        assert "Positional Accuracy" in SYSTEM_PROMPTS["spatial"]
        assert "Depth and Perspective" in SYSTEM_PROMPTS["spatial"]
        assert "Freeze-Frame" in SYSTEM_PROMPTS["temporal"]
        assert "Motion Representation" in SYSTEM_PROMPTS["temporal"]

    def test_prompts_request_concise_output(self):
        for dim in DIMENSIONS:
            assert "50-70 tokens" in SYSTEM_PROMPTS[dim]
            assert "Output only the final prompt" in SYSTEM_PROMPTS[dim]


class TestPromptEnhancerInterface:
    """Test PromptEnhancer without loading real models."""

    def test_enhance_rejects_unknown_dimension(self):
        """Ensure bad dimension raises ValueError before model call."""
        from act2i.prompt.enhance import PromptEnhancer

        enhancer = PromptEnhancer.__new__(PromptEnhancer)
        enhancer.pipeline = MagicMock()
        enhancer.max_new_tokens = 256
        enhancer.temperature = 0.7
        enhancer.top_p = 0.9

        with pytest.raises(ValueError, match="Unknown dimension"):
            enhancer.enhance("a Fox chasing a Rabbit", "invalid_dim")

    def test_enhance_accepts_custom_system_prompt(self):
        """Custom system_prompt should bypass dimension check."""
        from act2i.prompt.enhance import PromptEnhancer

        enhancer = PromptEnhancer.__new__(PromptEnhancer)
        enhancer.pipeline = MagicMock()
        enhancer.pipeline.return_value = [
            {"generated_text": [{"content": "enhanced output"}]}
        ]
        enhancer.max_new_tokens = 256
        enhancer.temperature = 0.7
        enhancer.top_p = 0.9

        result = enhancer.enhance(
            "a Fox chasing a Rabbit",
            "custom",
            system_prompt="You are a custom enhancer.",
        )
        assert result == "enhanced output"
        enhancer.pipeline.assert_called_once()

    def test_enhance_batch_returns_dict(self):
        """enhance_batch should return {dim: [str, ...]}."""
        from act2i.prompt.enhance import PromptEnhancer

        enhancer = PromptEnhancer.__new__(PromptEnhancer)
        enhancer.pipeline = MagicMock()
        enhancer.pipeline.return_value = [{"generated_text": [{"content": "enhanced"}]}]
        enhancer.max_new_tokens = 256
        enhancer.temperature = 0.7
        enhancer.top_p = 0.9

        results = enhancer.enhance_batch(
            prompts=["prompt1", "prompt2"],
            dimensions=["emotional"],
        )
        assert "emotional" in results
        assert len(results["emotional"]) == 2
        assert all(r == "enhanced" for r in results["emotional"])
