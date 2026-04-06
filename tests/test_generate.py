"""Tests for act2i.generate module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock


class TestImageGeneratorInterface:
    """Test ImageGenerator logic without loading real T2I models."""

    def _make_generator(self):
        """Create a mocked ImageGenerator."""
        from act2i.generate.t2i import ImageGenerator

        gen = ImageGenerator.__new__(ImageGenerator)
        gen.model_id = "test/model"
        gen.device = "cpu"
        gen.num_inference_steps = 1
        gen.guidance_scale = 1.0

        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_pipe.return_value.images = [mock_image]
        gen.pipe = mock_pipe
        return gen, mock_image

    def test_generate_returns_image(self):
        gen, mock_image = self._make_generator()
        result = gen.generate("a Fox chasing a Rabbit", seed=42)
        assert result is mock_image
        gen.pipe.assert_called_once()

    def test_generate_batch_creates_files(self):
        gen, mock_image = self._make_generator()

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts = [
                {"id": "1", "phrase": "test prompt"},
            ]
            saved = gen.generate_batch(
                prompts=prompts,
                seeds=[42],
                output_dir=tmpdir,
                model_tag="testmodel",
                prompt_types=["phrase"],
            )
            assert len(saved) == 1
            assert saved[0].exists() or mock_image.save.called

    def test_generate_batch_skips_existing(self):
        gen, mock_image = self._make_generator()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create the file
            out = Path(tmpdir) / "testmodel" / "phrase"
            out.mkdir(parents=True)
            (out / "1_0.png").touch()

            prompts = [{"id": "1", "phrase": "test"}]
            saved = gen.generate_batch(
                prompts=prompts,
                seeds=[42],
                output_dir=tmpdir,
                model_tag="testmodel",
                prompt_types=["phrase"],
            )
            # Should skip, not generate
            assert len(saved) == 0
            gen.pipe.assert_not_called()

    def test_generate_batch_skips_missing_key(self):
        gen, mock_image = self._make_generator()

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts = [{"id": "1"}]  # no "phrase" key
            saved = gen.generate_batch(
                prompts=prompts,
                seeds=[42],
                output_dir=tmpdir,
                model_tag="testmodel",
                prompt_types=["phrase"],
            )
            assert len(saved) == 0
