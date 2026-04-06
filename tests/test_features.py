"""Tests for act2i.features module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch


class TestFeatureExtractorInterface:
    """Test FeatureExtractor without loading real models."""

    @staticmethod
    def _make_batch_feature(data: dict):
        """Create a mock that behaves like BatchFeature (.to returns self)."""
        m = MagicMock()
        m.__getitem__ = lambda self, k: data[k]
        m.__contains__ = lambda self, k: k in data
        m.to.return_value = m
        # Allow **inputs unpacking in model call
        m.keys.return_value = data.keys()
        for k, v in data.items():
            setattr(m, k, v)
        return m

    def _make_extractor(self):
        from act2i.features.extract import FeatureExtractor

        ext = FeatureExtractor.__new__(FeatureExtractor)
        ext.model_name = "facebook/dinov2-base"
        ext.device = "cpu"
        ext.batch_size = 2
        ext.num_workers = 0

        ext.processor = MagicMock()
        ext.model = MagicMock()
        return ext

    def test_forward_dinov2_mean_pools(self):
        ext = self._make_extractor()

        # Simulate model output: last_hidden_state (B, patches, D)
        fake_output = MagicMock()
        fake_output.last_hidden_state = torch.randn(2, 197, 768)
        ext.model.return_value = fake_output
        ext.processor.return_value = self._make_batch_feature(
            {"pixel_values": torch.randn(2, 3, 224, 224)}
        )

        result = ext._forward([MagicMock(), MagicMock()])
        assert result.shape == (2, 768)

    def test_forward_siglip_returns_embeds(self):
        ext = self._make_extractor()
        ext.model_name = "google/siglip-so400m-patch14-384"

        fake_output = MagicMock()
        fake_output.image_embeds = torch.randn(2, 1152)
        ext.model.return_value = fake_output
        ext.processor.return_value = self._make_batch_feature(
            {"pixel_values": torch.randn(2, 3, 384, 384)}
        )

        result = ext._forward(
            [MagicMock(), MagicMock()],
            text_context="test action",
        )
        assert result.shape == (2, 1152)

    def test_extract_and_save_creates_file(self):
        ext = self._make_extractor()

        fake_output = MagicMock()
        fake_output.last_hidden_state = torch.randn(1, 197, 768)
        ext.model.return_value = fake_output
        ext.processor.return_value = self._make_batch_feature(
            {"pixel_values": torch.randn(1, 3, 224, 224)}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy image file for the dataset
            from PIL import Image

            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (64, 64)).save(img_path)

            out_path = Path(tmpdir) / "feats.pt"
            result = ext.extract_and_save([img_path], out_path)
            assert result == out_path
            assert out_path.exists()

            loaded = torch.load(out_path, weights_only=True)
            assert loaded.shape[1] == 768


class TestImageDataset:
    """Test the internal _ImageDataset."""

    def test_loads_rgb(self):
        from PIL import Image

        from act2i.features.extract import _ImageDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "img.png"
            Image.new("L", (32, 32)).save(path)  # grayscale

            ds = _ImageDataset([path])
            assert len(ds) == 1
            img = ds[0]
            assert img.mode == "RGB"
