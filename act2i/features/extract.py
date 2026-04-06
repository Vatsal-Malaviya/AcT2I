"""Image feature extraction using DINOv2 or SigLIP.

Extracts per-image (or per-video) feature vectors from image datasets
such as Animal Kingdom, saving them as ``.pt`` files.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class _ImageDataset(Dataset):
    """Simple dataset that loads images as PIL RGB."""

    def __init__(self, image_paths: Sequence[Union[str, Path]]):
        self.image_paths = [Path(p) for p in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            return img.convert("RGB")


def _pil_collate(batch):
    return batch


class FeatureExtractor:
    """Extract image features using DINOv2 or SigLIP.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        Target device (default: "cuda").
    batch_size : int
        Batch size for feature extraction.
    num_workers : int
        DataLoader workers for image loading.
    """

    SUPPORTED_MODELS = {
        "facebook/dinov2-base",
        "google/siglip-so400m-patch14-384",
    }

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        cache_dir = os.environ.get("MODEL_CACHE_DIR")

        logger.info("Loading feature extractor: %s …", model_name)
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model.to(device)
        self.model.eval()
        logger.info("Feature extractor ready on %s.", device)

    def extract(
        self,
        image_paths: Sequence[Union[str, Path]],
        text_context: Optional[str] = None,
    ) -> torch.Tensor:
        """Extract features for a list of images.

        Parameters
        ----------
        image_paths : sequence of path-like
            Paths to image files.
        text_context : str, optional
            Text context for SigLIP (ignored for DINOv2).

        Returns
        -------
        torch.Tensor
            Shape ``(N, D)`` feature matrix.
        """
        dataset = _ImageDataset(image_paths)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=_pil_collate,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        all_feats: List[torch.Tensor] = []

        for images in loader:
            images = list(images)
            with torch.no_grad():
                feats = self._forward(images, text_context)
            all_feats.append(feats.cpu())
            del images
            torch.cuda.empty_cache()

        return torch.cat(all_feats, dim=0)

    def extract_and_save(
        self,
        image_paths: Sequence[Union[str, Path]],
        output_path: Union[str, Path],
        text_context: Optional[str] = None,
    ) -> Path:
        """Extract features and save as a ``.pt`` file.

        Returns the output path.
        """
        feats = self.extract(image_paths, text_context)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats, out)
        logger.info("Saved features (%s) → %s", feats.shape, out)
        return out

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _forward(
        self,
        images: list,
        text_context: Optional[str] = None,
    ) -> torch.Tensor:
        """Run model forward pass, returning pooled features."""
        is_dinov2 = "dinov2" in self.model_name.lower()

        if is_dinov2:
            inputs = self.processor(
                images=images, return_tensors="pt"
            ).to(self.device)
            out = self.model(**inputs).last_hidden_state
            return out.mean(dim=1)  # mean-pool patch tokens
        else:
            # SigLIP path
            text = text_context or ""
            inputs = self.processor(
                text=[text] * len(images),
                images=images,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            return self.model(**inputs).image_embeds
