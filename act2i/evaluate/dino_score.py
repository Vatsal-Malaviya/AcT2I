"""DINOv2-based visual similarity scoring.

Computes cosine similarity between generated image features and
pre-computed reference action features from the Animal Kingdom dataset.
"""

import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
from typing import Sequence
from typing import Union

import torch
from torch.nn import CosineSimilarity
from torchvision.io import read_image

logger = logging.getLogger(__name__)


def _batch_iter(iterable, batch_size: int):
    """Yield successive chunks of *batch_size* from *iterable*."""
    it = iter(iterable)
    for first in it:
        yield list(itertools.chain([first], itertools.islice(it, batch_size - 1)))


class DINOScorer:
    """Compute DINOv2 cosine similarity against reference features.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (default: facebook/dinov2-base).
    device : str
        Target device.
    batch_size : int
        Number of images to process per forward pass.
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        device: str = "cuda",
        batch_size: int = 400,
    ):
        self.device = device
        self.batch_size = batch_size

        cache_dir = os.environ.get("MODEL_CACHE_DIR")

        logger.info("Loading DINOv2: %s …", model_id)
        from transformers import AutoImageProcessor
        from transformers import AutoModel

        self.processor = AutoImageProcessor.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
        self.model.to(device)
        self.model.eval()
        self.cos = CosineSimilarity(dim=1)
        logger.info("DINOv2 ready on %s.", device)

    def extract_features(
        self,
        image_paths: Sequence[Union[str, Path]],
    ) -> torch.Tensor:
        """Extract mean-pooled DINOv2 features for a batch of images.

        Parameters
        ----------
        image_paths : sequence of path-like
            Paths to image files.

        Returns
        -------
        torch.Tensor
            Shape ``(N, D)`` feature matrix.
        """
        all_feats: List[torch.Tensor] = []
        for batch_paths in _batch_iter(image_paths, self.batch_size):
            workers = min(self.batch_size, max(1, os.cpu_count() // 2))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                images = list(ex.map(read_image, [str(p) for p in batch_paths]))

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model(**inputs).last_hidden_state
                feats = feats.mean(dim=1)  # mean-pool patches
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)

    def score_against_reference(
        self,
        image_paths: Sequence[Union[str, Path]],
        reference_features: torch.Tensor,
    ) -> List[float]:
        """Compute cosine similarity of images vs a reference vector.

        Parameters
        ----------
        image_paths : sequence of path-like
            Generated image paths.
        reference_features : torch.Tensor
            Shape ``(1, D)`` or ``(D,)`` reference feature vector.

        Returns
        -------
        list of float
            Cosine similarity per image.
        """
        ref = reference_features.to(self.device)
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)

        sims: List[float] = []
        for batch_paths in _batch_iter(image_paths, self.batch_size):
            workers = min(self.batch_size, max(1, os.cpu_count() // 2))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                images = list(ex.map(read_image, [str(p) for p in batch_paths]))
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model(**inputs).last_hidden_state.mean(dim=1)
                batch_sims = self.cos(ref, feats)
            sims.extend(batch_sims.detach().cpu().tolist())
        return sims
