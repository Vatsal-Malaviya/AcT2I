"""CLIPScore computation for text-image alignment evaluation."""

import logging
from pathlib import Path
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import torchvision.io as io

logger = logging.getLogger(__name__)


class CLIPScorer:
    """Compute CLIPScore between images and text prompts.

    Parameters
    ----------
    model_name : str
        CLIP model identifier (default: openai/clip-vit-base-patch16).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
    ):
        logger.info("Loading CLIPScore model: %s …", model_name)
        from torchmetrics.multimodal.clip_score import CLIPScore

        self.metric = CLIPScore(model_name_or_path=model_name)
        logger.info("CLIPScore ready.")

    def score(
        self,
        image_path: Union[str, Path],
        text: str,
    ) -> float:
        """Compute CLIPScore for one image-text pair.

        Parameters
        ----------
        image_path : path-like
            Path to a PNG/JPEG image.
        text : str
            The text prompt to compare against.

        Returns
        -------
        float
            The CLIPScore value.
        """
        img = io.read_image(str(image_path))
        score = self.metric(img, text)
        return float(score.detach().cpu())

    def score_batch(
        self,
        pairs: Sequence[Tuple[Union[str, Path], str]],
    ) -> List[float]:
        """Compute CLIPScore for multiple (image_path, text) pairs.

        Parameters
        ----------
        pairs : sequence of (path, text)
            Each element is ``(image_path, prompt_text)``.

        Returns
        -------
        list of float
        """
        scores: List[float] = []
        for image_path, text in pairs:
            scores.append(self.score(image_path, text))
        return scores
