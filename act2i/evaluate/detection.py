"""OWLv2-based zero-shot object detection for atom verification.

Checks whether the two animals described in a prompt are actually
present in a generated image.
"""

import logging
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Sequence

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of detecting a single animal in an image."""

    found: bool = False
    score: float = 0.0
    box: List[float] = field(default_factory=lambda: [0, 0, 0, 0])


class ObjectDetector:
    """Zero-shot object detector using OWLv2.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    threshold : float
        Confidence threshold for detection (default: 0.1).
    device : str
        Target device (default: "cuda").
    """

    def __init__(
        self,
        model_id: str = "google/owlv2-large-patch14-ensemble",
        threshold: float = 0.1,
        device: str = "cuda",
    ):
        self.threshold = threshold
        self.device = device

        cache_dir = os.environ.get("MODEL_CACHE_DIR")

        logger.info("Loading OWLv2 model: %s …", model_id)
        from transformers import Owlv2ForObjectDetection
        from transformers import Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.model.to(device)
        logger.info("OWLv2 ready on %s.", device)

    def detect(
        self,
        image: Image.Image,
        labels: Sequence[str],
    ) -> List[DetectionResult]:
        """Run zero-shot detection for *labels* on *image*.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image (RGB).
        labels : sequence of str
            Text queries (e.g. ``["Fox", "Rabbit"]``).

        Returns
        -------
        list of DetectionResult
            One result per label, in the same order.
        """
        inputs = self.processor(
            text=list(labels),
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        processed = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
        )

        results: List[DetectionResult] = []
        det = processed[0]
        det_labels = det["labels"].tolist()

        for i in range(len(labels)):
            if i in det_labels:
                idx = det_labels.index(i)
                results.append(
                    DetectionResult(
                        found=True,
                        score=float(det["scores"][idx]),
                        box=det["boxes"][idx].cpu().tolist(),
                    )
                )
            else:
                results.append(DetectionResult())

        return results

    def detect_animals(
        self,
        image_path: str,
        animal1: str,
        animal2: str,
    ) -> Dict[str, DetectionResult]:
        """Convenience: detect two animals in an image file.

        Returns
        -------
        dict
            ``{"animal1": DetectionResult, "animal2": DetectionResult}``
        """
        image = Image.open(image_path).convert("RGB")
        results = self.detect(image, [animal1, animal2])
        return {"animal1": results[0], "animal2": results[1]}
