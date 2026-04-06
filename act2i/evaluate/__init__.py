"""Evaluation modules for AcT2I."""

from act2i.evaluate.detection import ObjectDetector
from act2i.evaluate.clip_score import CLIPScorer
from act2i.evaluate.dino_score import DINOScorer
from act2i.evaluate.metrics import compute_topk_accuracy
from act2i.evaluate.metrics import evaluate_predictions

__all__ = [
    "ObjectDetector",
    "CLIPScorer",
    "DINOScorer",
    "compute_topk_accuracy",
    "evaluate_predictions",
]
