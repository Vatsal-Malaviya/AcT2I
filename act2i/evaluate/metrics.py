"""Classification metrics for evaluating CLIP / DINO action prediction.

Computes top-k accuracy, precision, recall, and F1 from raw score
matrices (one score per action class per image).
"""

import logging
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.functional import softmax

logger = logging.getLogger(__name__)


def compute_topk_accuracy(
    probs: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Compute top-k accuracy.

    Parameters
    ----------
    probs : ndarray, shape (N, C)
        Probability/score matrix.
    labels : ndarray, shape (N,)
        True class indices.
    k : int
        Number of top predictions to consider.

    Returns
    -------
    float
    """
    topk = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(1 for preds, true in zip(topk, labels) if true in preds)
    return correct / len(labels)


def evaluate_predictions(
    scores: np.ndarray,
    true_labels: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Evaluate a score matrix against ground-truth labels.

    Parameters
    ----------
    scores : ndarray, shape (N, C)
        Raw score matrix (before softmax).
    true_labels : ndarray, shape (N,)
        Integer class indices.
    class_names : sequence of str, optional
        Human-readable class names (for logging only).

    Returns
    -------
    dict
        Keys: accuracy, top2–top5 accuracy, precision, recall, f1.
    """
    probs = softmax(torch.tensor(scores), dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(true_labels, preds)),
    }
    for k in range(2, 6):
        metrics[f"top{k}_accuracy"] = compute_topk_accuracy(probs, true_labels, k)

    prec, rec, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro", zero_division=0
    )
    metrics["precision"] = float(prec)
    metrics["recall"] = float(rec)
    metrics["f1"] = float(f1)

    logger.info(
        "Accuracy=%.4f  F1=%.4f  (N=%d, C=%d)",
        metrics["accuracy"],
        metrics["f1"],
        len(true_labels),
        scores.shape[1],
    )
    return metrics
