"""Tests for act2i.evaluate module."""

import numpy as np

from act2i.evaluate.metrics import compute_topk_accuracy
from act2i.evaluate.metrics import evaluate_predictions


class TestTopKAccuracy:
    """Test the top-k accuracy computation."""

    def test_top1_perfect(self):
        probs = np.array([[0.1, 0.9], [0.8, 0.2]])
        labels = np.array([1, 0])
        assert compute_topk_accuracy(probs, labels, 1) == 1.0

    def test_top1_half(self):
        probs = np.array([[0.1, 0.9], [0.3, 0.7]])
        labels = np.array([1, 0])  # second is wrong
        assert compute_topk_accuracy(probs, labels, 1) == 0.5

    def test_top2_always_correct(self):
        probs = np.array([[0.1, 0.9], [0.3, 0.7]])
        labels = np.array([1, 0])
        # With k=2 and 2 classes, everything is in top-2
        assert compute_topk_accuracy(probs, labels, 2) == 1.0

    def test_top1_all_wrong(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        labels = np.array([1, 0])
        assert compute_topk_accuracy(probs, labels, 1) == 0.0


class TestEvaluatePredictions:
    """Test the full evaluation pipeline."""

    def test_perfect_predictions(self):
        # Scores that clearly separate classes
        scores = np.array(
            [
                [10.0, -10.0, -10.0],
                [-10.0, 10.0, -10.0],
                [-10.0, -10.0, 10.0],
            ]
        )
        labels = np.array([0, 1, 2])

        metrics = evaluate_predictions(scores, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_random_predictions_below_perfect(self):
        rng = np.random.RandomState(42)
        scores = rng.randn(100, 5)
        labels = rng.randint(0, 5, 100)

        metrics = evaluate_predictions(scores, labels)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        # Random should be around 20% for 5 classes
        assert metrics["accuracy"] < 0.5

    def test_topk_increases_with_k(self):
        rng = np.random.RandomState(42)
        scores = rng.randn(100, 10)
        labels = rng.randint(0, 10, 100)

        metrics = evaluate_predictions(scores, labels)
        assert metrics["accuracy"] <= metrics["top2_accuracy"]
        assert metrics["top2_accuracy"] <= metrics["top3_accuracy"]
        assert metrics["top3_accuracy"] <= metrics["top4_accuracy"]
        assert metrics["top4_accuracy"] <= metrics["top5_accuracy"]

    def test_class_names_optional(self):
        scores = np.array([[10.0, -10.0], [-10.0, 10.0]])
        labels = np.array([0, 1])
        metrics = evaluate_predictions(scores, labels, class_names=["cat", "dog"])
        assert metrics["accuracy"] == 1.0


class TestDetectionResult:
    """Test the DetectionResult dataclass."""

    def test_default_values(self):
        from act2i.evaluate.detection import DetectionResult

        r = DetectionResult()
        assert r.found is False
        assert r.score == 0.0
        assert r.box == [0, 0, 0, 0]

    def test_custom_values(self):
        from act2i.evaluate.detection import DetectionResult

        r = DetectionResult(found=True, score=0.95, box=[10, 20, 100, 200])
        assert r.found is True
        assert r.score == 0.95
        assert r.box == [10, 20, 100, 200]
