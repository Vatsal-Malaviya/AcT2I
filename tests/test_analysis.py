"""Tests for act2i.analysis module."""

import pytest

spacy = pytest.importorskip("spacy", reason="spacy not installed")


class TestPromptAnalyzer:
    """Test PromptAnalyzer with real spaCy processing."""

    @pytest.fixture
    def analyzer(self):
        from act2i.analysis.prompt_analysis import PromptAnalyzer

        prompts = [
            "a playful puppy with a wagging tail chases a kitten",
            "a fierce lion attacks a zebra on the savanna",
            "a small bird gently lands atop a towering elephant",
        ]
        return PromptAnalyzer(prompts, spacy_model="en_core_web_sm")

    def test_structural_metrics_keys(self, analyzer):
        metrics = analyzer.structural_metrics()
        assert "lexical_diversity" in metrics
        assert "avg_length" in metrics
        assert "syntactic_complexity" in metrics
        # Should have at least some POS ratios
        pos_keys = [k for k in metrics if k.startswith("pos_ratio_")]
        assert len(pos_keys) > 0
        # Should have dependency ratios
        dep_keys = [k for k in metrics if k.startswith("dep_ratio_")]
        assert len(dep_keys) > 0

    def test_lexical_diversity_range(self, analyzer):
        metrics = analyzer.structural_metrics()
        assert 0.0 < metrics["lexical_diversity"] <= 1.0

    def test_avg_length_positive(self, analyzer):
        metrics = analyzer.structural_metrics()
        assert metrics["avg_length"] > 0

    def test_pos_ratios_sum_to_one(self, analyzer):
        metrics = analyzer.structural_metrics()
        pos_ratios = [v for k, v in metrics.items() if k.startswith("pos_ratio_")]
        assert abs(sum(pos_ratios) - 1.0) < 1e-6

    def test_dep_ratios_sum_to_one(self, analyzer):
        metrics = analyzer.structural_metrics()
        dep_ratios = [v for k, v in metrics.items() if k.startswith("dep_ratio_")]
        assert abs(sum(dep_ratios) - 1.0) < 1e-6


class TestCompareCategories:
    """Test the static compare_categories method."""

    def test_returns_dict_per_category(self):
        from act2i.analysis.prompt_analysis import PromptAnalyzer

        cats = {
            "spatial": ["a bird lands on an elephant's back"],
            "temporal": ["a cheetah mid-stride pursues a gazelle"],
        }
        results = PromptAnalyzer.compare_categories(cats)
        assert "spatial" in results
        assert "temporal" in results
        assert "lexical_diversity" in results["spatial"]
        assert "lexical_diversity" in results["temporal"]

    def test_empty_prompts_returns_zero_diversity(self):
        from act2i.analysis.prompt_analysis import PromptAnalyzer

        # Edge case: prompts with only punctuation
        cats = {"empty": ["...", "!!!"]}
        results = PromptAnalyzer.compare_categories(cats)
        assert results["empty"]["lexical_diversity"] == 0.0
