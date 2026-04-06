"""Structural NLP analysis of enhanced prompts.

Measures lexical diversity, POS-tag ratios, syntactic complexity,
and dependency-pattern distributions across prompt categories.
"""

import logging
from collections import Counter
from typing import Dict
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class PromptAnalyzer:
    """Compute structural metrics for a set of prompts.

    Parameters
    ----------
    prompts : list of str
        The prompts to analyze.
    spacy_model : str
        spaCy model name (default: ``en_core_web_sm``).
    """

    def __init__(
        self,
        prompts: List[str],
        spacy_model: str = "en_core_web_sm",
    ):
        import spacy

        nlp = spacy.load(spacy_model)
        self.prompts = prompts
        self.docs = [nlp(p) for p in prompts]

    def structural_metrics(self) -> Dict[str, float]:
        """Compute structural metrics for the prompt set.

        Returns
        -------
        dict
            Keys include:

            - ``lexical_diversity`` – unique / total words
            - ``avg_length`` – mean token count (excl. punct)
            - ``syntactic_complexity`` – mean dep count per prompt
            - ``pos_ratio_{tag}`` – proportion of each POS tag
            - ``dep_ratio_{dep}`` – proportion of each dep label
        """
        metrics: Dict[str, float] = {}

        # Lexical diversity
        all_tokens = [
            t.text.lower() for doc in self.docs for t in doc if not t.is_punct
        ]
        metrics["lexical_diversity"] = (
            len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
        )

        # Average prompt length
        lengths = [len([t for t in doc if not t.is_punct]) for doc in self.docs]
        metrics["avg_length"] = float(np.mean(lengths))

        # POS tag distribution
        pos_tags = [t.pos_ for doc in self.docs for t in doc]
        pos_counts = Counter(pos_tags)
        total_tags = len(pos_tags)
        for pos, count in pos_counts.items():
            metrics[f"pos_ratio_{pos.lower()}"] = count / total_tags

        # Syntactic complexity
        dep_lengths = [len([t for t in doc if t.dep_ != "punct"]) for doc in self.docs]
        metrics["syntactic_complexity"] = float(np.mean(dep_lengths))

        # Dependency pattern distribution
        dep_patterns = [t.dep_ for doc in self.docs for t in doc if t.dep_ != "punct"]
        dep_counts = Counter(dep_patterns)
        total_deps = len(dep_patterns)
        for dep, count in dep_counts.items():
            metrics[f"dep_ratio_{dep.lower()}"] = count / total_deps

        return metrics

    @staticmethod
    def compare_categories(
        category_prompts: Dict[str, List[str]],
        spacy_model: str = "en_core_web_sm",
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics per category for side-by-side comparison.

        Parameters
        ----------
        category_prompts : dict
            ``{category_name: [prompt, ...]}``
        spacy_model : str
            spaCy model name.

        Returns
        -------
        dict
            ``{category_name: {metric: value, ...}}``
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, prompts in category_prompts.items():
            analyzer = PromptAnalyzer(prompts, spacy_model)
            results[name] = analyzer.structural_metrics()
            logger.info(
                "Category '%s': %d prompts, diversity=%.3f",
                name,
                len(prompts),
                results[name]["lexical_diversity"],
            )
        return results
