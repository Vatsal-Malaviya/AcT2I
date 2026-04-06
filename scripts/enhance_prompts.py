#!/usr/bin/env python
"""CLI: Enhance baseline prompts using an instruction-tuned LLM.

Usage
-----
    python scripts/enhance_prompts.py \
        --prompts data/prompt/phrases_reviewed_final.json \
        --output data/prompt/enhanced.json \
        --dimensions emotional spatial temporal \
        --model meta-llama/Llama-3.3-70B-Instruct
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Enhance T2I prompts via knowledge distillation."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="JSON file with prompt entries (list of dicts with 'phrase').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompt/enhanced.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["emotional", "spatial", "temporal"],
        help="Enhancement dimensions.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="HuggingFace model ID.",
    )
    args = parser.parse_args()

    with open(args.prompts) as f:
        data = json.load(f)

    baseline_prompts = [entry["phrase"] for entry in data]
    logger.info("Loaded %d prompts from %s", len(baseline_prompts), args.prompts)

    from act2i.prompt import PromptEnhancer

    enhancer = PromptEnhancer(model_id=args.model)
    results = enhancer.enhance_batch(
        prompts=baseline_prompts,
        dimensions=args.dimensions,
        output_path=args.output,
    )

    logger.info("Done. Results saved to %s", args.output)
    for dim, texts in results.items():
        logger.info("  %s: %d prompts", dim, len(texts))


if __name__ == "__main__":
    main()
