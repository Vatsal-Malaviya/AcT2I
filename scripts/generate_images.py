#!/usr/bin/env python
"""CLI: Generate images from prompts using a T2I model.

Usage
-----
    python scripts/generate_images.py \
        --prompts data/prompt/phrases_reviewed_final.json \
        --output-dir data/images \
        --model stabilityai/stable-diffusion-3.5-large \
        --seeds 42 43 44 45
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
        description="Generate images from enhanced prompts."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="JSON file with prompt entries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/images"),
    )
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-3.5-large",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Subfolder name (default: derived from model ID).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45],
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        default=["phrase", "emotional", "spatial", "temporal"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N prompts.",
    )
    args = parser.parse_args()

    with open(args.prompts) as f:
        data = json.load(f)

    if args.limit:
        data = data[: args.limit]

    logger.info(
        "Generating images for %d prompts × %d seeds × %d types",
        len(data),
        len(args.seeds),
        len(args.prompt_types),
    )

    from act2i.generate import ImageGenerator

    gen = ImageGenerator(model_id=args.model)
    saved = gen.generate_batch(
        prompts=data,
        seeds=args.seeds,
        output_dir=args.output_dir,
        model_tag=args.model_tag,
        prompt_types=args.prompt_types,
    )
    logger.info("Saved %d images to %s", len(saved), args.output_dir)


if __name__ == "__main__":
    main()
