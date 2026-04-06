#!/usr/bin/env python
"""CLI: Run OWLv2 object detection on generated images.

Usage
-----
    python scripts/run_detection.py \
        --prompts data/prompt/phrases_reviewed_final.json \
        --images-dir data/images \
        --output data/prompt/detection_results.csv
"""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Detect animals in generated images via OWLv2."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompt/detection_results.csv"),
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, default=4)
    args = parser.parse_args()

    with open(args.prompts) as f:
        phrases = {x["id"]: x for x in json.load(f)}

    from act2i.evaluate import ObjectDetector

    detector = ObjectDetector(threshold=args.threshold)
    results_list = []

    for idx, meta in phrases.items():
        for model_dir in sorted(os.listdir(args.images_dir)):
            model_path = args.images_dir / model_dir
            if not model_path.is_dir():
                continue
            for prompt_type in sorted(os.listdir(model_path)):
                prompt_path = model_path / prompt_type
                if not prompt_path.is_dir():
                    continue
                for seed in range(args.seeds):
                    img_path = prompt_path / f"{idx}_{seed}.png"
                    if not img_path.exists():
                        continue
                    try:
                        det = detector.detect_animals(
                            str(img_path),
                            meta["animal1"],
                            meta["animal2"],
                        )
                        results_list.append(
                            {
                                "animal1": meta["animal1"],
                                "animal2": meta["animal2"],
                                "action": meta["action"],
                                "model": model_dir,
                                "prompt_type": prompt_type,
                                "seed": seed,
                                "index": idx,
                                "animal1_found": det["animal1"].found,
                                "animal1_score": det["animal1"].score,
                                "animal1_box": det["animal1"].box,
                                "animal2_found": det["animal2"].found,
                                "animal2_score": det["animal2"].score,
                                "animal2_box": det["animal2"].box,
                            }
                        )
                    except Exception as e:
                        logger.error("Error %s: %s", img_path, e)

    df = pd.DataFrame(results_list)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved %d results to %s", len(df), args.output)


if __name__ == "__main__":
    main()
