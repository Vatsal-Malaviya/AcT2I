#!/usr/bin/env python
"""CLI: Extract image features using DINOv2 or SigLIP.

Usage
-----
    python scripts/extract_features.py \
        --annotation data/train.pkl \
        --image-dir /path/to/images \
        --output-dir data/features \
        --model facebook/dinov2-base
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract image features per video/group."
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        required=True,
        help="Pickle file with 'path' and 'original_vido_id' cols.",
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="facebook/dinov2-base",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    annot = pd.read_pickle(args.annotation)
    annot["path"] = annot["path"].apply(
        lambda x: args.image_dir / x
    )
    unique_videos = annot["original_vido_id"].unique().tolist()
    logger.info(
        "Found %d unique video groups in %s",
        len(unique_videos),
        args.annotation,
    )

    from act2i.features import FeatureExtractor

    extractor = FeatureExtractor(
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_dir = args.output_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    for idx, vid in enumerate(unique_videos):
        out_file = model_dir / f"{vid}.pt"
        if out_file.exists():
            continue

        paths = annot[annot["original_vido_id"] == vid][
            "path"
        ].tolist()
        text = ", ".join(
            annot[annot["original_vido_id"] == vid]["act"].iloc[0]
        )

        extractor.extract_and_save(
            image_paths=paths,
            output_path=out_file,
            text_context=text,
        )

        if (idx + 1) % 50 == 0:
            logger.info("Progress: %d / %d", idx + 1, len(unique_videos))

    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()
