#!/usr/bin/env python
"""CLI: Compute CLIP and/or DINO scores for generated images.

Usage
-----
    python scripts/compute_scores.py clip \
        --prompts data/prompt/phrases_reviewed_final.json \
        --images-dir data/images \
        --output data/prompt/clip_scores.csv

    python scripts/compute_scores.py dino \
        --prompts data/prompt/phrases_reviewed_final.json \
        --images-dir data/images \
        --act-map data/prompt/act_map.json \
        --ref-features /path/to/action_feats/ \
        --output data/prompt/dino.json
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


def cmd_clip(args):
    """Compute CLIPScore for all images."""
    with open(args.prompts) as f:
        phrases = {int(p["id"]): p for p in json.load(f)}

    from act2i.evaluate import CLIPScorer

    scorer = CLIPScorer()
    rows = []

    for model_dir in sorted(os.listdir(args.images_dir)):
        model_path = args.images_dir / model_dir
        if not model_path.is_dir():
            continue
        for ptype in sorted(os.listdir(model_path)):
            ppath = model_path / ptype
            if not ppath.is_dir():
                continue
            for fname in sorted(os.listdir(ppath)):
                if not fname.endswith(".png"):
                    continue
                idx_s, seed_s = fname.split(".")[0].split("_")
                prompt = phrases[int(idx_s)]["phrase"]
                score = scorer.score(ppath / fname, prompt)
                rows.append(
                    {
                        "model": model_dir,
                        "prompt_type": ptype,
                        "id": idx_s,
                        "seed": seed_s,
                        "clip": score,
                    }
                )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved %d CLIP scores → %s", len(df), args.output)


def cmd_dino(args):
    """Compute DINOv2 similarity scores."""
    import torch

    with open(args.prompts) as f:
        phrases = {x["id"]: x for x in json.load(f)}
    with open(args.act_map) as f:
        act_map = json.load(f)

    from act2i.evaluate import DINOScorer

    scorer = DINOScorer()
    archive = {}

    # Collect image paths grouped by action
    img_groups = {}
    for model_dir in sorted(os.listdir(args.images_dir)):
        model_path = args.images_dir / model_dir
        if not model_path.is_dir():
            continue
        for ptype in sorted(os.listdir(model_path)):
            ppath = model_path / ptype
            if not ppath.is_dir():
                continue
            for fname in sorted(os.listdir(ppath)):
                if not fname.endswith(".png"):
                    continue
                idx_s = fname.split("_")[0]
                action = phrases.get(idx_s, {}).get("action", "")
                act = act_map.get(action, "")
                if not act:
                    continue
                img_groups.setdefault(act, []).append(
                    ppath / fname
                )

    for act, img_paths in img_groups.items():
        feat_file = args.ref_features / act / "all_frames.pt"
        if not feat_file.exists():
            logger.warning("No ref features for '%s'", act)
            continue

        ref = torch.load(feat_file, weights_only=True)
        sims = scorer.score_against_reference(img_paths, ref)
        for path, sim in zip(img_paths, sims):
            rel = str(path.relative_to(args.images_dir))
            archive[rel] = sim

        with open(args.output, "w") as f:
            json.dump(archive, f)
        logger.info("Action '%s': %d images scored.", act, len(sims))

    logger.info("Saved %d DINO scores → %s", len(archive), args.output)


def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation scores."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # CLIP sub-command
    p_clip = sub.add_parser("clip", help="Compute CLIPScore.")
    p_clip.add_argument("--prompts", type=Path, required=True)
    p_clip.add_argument("--images-dir", type=Path, required=True)
    p_clip.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompt/clip_scores.csv"),
    )

    # DINO sub-command
    p_dino = sub.add_parser("dino", help="Compute DINOv2 scores.")
    p_dino.add_argument("--prompts", type=Path, required=True)
    p_dino.add_argument("--images-dir", type=Path, required=True)
    p_dino.add_argument("--act-map", type=Path, required=True)
    p_dino.add_argument("--ref-features", type=Path, required=True)
    p_dino.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompt/dino.json"),
    )

    args = parser.parse_args()

    if args.command == "clip":
        cmd_clip(args)
    elif args.command == "dino":
        cmd_dino(args)


if __name__ == "__main__":
    main()
