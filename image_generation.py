import json
import os
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm

hf_token = os.environ.get("HF_TOKEN", None)
cache_dir = os.environ.get("MODEL_CACHE_DIR", os.getcwd())

MODEL = "sd35large"

g = torch.Generator()

seeds = [42, 43, 44, 45]

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
    token=hf_token,
    low_cpu_mem_usage=True,
)
pipe = pipe.to("cuda")

torch.cuda.empty_cache()

with open("sample_prompts.json") as f:
    phrase_meta = json.load(f)

for run in tqdm(["phrase", "emotional", "spatial", "temporal"], desc="Run"):
    prompts = [p[run] for p in phrase_meta]
    ids = [p["id"] for p in phrase_meta]
    out_path = Path(f"data/images/{MODEL}/{run}")
    os.makedirs(out_path, exist_ok=True)

    for seed_id, seed in tqdm(enumerate(seeds), desc="Seed", total=len(seeds)):
        g.manual_seed(seed)

        for id, prompt in zip(ids, prompts):
            image_path = out_path / f"{id}_{seed_id}.png"
            torch.cuda.empty_cache()
            with torch.no_grad():
                pipe(
                    prompt,
                    num_inference_steps=40,
                    guidance_scale=3.5,
                    generator=g,
                ).images[0].save(image_path)
            torch.cuda.empty_cache()
