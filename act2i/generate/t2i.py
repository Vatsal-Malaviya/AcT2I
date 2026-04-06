"""Text-to-image generation pipeline.

Wraps diffusers-based T2I models (e.g. Stable Diffusion 3.5 Large) to
generate images from baseline and enhanced prompts across multiple seeds.
"""

import logging
import os
from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import torch

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images from prompts using a diffusers pipeline.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for the T2I model.
    torch_dtype : torch.dtype
        Weight precision (default: float16).
    device : str
        Target device (default: "cuda").
    num_inference_steps : int
        Number of denoising steps (default: 40).
    guidance_scale : float
        Classifier-free guidance scale (default: 3.5).
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-large",
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
    ):
        self.model_id = model_id
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        hf_token = os.environ.get("HF_TOKEN")
        cache_dir = os.environ.get("MODEL_CACHE_DIR")

        logger.info("Loading T2I pipeline: %s …", model_id)
        # Lazy import to avoid pulling diffusers at package-import time
        from diffusers import StableDiffusion3Pipeline

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            token=hf_token,
            low_cpu_mem_usage=True,
        )
        self.pipe = self.pipe.to(device)
        torch.cuda.empty_cache()
        logger.info("T2I pipeline ready on %s.", device)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        seed: int = 42,
    ):  # -> PIL.Image.Image
        """Generate a single image from *prompt* with *seed*.

        Returns a PIL Image.
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        torch.cuda.empty_cache()
        with torch.no_grad():
            image = self.pipe(
                prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=g,
            ).images[0]
        torch.cuda.empty_cache()
        return image

    def generate_batch(
        self,
        prompts: Sequence[dict],
        seeds: Sequence[int] = (42, 43, 44, 45),
        output_dir: Union[str, Path] = "data/images",
        model_tag: Optional[str] = None,
        prompt_types: Sequence[str] = ("phrase", "emotional", "spatial", "temporal"),
        skip_existing: bool = True,
    ) -> List[Path]:
        """Generate images for a list of prompt entries across seeds.

        Parameters
        ----------
        prompts : sequence of dict
            Each dict must have ``"id"`` and keys matching
            *prompt_types* (e.g. ``{"id": "1", "phrase": "..."}``).
        seeds : sequence of int
            Random seeds to generate for each prompt.
        output_dir : path-like
            Root output directory for images.
        model_tag : str, optional
            Sub-folder name for this model (default: derived from *model_id*).
        prompt_types : sequence of str
            Which prompt variants to generate.
        skip_existing : bool
            If True, skip images that already exist on disk.

        Returns
        -------
        list of Path
            Paths to all generated images.
        """
        tag = model_tag or self.model_id.split("/")[-1]
        output_dir = Path(output_dir)
        saved: List[Path] = []

        for run_type in prompt_types:
            for seed_idx, seed in enumerate(seeds):
                for entry in prompts:
                    fname = f"{entry['id']}_{seed_idx}.png"
                    out_path = output_dir / tag / run_type / fname
                    if skip_existing and out_path.exists():
                        continue
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    prompt_text = entry.get(run_type)
                    if prompt_text is None:
                        logger.warning(
                            "Prompt entry %s missing key '%s', skipping.",
                            entry["id"],
                            run_type,
                        )
                        continue

                    image = self.generate(prompt_text, seed=seed)
                    image.save(out_path)
                    saved.append(out_path)
                    logger.debug("Saved %s", out_path)

        logger.info("Generated %d images.", len(saved))
        return saved
