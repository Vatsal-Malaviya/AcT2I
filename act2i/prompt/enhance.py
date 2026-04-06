"""LLM-based prompt enhancement using knowledge distillation.

Wraps a causal-LM (e.g. Llama 3.3-70B-Instruct) to enrich baseline
text-to-image prompts along emotional, spatial, and temporal dimensions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import transformers

from act2i.prompt.templates import DIMENSIONS, SYSTEM_PROMPTS

logger = logging.getLogger(__name__)


class PromptEnhancer:
    """Enhance T2I prompts via an instruction-tuned LLM.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (default: Llama 3.3-70B-Instruct).
    torch_dtype : torch.dtype
        Weight precision (default: bfloat16).
    device_map : str
        Device placement strategy (default: "auto").
    max_new_tokens : int
        Maximum generation length per prompt.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling threshold.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading model %s …", model_id)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
            },
            device_map=device_map,
        )
        logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def enhance(
        self,
        prompt: str,
        dimension: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Enhance a single prompt along *dimension*.

        Parameters
        ----------
        prompt : str
            Baseline T2I prompt (e.g. "a Fox chasing a Rabbit").
        dimension : str
            One of ``"emotional"``, ``"spatial"``, ``"temporal"``.
        system_prompt : str, optional
            Override the built-in system prompt for *dimension*.

        Returns
        -------
        str
            The enhanced prompt text.
        """
        if dimension not in DIMENSIONS and system_prompt is None:
            raise ValueError(
                f"Unknown dimension '{dimension}'. "
                f"Choose from {DIMENSIONS} or supply a custom system_prompt."
            )

        sys_content = system_prompt or SYSTEM_PROMPTS[dimension]
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt},
        ]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            batch_size=1,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        enhanced = outputs[0]["generated_text"][-1]["content"]
        logger.debug("%s | %s → %s", dimension, prompt, enhanced)
        return enhanced

    def enhance_batch(
        self,
        prompts: Sequence[str],
        dimensions: Optional[Sequence[str]] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, List[str]]:
        """Enhance a list of prompts across one or more dimensions.

        Parameters
        ----------
        prompts : sequence of str
            Baseline prompts.
        dimensions : sequence of str, optional
            Dimensions to enhance along (default: all three).
        output_path : Path, optional
            If given, results are saved incrementally as JSON after every prompt.

        Returns
        -------
        dict
            ``{dimension: [enhanced_prompt, ...]}``
        """
        dims = dimensions or list(DIMENSIONS)
        results: Dict[str, List[str]] = {d: [] for d in dims}

        for dim in dims:
            logger.info("Enhancing %d prompts along '%s' …", len(prompts), dim)
            for prompt in prompts:
                enhanced = self.enhance(prompt, dim)
                results[dim].append(enhanced)

                if output_path is not None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        json.dump(results, f, indent=2)

        return results
