from __future__ import annotations

from typing import Any

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


class FashionGenerator:
    """Stable Diffusion wrapper tuned for low-VRAM local inference."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._pipeline: StableDiffusionPipeline | None = None

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load_pipeline(self) -> StableDiffusionPipeline:
        if self._pipeline is not None:
            return self._pipeline

        pipeline_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
            "safety_checker": None,
        }

        if self.device.type == "cuda":
            pipeline_kwargs["variant"] = "fp16"

        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            **pipeline_kwargs,
        )
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()

        if self.device.type == "cuda":
            total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            six_gb_threshold = int(6.5 * 1024**3)

            if total_vram_bytes <= six_gb_threshold:
                pipeline.enable_sequential_cpu_offload()
            else:
                pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(self.device)

        self._pipeline = pipeline
        return pipeline

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> dict[str, object]:
        self.load_pipeline()

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "alpha": alpha,
            "beta": beta,
            "status": "stub",
            "message": "Stable Diffusion is initialized, but image generation is not implemented yet.",
        }
