"""Explicitly loaded local image selector; optional GPU imports stay at this boundary."""

import time
from pathlib import Path
from typing import Any

from PIL import Image


class LocalSelector:
    def __init__(self, weights: Path) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for local semantic selection")
        self.model: Any = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(weights.resolve()),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        self.processor: Any = AutoProcessor.from_pretrained(
            str(weights.resolve()),
            min_pixels=64 * 28 * 28,
            max_pixels=384 * 384,
            local_files_only=True,
        )

    def select(self, prompt: str, views: list[tuple[int, Path]]) -> dict[str, Any]:
        if not 3 <= len(views) <= 12:
            raise ValueError("Expected 3–12 labeled candidate images")
        result = self.select_labeled(
            prompt, [(f"Candidate {key}", path) for key, path in views], 512
        )
        result["order"] = [key for key, _ in views]
        return result

    def select_labeled(
        self, prompt: str, views: list[tuple[str, Path]], max_new_tokens: int = 1024
    ) -> dict[str, Any]:
        import torch

        if not 3 <= len(views) <= 32 or not 1 <= max_new_tokens <= 1024:
            raise ValueError("Expected 3–32 labeled images and at most 1024 output tokens")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        images = []
        for key, path in views:
            with Image.open(path) as source:
                rgba = Image.new("RGBA", source.size, "white")
                rgba.alpha_composite(source.convert("RGBA"))
                images.append(rgba.convert("RGB"))
            content.extend([{"type": "text", "text": f"{key}:"}, {"type": "image"}])
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(
            "cuda"
        )
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.cuda.synchronize()
        reply = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return {
            "order": [key for key, _ in views],
            "raw_text": reply,
            "seconds": time.perf_counter() - started,
            "generated_tokens": int(generated.shape[1] - inputs.input_ids.shape[1]),
        }
