"""
cognitive/local_vlm.py
=======================
Local VLM — loads Qwen2.5-VL directly via transformers.
No vLLM server needed. For Colab / single-GPU use.

Usage:
    vlm = LocalVLM("Qwen/Qwen2.5-VL-7B-Instruct")
    response = vlm.chat("What do you see?", image=pil_image)
    response = vlm.chat("Write code to pick up the mug")
"""
import numpy as np
from typing import Optional
from PIL import Image


class LocalVLM:
    """
    Loads a VL model directly on GPU via transformers.
    Drop-in replacement for vLLM API calls.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"[LocalVLM] Loading {model_name}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_name = model_name
        self._torch = torch

        gpu_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[LocalVLM] ✅ Loaded ({gpu_gb:.1f} GB GPU)")

    def chat(self, prompt: str, image: Optional[Image.Image] = None,
             max_tokens: int = 500, temperature: float = 0.1) -> str:
        """
        Send a prompt (optionally with an image) to the VLM.
        Returns the model's text response.
        """
        content = []
        images = []

        if image is not None:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            content.append({"type": "image", "image": image})
            images.append(image)

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if images:
            inputs = self.processor(
                text=[text], images=images,
                return_tensors="pt", padding=True,
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=[text], return_tensors="pt", padding=True,
            ).to(self.model.device)

        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=True,
            )

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)
