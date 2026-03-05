"""
perception/object_tagger.py
===========================
VLM-based object labeling and physics property estimation.

v2 changes:
  - Sim-mode fallback: reads labels from MuJoCo body names when VLM is unavailable
  - tag_quality score field on ObjectTag
  - Graceful error handling with detailed quality tracking

Uses Qwen3-VL-8B via vLLM (self-hosted, no cloud API costs).
"""
import json
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict, List
import base64
import io


@dataclass
class ObjectTag:
    label: str                       # "red ceramic mug"
    material: str                    # "ceramic"
    estimated_mass_kg: float         # 0.3
    estimated_friction: float        # 0.4
    is_graspable: bool               # True
    is_deformable: bool              # False
    dimensions_cm: Dict[str, float]  # {"width": 8, "height": 10, "depth": 8}
    collision_shape: str             # "cylinder" | "box" | "sphere" | "capsule"
    tag_quality: float = 1.0         # 1.0 = VLM parsed cleanly, 0.5 = partial, 0.0 = fallback
    raw_response: str = ""           # full VLM response for debugging


TAGGER_PROMPT = """You are a robotics perception system. You are looking at a cropped image of a single object on a table.

Analyze this object and respond ONLY with valid JSON (no markdown, no explanation):
{
  "label": "<descriptive name like 'red ceramic mug'>",
  "material": "<ceramic|plastic|metal|wood|glass|rubber|fabric|unknown>",
  "estimated_mass_kg": <float>,
  "estimated_friction_coefficient": <float between 0.1 and 1.0>,
  "is_graspable": <true|false>,
  "is_deformable": <true|false>,
  "approximate_dimensions_cm": {"width": <float>, "height": <float>, "depth": <float>},
  "simplified_collision_shape": "<cylinder|box|sphere|capsule>"
}"""


# Known sim objects — maps MuJoCo body name prefix to physics properties
_SIM_OBJECT_DB = {
    "red_mug": ObjectTag(
        label="red ceramic mug", material="ceramic",
        estimated_mass_kg=0.3, estimated_friction=0.5,
        is_graspable=True, is_deformable=False,
        dimensions_cm={"width": 8, "height": 10, "depth": 8},
        collision_shape="cylinder", tag_quality=0.9,
    ),
    "blue_box": ObjectTag(
        label="blue plastic box", material="plastic",
        estimated_mass_kg=0.15, estimated_friction=0.4,
        is_graspable=True, is_deformable=False,
        dimensions_cm={"width": 6, "height": 6, "depth": 6},
        collision_shape="box", tag_quality=0.9,
    ),
    "green_bottle": ObjectTag(
        label="green glass bottle", material="glass",
        estimated_mass_kg=0.25, estimated_friction=0.3,
        is_graspable=True, is_deformable=False,
        dimensions_cm={"width": 6, "height": 20, "depth": 6},
        collision_shape="cylinder", tag_quality=0.9,
    ),
    "yellow_ball": ObjectTag(
        label="yellow rubber ball", material="rubber",
        estimated_mass_kg=0.1, estimated_friction=0.6,
        is_graspable=True, is_deformable=True,
        dimensions_cm={"width": 7, "height": 7, "depth": 7},
        collision_shape="sphere", tag_quality=0.9,
    ),
}


class ObjectTagger:
    """
    Tags objects using VLM for physics property estimation.
    
    Priority order:
      1. VLM API (vLLM server) — if reachable
      2. Local VLM (transformers on GPU) — if torch + transformers available
      3. Sim-mode tagger (MuJoCo body names) — always available
    """

    def __init__(self, api_base: str = "http://localhost:8000/v1",
                 model_name: str = "Qwen/Qwen3-VL-8B",
                 sim_mode: bool = False):
        self.api_base = api_base
        self.model_name = model_name
        self.sim_mode = sim_mode
        self._vlm_available = None  # lazy check
        self._local_vlm = None      # local transformers model
        self._local_vlm_checked = False

    def _check_vlm(self) -> bool:
        """Check if VLM API is reachable."""
        if self._vlm_available is not None:
            return self._vlm_available
        try:
            from openai import OpenAI
            client = OpenAI(base_url=self.api_base, api_key="dummy")
            client.models.list()
            self._vlm_available = True
        except Exception:
            self._vlm_available = False
        return self._vlm_available

    def _check_local_vlm(self) -> bool:
        """Try to load VLM locally via transformers (for Colab GPU)."""
        if self._local_vlm_checked:
            return self._local_vlm is not None
        self._local_vlm_checked = True
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            print("[VLM Tagger] Loading Qwen2.5-VL-3B locally on GPU...")
            self._local_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._local_processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct"
            )
            print("[VLM Tagger] ✅ Local VLM loaded")
            return True
        except Exception as e:
            print(f"[VLM Tagger] Local VLM not available: {e}")
            return False

    def tag(self, crop: Image.Image, sim_body_name: str = None) -> ObjectTag:
        """
        Tag a single object crop with semantic label and physics properties.

        Priority: VLM API → Local VLM (transformers) → Sim-mode tagger
        """
        # Sim-mode forced
        if self.sim_mode:
            return self.tag_from_sim_name(sim_body_name)

        # Try VLM API first
        if self._check_vlm():
            return self._tag_with_vlm(crop)

        # Try local VLM (transformers on GPU)
        if self._check_local_vlm():
            return self._tag_with_local_vlm(crop)

        # Fallback: sim-mode
        return self.tag_from_sim_name(sim_body_name)

    def _tag_with_local_vlm(self, crop: Image.Image) -> ObjectTag:
        """Tag using local Qwen2.5-VL via transformers (runs on GPU)."""
        try:
            import torch
            from qwen_vl_utils import process_vision_info

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": TAGGER_PROMPT},
                ],
            }]

            text = self._local_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._local_processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(self._local_vlm.device)

            with torch.no_grad():
                generated_ids = self._local_vlm.generate(
                    **inputs, max_new_tokens=300, temperature=0.1,
                    do_sample=False
                )
            # Trim input tokens
            generated_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw = self._local_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            print(f"[VLM Local] Response: {raw[:100]}...")
            return self._parse_response(raw)

        except Exception as e:
            print(f"[VLM Local] Error: {e}")
            return self._default_tag()

    def _tag_with_vlm(self, crop: Image.Image) -> ObjectTag:
        """Tag using VLM API."""
        from openai import OpenAI

        client = OpenAI(base_url=self.api_base, api_key="dummy")

        # Encode image to base64
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TAGGER_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }}
                    ]
                }],
                max_tokens=300,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_response(raw)

        except Exception as e:
            print(f"[VLM Tagger] API error: {e}")
            return self._default_tag()

    def tag_from_sim_name(self, body_name: str = None) -> ObjectTag:
        """
        Tag an object using its MuJoCo body name.
        Useful for testing without VLM / GPU.

        e.g., "obj_red_mug" → looks up "red_mug" in the sim DB.
        """
        if body_name is None:
            return self._default_tag()

        # Strip "obj_" prefix
        clean = body_name.replace("obj_", "").strip()

        # Look up in known DB
        if clean in _SIM_OBJECT_DB:
            import copy
            return copy.deepcopy(_SIM_OBJECT_DB[clean])

        # Try partial matching
        for key, tag in _SIM_OBJECT_DB.items():
            if key in clean or clean in key:
                import copy
                return copy.deepcopy(tag)

        # Generate a generic tag from the name
        words = clean.replace("_", " ")
        return ObjectTag(
            label=words,
            material="unknown",
            estimated_mass_kg=0.2,
            estimated_friction=0.4,
            is_graspable=True,
            is_deformable=False,
            dimensions_cm={"width": 5, "height": 5, "depth": 5},
            collision_shape="box",
            tag_quality=0.5,
            raw_response=f"SIM_NAME:{body_name}",
        )

    def tag_batch(self, crops: list, sim_body_names: list = None) -> list:
        """Tag multiple object crops."""
        tags = []
        for i, crop in enumerate(crops):
            sim_name = sim_body_names[i] if sim_body_names and i < len(sim_body_names) else None
            print(f"[VLM] Tagging object {i+1}/{len(crops)}...")
            try:
                tag = self.tag(crop, sim_body_name=sim_name)
                tags.append(tag)
            except Exception as e:
                print(f"[VLM] Tagging failed for object {i+1}: {e}")
                tags.append(self._default_tag())
        return tags

    def _parse_response(self, raw: str) -> ObjectTag:
        """Parse VLM JSON response into ObjectTag."""
        try:
            # Strip markdown code blocks if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

            data = json.loads(raw)
            dims = data.get("approximate_dimensions_cm", {})

            return ObjectTag(
                label=data.get("label", "unknown object"),
                material=data.get("material", "unknown"),
                estimated_mass_kg=float(data.get("estimated_mass_kg", 0.2)),
                estimated_friction=float(data.get("estimated_friction_coefficient", 0.4)),
                is_graspable=bool(data.get("is_graspable", True)),
                is_deformable=bool(data.get("is_deformable", False)),
                dimensions_cm={
                    "width": float(dims.get("width", 5)),
                    "height": float(dims.get("height", 5)),
                    "depth": float(dims.get("depth", 5)),
                },
                collision_shape=data.get("simplified_collision_shape", "box"),
                tag_quality=1.0,
                raw_response=raw,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[VLM] Parse error: {e}\nRaw: {raw}")
            tag = self._default_tag()
            tag.tag_quality = 0.0
            tag.raw_response = raw
            return tag

    @staticmethod
    def _default_tag() -> ObjectTag:
        return ObjectTag(
            label="unknown object",
            material="unknown",
            estimated_mass_kg=0.2,
            estimated_friction=0.4,
            is_graspable=True,
            is_deformable=False,
            dimensions_cm={"width": 5, "height": 5, "depth": 5},
            collision_shape="box",
            tag_quality=0.0,
            raw_response="FALLBACK",
        )
