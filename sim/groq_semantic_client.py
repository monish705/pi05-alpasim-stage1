import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SEMANTIC_ACTIONS = [
    "follow_route",
    "yield",
    "change_lane_left",
    "change_lane_right",
    "creep_forward",
    "reroute",
]


def _mime_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "image/png"


def image_path_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type = _mime_type_for(path)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


class GroqSemanticClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError("GROQ_API_KEY is not set.")
        self.model = model
        self.client = OpenAI(
            api_key=resolved_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def _json_completion(
        self,
        messages: list[dict[str, Any]],
        max_completion_tokens: int = 700,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        content = completion.choices[0].message.content or "{}"
        return {
            "parsed": json.loads(content),
            "raw_text": content,
            "model": self.model,
            "usage": completion.usage.model_dump() if completion.usage else None,
        }

    def vision_json(self, prompt: str, image_path: str, max_completion_tokens: int = 700) -> dict[str, Any]:
        data_url = image_path_to_data_url(image_path)
        return self._json_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_completion_tokens=max_completion_tokens,
            temperature=0.2,
        )

    def mission_json(
        self,
        mission_text: str,
        allowed_behaviors: list[str] | None = None,
        default_speed_kmh: int = 25,
        max_completion_tokens: int = 500,
    ) -> dict[str, Any]:
        behaviors = allowed_behaviors or ["cautious", "normal", "aggressive"]
        prompt = (
            "You are compiling a plain-English driving task into a compact machine-readable mission spec "
            "for a fixed-route CARLA autonomy run. "
            "Return only a JSON object with these fields: "
            "`mission_title` (string), "
            f"`behavior` (one of: {', '.join(behaviors)}), "
            "`target_speed_kmh` (integer from 15 to 40), "
            "`completion_radius_m` (number from 2 to 8), "
            "`risk_posture` (low|medium|high), "
            "`fallback_if_blocked` (string), "
            "`notes` (array of short strings). "
            "Prefer conservative choices when the instruction is ambiguous. "
            f"Default target speed is {default_speed_kmh} km/h. "
            f"Mission text: {mission_text}"
        )
        result = self._json_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_completion_tokens=max_completion_tokens,
            temperature=0.1,
        )
        parsed = result["parsed"]
        behavior = parsed.get("behavior")
        if behavior not in behaviors:
            raise ValueError(f"Unexpected behavior: {behavior!r}")
        speed = int(parsed.get("target_speed_kmh", default_speed_kmh))
        parsed["target_speed_kmh"] = min(40, max(15, speed))
        radius = float(parsed.get("completion_radius_m", 3.0))
        parsed["completion_radius_m"] = min(8.0, max(2.0, radius))
        if parsed.get("risk_posture") not in {"low", "medium", "high"}:
            raise ValueError(f"Unexpected risk_posture: {parsed.get('risk_posture')!r}")
        return result

    def semantic_decision(self, image_path: str, mission: str, extra_context: str = "") -> dict[str, Any]:
        prompt = (
            "You are the semantic runtime layer above a low-level autonomy policy. "
            "Given one driving image and mission context, decide the next tactical action. "
            "Return only a JSON object with these fields: "
            "`scene_summary` (string), `risk_level` (low|medium|high), "
            f"`recommended_action` (one of: {', '.join(SEMANTIC_ACTIONS)}), "
            "`reason` (string), `confidence` (number from 0 to 1). "
            "Prefer `follow_route` unless the scene strongly suggests otherwise. "
            f"Mission: {mission}. "
        )
        if extra_context:
            prompt += f"Extra context: {extra_context}."
        result = self.vision_json(prompt=prompt, image_path=image_path)
        parsed = result["parsed"]
        action = parsed.get("recommended_action")
        if action not in SEMANTIC_ACTIONS:
            raise ValueError(f"Unexpected recommended_action: {action!r}")
        return result
