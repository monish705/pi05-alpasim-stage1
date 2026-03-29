from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ACTION_DIM, ACTION_HORIZON

try:
    import openpi.models.model as _model
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as training_config
    import openpi.training.weight_loaders as weight_loaders
    import openpi.transforms as transforms
except ImportError:  # pragma: no cover - imported on the remote training box.
    _model = None
    pi0_config = None
    training_config = None
    weight_loaders = None
    transforms = None


def _require_openpi() -> None:
    if _model is None:
        raise ImportError("openpi is required for Stage 0 training utilities. Run this on the remote openpi env.")


def _parse_image(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = (255.0 * array).clip(0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return array


@dataclasses.dataclass(frozen=True)
class DrivingInputs:
    model_type: Any

    def __call__(self, data: dict) -> dict:
        front = _parse_image(data["image"]["front"])
        left = _parse_image(data["image"]["left"])
        right = _parse_image(data["image"]["right"])
        ego_state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
        route = np.asarray(data["route"], dtype=np.float32).reshape(-1)

        inputs = {
            "state": np.concatenate([ego_state, route], axis=0),
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                ),
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class DrivingOutputs:
    active_action_dim: int = 3

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {"actions": actions[:, : self.active_action_dim]}


@dataclasses.dataclass(frozen=True)
class Stage0DrivingDataConfig(training_config.DataConfigFactory if training_config else object):
    repo_id: str
    assets: Any = dataclasses.field(default_factory=lambda: training_config.AssetsConfig() if training_config else None)
    base_config: Any = None

    def create(self, assets_dirs: Path, model_config: Any) -> Any:
        _require_openpi()
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "image": {
                            "front": "observation.images.front",
                            "left": "observation.images.left",
                            "right": "observation.images.right",
                        },
                        "state": "observation.state",
                        "route": "observation.route",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[DrivingInputs(model_type=model_config.model_type)],
            outputs=[DrivingOutputs()],
        )
        model_transforms = training_config.ModelTransformFactory(default_prompt="drive the route")(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def make_stage0_train_config(
    *,
    repo_id: str,
    assets_base_dir: str,
    checkpoint_base_dir: str,
    exp_name: str = "pi05_stage0_av",
    num_train_steps: int = 2500,
    batch_size: int = 8,
) -> Any:
    _require_openpi()
    model = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        max_token_len=1024,
    )
    return training_config.TrainConfig(
        name="pi05_stage0_av",
        exp_name=exp_name,
        model=model,
        data=Stage0DrivingDataConfig(
            repo_id=repo_id,
            base_config=training_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
        ).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir=assets_base_dir,
        checkpoint_base_dir=checkpoint_base_dir,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        save_interval=max(100, min(500, num_train_steps // 5)),
        log_interval=25,
        keep_period=max(250, min(1000, num_train_steps)),
        overwrite=True,
        wandb_enabled=False,
        num_workers=0,
        policy_metadata={
            "stage": "stage0",
            "action_semantics": ["delta_s", "delta_yaw", "target_speed"],
        },
    )
