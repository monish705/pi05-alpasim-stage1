"""
Capture-to-simulation compiler.

This module turns a small set of images into:
  - a fused scene graph
  - a MuJoCo scene XML
  - a randomization config
  - a compiler report describing what was inferred vs. assumed

Intended architecture:
  RGB / video frames
    -> geometry backbone (VGGT / MASt3R / DUSt3R / SLAM)
    -> segmentation (SAM-family)
    -> object reconstruction (SAM3D Objects)
    -> semantics + physics priors
    -> scene graph fusion
    -> simulator export

The current implementation supports metric captures where depth and camera
extrinsics are already known. RGB-only geometry backbones can be plugged in
later without changing the export contract.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from perception.multi_view_fusion import MultiViewFusion
from perception.pipeline import PerceptionPipeline, PipelineConfig
from perception.scene_graph import SceneGraph


@dataclass
class CaptureFrame:
    """One captured frame plus its calibration."""

    camera_name: str
    rgb_path: Path
    intrinsics: Dict[str, float]
    depth_path: Optional[Path] = None
    camera_extrinsics: Optional[np.ndarray] = None
    prompts: List[str] = field(default_factory=list)


@dataclass
class CaptureManifest:
    """Input spec for a scene compilation run."""

    scene_name: str
    frames: List[CaptureFrame]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilerArtifactPaths:
    """Where the exported artifacts were written."""

    scene_graph_json: Path
    mujoco_xml: Path
    randomization_json: Path
    compiler_report_json: Path


@dataclass
class CompiledScene:
    """Result of compiling a capture set into simulator artifacts."""

    scene_name: str
    scene_graph: SceneGraph
    mujoco_xml: str
    randomization_config: Dict[str, Any]
    compiler_report: Dict[str, Any]
    artifact_paths: Optional[CompilerArtifactPaths] = None


@dataclass
class PhotoToSimCompilerConfig:
    """Runtime settings for the compiler."""

    vlm_api_base: str = "http://localhost:8000/v1"
    vlm_model: str = "Qwen/Qwen3-VL-8B"
    pipeline: PipelineConfig = field(
        default_factory=lambda: PipelineConfig(
            use_sim_tagger=False,
            depth_fallback=True,
            min_confidence=0.10,
        )
    )
    fusion_merge_threshold_m: float = 0.15
    support_thickness_m: float = 0.04
    support_margin_m: float = 0.25
    default_floor_z_m: float = 0.0
    allow_identity_extrinsics: bool = True


class PhotoToSimCompiler:
    """
    Compile calibrated captures into a physics-friendly MuJoCo scene.

    Current requirement:
      - depth is provided per frame
      - intrinsics are known
      - camera extrinsics are known or explicitly assumed
    """

    def __init__(self, config: Optional[PhotoToSimCompilerConfig] = None):
        self.config = config or PhotoToSimCompilerConfig()

    def compile_manifest(
        self,
        manifest_path: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> CompiledScene:
        """Load a capture manifest from disk and compile it."""
        manifest_path = Path(manifest_path).resolve()
        manifest = self.load_manifest(manifest_path)
        compiled = self.compile(manifest)
        if output_dir is not None:
            return self.export(compiled, output_dir)
        return compiled

    def load_manifest(self, manifest_path: str | Path) -> CaptureManifest:
        """Load JSON capture manifest and resolve relative paths."""
        manifest_path = Path(manifest_path).resolve()
        root = manifest_path.parent
        data = json.loads(manifest_path.read_text(encoding="utf-8"))

        scene_name = data.get("scene_name", manifest_path.stem)
        metadata = data.get("metadata", {})
        frames: List[CaptureFrame] = []

        for idx, frame_data in enumerate(data.get("frames", [])):
            rgb_path = self._resolve_path(root, frame_data["rgb_path"])
            depth_path = self._resolve_path(root, frame_data["depth_path"]) if frame_data.get("depth_path") else None

            intrinsics = self._load_intrinsics(root, frame_data)
            extrinsics = self._load_extrinsics(root, frame_data)
            if extrinsics is None and self.config.allow_identity_extrinsics:
                extrinsics = np.eye(4, dtype=np.float64)

            frames.append(
                CaptureFrame(
                    camera_name=frame_data.get("camera_name", f"frame_{idx}"),
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    intrinsics=intrinsics,
                    camera_extrinsics=extrinsics,
                    prompts=list(frame_data.get("prompts", [])),
                )
            )

        if not frames:
            raise ValueError(f"No frames found in manifest: {manifest_path}")

        return CaptureManifest(scene_name=scene_name, frames=frames, metadata=metadata)

    def compile(self, manifest: CaptureManifest) -> CompiledScene:
        """Compile the manifest into a fused scene graph and MuJoCo XML."""
        per_camera_objects: Dict[str, List[Any]] = {}
        warnings: List[str] = []

        for frame in manifest.frames:
            if frame.depth_path is None:
                raise ValueError(
                    f"Frame '{frame.camera_name}' has no depth_path. "
                    "RGB-only compilation is not wired yet; attach a geometry "
                    "backend or provide precomputed depth/extrinsics."
                )

            if frame.camera_extrinsics is None:
                raise ValueError(f"Frame '{frame.camera_name}' has no camera extrinsics.")

            if np.allclose(frame.camera_extrinsics, np.eye(4)):
                warnings.append(
                    f"{frame.camera_name}: using identity extrinsics; scene will "
                    "remain in camera coordinates unless a real pose is provided."
                )

            pipeline = PerceptionPipeline(
                vlm_api_base=self.config.vlm_api_base,
                vlm_model=self.config.vlm_model,
                config=self.config.pipeline,
            )

            rgb = np.array(Image.open(frame.rgb_path).convert("RGB"))
            depth = np.load(frame.depth_path).astype(np.float32)

            scene_graph = pipeline.perceive(
                rgb=rgb,
                depth=depth,
                camera_params=frame.intrinsics,
                camera_extrinsics=frame.camera_extrinsics,
                prompts=frame.prompts or None,
            )
            per_camera_objects[frame.camera_name] = copy.deepcopy(scene_graph.get_all_objects())

        if len(per_camera_objects) == 1:
            fused_scene_graph = SceneGraph()
            fused_scene_graph.update(copy.deepcopy(next(iter(per_camera_objects.values()))))
        else:
            fusion = MultiViewFusion(merge_threshold_m=self.config.fusion_merge_threshold_m)
            fused_scene_graph = fusion.fuse(per_camera_objects)

        mujoco_xml = self._build_mujoco_xml(manifest.scene_name, fused_scene_graph)
        randomization = self._build_randomization_config(fused_scene_graph)
        compiler_report = self._build_compiler_report(
            manifest=manifest,
            fused_scene_graph=fused_scene_graph,
            warnings=warnings,
        )

        return CompiledScene(
            scene_name=manifest.scene_name,
            scene_graph=fused_scene_graph,
            mujoco_xml=mujoco_xml,
            randomization_config=randomization,
            compiler_report=compiler_report,
        )

    def export(self, compiled: CompiledScene, output_dir: str | Path) -> CompiledScene:
        """Write compiler artifacts to disk."""
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_graph_json = output_dir / "scene_graph.json"
        mujoco_xml = output_dir / "scene.xml"
        randomization_json = output_dir / "randomization.json"
        compiler_report_json = output_dir / "compiler_report.json"

        compiled.scene_graph.save_json(str(scene_graph_json))
        mujoco_xml.write_text(compiled.mujoco_xml, encoding="utf-8")
        randomization_json.write_text(json.dumps(compiled.randomization_config, indent=2), encoding="utf-8")
        compiler_report_json.write_text(json.dumps(compiled.compiler_report, indent=2), encoding="utf-8")

        compiled.artifact_paths = CompilerArtifactPaths(
            scene_graph_json=scene_graph_json,
            mujoco_xml=mujoco_xml,
            randomization_json=randomization_json,
            compiler_report_json=compiler_report_json,
        )
        return compiled

    def _build_mujoco_xml(self, scene_name: str, scene_graph: SceneGraph) -> str:
        """Export the fused scene graph to MuJoCo XML."""
        objects = scene_graph.get_all_objects()
        support = self._infer_support_surface(objects)

        bodies: List[str] = []
        for obj in objects:
            geom_xml = self._scene_object_to_geom_xml(obj)
            pos = obj.position_world
            body = (
                f'    <body name="obj_{obj.track_id}" '
                f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}">\n'
                f'      <freejoint name="joint_{obj.track_id}"/>\n'
                f"      {geom_xml}\n"
                f"    </body>"
            )
            bodies.append(body)

        support_geom = (
            f'    <geom name="support_surface" type="box" '
            f'pos="{support["center_x"]:.4f} {support["center_y"]:.4f} '
            f'{support["center_z"]:.4f}" '
            f'size="{support["size_x"]:.4f} {support["size_y"]:.4f} '
            f'{support["size_z"]:.4f}" '
            'rgba="0.55 0.42 0.28 1" friction="0.8 0.005 0.0001"/>'
        )
        floor_geom = (
            f'    <geom name="floor" type="plane" '
            f'pos="0 0 {self.config.default_floor_z_m:.4f}" '
            'size="5 5 0.02" rgba="0.9 0.9 0.9 1" '
            'friction="0.9 0.005 0.0001"/>'
        )

        xml = (
            f'<mujoco model="{scene_name}">\n'
            "  <worldbody>\n"
            '    <light pos="0 0 3" dir="0 0 -1"/>\n'
            f"{floor_geom}\n"
            f"{support_geom}\n"
            f"{chr(10).join(bodies)}\n"
            "  </worldbody>\n"
            "</mujoco>\n"
        )
        return xml

    def _scene_object_to_geom_xml(self, obj: Any) -> str:
        """Convert one scene object into a MuJoCo geom."""
        dims = obj.dimensions_m
        width = max(float(dims.get("width", 0.05)), 0.01)
        height = max(float(dims.get("height", 0.05)), 0.01)
        depth = max(float(dims.get("depth", 0.05)), 0.01)
        mass = max(float(getattr(obj, "mass_kg", 0.2)), 0.01)
        friction = max(float(getattr(obj, "friction", 0.4)), 0.05)
        rgba = "0.62 0.62 0.62 1"

        shape = getattr(obj, "collision_primitive", "box")
        if shape == "sphere":
            radius = max(width, height, depth) / 2
            return (
                f'<geom name="geom_{obj.track_id}" type="sphere" '
                f'size="{radius:.4f}" mass="{mass:.4f}" '
                f'friction="{friction:.4f} 0.005 0.0001" rgba="{rgba}"/>'
            )

        if shape == "cylinder":
            radius = max(width, depth) / 2
            half_height = height / 2
            return (
                f'<geom name="geom_{obj.track_id}" type="cylinder" '
                f'size="{radius:.4f} {half_height:.4f}" mass="{mass:.4f}" '
                f'friction="{friction:.4f} 0.005 0.0001" rgba="{rgba}"/>'
            )

        return (
            f'<geom name="geom_{obj.track_id}" type="box" '
            f'size="{width/2:.4f} {height/2:.4f} {depth/2:.4f}" '
            f'mass="{mass:.4f}" friction="{friction:.4f} 0.005 0.0001" '
            f'rgba="{rgba}"/>'
        )

    def _infer_support_surface(self, objects: List[Any]) -> Dict[str, float]:
        """
        Infer one dominant support surface from object bottoms.

        This keeps compiled scenes dynamically stable even when the capture
        came from a tabletop or shelf environment rather than the floor.
        """
        if not objects:
            thickness = self.config.support_thickness_m / 2
            return {
                "center_x": 0.0,
                "center_y": 0.0,
                "center_z": self.config.default_floor_z_m - thickness,
                "size_x": 0.6,
                "size_y": 0.6,
                "size_z": thickness,
            }

        xs = np.array([float(obj.position_world[0]) for obj in objects])
        ys = np.array([float(obj.position_world[1]) for obj in objects])
        bottoms = []
        for obj in objects:
            height = max(float(obj.dimensions_m.get("height", 0.05)), 0.01)
            bottoms.append(float(obj.position_world[2]) - height / 2)

        support_z = float(np.median(bottoms))
        margin = self.config.support_margin_m
        size_x = max((xs.max() - xs.min()) / 2 + margin, 0.25)
        size_y = max((ys.max() - ys.min()) / 2 + margin, 0.25)
        thickness = self.config.support_thickness_m / 2

        return {
            "center_x": float((xs.max() + xs.min()) / 2),
            "center_y": float((ys.max() + ys.min()) / 2),
            "center_z": support_z - thickness,
            "size_x": float(size_x),
            "size_y": float(size_y),
            "size_z": float(thickness),
        }

    def _build_randomization_config(self, scene_graph: SceneGraph) -> Dict[str, Any]:
        """Generate practical scene randomization knobs around compiled objects."""
        objects_cfg = []
        for obj in scene_graph.get_all_objects():
            objects_cfg.append(
                {
                    "track_id": int(obj.track_id),
                    "label": obj.label,
                    "position_jitter_m": {"x": 0.03, "y": 0.03, "z": 0.01},
                    "yaw_jitter_rad": 0.35,
                    "mass_scale_range": [0.85, 1.15],
                    "friction_scale_range": [0.85, 1.15],
                    "size_scale_range": [0.95, 1.05],
                }
            )

        return {
            "scene_randomization": {
                "support_surface_xy_jitter_m": 0.02,
                "support_surface_height_jitter_m": 0.01,
                "lighting_profile": ["neutral", "warm", "cool"],
            },
            "objects": objects_cfg,
        }

    def _build_compiler_report(
        self,
        manifest: CaptureManifest,
        fused_scene_graph: SceneGraph,
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Summarize what the compiler did and what it assumed."""
        return {
            "scene_name": manifest.scene_name,
            "num_frames": len(manifest.frames),
            "num_objects": len(fused_scene_graph.get_all_objects()),
            "geometry_mode": "metric_depth_capture",
            "fusion_mode": "multi_view_fusion" if len(manifest.frames) > 1 else "single_view",
            "research_backbone_notes": [
                "Current path expects depth/extrinsics or a precomputed geometry backend.",
                "Recommended RGB-only backbones: VGGT, MASt3R, DUSt3R, or calibrated SLAM.",
                "SAM3D Objects should be treated as an object-level prior, not the sole source of global geometry.",
            ],
            "warnings": warnings,
            "metadata": manifest.metadata,
        }

    @staticmethod
    def _resolve_path(root: Path, value: str) -> Path:
        """Resolve a manifest path relative to the manifest directory."""
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        return path.resolve()

    @staticmethod
    def _load_intrinsics(root: Path, frame_data: Dict[str, Any]) -> Dict[str, float]:
        """Load intrinsics either inline or from JSON."""
        if "intrinsics" in frame_data:
            data = frame_data["intrinsics"]
        elif "intrinsics_path" in frame_data:
            path = PhotoToSimCompiler._resolve_path(root, frame_data["intrinsics_path"])
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Each frame must provide 'intrinsics' or 'intrinsics_path'")

        return {
            "fx": float(data["fx"]),
            "fy": float(data["fy"]),
            "cx": float(data["cx"]),
            "cy": float(data["cy"]),
        }

    @staticmethod
    def _load_extrinsics(root: Path, frame_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load a 4x4 camera-to-world extrinsics matrix."""
        data = None
        if "camera_extrinsics" in frame_data:
            data = frame_data["camera_extrinsics"]
        elif "extrinsics_path" in frame_data:
            path = PhotoToSimCompiler._resolve_path(root, frame_data["extrinsics_path"])
            data = json.loads(path.read_text(encoding="utf-8"))

        if data is None:
            return None

        arr = np.array(data, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError(f"camera_extrinsics must be 4x4, got {arr.shape}")
        return arr
