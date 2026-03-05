"""
perception/pipeline.py
======================
Orchestrates the full perception pipeline:
  RGB → SAM3 → SAM3D Objects → Qwen3-VL → Depth Validation → Scene Graph

v2 changes:
  - Unified confidence score combining all pipeline stages
  - PipelineConfig for tuning thresholds and weights
  - Graceful degradation: runs with whatever models are available
  - Camera-to-world transform support for multi-camera fusion
  - Sim-mode: can tag objects from MuJoCo body names without VLM
"""
import uuid
import numpy as np
from PIL import Image
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from perception.segmentor import SAM3Segmentor
from perception.reconstructor_3d import SAM3DReconstructor
from perception.object_tagger import ObjectTagger
from perception.depth_validator import DepthValidator
from perception.scene_graph import SceneGraph, SceneObject


@dataclass
class PipelineConfig:
    """Tunable parameters for the perception pipeline."""
    # Confidence weights (must sum to 1.0)
    w_segmentation: float = 0.30    # SAM3 segmentation score
    w_reconstruction: float = 0.15  # SAM3D success
    w_tagging: float = 0.20        # VLM tag parse quality
    w_depth: float = 0.35          # depth agreement

    # Thresholds
    min_confidence: float = 0.15    # discard objects below this
    depth_tolerance_m: float = 0.05 # max discrepancy before depth correction

    # Fallback behaviour
    use_sim_tagger: bool = False    # use MuJoCo body names instead of VLM
    depth_fallback: bool = True     # use depth segmentation if SAM3 unavailable


class PerceptionPipeline:
    """
    Full active perception pipeline.

    Chains: SAM3 segmentation → SAM3D 3D reconstruction →
            Qwen3-VL tagging → Depth validation → Scene Graph update

    Degrades gracefully when models are unavailable:
      - No SAM3: uses depth-based connected components
      - No SAM3D: uses depth-projected centroids for 3D position
      - No VLM: uses sim-mode tagger (MuJoCo body names)
    """

    def __init__(self, vlm_api_base: str = "http://localhost:8000/v1",
                 vlm_model: str = "Qwen/Qwen3-VL-8B",
                 config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        self.segmentor = SAM3Segmentor()
        self.reconstructor = SAM3DReconstructor()
        self.tagger = ObjectTagger(
            api_base=vlm_api_base,
            model_name=vlm_model,
            sim_mode=self.config.use_sim_tagger,
        )
        self.depth_validator = DepthValidator(
            tolerance_m=self.config.depth_tolerance_m
        )
        self.scene_graph = SceneGraph()

        # Report capabilities
        caps = []
        if SAM3Segmentor.is_available():
            caps.append("SAM3")
        else:
            caps.append("depth-segmentation")
        if SAM3DReconstructor.is_available():
            caps.append("SAM3D")
        else:
            caps.append("depth-projection")
        if not self.config.use_sim_tagger:
            caps.append("VLM-tagger")
        else:
            caps.append("sim-tagger")
        print(f"[Pipeline] Capabilities: {' + '.join(caps)}")

    def perceive(self, rgb: np.ndarray, depth: np.ndarray,
                 camera_params: Dict[str, float],
                 camera_extrinsics: np.ndarray = None,
                 prompts: list = None,
                 sim_body_names: list = None) -> SceneGraph:
        """
        Run full perception on a single frame.

        Args:
            rgb: (H, W, 3) uint8 RGB image
            depth: (H, W) float32 depth map
            camera_params: dict with fx, fy, cx, cy
            camera_extrinsics: optional (4, 4) camera-to-world transform matrix
            prompts: optional list of text prompts for SAM3
            sim_body_names: optional list of MuJoCo body names for sim-mode tagging

        Returns:
            Updated SceneGraph with all detected objects
        """
        pil_image = Image.fromarray(rgb)

        # Stage 1: Segmentation (SAM3 or depth fallback)
        print("\n[Pipeline] Stage 1: Segmentation...")
        segments = self.segmentor.segment_all(
            pil_image, prompts=prompts, depth=depth
        )
        print(f"[Pipeline] Found {len(segments)} objects "
              f"(method: {segments[0].method if segments else 'none'})")

        if not segments:
            print("[Pipeline] No objects detected.")
            return self.scene_graph

        # Stage 2: 3D Reconstruction (SAM3D or skip)
        print("[Pipeline] Stage 2: 3D Reconstruction...")
        masks = [seg.mask for seg in segments]
        reconstructions = self.reconstructor.reconstruct_batch(pil_image, masks)

        # Stage 3: Object Tagging (VLM or sim-mode)
        print("[Pipeline] Stage 3: Object Tagging...")
        crops = [seg.crop for seg in segments]
        tags = self.tagger.tag_batch(crops, sim_body_names=sim_body_names)

        # Stage 4: Depth Validation + Confidence + Assembly
        print("[Pipeline] Stage 4: Depth Validation + Assembly...")
        new_objects = []

        for i, (seg, recon, tag) in enumerate(zip(segments, reconstructions, tags)):
            # Get 3D position
            if recon.success and recon.translation is not None:
                position = recon.translation
            else:
                # Fallback: depth-project the mask centroid
                position = self.depth_validator._depth_project(
                    seg.mask, depth, camera_params
                )
                if position is None:
                    print(f"[Pipeline] Skipping object {i}: no valid position")
                    continue

            # Validate against depth
            corrected_pos, was_corrected, discrepancy = \
                self.depth_validator.validate_position(
                    position, seg.mask, depth, camera_params
                )

            # Transform to world frame if camera extrinsics provided
            if camera_extrinsics is not None:
                corrected_pos = self._camera_to_world(
                    corrected_pos, camera_extrinsics
                )

            # Measure dimensions from depth
            depth_dims = self.depth_validator.measure_object_dimensions(
                seg.mask, depth, camera_params
            )

            # Merge VLM dimensions with depth-measured dimensions
            dims_m = {k: v / 100.0 for k, v in tag.dimensions_cm.items()}
            if depth_dims:
                dims_m = depth_dims

            # Compute unified confidence score
            confidence = self._compute_confidence(
                seg_score=seg.score,
                recon_success=recon.success,
                tag_quality=tag.tag_quality,
                depth_discrepancy=discrepancy,
            )

            if confidence < self.config.min_confidence:
                print(f"[Pipeline] Skipping object {i} '{tag.label}': "
                      f"confidence {confidence:.2f} < {self.config.min_confidence}")
                continue

            obj = SceneObject(
                id=str(uuid.uuid4())[:8],
                label=tag.label,
                position_world=corrected_pos.tolist(),
                collision_primitive=tag.collision_shape,
                dimensions_m=dims_m,
                mass_kg=tag.estimated_mass_kg,
                friction=tag.estimated_friction,
                material=tag.material,
                is_graspable=tag.is_graspable,
                is_deformable=tag.is_deformable,
                confidence=confidence,
                _mesh_vertices=recon.mesh_vertices if recon.success else None,
                _mesh_faces=recon.mesh_faces if recon.success else None,
            )
            new_objects.append(obj)

        # Stage 5: Update Scene Graph (with temporal tracking)
        self.scene_graph.update(new_objects)
        print(f"[Pipeline] Scene Graph updated: "
              f"{len(self.scene_graph.objects)} total objects")
        self.scene_graph.print_all()

        return self.scene_graph

    def _compute_confidence(self, seg_score: float,
                            recon_success: bool,
                            tag_quality: float,
                            depth_discrepancy: float) -> float:
        """
        Compute unified confidence combining all pipeline stages.

        Returns a value in [0, 1].
        """
        cfg = self.config

        # Segmentation contribution [0, 1]
        c_seg = min(seg_score, 1.0)

        # Reconstruction contribution [0, 1]
        c_recon = 1.0 if recon_success else 0.3  # partial credit for depth-only

        # Tag quality [0, 1]
        c_tag = tag_quality

        # Depth agreement [0, 1] — maps discrepancy to agreement score
        if depth_discrepancy < 0.01:
            c_depth = 1.0
        elif depth_discrepancy < self.config.depth_tolerance_m:
            c_depth = 0.8
        elif depth_discrepancy < 0.1:
            c_depth = 0.5
        else:
            c_depth = 0.2

        confidence = (
            cfg.w_segmentation * c_seg +
            cfg.w_reconstruction * c_recon +
            cfg.w_tagging * c_tag +
            cfg.w_depth * c_depth
        )

        return round(min(max(confidence, 0.0), 1.0), 3)

    @staticmethod
    def _camera_to_world(pos_camera: np.ndarray,
                         extrinsics: np.ndarray) -> np.ndarray:
        """
        Transform a position from camera optical frame to world frame.

        The depth projector outputs coords in OpenCV optical convention:
          x = right, y = down, z = forward

        MuJoCo camera convention:
          x = right, y = up, z = backward (out of screen)

        The extrinsics matrix encodes: cam_pos + cam_rot @ flip @ pos_optical
        where flip converts optical → MuJoCo camera, and cam_rot maps to world.

        Args:
            pos_camera: (3,) position in optical frame [x_right, y_down, z_fwd]
            extrinsics: (4, 4) camera-to-world homogeneous transform
                        (built as rot @ flip, with flip= diag(1,-1,-1))

        Returns:
            (3,) position in world frame
        """
        pos_h = np.array([pos_camera[0], pos_camera[1], pos_camera[2], 1.0])
        pos_world = extrinsics @ pos_h
        return pos_world[:3]
