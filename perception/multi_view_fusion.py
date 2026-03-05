"""
perception/multi_view_fusion.py
================================
Multi-View Fusion: fuses perception results from N cameras into a
single unified scene graph with corrected 3D bounding boxes.

Solves the single-view limitation where depth (thickness going away
from camera) must be guessed. By combining views from multiple angles,
true 3D dimensions are computed from intersected observations.

Pipeline:
  For each camera:
    1. Run SAM3 → segments
    2. Depth-project centroids → 3D positions (in world frame)
    3. Run VLM → labels + physics
  Then:
    4. Match objects across cameras (Hungarian on 3D centroids)
    5. Fuse positions (weighted average by confidence)
    6. Fuse dimensions (union of per-view measurements)
    7. Vote on labels (majority wins)
    8. Build unified SceneGraph
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from perception.scene_graph import SceneGraph, SceneObject
import uuid


@dataclass
class CameraObservation:
    """A single object observed from a single camera."""
    camera_name: str
    position_world: np.ndarray       # (3,) world-frame position
    dimensions_m: Dict[str, float]   # width, height, depth
    label: str
    collision_shape: str
    confidence: float
    mass_kg: float
    friction: float
    material: str
    is_graspable: bool
    is_deformable: bool
    crop: object = None              # PIL Image crop for color extraction
    mask: np.ndarray = None          # for point cloud extraction


@dataclass
class FusedObject:
    """An object reconstructed from multiple camera views."""
    observations: List[CameraObservation] = field(default_factory=list)
    fused_position: np.ndarray = None
    fused_dimensions: Dict[str, float] = None
    fused_label: str = ""
    fused_shape: str = "box"
    fused_confidence: float = 0.0
    num_views: int = 0


class MultiViewFusion:
    """
    Fuses per-camera SceneObjects into a unified world model.

    Algorithm:
      1. Collect all objects from all cameras (each has world-frame position)
      2. Build a cross-camera cost matrix based on 3D distance
      3. Hungarian matching to find same-object-across-views
      4. For each matched cluster, fuse position/dims/labels
      5. Output a single unified SceneGraph
    """

    def __init__(self, merge_threshold_m: float = 0.15):
        """
        Args:
            merge_threshold_m: max 3D distance to consider two observations
                               as the same object across cameras
        """
        self.merge_threshold = merge_threshold_m

    def fuse(self, per_camera_objects: Dict[str, List[SceneObject]],
             per_camera_segments: Dict[str, list] = None) -> SceneGraph:
        """
        Fuse objects from multiple cameras into a single SceneGraph.

        Args:
            per_camera_objects: {camera_name: [SceneObject, ...]}
            per_camera_segments: optional {camera_name: [SegmentationResult, ...]}
                                 for color/crop extraction

        Returns:
            Unified SceneGraph with fused objects
        """
        # Collect all observations
        all_obs = []
        for cam_name, objects in per_camera_objects.items():
            segs = per_camera_segments.get(cam_name, []) if per_camera_segments else []
            for i, obj in enumerate(objects):
                obs = CameraObservation(
                    camera_name=cam_name,
                    position_world=np.array(obj.position_world),
                    dimensions_m=dict(obj.dimensions_m),
                    label=obj.label,
                    collision_shape=obj.collision_primitive,
                    confidence=obj.confidence,
                    mass_kg=obj.mass_kg,
                    friction=obj.friction,
                    material=obj.material,
                    is_graspable=obj.is_graspable,
                    is_deformable=obj.is_deformable,
                    crop=segs[i].crop if i < len(segs) else None,
                )
                all_obs.append(obs)

        if not all_obs:
            return SceneGraph()

        print(f"\n[MultiView] Fusing {len(all_obs)} observations from "
              f"{len(per_camera_objects)} cameras")

        # Cluster observations into objects
        clusters = self._cluster_observations(all_obs)
        print(f"[MultiView] Found {len(clusters)} unique objects across views")

        # Fuse each cluster
        fused_objects = []
        for cluster in clusters:
            fused = self._fuse_cluster(cluster)
            fused_objects.append(fused)

        # Build unified scene graph
        sg = SceneGraph()
        for fused in fused_objects:
            obj = SceneObject(
                id=str(uuid.uuid4())[:8],
                label=fused.fused_label,
                position_world=fused.fused_position.tolist(),
                collision_primitive=fused.fused_shape,
                dimensions_m=fused.fused_dimensions,
                mass_kg=np.mean([o.mass_kg for o in fused.observations]),
                friction=np.mean([o.friction for o in fused.observations]),
                material=self._vote_string([o.material for o in fused.observations]),
                is_graspable=any(o.is_graspable for o in fused.observations),
                is_deformable=any(o.is_deformable for o in fused.observations),
                confidence=fused.fused_confidence,
            )
            sg.add_object(obj)

        # Compute spatial relationships
        sg._compute_relationships()
        sg.frame_count = 1

        # Print fusion report
        self._print_report(fused_objects)

        return sg

    def _cluster_observations(self, observations: List[CameraObservation]
                              ) -> List[List[CameraObservation]]:
        """
        Cluster observations from different cameras into same-object groups.

        Uses greedy nearest-neighbor clustering on 3D positions.
        Two observations are merged if:
          - They are from DIFFERENT cameras
          - Their 3D distance < merge_threshold
        """
        used = [False] * len(observations)
        clusters = []

        # Sort by confidence descending (seed clusters with best observations)
        indices = sorted(range(len(observations)),
                         key=lambda i: observations[i].confidence, reverse=True)

        for seed_idx in indices:
            if used[seed_idx]:
                continue

            cluster = [observations[seed_idx]]
            used[seed_idx] = True
            seed_pos = observations[seed_idx].position_world
            cameras_in_cluster = {observations[seed_idx].camera_name}

            # Find matches from other cameras
            for j in indices:
                if used[j]:
                    continue
                obs_j = observations[j]

                # Must be from a different camera
                if obs_j.camera_name in cameras_in_cluster:
                    continue

                dist = np.linalg.norm(obs_j.position_world - seed_pos)
                if dist < self.merge_threshold:
                    cluster.append(obs_j)
                    used[j] = True
                    cameras_in_cluster.add(obs_j.camera_name)

            clusters.append(cluster)

        return clusters

    def _fuse_cluster(self, cluster: List[CameraObservation]) -> FusedObject:
        """Fuse a cluster of observations into a single object."""
        fused = FusedObject(observations=cluster, num_views=len(cluster))

        # --- Position: confidence-weighted average ---
        weights = np.array([o.confidence for o in cluster])
        if weights.sum() == 0:
            weights = np.ones(len(cluster))
        weights = weights / weights.sum()

        positions = np.array([o.position_world for o in cluster])
        fused.fused_position = np.average(positions, axis=0, weights=weights)

        # --- Dimensions: cross-view fusion ---
        fused.fused_dimensions = self._fuse_dimensions(cluster)

        # --- Label: majority vote ---
        fused.fused_label = self._vote_string([o.label for o in cluster])

        # --- Shape: majority vote ---
        fused.fused_shape = self._vote_string([o.collision_shape for o in cluster])

        # --- Confidence: boost for multi-view (more views = more confident) ---
        base_conf = max(o.confidence for o in cluster)
        view_bonus = min(0.1 * (len(cluster) - 1), 0.15)  # up to 15% boost
        fused.fused_confidence = min(base_conf + view_bonus, 1.0)

        return fused

    def _fuse_dimensions(self, cluster: List[CameraObservation]
                         ) -> Dict[str, float]:
        """
        Fuse dimensions from multiple views.

        Key insight: each camera accurately measures width and height
        (perpendicular to its view axis) but poorly estimates depth
        (along its view axis). Multi-view fusion resolves this by
        using each camera's accurate measurements.

        Strategy: for each dimension, use the MEDIAN across views.
        This is more robust than mean (avoids outliers from bad depth).
        """
        widths = [o.dimensions_m.get("width", 0.05) for o in cluster]
        heights = [o.dimensions_m.get("height", 0.05) for o in cluster]
        depths = [o.dimensions_m.get("depth", 0.05) for o in cluster]

        return {
            "width": float(np.median(widths)),
            "height": float(np.median(heights)),
            "depth": float(np.median(depths)),
        }

    @staticmethod
    def _vote_string(candidates: List[str]) -> str:
        """Majority vote on string values. Ignores 'unknown'."""
        filtered = [c for c in candidates if c and c != "unknown"]
        if not filtered:
            return candidates[0] if candidates else "unknown"

        from collections import Counter
        counts = Counter(filtered)
        return counts.most_common(1)[0][0]

    def _print_report(self, fused_objects: List[FusedObject]):
        """Print a summary of the fusion results."""
        print(f"\n{'='*60}")
        print(f"  MULTI-VIEW FUSION REPORT")
        print(f"{'='*60}")
        for i, fused in enumerate(fused_objects):
            pos = fused.fused_position
            dims = fused.fused_dimensions
            cams = [o.camera_name for o in fused.observations]
            print(f"\n  [{i+1}] {fused.fused_label}")
            print(f"      Views: {len(cams)} cameras — {', '.join(cams)}")
            print(f"      Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            print(f"      Dimensions: {dims['width']*100:.1f}×"
                  f"{dims['height']*100:.1f}×{dims['depth']*100:.1f} cm")
            print(f"      Shape: {fused.fused_shape} | "
                  f"Conf: {fused.fused_confidence:.2f}")

            # Show per-view position spread
            if len(fused.observations) > 1:
                positions = np.array([o.position_world for o in fused.observations])
                spread = np.max(positions, axis=0) - np.min(positions, axis=0)
                print(f"      View spread: Δx={spread[0]*100:.1f}cm, "
                      f"Δy={spread[1]*100:.1f}cm, Δz={spread[2]*100:.1f}cm")
        print(f"\n{'='*60}\n")
