"""
perception/depth_validator.py
=============================
Cross-checks SAM3D's single-image 3D estimates against actual depth sensor data.
Catches errors in 3D reconstruction and provides corrected world-frame positions.

v2 changes:
  - Shape validation: checks depth contour against claimed collision primitive
  - Dimension validation: cross-checks VLM vs depth measurements, returns agreement
  - Ground plane detection: identifies table/floor surface from depth map
"""
import numpy as np
from typing import Dict, Optional, Tuple


class DepthValidator:
    """
    Validates and corrects 3D position estimates using depth sensor data.

    SAM3D reconstructs 3D from a single RGB image (monocular estimation).
    The depth sensor provides ground-truth distance measurements.
    When they disagree, we trust the depth sensor.
    """

    def __init__(self, tolerance_m: float = 0.05):
        """
        Args:
            tolerance_m: maximum allowed discrepancy in metres before
                         overriding SAM3D's estimate with depth-based position
        """
        self.tolerance = tolerance_m

    def validate_position(
        self,
        sam3d_position: np.ndarray,     # camera-relative [x, y, z]
        mask: np.ndarray,               # (H, W) bool mask
        depth_map: np.ndarray,          # (H, W) float depth
        intrinsics: Dict[str, float],   # fx, fy, cx, cy
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Compare SAM3D position against depth-projected position.

        Returns:
            (corrected_position, was_corrected, discrepancy_m)
        """
        # Get depth-based position from mask centroid
        depth_position = self._depth_project(mask, depth_map, intrinsics)

        if depth_position is None:
            # Depth map invalid at mask location — trust SAM3D
            return sam3d_position, False, 0.0

        # Compare distances
        discrepancy = np.linalg.norm(sam3d_position - depth_position)

        if discrepancy > self.tolerance:
            print(f"[DEPTH] Position corrected: SAM3D={sam3d_position} → "
                  f"Depth={depth_position} (Δ={discrepancy:.3f}m)")
            return depth_position, True, discrepancy

        return sam3d_position, False, discrepancy

    def measure_object_dimensions(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        """
        Estimate real-world object dimensions from depth contour.

        Uses the mask boundary + depth values to compute width/height/depth
        in metres. Cross-references VLM's dimension estimates.
        """
        if mask.sum() < 10:
            return None

        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        # Get mask bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Average depth within mask
        masked_depth = depth_map[mask]
        valid_depth = masked_depth[masked_depth > 0]
        if len(valid_depth) == 0:
            return None
        avg_depth = np.median(valid_depth)

        # Convert pixel span to metres using pinhole model
        width_m = (x_max - x_min) * avg_depth / fx
        height_m = (y_max - y_min) * avg_depth / fy

        # Depth extent (front-to-back) from depth variance within mask
        if len(valid_depth) > 1:
            depth_extent_m = float(valid_depth.max() - valid_depth.min())
        else:
            depth_extent_m = width_m  # assume roughly cubic

        return {
            "width": float(width_m),
            "height": float(height_m),
            "depth": float(depth_extent_m),
        }

    def validate_dimensions(
        self,
        vlm_dims_m: Dict[str, float],
        depth_dims_m: Dict[str, float],
        tolerance_ratio: float = 0.5,
    ) -> Tuple[Dict[str, float], float]:
        """
        Cross-check VLM-estimated dimensions against depth-measured dimensions.

        Returns:
            (best_dims, agreement_score)
            agreement_score: 1.0 = perfect match, 0.0 = completely off
        """
        if not vlm_dims_m or not depth_dims_m:
            return depth_dims_m or vlm_dims_m or {}, 0.5

        agreements = []
        best_dims = {}

        for key in ["width", "height", "depth"]:
            v = vlm_dims_m.get(key, 0.05)
            d = depth_dims_m.get(key, 0.05)

            if v <= 0 or d <= 0:
                best_dims[key] = max(v, d, 0.01)
                agreements.append(0.5)
                continue

            ratio = min(v, d) / max(v, d)
            agreements.append(ratio)

            if ratio > tolerance_ratio:
                # Good agreement — average them
                best_dims[key] = (v + d) / 2
            else:
                # Poor agreement — trust depth
                best_dims[key] = d

        agreement = sum(agreements) / len(agreements) if agreements else 0.5
        return best_dims, agreement

    def validate_shape(
        self,
        mask: np.ndarray,
        claimed_shape: str,
    ) -> Tuple[str, float]:
        """
        Validate the claimed collision shape against the mask contour.

        Compares the mask shape to expected shapes:
          - cylinder/sphere: high circularity
          - box: low circularity, rectangular aspect ratio
          - capsule: elongated with rounded ends

        Returns:
            (suggested_shape, shape_confidence)
        """
        if mask.sum() < 50:
            return claimed_shape, 0.5

        # Compute mask properties
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return claimed_shape, 0.5

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        bbox_area = bbox_w * bbox_h
        mask_area = mask.sum()

        # Circularity: mask area / bbox area (1.0 = fills bbox, π/4 ≈ 0.785 for circle)
        fill_ratio = mask_area / max(bbox_area, 1)
        aspect_ratio = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)

        # Classify shape
        if fill_ratio > 0.7 and aspect_ratio < 1.3:
            suggested = "sphere"
            conf = fill_ratio
        elif fill_ratio > 0.6 and aspect_ratio > 1.5:
            suggested = "cylinder"
            conf = 0.7
        elif fill_ratio > 0.5 and aspect_ratio > 2.0:
            suggested = "capsule"
            conf = 0.6
        elif fill_ratio > 0.75:
            suggested = "box"
            conf = fill_ratio * 0.9
        else:
            suggested = "box"  # default
            conf = 0.5

        # If claimed shape matches our suggestion, boost confidence
        if claimed_shape == suggested:
            conf = min(1.0, conf + 0.2)

        return suggested, conf

    def detect_ground_plane(
        self,
        depth_map: np.ndarray,
        intrinsics: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        """
        Detect the dominant horizontal surface (table/floor) from the depth map.

        Returns:
            dict with 'height_m', 'normal', 'confidence' or None if not found
        """
        if depth_map.size == 0:
            return None

        valid = (depth_map > 0.1) & (depth_map < 5.0)
        if valid.sum() < 100:
            return None

        valid_depths = depth_map[valid]

        # Find the most common depth band (table surface)
        hist, edges = np.histogram(valid_depths, bins=100)
        peak_idx = hist.argmax()
        surface_depth = (edges[peak_idx] + edges[peak_idx + 1]) / 2
        surface_count = hist[peak_idx]

        # Get the 3D height of this surface
        # Assuming camera looks roughly downward, surface depth ≈ height
        fy = intrinsics["fy"]
        cy = intrinsics["cy"]
        h = depth_map.shape[0]

        # Pixels at surface depth
        surface_mask = (depth_map > surface_depth - 0.02) & \
                       (depth_map < surface_depth + 0.02)
        if surface_mask.sum() < 50:
            return None

        ys, _ = np.where(surface_mask)
        avg_y = ys.mean()

        # Height in camera frame
        height_m = (avg_y - cy) * surface_depth / fy

        confidence = min(surface_count / valid.sum() * 3, 1.0)

        return {
            "height_m": float(height_m),
            "depth_m": float(surface_depth),
            "confidence": float(confidence),
        }

    def _depth_project(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: Dict[str, float],
    ) -> Optional[np.ndarray]:
        """
        Project mask centroid through depth map to get 3D position
        in camera optical frame.

        Returns position in OpenCV/optical convention:
          x = right, y = down, z = forward (into scene)

        The pipeline._camera_to_world() then transforms this to world frame
        using extrinsics that include the MuJoCo→OpenCV flip.
        """
        if mask.sum() == 0:
            return None

        # Centroid of mask
        ys, xs = np.where(mask)
        cx_pixel = xs.mean()
        cy_pixel = ys.mean()

        # Depth at centroid (use median of masked pixels for robustness)
        masked_depth = depth_map[mask]
        valid = masked_depth[masked_depth > 0]
        if len(valid) == 0:
            return None
        z = float(np.median(valid))

        # Unproject using pinhole model → camera optical frame
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        x = (cx_pixel - cx) * z / fx
        y = (cy_pixel - cy) * z / fy

        return np.array([x, y, z])
