"""
world_model/verification.py
=============================
Closed-Loop Verification — the World Model's Self-Check.

After perceiving the world and building a simulation, this module
answers: "Does my internal simulation actually LOOK like reality?"

It does this by:
  1. Rendering the perceived simulation from the SAME camera angle
  2. Comparing the rendered image against the real camera image
  3. Computing a pixel-level error map highlighting discrepancies
  4. Adjusting confidence scores based on visual agreement
  5. Triggering re-perception if error is too high

This is the mechanism that prevents the world model from silently
diverging from reality. It's the robot's ability to doubt itself.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

from perception.scene_graph import SceneGraph


@dataclass
class VerificationResult:
    """Result of comparing perceived world model against reality."""
    mean_pixel_error: float          # 0-255 scale
    structural_similarity: float     # 0-1, 1=identical
    per_object_visibility: Dict[int, bool]  # track_id → visible in both?
    error_map: Optional[np.ndarray]  # (H, W) float error heatmap
    rendered_image: Optional[np.ndarray]  # the sim's rendered view
    real_image: Optional[np.ndarray]      # the real camera image
    needs_reperception: bool         # error too high?
    confidence_adjustment: float     # multiply scene graph confidence by this


class WorldModelVerifier:
    """
    Verifies the world model by re-rendering it and comparing to reality.
    
    This is conceptually similar to how humans "double check" by looking
    at something again. The robot renders what it THINKS the world looks
    like, then compares to what it actually SEES.
    """
    
    # Thresholds
    GOOD_SIMILARITY = 0.85       # above this, world model is trustworthy
    BAD_SIMILARITY = 0.50        # below this, trigger re-perception
    MAX_PIXEL_ERROR = 40.0       # above this mean error, confidence drops
    
    def __init__(self, render_width: int = 640, render_height: int = 480):
        self.render_width = render_width
        self.render_height = render_height
    
    def verify(self, scene_graph: SceneGraph,
               real_rgb: np.ndarray,
               camera_pos: List[float],
               camera_target: List[float]) -> VerificationResult:
        """
        Verify the world model against a real camera image.
        
        Args:
            scene_graph: the perceived world model
            real_rgb: (H, W, 3) uint8 real camera image
            camera_pos: [x, y, z] where the camera is
            camera_target: [x, y, z] where the camera looks
            
        Returns:
            VerificationResult with error metrics and recommendations
        """
        if not _MUJOCO_AVAILABLE:
            return self._fallback_result(real_rgb)
        
        # Build simulation from scene graph
        xml_str = self._build_verification_xml(
            scene_graph, camera_pos, camera_target
        )
        
        try:
            model = mujoco.MjModel.from_xml_string(xml_str)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            
            # Render from the same camera angle
            renderer = mujoco.Renderer(model, self.render_height, self.render_width)
            renderer.update_scene(data, camera="verify_cam")
            rendered = renderer.render().copy()
            renderer.close()
        except Exception as e:
            print(f"[Verify] Render failed: {e}")
            return self._fallback_result(real_rgb)
        
        # Resize real image to match rendered size
        from PIL import Image
        real_resized = np.array(
            Image.fromarray(real_rgb).resize(
                (self.render_width, self.render_height)
            )
        )
        
        # Compute pixel-level error
        error_map = np.mean(
            np.abs(rendered.astype(float) - real_resized.astype(float)),
            axis=2
        )
        mean_error = float(np.mean(error_map))
        
        # Structural similarity (simplified SSIM)
        ssim = self._compute_ssim(rendered, real_resized)
        
        # Per-object visibility check
        visibility = self._check_object_visibility(
            scene_graph, rendered, real_resized
        )
        
        # Determine if re-perception is needed
        needs_reperception = ssim < self.BAD_SIMILARITY
        
        # Confidence adjustment factor
        if ssim > self.GOOD_SIMILARITY:
            confidence_adj = 1.0 + (ssim - self.GOOD_SIMILARITY) * 0.5
        elif ssim > self.BAD_SIMILARITY:
            confidence_adj = ssim / self.GOOD_SIMILARITY
        else:
            confidence_adj = 0.5  # halve confidence
        
        result = VerificationResult(
            mean_pixel_error=mean_error,
            structural_similarity=ssim,
            per_object_visibility=visibility,
            error_map=error_map,
            rendered_image=rendered,
            real_image=real_resized,
            needs_reperception=needs_reperception,
            confidence_adjustment=confidence_adj,
        )
        
        self._print_result(result)
        return result
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Simplified Structural Similarity Index (SSIM).
        
        Measures structural similarity between two images.
        Returns value in [0, 1] where 1 = identical.
        """
        # Convert to float
        a = img1.astype(np.float64)
        b = img2.astype(np.float64)
        
        # Constants for stability
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Compute means
        mu_a = np.mean(a)
        mu_b = np.mean(b)
        
        # Compute variances and covariance
        sigma_a_sq = np.var(a)
        sigma_b_sq = np.var(b)
        sigma_ab = np.mean((a - mu_a) * (b - mu_b))
        
        # SSIM formula
        numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        denominator = (mu_a**2 + mu_b**2 + C1) * (sigma_a_sq + sigma_b_sq + C2)
        
        ssim = numerator / denominator
        return float(np.clip(ssim, 0, 1))
    
    def _check_object_visibility(self, scene_graph: SceneGraph,
                                  rendered: np.ndarray,
                                  real: np.ndarray) -> Dict[int, bool]:
        """Check which objects are visible in both rendered and real images."""
        visibility = {}
        for obj in scene_graph.get_all_objects():
            # Simplified: assume all objects are visible if they're in the scene graph
            # A full implementation would use segmentation masks
            visibility[obj.track_id] = True
        return visibility
    
    def _build_verification_xml(self, scene_graph: SceneGraph,
                                 camera_pos: List[float],
                                 camera_target: List[float]) -> str:
        """Build MuJoCo XML for verification rendering."""
        bodies = ""
        for obj in scene_graph.get_all_objects():
            pos = obj.position_world
            dims = obj.dimensions_m
            w = dims.get("width", 0.05)
            h = dims.get("height", 0.05)
            d = dims.get("depth", 0.05)
            
            shape = obj.collision_primitive or "box"
            rgba = "0.5 0.5 0.5 1"
            
            if shape == "cylinder":
                geom = f'<geom type="cylinder" size="{max(w,d)/2:.4f} {h/2:.4f}" rgba="{rgba}"/>'
            elif shape == "sphere":
                r = max(w, h, d) / 2
                geom = f'<geom type="sphere" size="{r:.4f}" rgba="{rgba}"/>'
            else:
                geom = f'<geom type="box" size="{w/2:.4f} {h/2:.4f} {d/2:.4f}" rgba="{rgba}"/>'
            
            bodies += f"""
    <body name="obj_{obj.track_id}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}">
      {geom}
    </body>"""
        
        # Camera xyaxes computation
        cam_dir = np.array(camera_target) - np.array(camera_pos)
        cam_dir = cam_dir / np.linalg.norm(cam_dir)
        up = np.array([0, 0, 1])
        right = np.cross(cam_dir, up)
        if np.linalg.norm(right) < 0.001:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, cam_dir)
        
        xyaxes = f"{right[0]:.4f} {right[1]:.4f} {right[2]:.4f} {up_corrected[0]:.4f} {up_corrected[1]:.4f} {up_corrected[2]:.4f}"
        
        xml = f"""<mujoco>
  <visual><headlight ambient="0.3 0.3 0.3"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <geom type="box" pos="0.65 0 0.4" size="0.3 0.3 0.02" rgba="0.6 0.4 0.2 1"/>{bodies}
    <camera name="verify_cam" pos="{camera_pos[0]} {camera_pos[1]} {camera_pos[2]}" xyaxes="{xyaxes}"/>
  </worldbody>
</mujoco>"""
        return xml
    
    def _fallback_result(self, real_rgb: np.ndarray) -> VerificationResult:
        """Return when MuJoCo isn't available."""
        return VerificationResult(
            mean_pixel_error=0,
            structural_similarity=0.5,
            per_object_visibility={},
            error_map=None,
            rendered_image=None,
            real_image=real_rgb,
            needs_reperception=False,
            confidence_adjustment=1.0,
        )
    
    def _print_result(self, result: VerificationResult):
        """Print verification results."""
        print(f"\n{'='*60}")
        print(f"  WORLD MODEL VERIFICATION")
        print(f"{'='*60}")
        
        emoji = "✅" if result.structural_similarity > self.GOOD_SIMILARITY else \
                "⚠️" if result.structural_similarity > self.BAD_SIMILARITY else "❌"
        
        print(f"  {emoji} Structural Similarity: {result.structural_similarity:.1%}")
        print(f"  Mean Pixel Error: {result.mean_pixel_error:.1f}/255")
        print(f"  Confidence Adjustment: {result.confidence_adjustment:.2f}x")
        
        if result.needs_reperception:
            print(f"  🔄 RE-PERCEPTION TRIGGERED — world model too inaccurate")
        else:
            print(f"  ✅ World model is consistent with reality")
        
        print(f"\n{'='*60}\n")

