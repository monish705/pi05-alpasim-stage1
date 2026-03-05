"""
perception/reconstructor_3d.py
==============================
SAM3D Objects wrapper for single-image 3D reconstruction.
Takes an image + mask (from SAM3) and produces 3D mesh + pose.

Uses the REAL Meta SAM3D Objects model:
  - Checkpoints from HuggingFace (facebook/sam-3d-objects)
  - Inference via notebook/inference.py from the sam-3d-objects repo
  - Outputs: Gaussian splat, mesh, camera-relative transform

Falls back to depth-only position estimation when SAM3D is unavailable.

Requires: facebookresearch/sam-3d-objects cloned to /content/sam3d
          + checkpoints downloaded via `hf download facebook/sam-3d-objects`
"""
import os
import sys
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Reconstruction3D:
    """3D reconstruction output from SAM3D Objects."""
    mesh_vertices: Optional[np.ndarray] = None   # (N, 3)
    mesh_faces: Optional[np.ndarray] = None      # (M, 3)
    gaussian_splat: object = None                 # GaussianSplat object
    rotation: Optional[np.ndarray] = None         # (3, 3) rotation matrix
    translation: Optional[np.ndarray] = None      # (3,) translation vector
    scale: Optional[float] = None                 # scale factor
    bbox_3d: Optional[np.ndarray] = None          # (6,) [x,y,z, w,h,d]
    success: bool = False


# Check SAM3D availability — look for the inference module in the cloned repo
_SAM3D_AVAILABLE = False
_SAM3D_PATH = None

# Search common locations for the sam3d repo
for candidate in ["/content/sam3d", os.path.expanduser("~/sam3d"),
                  os.path.join(os.getcwd(), "sam3d")]:
    if os.path.isfile(os.path.join(candidate, "notebook", "inference.py")):
        _SAM3D_PATH = candidate
        break

if _SAM3D_PATH:
    try:
        sys.path.insert(0, os.path.join(_SAM3D_PATH, "notebook"))
        from inference import Inference as _SAM3DInference
        _SAM3D_AVAILABLE = True
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[SAM3D] Import failed: {e}")


class SAM3DReconstructor:
    """
    Wraps Meta's SAM3D Objects for per-object 3D reconstruction.
    
    REAL model pipeline:
      Image + Mask → SAM3D Inference → Gaussian Splat + Mesh + Pose
    
    Fallback when SAM3D unavailable:
      Returns empty Reconstruction3D(success=False)
      Pipeline continues with depth-only position estimation.
    """

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: path to SAM3D pipeline.yaml config.
                         Auto-detected from checkpoints/hf/pipeline.yaml
        """
        self.config_path = config_path
        self.inference = None
        self._loaded = False

        # Auto-detect config path
        if self.config_path is None and _SAM3D_PATH:
            # Check for HF checkpoints
            hf_config = os.path.join(_SAM3D_PATH, "checkpoints", "hf", "pipeline.yaml")
            if os.path.exists(hf_config):
                self.config_path = hf_config

    @staticmethod
    def is_available() -> bool:
        """Check if SAM3D Objects is installed and checkpoints are present."""
        return _SAM3D_AVAILABLE

    @staticmethod
    def checkpoints_ready() -> bool:
        """Check if SAM3D checkpoints have been downloaded."""
        if not _SAM3D_PATH:
            return False
        return os.path.exists(
            os.path.join(_SAM3D_PATH, "checkpoints", "hf", "pipeline.yaml")
        )

    @staticmethod
    def download_checkpoints(hf_token: str = None):
        """
        Download SAM3D checkpoints from HuggingFace.
        Requires HF authentication (token with access to facebook/sam-3d-objects).
        """
        if not _SAM3D_PATH:
            print("[SAM3D] ❌ sam-3d-objects repo not found. Clone it first:")
            print("  git clone https://github.com/facebookresearch/sam-3d-objects.git /content/sam3d")
            return False

        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            import shutil

            print("[SAM3D] Downloading checkpoints from facebook/sam-3d-objects...")
            dl_path = os.path.join(_SAM3D_PATH, "checkpoints", "hf-download")
            snapshot_download(
                repo_id="facebook/sam-3d-objects",
                repo_type="model",
                local_dir=dl_path,
                token=hf_token,
            )
            # Move checkpoints to expected location
            src = os.path.join(dl_path, "checkpoints")
            dst = os.path.join(_SAM3D_PATH, "checkpoints", "hf")
            if os.path.exists(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
            print("[SAM3D] ✅ Checkpoints downloaded")
            return True
        except Exception as e:
            print(f"[SAM3D] ❌ Checkpoint download failed: {e}")
            return False

    def load(self):
        """Load SAM3D Objects model. Call once."""
        if not _SAM3D_AVAILABLE:
            print("[SAM3D] Not installed — 3D reconstruction disabled, "
                  "using depth-only position estimation")
            return

        if self.config_path is None or not os.path.exists(self.config_path):
            print(f"[SAM3D] ❌ Checkpoints not found at {self.config_path}")
            print("[SAM3D] Run SAM3DReconstructor.download_checkpoints(hf_token) first")
            return

        from inference import Inference

        print("[SAM3D] Loading model (this may take 30-60s on first load)...")
        self.inference = Inference(self.config_path, compile=False)
        self._loaded = True
        print("[SAM3D] ✅ Model loaded — REAL 3D reconstruction active")

    def reconstruct(self, image: Image.Image, mask: np.ndarray,
                    seed: int = 42) -> Reconstruction3D:
        """
        Reconstruct 3D from a single image + mask using REAL SAM3D.

        Args:
            image: PIL Image (full scene)
            mask: (H, W) binary mask of the target object
            seed: random seed for deterministic output

        Returns:
            Reconstruction3D with mesh, pose, gaussian splat
        """
        if not _SAM3D_AVAILABLE:
            return Reconstruction3D(success=False)

        if not self._loaded:
            self.load()
            if not self._loaded:
                return Reconstruction3D(success=False)

        try:
            # Run SAM3D inference — the real model
            output = self.inference(image, mask, seed=seed)

            result = Reconstruction3D(success=True)

            # Gaussian splat
            if "gs" in output:
                result.gaussian_splat = output["gs"]

            # Mesh (extract vertices/faces)
            if "mesh" in output and output["mesh"] is not None:
                mesh = output["mesh"]
                if hasattr(mesh, 'vertices'):
                    result.mesh_vertices = np.array(mesh.vertices)
                if hasattr(mesh, 'faces'):
                    result.mesh_faces = np.array(mesh.faces)

                # Compute 3D bounding box from vertices
                if result.mesh_vertices is not None and len(result.mesh_vertices) > 0:
                    mins = result.mesh_vertices.min(axis=0)
                    maxs = result.mesh_vertices.max(axis=0)
                    center = (mins + maxs) / 2
                    dims = maxs - mins
                    result.bbox_3d = np.concatenate([center, dims])
                    result.translation = center

            # Camera-relative transform
            if "transform" in output:
                xform = output["transform"]
                if hasattr(xform, 'rotation'):
                    result.rotation = np.array(xform.rotation)
                if hasattr(xform, 'translation'):
                    result.translation = np.array(xform.translation)
                if hasattr(xform, 'scale'):
                    result.scale = float(xform.scale)

            return result

        except Exception as e:
            print(f"[SAM3D] Reconstruction failed: {e}")
            return Reconstruction3D(success=False)

    def reconstruct_batch(self, image: Image.Image,
                          masks: list, seed: int = 42) -> list:
        """Reconstruct 3D for multiple masks from the same image."""
        if not _SAM3D_AVAILABLE:
            return [Reconstruction3D(success=False) for _ in masks]

        results = []
        for i, mask in enumerate(masks):
            print(f"[SAM3D] Reconstructing object {i+1}/{len(masks)}...")
            result = self.reconstruct(image, mask, seed=seed)
            results.append(result)
        return results
