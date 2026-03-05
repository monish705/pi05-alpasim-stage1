"""
perception/segmentor.py
=======================
SAM3 (Segment Anything with Concepts) wrapper for 2D segmentation.
Produces pixel-level masks for each detected object in an RGB image.

v2: Graceful degradation — falls back to depth-based connected components
    if SAM3 is not installed. Both paths produce the same SegmentationResult.

Requires (optional): facebookresearch/sam3 (Python 3.12+, PyTorch 2.7+, CUDA 12.6+)
"""
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SegmentationResult:
    mask: np.ndarray         # (H, W) bool mask
    bbox_2d: List[float]     # [x1, y1, x2, y2]
    score: float             # confidence score
    crop: Image.Image        # cropped RGB region
    prompt: Optional[str] = None
    method: str = "sam3"     # "sam3" or "depth_fallback"


# Check SAM3 availability at import time
_SAM3_AVAILABLE = False
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    _SAM3_AVAILABLE = True
except ImportError:
    pass


class SAM3Segmentor:
    """
    Wraps Meta's SAM3 for text-prompted or automatic segmentation.
    Falls back to depth-based segmentation if SAM3 is not available.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self._loaded = False

    @staticmethod
    def is_available() -> bool:
        """Check if SAM3 is installed."""
        return _SAM3_AVAILABLE

    def load(self):
        """Load SAM3 model. Call once before inference."""
        if not _SAM3_AVAILABLE:
            print("[SAM3] Not installed — using depth fallback segmentor")
            return

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print("[SAM3] Loading model (downloading checkpoint from HuggingFace)...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        self._loaded = True
        print("[SAM3] ✅ Model loaded — REAL segmentation active")

    def segment_with_text(self, image: Image.Image, prompt: str,
                          score_threshold: float = 0.3) -> List[SegmentationResult]:
        """
        Segment objects matching a text prompt.

        Args:
            image: PIL Image (RGB)
            prompt: text description e.g. "red mug", "table"
            score_threshold: minimum confidence to keep

        Returns:
            List of SegmentationResult with masks, boxes, crops
        """
        if not _SAM3_AVAILABLE:
            return []

        if not self._loaded:
            self.load()

        state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=state, prompt=prompt)

        masks = output["masks"]   # tensor of masks
        boxes = output["boxes"]   # bounding boxes
        scores = output["scores"] # confidence scores

        results = []
        img_array = np.array(image)

        for i in range(len(scores)):
            score = float(scores[i])
            if score < score_threshold:
                continue

            mask = masks[i].cpu().numpy().astype(bool)
            if mask.ndim == 3:
                mask = mask[0]  # take first channel if multi-dim

            box = boxes[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(b) for b in box]

            # Crop the RGB region
            crop = image.crop((x1, y1, x2, y2))

            results.append(SegmentationResult(
                mask=mask,
                bbox_2d=[x1, y1, x2, y2],
                score=score,
                crop=crop,
                prompt=prompt,
                method="sam3",
            ))

        return results

    def segment_all(self, image: Image.Image,
                    prompts: List[str] = None,
                    score_threshold: float = 0.3,
                    depth: np.ndarray = None) -> List[SegmentationResult]:
        """
        Segment all objects of interest.

        Args:
            image: PIL Image (RGB)
            prompts: list of text prompts to try. If None, uses generic prompts
            score_threshold: minimum confidence
            depth: optional depth map for fallback segmentation

        Returns:
            All detected objects across all prompts, deduplicated
        """
        if not _SAM3_AVAILABLE:
            # Fallback to depth-based segmentation
            if depth is not None:
                return self._segment_from_depth(image, depth)
            return []

        if prompts is None:
            prompts = [
                "cup", "mug", "bowl", "bottle", "box", "cube",
                "ball", "sphere", "cylinder", "container", "table"
            ]

        all_results = []
        for prompt in prompts:
            results = self.segment_with_text(image, prompt, score_threshold)
            all_results.extend(results)

        # Deduplicate by IoU
        return self._deduplicate(all_results, iou_threshold=0.5)

    def _segment_from_depth(self, image: Image.Image,
                            depth: np.ndarray) -> List[SegmentationResult]:
        """
        Fallback segmentation using depth map connected components.
        Groups pixels with similar depth into object clusters.
        No ML required — works on CPU.
        """
        from scipy import ndimage

        img_array = np.array(image)
        h, w = depth.shape[:2]

        # Filter out background (too far or too close)
        valid = (depth > 0.1) & (depth < 3.0)
        if valid.sum() < 100:
            return []

        # Estimate table/floor plane as the mode depth
        valid_depths = depth[valid]
        hist, edges = np.histogram(valid_depths, bins=50)
        bg_depth = (edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2

        # Objects are things significantly closer than background
        # Use a small threshold to catch objects barely above the surface
        foreground = valid & (depth < bg_depth - 0.015)
        if foreground.sum() < 30:
            return []

        # Connected components on foreground mask
        labeled, n_labels = ndimage.label(foreground)

        results = []
        for label_id in range(1, n_labels + 1):
            mask = labeled == label_id
            area = mask.sum()
            if area < 50 or area > (h * w * 0.5):
                continue

            # Bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Crop
            crop = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

            # Confidence based on mask compactness
            bbox_area = (y_max - y_min) * (x_max - x_min)
            compactness = area / max(bbox_area, 1)
            score = min(compactness * 0.8, 0.8)  # cap at 0.8 for fallback

            results.append(SegmentationResult(
                mask=mask,
                bbox_2d=[int(x_min), int(y_min), int(x_max), int(y_max)],
                score=score,
                crop=crop,
                prompt="depth_object",
                method="depth_fallback",
            ))

        return results

    def _deduplicate(self, results: List[SegmentationResult],
                     iou_threshold: float = 0.5) -> List[SegmentationResult]:
        """Remove duplicate detections based on mask IoU."""
        if not results:
            return []

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        keep = []

        for r in results:
            is_dup = False
            for kept in keep:
                iou = self._mask_iou(r.mask, kept.mask)
                if iou > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(r)

        return keep

    @staticmethod
    def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return intersection / max(union, 1)
