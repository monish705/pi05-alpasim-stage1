# Phase 2 Perception Pipeline — Colab Test Notebook
# ==================================================
# Run this on Google Colab (GPU runtime: T4 or A100)
# Upload the `test_data/` folder and `perception/` folder from your local machine first.
#
# This script tests the full chain:
#   SAM3 → SAM3D Objects → Qwen3-VL-8B → Depth Validation → Scene Graph
#
# To convert to .ipynb, open this file in Colab or use jupytext.

# %% [markdown]
# # Phase 2: Perception Pipeline Test
# 
# Tests the full perception stack on Phase 1 sim-rendered frames.
# 
# **Pipeline:** `RGB → SAM3 (segment) → SAM3D Objects (3D reconstruct) → Qwen3-VL (tag) → Depth validate → Scene Graph`

# %% Cell 1: Install Dependencies
# ================================

# !pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# !git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .
# !git clone https://github.com/facebookresearch/sam-3d-objects.git && cd sam-3d-objects && pip install -r requirements.txt
# !pip install vllm openai pillow numpy opencv-python-headless matplotlib

# Launch vLLM server in background (takes ~2 min to load model)
# !nohup vllm serve Qwen/Qwen3-VL-8B --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.85 > vllm.log 2>&1 &

# %% Cell 2: Upload Test Data
# ============================
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Paths (adjust if uploaded differently)
TEST_DATA_DIR = "test_data"
PERCEPTION_DIR = "perception"

# Load test frame
rgb = np.array(Image.open(f"{TEST_DATA_DIR}/overview_cam_rgb.png").convert("RGB"))
depth = np.load(f"{TEST_DATA_DIR}/overview_cam_depth.npy")

with open(f"{TEST_DATA_DIR}/overview_cam_params.json") as f:
    camera_params = json.load(f)

with open(f"{TEST_DATA_DIR}/ground_truth.json") as f:
    ground_truth = json.load(f)

print(f"RGB: {rgb.shape}, Depth: {depth.shape}")
print(f"Camera: {camera_params}")
print(f"Ground truth objects: {list(ground_truth.keys())}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.imshow(rgb); ax1.set_title("RGB (overview_cam)")
ax2.imshow(depth, cmap='viridis'); ax2.set_title("Depth Map")
plt.tight_layout(); plt.show()


# %% Cell 3: SAM3 Segmentation
# ==============================
import sys
import os

# Add both the current and parent directories to your Python path 
# so it can find the 'perception' module no matter where you are in Google Drive
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from perception.segmentor import SAM3Segmentor

segmentor = SAM3Segmentor()
segmentor.load()

pil_image = Image.fromarray(rgb)
segments = segmentor.segment_all(pil_image)

print(f"\nDetected {len(segments)} objects:")
for i, seg in enumerate(segments):
    print(f"  [{i}] prompt='{seg.prompt}', score={seg.score:.2f}, "
          f"bbox={seg.bbox_2d}, mask_area={seg.mask.sum()}")

# Visualize masks
fig, axes = plt.subplots(1, min(len(segments), 5), figsize=(20, 4))
if len(segments) == 1:
    axes = [axes]
for i, (seg, ax) in enumerate(zip(segments[:5], axes)):
    overlay = rgb.copy()
    overlay[seg.mask] = overlay[seg.mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title(f"[{i}] {seg.prompt} ({seg.score:.2f})")
    ax.axis('off')
plt.suptitle("SAM3 Segmentation Results")
plt.tight_layout(); plt.show()


# %% Cell 4: SAM3D Objects — 3D Reconstruction
# ==============================================
from perception.reconstructor_3d import SAM3DReconstructor

reconstructor = SAM3DReconstructor()
reconstructor.load()

masks = [seg.mask for seg in segments]
reconstructions = reconstructor.reconstruct_batch(pil_image, masks)

print(f"\n3D Reconstructions:")
for i, recon in enumerate(reconstructions):
    if recon.success:
        verts = recon.mesh_vertices
        n_verts = len(verts) if verts is not None else 0
        bbox = recon.bbox_3d
        print(f"  [{i}] ✅ mesh={n_verts} vertices, "
              f"translation={recon.translation}, "
              f"bbox_3d={bbox}")
    else:
        print(f"  [{i}] ❌ Reconstruction failed")


# %% Cell 5: Qwen3-VL-8B Object Tagging
# =======================================
import time

# Wait for vLLM to be ready
print("Waiting for vLLM server...")
for attempt in range(30):
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
        client.models.list()
        print("vLLM ready!")
        break
    except Exception:
        time.sleep(5)
        print(f"  attempt {attempt+1}/30...")
else:
    print("ERROR: vLLM server did not start. Check vllm.log")

from perception.object_tagger import ObjectTagger

tagger = ObjectTagger()
crops = [seg.crop for seg in segments]
tags = tagger.tag_batch(crops)

print(f"\nObject Tags:")
for i, tag in enumerate(tags):
    print(f"  [{i}] {tag.label}")
    print(f"       material={tag.material}, mass={tag.estimated_mass_kg:.2f}kg, "
          f"friction={tag.estimated_friction:.2f}")
    print(f"       shape={tag.collision_shape}, "
          f"dims={tag.dimensions_cm}, graspable={tag.is_graspable}")


# %% Cell 6: Depth Validation + Scene Graph Assembly
# ===================================================
from perception.depth_validator import DepthValidator
from perception.scene_graph import SceneGraph, SceneObject
import uuid

validator = DepthValidator(tolerance_m=0.05)
scene_graph = SceneGraph()
new_objects = []

for i, (seg, recon, tag) in enumerate(zip(segments, reconstructions, tags)):
    # Get position
    if recon.success and recon.translation is not None:
        position = recon.translation
    else:
        position = validator._depth_project(seg.mask, depth, camera_params)
        if position is None:
            print(f"  [{i}] Skipped: no valid position")
            continue

    # Validate
    corrected_pos, was_corrected, discrepancy = validator.validate_position(
        position, seg.mask, depth, camera_params
    )

    # Measure dimensions
    depth_dims = validator.measure_object_dimensions(seg.mask, depth, camera_params)
    dims_m = {k: v/100.0 for k, v in tag.dimensions_cm.items()}
    if depth_dims:
        dims_m = depth_dims

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
        confidence=seg.score,
    )
    new_objects.append(obj)
    status = "CORRECTED" if was_corrected else "OK"
    print(f"  [{i}] {tag.label} → pos={corrected_pos} [{status}, Δ={discrepancy:.3f}m]")

scene_graph.update(new_objects)
scene_graph.print_all()


# %% Cell 7: Validation Against Ground Truth
# ============================================
print("\n" + "="*60)
print("  VALIDATION: Perception vs MuJoCo Ground Truth")
print("="*60)

keyword_map = {
    "obj_red_mug": "mug",
    "obj_blue_box": "box",
    "obj_green_bottle": "bottle",
    "obj_yellow_ball": "ball",
}

all_pass = True
for gt_name, gt_pos in ground_truth.items():
    keyword = keyword_map.get(gt_name, gt_name.split("_")[-1])
    match = scene_graph.query(keyword)
    if match:
        error = np.linalg.norm(np.array(match.position_world) - np.array(gt_pos))
        status = "✅" if error < 0.05 else "❌"
        if error >= 0.05:
            all_pass = False
        print(f"  {gt_name}: {keyword} → {match.label}")
        print(f"    GT:  {gt_pos}")
        print(f"    Det: {match.position_world}")
        print(f"    Error: {error:.3f}m {status}")
    else:
        print(f"  {gt_name}: {keyword} → NOT FOUND ❌")
        all_pass = False

print(f"\n{'='*60}")
print(f"  RESULT: {'ALL PASSED ✅' if all_pass else 'SOME FAILED ❌'}")
print(f"{'='*60}")


# %% Cell 8: Export Scene Graph JSON
# ====================================
scene_graph.save_json("scene_graph_export.json")
print("\nScene Graph exported to scene_graph_export.json")
print("Download this file to your local machine for Phase 3+ integration.")

# Also save a summary
summary = {
    "num_objects": len(scene_graph.objects),
    "objects": [
        {"label": o.label, "position": o.position_world,
         "mass": o.mass_kg, "shape": o.collision_primitive}
        for o in scene_graph.objects.values()
    ],
    "validation": {
        gt_name: {
            "matched": scene_graph.query(keyword_map.get(gt_name, gt_name.split("_")[-1])) is not None,
        }
        for gt_name in ground_truth
    }
}
with open("perception_test_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Summary saved to perception_test_summary.json")
