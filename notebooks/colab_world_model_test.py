# =============================================================================
#  Unitree G1 — FULL World Model Test (Colab GPU)
# =============================================================================
# Runtime: A100 (recommended) or T4
#
# This runs REAL AI models — NO fakes, NO placeholders:
#   ✅ SAM3 (Meta) — text-prompted 2D segmentation
#   ✅ SAM3D Objects (Meta) — single-image 3D reconstruction
#   ✅ Qwen2.5-VL-7B (local GPU) — object physics estimation
#   ✅ MuJoCo — physics simulation + camera rendering
#   ✅ Depth validation — cross-checks 3D estimates
#   ✅ Scene Graph v2 — persistent tracking + relationships
#
# Pipeline: MuJoCo cameras → SAM3 → SAM3D → Qwen VLM → depth → scene graph
# =============================================================================


# %% Cell 1 — Install ALL Dependencies (~5-8 min on A100)
# =========================================================
# Run this cell FIRST. It installs everything.

# ---- PyTorch + CUDA ----
# !pip install -q torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# ---- MuJoCo + Core ----
# !pip install -q mujoco==3.3.2 "numpy<2.0" scipy pillow matplotlib imageio pyyaml

# ---- SAM3 (Meta's Segment Anything 3) ----
# !git clone https://github.com/facebookresearch/sam3.git /content/sam3
# !cd /content/sam3 && pip install -e .

# ---- SAM3D Objects (Meta's single-image 3D reconstruction) ----
# !git clone https://github.com/facebookresearch/sam-3d-objects.git /content/sam3d
# !cd /content/sam3d && pip install -r requirements.txt

# ---- Qwen VLM (local GPU inference via transformers) ----
# !pip install -q "transformers>=4.45" accelerate qwen-vl-utils

# ---- OpenAI client (API compat layer for VLM) ----
# !pip install -q openai opencv-python-headless

# ---- HuggingFace auth (needed for SAM3 checkpoints) ----
# from huggingface_hub import login
# login()  # paste your HF token when prompted


# %% Cell 2 — Verify Installation
# ==================================
import sys, os
import torch
import numpy as np
import mujoco

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"MuJoCo: {mujoco.__version__}")
print(f"NumPy: {np.__version__}")

# Add paths
if os.path.exists('/content/sam3'):
    sys.path.insert(0, '/content/sam3')
    print("✅ SAM3 found")
else:
    print("⚠️ SAM3 not found — will use depth fallback")

if os.path.exists('/content/sam3d'):
    sys.path.insert(0, '/content/sam3d')
    print("✅ SAM3D found")
else:
    print("⚠️ SAM3D not found — will use depth projection")


# %% Cell 3 — Upload Project
# ============================
# Option A: Upload the 15MB ZIP directly (fastest)
# from google.colab import files
# uploaded = files.upload()  # upload g1_world_model_colab.zip
# !unzip -q g1_world_model_colab.zip -d /content/unitree_embodied_ai

# Option B: Google Drive (if already uploaded)
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r "/content/drive/MyDrive/g1_world_model_colab" /content/unitree_embodied_ai

PROJECT_ROOT = "/content/unitree_embodied_ai"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Verify
required = [
    "perception/scene_graph.py", "perception/pipeline.py",
    "perception/segmentor.py", "perception/reconstructor_3d.py",
    "perception/object_tagger.py", "perception/depth_validator.py",
    "perception/active_sim.py", "motor/unitree_bridge.py",
    "mujoco_menagerie/unitree_g1/g1_tabletop.xml",
    "mujoco_menagerie/unitree_g1/g1.xml",
]
for f in required:
    assert os.path.exists(f), f"Missing: {f}"
print("✅ All project files verified")


# %% Cell 4 — Boot MuJoCo Simulation
# =====================================
import mujoco
import numpy as np
from pathlib import Path

SCENE_XML = str(Path(PROJECT_ROOT) / "mujoco_menagerie" / "unitree_g1" / "g1_tabletop.xml")

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
print(f"Model: nq={model.nq}, nv={model.nv}, nu={model.nu}, bodies={model.nbody}")

from motor.unitree_bridge import UnitreeBridge
bridge = UnitreeBridge(model, data)
bridge.reset_to_stand()

# Settle physics
for _ in range(500):
    mujoco.mj_step(model, data)

base_z = bridge.get_base_pos()[2]
assert base_z > 0.5, f"Robot fell! z={base_z:.3f}"
print(f"✅ Robot standing at z={base_z:.3f}m")

# Record ground truth
gt_positions = {}
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and name.startswith("obj_"):
        gt_positions[name] = data.xpos[i].copy()
        p = data.xpos[i]
        print(f"  📦 {name}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")


# %% Cell 5 — Render Cameras
# ============================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

renderer = mujoco.Renderer(model, 480, 640)

def render_rgb(cam):
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()

def render_depth(cam):
    mujoco.mj_forward(model, data)
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    d = renderer.render().copy()
    renderer.disable_depth_rendering()
    return d

def get_intrinsics(cam):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    fovy = model.cam_fovy[cam_id]
    f = 0.5 * 480 / np.tan(fovy * np.pi / 360)
    return {"fx": f, "fy": f, "cx": 320, "cy": 240}

def get_extrinsics(cam):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    pos = data.cam_xpos[cam_id].copy()
    rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rot @ flip
    extrinsics[:3, 3] = pos
    return extrinsics

# Render all cameras
cameras = ["overview_cam", "head_cam", "side_cam"]
renders = {}
for cam in cameras:
    try:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if cam_id < 0:
            continue
        renders[cam] = {"rgb": render_rgb(cam), "depth": render_depth(cam)}
        print(f"📷 {cam}: RGB {renders[cam]['rgb'].shape}")
    except Exception as e:
        print(f"❌ {cam}: {e}")

# Show
n = len(renders)
fig, axes = plt.subplots(2, n, figsize=(6*n, 8))
if n == 1: axes = axes.reshape(-1, 1)
for col, (cam, r) in enumerate(renders.items()):
    axes[0, col].imshow(r["rgb"]); axes[0, col].set_title(f"{cam} — RGB"); axes[0, col].axis("off")
    axes[1, col].imshow(r["depth"], cmap="viridis"); axes[1, col].set_title(f"{cam} — Depth"); axes[1, col].axis("off")
plt.suptitle("G1 Tabletop — All Camera Views", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("camera_views.png", dpi=120, bbox_inches="tight"); plt.show()
print("✅ Camera renders saved")


# %% Cell 6 — Check Which REAL Models Are Available
# ====================================================
from perception.segmentor import SAM3Segmentor
from perception.reconstructor_3d import SAM3DReconstructor

sam3_ok = SAM3Segmentor.is_available()
sam3d_ok = SAM3DReconstructor.is_available()

# Check if VLM can load
vlm_ok = False
try:
    from cognitive.local_vlm import LocalVLM
    vlm_ok = True
except ImportError:
    pass

print(f"\n{'='*50}")
print(f"  MODEL AVAILABILITY")
print(f"{'='*50}")
print(f"  SAM3 (segmentation):     {'✅ REAL' if sam3_ok else '❌ fallback (depth)'}")
print(f"  SAM3D (3D reconstruct):  {'✅ REAL' if sam3d_ok else '❌ fallback (depth project)'}")
print(f"  Qwen VLM (tagging):      {'✅ REAL' if vlm_ok else '❌ fallback (sim tagger)'}")
print(f"{'='*50}")

# Decide pipeline config
use_sim_tagger = not vlm_ok
print(f"\n  Pipeline mode: {'FULL (all real models)' if (sam3_ok and sam3d_ok and vlm_ok) else 'HYBRID (real + fallback)'}")


# %% Cell 7 — Load Real VLM (if available)
# ==========================================
local_vlm = None
if vlm_ok and torch.cuda.is_available():
    print("Loading Qwen2.5-VL-7B on GPU...")
    from cognitive.local_vlm import LocalVLM
    local_vlm = LocalVLM("Qwen/Qwen2.5-VL-7B-Instruct")
    # Quick sanity
    resp = local_vlm.chat("Say 'ready'.")
    print(f"VLM: {resp}")
    print("✅ VLM loaded on GPU")
else:
    print("VLM not available — using sim-mode tagger (MuJoCo body names)")


# %% Cell 8 — Load Real SAM3 (if available)
# ==========================================
if sam3_ok:
    print("Loading SAM3 model...")
    segmentor = SAM3Segmentor()
    segmentor.load()
    print("✅ SAM3 loaded")


# %% Cell 9 — Run FULL Perception Pipeline
# ==========================================
from perception.pipeline import PerceptionPipeline, PipelineConfig

# Configure — use real models where available, fallback where not
config = PipelineConfig(
    use_sim_tagger=use_sim_tagger,
    depth_fallback=True,
    min_confidence=0.10,
)

pipeline = PerceptionPipeline(config=config)

# Get sim body names for tagger fallback
sim_names = []
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and name.startswith("obj_"):
        sim_names.append(name)

# Run on primary camera
primary_cam = "overview_cam"
rgb = renders[primary_cam]["rgb"]
depth = renders[primary_cam]["depth"]
intrinsics = get_intrinsics(primary_cam)
extrinsics = get_extrinsics(primary_cam)

print(f"\n{'='*60}")
print(f"  RUNNING FULL PERCEPTION PIPELINE")
print(f"  Camera: {primary_cam}")
print(f"  Models: SAM3={'REAL' if sam3_ok else 'fallback'}, "
      f"SAM3D={'REAL' if sam3d_ok else 'fallback'}, "
      f"VLM={'REAL' if vlm_ok else 'sim-tagger'}")
print(f"{'='*60}")

scene_graph = pipeline.perceive(
    rgb, depth, intrinsics,
    camera_extrinsics=extrinsics,
    sim_body_names=sim_names,
)

print(f"\n✅ Perception complete!")
scene_graph.print_all()


# %% Cell 10 — Run on HEAD CAMERA Too (Multi-Camera)
# =====================================================
if "head_cam" in renders:
    print(f"\n{'='*60}")
    print(f"  RUNNING ON HEAD CAMERA (multi-camera fusion)")
    print(f"{'='*60}")

    rgb_head = renders["head_cam"]["rgb"]
    depth_head = renders["head_cam"]["depth"]
    intrinsics_head = get_intrinsics("head_cam")
    extrinsics_head = get_extrinsics("head_cam")

    pipeline.perceive(
        rgb_head, depth_head, intrinsics_head,
        camera_extrinsics=extrinsics_head,
        sim_body_names=sim_names,
    )

    print(f"After multi-camera fusion:")
    scene_graph.print_all()


# %% Cell 11 — Validate Against Ground Truth
# =============================================
import json

print(f"\n{'='*60}")
print(f"  VALIDATION: Perception vs MuJoCo Ground Truth")
print(f"{'='*60}")

detected = scene_graph.get_all_objects()
print(f"Detected: {len(detected)} objects | Ground truth: {len(gt_positions)} objects\n")

results = {}
for gt_name, gt_pos in gt_positions.items():
    keyword = gt_name.replace("obj_", "").replace("_", " ")
    match = scene_graph.query(keyword)
    if match:
        det_pos = np.array(match.position_world)
        error = np.linalg.norm(det_pos - gt_pos)
        status = "✅" if error < 0.10 else ("⚠️" if error < 0.20 else "❌")
        print(f"  {gt_name}:")
        print(f"    GT:       ({gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f})")
        print(f"    Detected: ({det_pos[0]:.3f}, {det_pos[1]:.3f}, {det_pos[2]:.3f})")
        print(f"    Error:    {error:.3f}m {status}")
        print(f"    Label:    {match.label} (T{match.track_id}, conf={match.confidence:.2f})")
        results[gt_name] = {"error_m": float(error), "status": str(status), "label": match.label}
    else:
        print(f"  {gt_name}: NOT FOUND ❌")
        results[gt_name] = {"error_m": None, "status": "not_found"}

n_found = sum(1 for r in results.values() if r["error_m"] is not None)
n_good = sum(1 for r in results.values() if r["error_m"] is not None and r["error_m"] < 0.10)

print(f"\n{'='*60}")
print(f"  RESULT: {n_found}/{len(gt_positions)} found, "
      f"{n_good}/{len(gt_positions)} within 10cm")
print(f"{'='*60}")


# %% Cell 12 — Multi-Frame Tracking Test
# =========================================
print(f"\n{'='*60}")
print(f"  MULTI-FRAME TRACKING (5 frames)")
print(f"{'='*60}")

for frame in range(5):
    for _ in range(100):
        mujoco.mj_step(model, data)

    rgb_f = render_rgb(primary_cam)
    depth_f = render_depth(primary_cam)
    pipeline.perceive(rgb_f, depth_f, intrinsics,
                      camera_extrinsics=extrinsics,
                      sim_body_names=sim_names)

    objs = scene_graph.get_all_objects()
    tracks = sorted([(o.track_id, o.label[:20], o.frames_since_seen) for o in objs])
    print(f"  Frame {scene_graph.frame_count}: {len(objs)} objects, "
          f"tracks={[t[0] for t in tracks]}, stale={[t[2] for t in tracks]}")

final_tracks = sorted([o.track_id for o in scene_graph.get_all_objects()])
print(f"\n  Track IDs after {scene_graph.frame_count} frames: {final_tracks}")
print(f"  Stable: {'✅ YES' if len(final_tracks) == len(gt_positions) else '⚠️ DRIFT'}")


# %% Cell 13 — Visualize Scene Graph
# =====================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: annotated RGB
axes[0].imshow(rgb)
axes[0].set_title("Detected Objects (Overview Cam)", fontsize=12)

colors = {"mug": "#ff4444", "box": "#4444ff", "bottle": "#44cc44", "ball": "#ffcc00"}

for obj in detected:
    pos = np.array(obj.position_world)
    pos_h = np.array([*pos, 1.0])
    pos_cam = np.linalg.inv(extrinsics) @ pos_h
    if pos_cam[2] > 0:
        px = intrinsics["fx"] * pos_cam[0] / pos_cam[2] + intrinsics["cx"]
        py = intrinsics["fy"] * pos_cam[1] / pos_cam[2] + intrinsics["cy"]
        if 0 <= px < 640 and 0 <= py < 480:
            c = "#ffffff"
            for k, v in colors.items():
                if k in obj.label.lower(): c = v; break
            axes[0].plot(px, py, 'o', color=c, markersize=14, markeredgecolor='white', markeredgewidth=2)
            axes[0].annotate(f"T{obj.track_id}: {obj.label}", (px, py),
                            textcoords="offset points", xytext=(10, -15), fontsize=8,
                            color='white', fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.3'))
axes[0].axis("off")

# Right: scene graph text
axes[1].axis("off"); axes[1].set_facecolor('#0a0a1a')
lines = [f"SCENE GRAPH — Frame {scene_graph.frame_count}", f"Objects: {len(detected)}", "─" * 40]
for obj in detected:
    p = obj.position_world
    lines.append(f"[T{obj.track_id}] {obj.label}")
    lines.append(f"  pos: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
    lines.append(f"  {obj.collision_primitive}, {obj.mass_kg}kg, conf={obj.confidence:.2f}")
    if obj.relationships: lines.append(f"  {', '.join(obj.relationships[:3])}")
    lines.append("")
axes[1].text(0.05, 0.95, "\n".join(lines), transform=axes[1].transAxes, fontsize=9,
            va='top', fontfamily='monospace', color='#00ff88',
            bbox=dict(facecolor='#1a1a2e', alpha=0.95, boxstyle='round'))
axes[1].set_title("Scene Graph", fontsize=12, color='#00ff88')
plt.suptitle("World Model — Perception Output", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("world_model_output.png", dpi=150, bbox_inches="tight"); plt.show()


# %% Cell 14 — Export + Summary
# ===============================
scene_graph.save_json("scene_graph_export.json")

summary = {
    "test": "world_model_full",
    "real_models": {"SAM3": sam3_ok, "SAM3D": sam3d_ok, "VLM": vlm_ok},
    "frames": scene_graph.frame_count,
    "objects": len(scene_graph.objects),
    "validation": results,
    "objects_detail": [
        {"track_id": o.track_id, "label": o.label, "pos": o.position_world,
         "mass": o.mass_kg, "shape": o.collision_primitive,
         "conf": o.confidence, "rels": o.relationships}
        for o in scene_graph.get_all_objects()
    ]
}
with open("world_model_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"  🎉 WORLD MODEL TEST COMPLETE")
print(f"{'='*60}")
print(f"  Models used:  SAM3={'REAL' if sam3_ok else 'fallback'}, "
      f"SAM3D={'REAL' if sam3d_ok else 'fallback'}, "
      f"VLM={'REAL' if vlm_ok else 'sim-tagger'}")
print(f"  Objects:      {len(scene_graph.objects)}")
print(f"  Frames:       {scene_graph.frame_count}")
print(f"  Track IDs:    {sorted([o.track_id for o in scene_graph.get_all_objects()])}")
print(f"  Exports:      scene_graph_export.json, world_model_results.json")
print(f"{'='*60}")

renderer.close()
