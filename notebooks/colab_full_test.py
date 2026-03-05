# =============================================================================
#  Unitree G1 — End-to-End Autonomous Task Test (Colab)
# =============================================================================
# GPU: A100 recommended (SAM3 + SAM3D + VLM all need GPU memory)
#      T4 may work with reduced batch sizes
#
# This runs the REAL system. Full pipeline:
#   MuJoCo → SAM3 segmentation → SAM3D 3D mesh → Qwen VLM tagging →
#   depth validation → scene graph → VLM code generation →
#   motor execution → verification
#
# No fakes. No bypasses. No placeholders.
# =============================================================================

# %% Cell 1 — Install ALL dependencies
# =========================================
# ORDER MATTERS. Read the comments.

"""
# === Step 1: Core (MuJoCo, ONNX, numpy<2) ===
!pip install -q mujoco==3.3.2 onnxruntime pyyaml imageio pillow scipy

# === Step 2: SAM3 (Meta segment-anything-3) ===
# Clone and install as editable package
!git clone https://github.com/facebookresearch/sam3.git /content/sam3
!cd /content/sam3 && pip install -e .

# === Step 3: SAM3D Objects (Meta single-image 3D) ===
# Clone the repo but DO NOT run requirements.txt — it has a broken
# auto_gptq==0.7.1 that fails on Colab (metadata version mismatch).
# Instead, install only the packages SAM3D actually imports.
!git clone https://github.com/facebookresearch/sam-3d-objects.git /content/sam3d
!pip install -q trimesh pyglet pyrender open3d

# === Step 4: VLM (Qwen2.5-VL via transformers) ===
# Uses Colab's pre-installed PyTorch 2.7+cu126. Do NOT install cu121.
!pip install -q "transformers>=4.45" accelerate qwen-vl-utils

# === Step 5: Utils ===
!pip install -q openai opencv-python-headless matplotlib

# === Step 6: RE-PIN numpy LAST ===
# SAM3 needs numpy<2. Some deps above pull in numpy>=2.
# Force it back down AFTER everything else is installed.
!pip install -q "numpy>=1.26,<2.0"
"""

# %% Cell 2 — Upload project
"""
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/unitree_embodied_ai" /content/unitree_embodied_ai
"""

import sys, os
sys.path.insert(0, '/content/unitree_embodied_ai')
# Add SAM3D to path
if os.path.exists('/content/sam3d'):
    sys.path.insert(0, '/content/sam3d')
if os.path.exists('/content/sam3'):
    sys.path.insert(0, '/content/sam3')
os.chdir('/content/unitree_embodied_ai')

# Verify
assert os.path.exists('main.py'), "Project not found!"
assert os.path.exists('perception/pipeline.py'), "Perception pipeline missing!"
assert os.path.exists('perception/segmentor.py'), "SAM3 segmentor missing!"
assert os.path.exists('perception/reconstructor_3d.py'), "SAM3D reconstructor missing!"
assert os.path.exists('perception/object_tagger.py'), "VLM tagger missing!"
assert os.path.exists('motor/locomotion.py'), "Motor layer missing!"
assert os.path.exists('unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx'), "RL policy missing!"
print("✅ All project files verified")


# %% Cell 3 — Load VLM on GPU
from cognitive.local_vlm import LocalVLM

vlm = LocalVLM("Qwen/Qwen2.5-VL-7B-Instruct")

# Quick sanity test
resp = vlm.chat("Say 'ready' if you can respond.")
print(f"VLM: {resp}")


# %% Cell 4 — Run the FULL end-to-end test
from test_e2e_autonomous import run_end_to_end_test

results = run_end_to_end_test(local_vlm=vlm)

# The test runs 6 phases:
# Phase 1: Boot MuJoCo sim + motors
# Phase 2: Assign task ("Walk over to the table and pick up the red mug")
# Phase 3: FULL perception: SAM3 → SAM3D → VLM tagging → depth → scene graph
# Phase 4: VLM generates Python code for the task
# Phase 5: Execute VLM's code (robot walks, reaches, grasps)
# Phase 6: Verify (is object in hand? robot stable?)


# %% Cell 5 — Visualise results
import matplotlib.pyplot as plt
import mujoco
import numpy as np

# Re-render the final state
from sim.mujoco_env import MuJoCoEnv
from motor.unitree_bridge import UnitreeBridge

env = MuJoCoEnv("mujoco_menagerie/unitree_g1/g1_tabletop.xml", render_mode="headless")
bridge = UnitreeBridge(env.model, env.data)

renderer = mujoco.Renderer(env.model, 480, 640)

def snap(cam):
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, camera=cam)
    return renderer.render().copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(snap('overview_cam')); axes[0].set_title('Overview')
axes[1].imshow(snap('head_cam'));     axes[1].set_title('Head Camera')
for a in axes: a.axis('off')

status = '✅ PASSED' if results.get('grasping') else '❌ FAILED'
plt.suptitle(f'Task Result: {status}', fontsize=14)
plt.tight_layout(); plt.show()


# %% Cell 6 — Print detailed results
import json
print("\n" + "="*60)
print("  DETAILED RESULTS")
print("="*60)
print(json.dumps(results, indent=2))
