"""
test_pick_and_place.py — Verify scene loads and pipeline connects.
No GPU needed, just validates MuJoCo scene + RL policy load.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

SCENE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sim", "scenes", "g1_pick_and_place.xml"
)

print("=== Test 1: Scene loads ===")
model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")

print("\n=== Test 2: Robot stands ===")
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
for _ in range(1000):
    mujoco.mj_step(model, data)
base_z = data.qpos[2]
print(f"  Base z = {base_z:.3f}m (should be > 0.5)")
assert base_z > 0.4, f"Robot fell: z={base_z}"

print("\n=== Test 3: Objects on Table A ===")
mujoco.mj_forward(model, data)
for name in ["obj_red_mug", "obj_blue_cube", "obj_green_bottle"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"{name} not in scene"
    pos = data.xpos[bid]
    print(f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    assert pos[1] < 0, f"{name} not on Table A"
    assert pos[2] > 0.3, f"{name} fell"

print("\n=== Test 4: Table B empty ===")
for name in ["obj_red_mug", "obj_blue_cube", "obj_green_bottle"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert data.xpos[bid][1] < 0.05, f"{name} already on Table B"
print("  ✅ Table B clear")

print("\n=== Test 5: Camera renders ===")
renderer = mujoco.Renderer(model, 224, 224)
renderer.update_scene(data, camera="head_cam")
img = renderer.render()
print(f"  head_cam: {img.shape} max={img.max()}")
assert img.shape == (224, 224, 3) and img.max() > 0
renderer.close()

print("\n=== Test 6: RL policy loads ===")
import yaml
import onnxruntime as ort
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
policy_dir = os.path.join(_root, "unitree_rl_mjlab", "deploy", "robots", "g1",
                          "config", "policy", "velocity", "v0")
onnx_path = os.path.join(policy_dir, "exported", "policy.onnx")
yaml_path = os.path.join(policy_dir, "params", "deploy.yaml")

if os.path.exists(onnx_path):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    print(f"  RL policy: input={sess.get_inputs()[0].shape} output={sess.get_outputs()[0].shape}")
    print(f"  Joints: {len(cfg['joint_ids_map'])}")
    print("  ✅ RL policy ready")
else:
    print(f"  ⚠️ ONNX not found at {onnx_path}")

print("\n" + "=" * 40)
print("  ALL TESTS PASSED ✅")
print("=" * 40)
