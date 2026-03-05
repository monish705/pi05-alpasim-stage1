"""
tests/render_e2e_demo.py
=========================
End-to-end video: Stand → RL Walk → Arm Reach → Grasp
Multi-camera: overview + head cam + wrist cam
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import imageio

SCENE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mujoco_menagerie", "unitree_g1", "g1_tabletop.xml"
)
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from motor.unitree_bridge import UnitreeBridge
from motor.locomotion import LocomotionController
from motor.arm_controller import ArmController
from motor.grasp_controller import GraspController

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 480, 640)
bridge = UnitreeBridge(model, data)
bridge.reset_to_stand()

frames = []

def cap(cam="overview_cam"):
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())

# ======= PHASE 1: Stand with RL policy (1.5s) =======
print("[Phase 1] Standing with RL policy...")
loco = LocomotionController(bridge)

# Capture initial overview
for _ in range(5): cap("overview_cam")
for _ in range(3): cap("head_cam")

# Stand for 1.5s, capturing frames
n_stand = int(1.5 / loco.step_dt)
step_fn = loco._policy_step if loco._has_policy else loco._fallback_step
for i in range(n_stand):
    step_fn(0.0, 0.0, 0.0)
    if i % 3 == 0:
        cap("overview_cam")

base = bridge.get_base_pos()
print(f"  Standing at z={base[2]:.3f}m")

# Capture head view after settling
for _ in range(5): cap("head_cam")

# ======= PHASE 2: Walk toward table with RL policy =======
print("[Phase 2] Walking toward table...")
target_x, target_y = 0.35, 0.0
n_max = int(8.0 / loco.step_dt)

for step in range(n_max):
    mujoco.mj_forward(model, data)
    cx, cy, cyaw = loco._get_base_xytheta()
    dist = np.sqrt((target_x - cx)**2 + (target_y - cy)**2)

    if dist < 0.15:
        step_fn(0.0, 0.0, 0.0)
        print(f"  Arrived at step {step} (dist={dist:.3f}m)")
        break

    dx = target_x - cx
    dy = target_y - cy
    angle = np.arctan2(dy, dx)
    angle_err = (angle - cyaw + np.pi) % (2*np.pi) - np.pi

    if abs(angle_err) > 0.4:
        cmd_vx, cmd_vy, cmd_wz = 0.0, 0.0, np.clip(2.0*angle_err, -1.0, 1.0)
    else:
        cmd_vx = np.clip(1.5*dist, 0, 1.0)
        cmd_vy = 0.0
        cmd_wz = np.clip(2.0*angle_err, -1.0, 1.0)

    step_fn(cmd_vx, cmd_vy, cmd_wz)

    if step % 2 == 0:
        cap("overview_cam")
    if step % 8 == 0:
        cap("head_cam")

# Settle
for i in range(20):
    step_fn(0.0, 0.0, 0.0)
    if i % 4 == 0: cap("overview_cam")

base = bridge.get_base_pos()
print(f"  Final pos: ({base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f})")

# Head cam view of table
for _ in range(8): cap("head_cam")

# ======= PHASE 3: Arm reach toward nearest object =======
print("[Phase 3] Arm reaching toward object...")
arm = ArmController(bridge, hand="right")
grasp = GraspController(bridge)

# Find closest object
mujoco.mj_forward(model, data)
nearby = grasp.detect_nearby("right", max_dist=1.0)
if nearby:
    target_obj = nearby[0]
    print(f"  Targeting: {target_obj['name']} at dist={target_obj['distance']:.3f}m")
    # Reach 5cm above the object
    arm_target = target_obj['position'] + np.array([0.0, 0.0, 0.05])
else:
    # Fallback: reach forward from current base
    arm_target = base + np.array([0.25, -0.15, 0.05])
    print(f"  No objects nearby, reaching to ({arm_target[0]:.2f}, {arm_target[1]:.2f}, {arm_target[2]:.2f})")

print(f"  Target: ({arm_target[0]:.3f}, {arm_target[1]:.3f}, {arm_target[2]:.3f})")

# Do IK with frame capture
for step in range(250):
    mujoco.mj_forward(model, data)
    ee = arm.get_ee_pos()
    err_vec = arm_target - ee
    err = np.linalg.norm(err_vec)

    if err < 0.04:
        print(f"  ✅ Reached at step {step} (error={err:.4f}m)")
        for _ in range(15): cap("overview_cam")
        for _ in range(10): cap("wrist_cam")
        break

    # CoM check
    if step > 10 and not arm.is_com_stable(margin=0.01):
        print(f"  ⚠️ CoM unstable at step {step}, stopping (err={err:.4f}m)")
        break

    max_dx = 0.015
    if err > max_dx:
        err_vec = err_vec / err * max_dx

    J = arm.compute_jacobian()
    n_j = J.shape[1]
    W = np.eye(n_j)
    W[0,0], W[1,1], W[2,2] = 0.3, 0.2, 0.2  # penalize waist
    JW = J @ W
    JJT = JW @ J.T + 0.05**2 * np.eye(3)
    dq = W @ J.T @ np.linalg.solve(JJT, err_vec)

    for i, idx in enumerate(arm._wb_indices):
        new_q = data.ctrl[idx] + dq[i]
        new_q = np.clip(new_q, arm._joint_limits[i, 0], arm._joint_limits[i, 1])
        data.ctrl[idx] = new_q

    for _ in range(5):
        mujoco.mj_step(model, data)

    if step % 3 == 0:
        cap("overview_cam")
    if step % 10 == 0:
        cap("wrist_cam")

# ======= PHASE 4: Final views =======
print("[Phase 4] Final poses...")
for _ in range(10): cap("overview_cam")
for _ in range(5): cap("side_cam")
for _ in range(5): cap("head_cam")
for _ in range(5): cap("wrist_cam")

renderer.close()

# ======= SAVE VIDEO =======
print(f"\nTotal frames: {len(frames)}")
video_path = os.path.join(OUT, "phase3_e2e_demo.mp4")
writer = imageio.get_writer(video_path, fps=30, codec='libx264',
                             quality=8, pixelformat='yuv420p')
for f in frames:
    writer.append_data(f)
writer.close()
print(f"✅ Video saved: {video_path}")

# Key frames
from PIL import Image
for name, idx in [("stand", 5), ("walk", len(frames)//3), ("reach", 2*len(frames)//3), ("final", -1)]:
    p = os.path.join(OUT, f"e2e_{name}.png")
    Image.fromarray(frames[idx]).save(p)
    print(f"  {name} → {p}")

print("\n🎉 End-to-end demo complete!")
