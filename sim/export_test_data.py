"""
sim/export_test_data.py
=======================
Exports RGB frames, depth maps, and camera parameters from Phase 1 sim
for Phase 2 perception testing on Google Colab.
"""
import numpy as np
import cv2
import json
import time
from pathlib import Path

def export():
    from sim.mujoco_env import MuJoCoEnv

    out_dir = Path("test_data")
    out_dir.mkdir(exist_ok=True)

    scene_path = Path(__file__).parent / "scenes" / "tabletop.xml"
    print(f"Loading scene: {scene_path}")
    env = MuJoCoEnv(str(scene_path), render_mode="headless")
    env.reset()

    # Let objects settle
    print("Settling physics (500 steps)...")
    for _ in range(500):
        env.step()

    cameras = ["overview_cam", "wrist_cam_proxy"]
    
    for cam_name in cameras:
        print(f"\nExporting camera: {cam_name}")
        
        # RGB
        rgb = env.get_rgb(cam_name)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{cam_name}_rgb.png"), bgr)
        print(f"  RGB: {rgb.shape}, saved as PNG")

        # Depth
        depth = env.get_depth(cam_name)
        np.save(str(out_dir / f"{cam_name}_depth.npy"), depth)
        print(f"  Depth: {depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]")

        # Camera intrinsics
        K = env.get_camera_intrinsics(cam_name)
        print(f"  Intrinsics: {K}")
        
        with open(str(out_dir / f"{cam_name}_params.json"), "w") as f:
            json.dump({k: float(v) for k, v in K.items()}, f, indent=2)

    # Ground truth object positions from MuJoCo
    import mujoco
    ground_truth = {}
    obj_names = ["obj_red_mug", "obj_blue_box", "obj_green_bottle", "obj_yellow_ball"]
    for name in obj_names:
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            pos = env.data.xpos[body_id].tolist()
            ground_truth[name] = pos
            print(f"  GT {name}: {pos}")

    with open(str(out_dir / "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nAll test data exported to {out_dir.resolve()}")
    print("Upload this folder to Colab for Phase 2 testing.")

if __name__ == "__main__":
    export()
