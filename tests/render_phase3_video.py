"""
tests/render_phase3_video.py
=============================
Renders a video of the G1 robot:
1. Standing up from keyframe
2. Right arm reaching toward the table
3. Overview and side camera views

Outputs: phase3_demo.mp4 (or .gif if no ffmpeg)
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco

SCENE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mujoco_menagerie", "unitree_g1", "g1_tabletop.xml"
)

def main():
    print("=" * 60)
    print("  PHASE 3: G1 DEMO VIDEO RENDER")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)

    from motor.unitree_bridge import UnitreeBridge
    from motor.arm_controller import ArmController

    bridge = UnitreeBridge(model, data)
    bridge.reset_to_stand()
    arm = ArmController(bridge, hand="right")

    frames = []

    def capture(camera="overview_cam"):
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        frames.append(rgb.copy())

    # ---- PHASE A: Stand and settle (2 seconds) ----
    print("\n[Phase A] Standing and settling...")
    for i in range(500):
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            capture("overview_cam")
    print(f"  Captured {len(frames)} frames so far")

    # ---- PHASE B: Look at objects from side ----
    print("\n[Phase B] Side view...")
    for i in range(100):
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            capture("side_cam")

    # ---- PHASE C: Right arm reaches toward the red mug ----
    print("\n[Phase C] Reaching toward red mug...")
    # Target: above the mug position
    mug_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj_red_mug")
    mujoco.mj_forward(model, data)
    mug_pos = data.xpos[mug_bid].copy()
    print(f"  Mug position: {mug_pos}")

    # Reach target: slightly above the mug
    target = mug_pos + np.array([0.0, 0.0, 0.10])
    print(f"  Reach target: {target}")

    # Do IK iterations, capturing frames
    for step in range(300):
        mujoco.mj_forward(model, data)

        current_pos = arm.get_ee_pos()
        error_vec = target - current_pos
        error = np.linalg.norm(error_vec)

        if error < 0.03:
            print(f"  ✅ Reached target at step {step} (error={error:.4f}m)")
            # Hold position for a beat
            for _ in range(50):
                mujoco.mj_step(model, data)
                capture("overview_cam")
            break

        # Clamp step
        max_dx = 0.015
        if error > max_dx:
            error_vec = error_vec / error * max_dx

        J = arm.compute_jacobian()
        JJT = J @ J.T + 0.05**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error_vec)

        for i, idx in enumerate(arm._arm_indices):
            new_q = data.ctrl[idx] + dq[i]
            new_q = np.clip(new_q, arm._joint_limits[i, 0], arm._joint_limits[i, 1])
            data.ctrl[idx] = new_q

        for _ in range(5):
            mujoco.mj_step(model, data)

        if step % 3 == 0:
            capture("overview_cam")

    # ---- PHASE D: Show final pose from multiple angles ----
    print("\n[Phase D] Final pose views...")
    for i in range(60):
        mujoco.mj_step(model, data)
        if i % 3 == 0:
            capture("side_cam")

    # Back to overview
    for i in range(60):
        mujoco.mj_step(model, data)
        if i % 3 == 0:
            capture("overview_cam")

    renderer.close()

    print(f"\nTotal frames: {len(frames)}")
    print(f"Frame size: {frames[0].shape}")

    # ---- Save as video ----
    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Try MP4 first via imageio
    try:
        import imageio
        video_path = os.path.join(out_dir, "phase3_demo.mp4")
        writer = imageio.get_writer(video_path, fps=30, codec='libx264',
                                     quality=8, pixelformat='yuv420p')
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"✅ Video saved to {video_path}")
    except Exception as e:
        print(f"MP4 failed ({e}), trying GIF...")
        # Fallback: GIF via PIL
        try:
            from PIL import Image
            gif_path = os.path.join(out_dir, "phase3_demo.gif")
            pil_frames = [Image.fromarray(f) for f in frames[::2]]  # Skip every other for size
            pil_frames[0].save(gif_path, save_all=True,
                              append_images=pil_frames[1:],
                              duration=66, loop=0, optimize=True)
            print(f"✅ GIF saved to {gif_path}")
        except Exception as e2:
            print(f"GIF failed too: {e2}")
            # Last resort: save individual frames
            frames_dir = os.path.join(out_dir, "phase3_frames")
            os.makedirs(frames_dir, exist_ok=True)
            from PIL import Image
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            print(f"✅ {len(frames)} frames saved to {frames_dir}/")

    # Also save key frames as individual images
    from PIL import Image
    for name, idx in [("start", 0), ("reach", len(frames)//2), ("final", -1)]:
        path = os.path.join(out_dir, f"phase3_{name}.png")
        Image.fromarray(frames[idx]).save(path)
        print(f"  {name} → {path}")

    print("\n🎉 Phase 3 demo video complete!")


if __name__ == "__main__":
    main()
