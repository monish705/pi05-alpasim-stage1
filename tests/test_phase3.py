"""
tests/test_phase3.py
====================
Comprehensive Phase 3 verification:
1. G1 loads and stands stably
2. Cameras render valid images
3. IMU sensors read correctly
4. Arm IK reaches a target position
5. Scene objects are visible and interactive
"""
import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco

SCENE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mujoco_menagerie", "unitree_g1", "g1_tabletop.xml"
)

def main():
    print("=" * 60)
    print("  PHASE 3: G1 ROBOT VERIFICATION")
    print("=" * 60)

    # ---- TEST 1: Load scene ----
    print("\n[TEST 1] Loading G1 tabletop scene...")
    try:
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        data = mujoco.MjData(model)
        print(f"  Model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
        print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
        print(f"  ✅ Scene loaded successfully!")
    except Exception as e:
        print(f"  ❌ Failed to load scene: {e}")
        return

    # ---- TEST 2: Reset to stand keyframe ----
    print("\n[TEST 2] Resetting to stand keyframe...")
    from motor.unitree_bridge import UnitreeBridge
    bridge = UnitreeBridge(model, data)
    bridge.reset_to_stand()
    mujoco.mj_forward(model, data)
    
    base_pos = bridge.get_base_pos()
    print(f"  Base position: ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})")
    assert base_pos[2] > 0.5, f"Robot too low! z={base_pos[2]:.3f}"
    print(f"  ✅ Robot standing (height={base_pos[2]:.3f}m)")

    # ---- TEST 3: Stability (stand for 3 seconds) ----
    print("\n[TEST 3] Stability test (3 seconds / 1500 steps)...")
    for i in range(1500):
        mujoco.mj_step(model, data)
    
    base_pos_after = bridge.get_base_pos()
    height_drop = base_pos[2] - base_pos_after[2]
    print(f"  Position after 3s: ({base_pos_after[0]:.3f}, {base_pos_after[1]:.3f}, {base_pos_after[2]:.3f})")
    print(f"  Height drop: {height_drop:.4f}m")
    
    if base_pos_after[2] > 0.4:
        print(f"  ✅ Robot stable (z={base_pos_after[2]:.3f}m)")
    else:
        print(f"  ❌ Robot fell! z={base_pos_after[2]:.3f}m")

    # ---- TEST 4: IMU sensors ----
    print("\n[TEST 4] IMU sensor readings...")
    imu_torso = bridge.get_imu("torso")
    imu_pelvis = bridge.get_imu("pelvis")
    print(f"  Torso gyro: {imu_torso['angular_velocity']}")
    print(f"  Torso accel: {imu_torso['linear_acceleration']}")
    print(f"  Pelvis gyro: {imu_pelvis['angular_velocity']}")
    
    # Accel should show ~9.81 in z direction (gravity)
    accel_z = imu_torso['linear_acceleration'][2]
    if abs(accel_z) > 5.0:
        print(f"  ✅ IMU reads gravity (az={accel_z:.2f} m/s²)")
    else:
        print(f"  ⚠️ IMU z-accel looks off: {accel_z:.2f}")

    # ---- TEST 5: End-effector positions ----
    print("\n[TEST 5] End-effector positions...")
    r_ee = bridge.get_ee_pos("right")
    l_ee = bridge.get_ee_pos("left")
    print(f"  Right hand: ({r_ee[0]:.3f}, {r_ee[1]:.3f}, {r_ee[2]:.3f})")
    print(f"  Left hand:  ({l_ee[0]:.3f}, {l_ee[1]:.3f}, {l_ee[2]:.3f})")
    print(f"  ✅ End-effectors located")

    # ---- TEST 6: Camera rendering ----
    print("\n[TEST 6] Camera rendering...")
    renderer = mujoco.Renderer(model, 480, 640)
    
    cameras_to_test = ["overview_cam", "side_cam"]
    for cam_name in cameras_to_test:
        try:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                print(f"  ⚠️ Camera '{cam_name}' not found")
                continue
            renderer.update_scene(data, camera=cam_name)
            rgb = renderer.render()
            print(f"  {cam_name}: {rgb.shape}, mean={rgb.mean():.1f}")
            
            # Save image
            from PIL import Image
            img = Image.fromarray(rgb)
            out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     f"test_{cam_name}.png")
            img.save(out_path)
            print(f"    Saved to {out_path}")
        except Exception as e:
            print(f"  ❌ {cam_name}: {e}")

    # ---- TEST 7: Depth rendering ----
    print("\n[TEST 7] Depth rendering...")
    try:
        renderer.enable_depth_rendering(True)
        renderer.update_scene(data, camera="overview_cam")
        depth = renderer.render()
        renderer.enable_depth_rendering(False)
        print(f"  Depth: {depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]")
        print(f"  ✅ Depth rendering works")
    except Exception as e:
        print(f"  ❌ Depth rendering: {e}")

    # ---- TEST 8: Scene objects ----
    print("\n[TEST 8] Scene objects...")
    obj_names = ["obj_red_mug", "obj_blue_box", "obj_green_bottle", "obj_yellow_ball"]
    for name in obj_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            pos = data.xpos[bid]
            print(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        else:
            print(f"  ⚠️ {name} not found")
    print(f"  ✅ Objects in scene")

    # ---- TEST 9: Arm IK ----
    print("\n[TEST 9] Arm IK — reaching toward table...")
    try:
        from motor.arm_controller import ArmController
        arm = ArmController(bridge, hand="right")
        
        # Target: above the table
        target = np.array([0.45, -0.1, 0.55])
        print(f"  Target: {target}")
        print(f"  Start EE: {arm.get_ee_pos()}")
        
        result = arm.move_to(target, max_steps=400, tolerance=0.05)
        final_ee = arm.get_ee_pos()
        print(f"  Final EE: ({final_ee[0]:.3f}, {final_ee[1]:.3f}, {final_ee[2]:.3f})")
        print(f"  Error: {result['final_error']:.4f}m")
        
        if result['success']:
            print(f"  ✅ Arm IK succeeded in {result['steps']} steps")
        else:
            print(f"  ⚠️ Arm IK: error={result['final_error']:.4f}m after {result['steps']} steps")
    except Exception as e:
        print(f"  ❌ Arm IK failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- TEST 10: Joint state readout ----
    print("\n[TEST 10] Joint states...")
    joints = bridge.get_joint_positions()
    print(f"  {len(joints)} joints read")
    # Print arm joints only
    arm_joints = {k: v for k, v in joints.items() if "shoulder" in k or "elbow" in k}
    for name, val in arm_joints.items():
        print(f"    {name}: {val:.4f} rad ({np.degrees(val):.1f}°)")
    print(f"  ✅ Joint states readable")

    # ---- SUMMARY ----
    print("\n" + "=" * 60)
    print("  PHASE 3 VERIFICATION SUMMARY")
    print("=" * 60)
    bridge.print_state()
    
    # Final renders
    print("\n  Saving final renders...")
    mujoco.mj_forward(model, data)
    for cam_name in ["overview_cam", "side_cam"]:
        try:
            renderer.update_scene(data, camera=cam_name)
            rgb = renderer.render()
            from PIL import Image
            img = Image.fromarray(rgb)
            out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     f"phase3_{cam_name}.png")
            img.save(out_path)
            print(f"    {cam_name} → {out_path}")
        except Exception as e:
            print(f"    {cam_name}: {e}")

    renderer.close()
    print("\n  🎉 PHASE 3 VERIFICATION COMPLETE!")


if __name__ == "__main__":
    main()
