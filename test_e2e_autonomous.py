"""
test_e2e_autonomous.py
=======================
End-to-end autonomous task test.

Runs the FULL system:
  MuJoCo sim → cameras → SAM3 segmentation → SAM3D 3D mesh →
  Qwen3-VL tagging → depth validation → scene graph →
  VLM code generation → motor execution → verification

No fakes. No placeholders. Every component is the real one.

Usage:
    # On Colab (local VLM on GPU):
    from cognitive.local_vlm import LocalVLM
    vlm = LocalVLM("Qwen/Qwen2.5-VL-7B-Instruct")
    run_end_to_end_test(local_vlm=vlm)

    # On server with vLLM:
    run_end_to_end_test(vlm_api="http://gpu:8000/v1")
"""
import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
from sim.mujoco_env import MuJoCoEnv
from motor.unitree_bridge import UnitreeBridge
from motor.locomotion import LocomotionController
from motor.arm_controller import ArmController
from motor.grasp_controller import GraspController
from perception.pipeline import PerceptionPipeline
from perception.scene_graph import SceneGraph
from cognitive.vlm_executor import VLMExecutor


SCENE_XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mujoco_menagerie", "unitree_g1", "g1_tabletop.xml"
)


def run_end_to_end_test(scene_xml: str = SCENE_XML,
                        vlm_api: str = "http://localhost:8000/v1",
                        vlm_model: str = "Qwen/Qwen3-VL-8B",
                        local_vlm=None):
    """
    Run the full autonomous task test.

    Args:
        scene_xml: path to MuJoCo scene
        vlm_api: vLLM API URL (used if local_vlm is None)
        vlm_model: model name
        local_vlm: optional LocalVLM instance for Colab
    """

    print("=" * 60)
    print("  END-TO-END AUTONOMOUS TASK TEST")
    print("=" * 60)

    # =========================================================
    # PHASE 1: INITIALISE REALITY (SIMULATION)
    # =========================================================
    print("\n--- PHASE 1: INITIALISING SIMULATION ---")

    env = MuJoCoEnv(scene_xml, render_mode="headless")
    bridge = UnitreeBridge(env.model, env.data)
    bridge.reset_to_stand()

    # Motor controllers
    loco = LocomotionController(bridge)
    arm = ArmController(bridge, hand="right")
    grasp = GraspController(bridge)

    # Let robot settle
    loco.stand_still(1.0)
    base_z = bridge.get_base_pos()[2]
    assert base_z > 0.7, f"Robot fell! z={base_z:.3f}"
    print(f"  ✅ Robot standing at z={base_z:.3f}m")

    # Renderer for camera images
    renderer = mujoco.Renderer(env.model, 480, 640)

    def render_rgb(cam):
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, camera=cam)
        return renderer.render().copy()

    def render_depth(cam):
        mujoco.mj_forward(env.model, env.data)
        renderer.enable_depth_rendering()
        renderer.update_scene(env.data, camera=cam)
        d = renderer.render().copy()
        renderer.disable_depth_rendering()
        return d

    def get_intrinsics(cam):
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        fovy = env.model.cam_fovy[cam_id]
        f = 0.5 * 480 / np.tan(fovy * np.pi / 360)
        return {"fx": f, "fy": f, "cx": 320, "cy": 240}

    print(f"  ✅ Simulation initialised")

    # =========================================================
    # PHASE 2: ASSIGN THE TASK
    # =========================================================
    print("\n--- PHASE 2: TASK ASSIGNMENT ---")
    task = "Walk over to the table and pick up the red mug."
    print(f"  Task: \"{task}\"")

    # =========================================================
    # PHASE 3: PERCEPTION (SAM3 → SAM3D → VLM → Scene Graph)
    # =========================================================
    print("\n--- PHASE 3: PERCEPTION ---")
    print("  Running: SAM3 → SAM3D → Qwen3-VL → Depth Validation → Scene Graph")

    # Render head camera
    rgb = render_rgb("head_cam")
    depth = render_depth("head_cam")
    intrinsics = get_intrinsics("head_cam")
    print(f"  Head cam: RGB {rgb.shape}, Depth {depth.shape}")

    # Run the FULL perception pipeline
    # This calls: SAM3 segmentor → SAM3D reconstructor → VLM tagger → depth validator
    perception = PerceptionPipeline(
        vlm_api_base=vlm_api,
        vlm_model=vlm_model,
    )
    scene_graph = perception.perceive(rgb, depth, intrinsics)

    # Validate: did perception find the mug?
    scene_json = scene_graph.to_json()
    print(f"\n  Scene Graph: {len(scene_graph.objects)} objects detected")
    scene_graph.print_all()

    mug = scene_graph.query("red mug")
    assert mug is not None, "PERCEPTION FAILED: 'red mug' not found in scene graph!"
    print(f"\n  ✅ Target found: '{mug.label}' at {mug.position_world}")

    # Add robot state to scene JSON
    scene_data = json.loads(scene_json)
    scene_data["robot"] = {
        "base_pos": bridge.get_base_pos().tolist(),
        "right_hand_pos": bridge.get_ee_pos("right").tolist(),
        "is_grasping": False,
    }
    full_scene_json = json.dumps(scene_data, indent=2)

    # =========================================================
    # PHASE 4: COGNITIVE REASONING (VLM Code Generation)
    # =========================================================
    print("\n--- PHASE 4: VLM CODE GENERATION ---")

    motor_context = {
        "loco": loco,
        "arm": arm,
        "grasp": grasp,
        "bridge": bridge,
        "np": np,
        "time": time,
    }

    executor = VLMExecutor(
        api_base=vlm_api,
        model_name=vlm_model,
        motor_context=motor_context,
        local_vlm=local_vlm,
    )

    # Generate code
    code = executor.generate_code(task, full_scene_json)
    print(f"  VLM generated {len(code)} chars of code")
    print("  --- GENERATED CODE ---")
    print(code)
    print("  --- END CODE ---")

    # =========================================================
    # PHASE 5: AUTONOMOUS EXECUTION
    # =========================================================
    print("\n--- PHASE 5: AUTONOMOUS EXECUTION ---")

    success, output, error = executor.execute_code(code)

    if output:
        print(f"  Robot output: {output}")

    if success:
        print("  ✅ Code executed without errors")
    else:
        print(f"  ❌ Execution error: {error}")
        # Retry with error feedback
        print("  Retrying with error feedback...")
        code2 = executor.generate_code(task, full_scene_json, error_feedback=error)
        success2, output2, error2 = executor.execute_code(code2)
        if success2:
            print("  ✅ Retry succeeded!")
        else:
            print(f"  ❌ Retry also failed: {error2}")

    # =========================================================
    # PHASE 6: VERIFICATION
    # =========================================================
    print("\n--- PHASE 6: VERIFICATION ---")

    final_pos = bridge.get_base_pos()
    final_hand = bridge.get_ee_pos("right")
    is_grasping = grasp.is_grasping("right")
    grasped_obj = grasp.get_grasped_object("right")
    is_stable = final_pos[2] > 0.5

    print(f"  Robot base:    ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
    print(f"  Right hand:    ({final_hand[0]:.2f}, {final_hand[1]:.2f}, {final_hand[2]:.2f})")
    print(f"  Grasping:      {is_grasping}")
    print(f"  Grasped obj:   {grasped_obj}")
    print(f"  Robot stable:  {is_stable}")

    # Re-run perception to verify world changed
    rgb_after = render_rgb("head_cam")
    depth_after = render_depth("head_cam")
    scene_after = perception.perceive(rgb_after, depth_after, intrinsics)

    # Check results
    results = {
        "task": task,
        "code_executed": success,
        "grasping": is_grasping,
        "grasped_object": grasped_obj,
        "robot_stable": is_stable,
        "robot_final_pos": final_pos.tolist(),
        "hand_final_pos": final_hand.tolist(),
        "objects_before": len(scene_graph.objects),
        "objects_after": len(scene_after.objects),
    }

    print(f"\n{'='*60}")
    if is_grasping and is_stable:
        print(f"  ✅ TEST PASSED")
        print(f"  Robot autonomously perceived, walked, reached, and grasped.")
        print(f"  Holding: {grasped_obj}")
    elif success and is_stable:
        print(f"  ⚠️  PARTIAL: Code ran, robot stable, but not grasping.")
        print(f"  The VLM code may not have completed the grasp.")
    else:
        print(f"  ❌ TEST FAILED")
        if not success:
            print(f"  Reason: VLM code failed to execute")
        if not is_stable:
            print(f"  Reason: Robot fell (z={final_pos[2]:.3f})")
        if not is_grasping:
            print(f"  Reason: Object not grasped")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-url", default="http://localhost:8000/v1")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B")
    args = parser.parse_args()
    run_end_to_end_test(vlm_api=args.vlm_url, vlm_model=args.vlm_model)
