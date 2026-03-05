"""
Local test of the perception world model pipeline.
Runs entirely on CPU — no GPU, no SAM3, no VLM.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
from pathlib import Path

SCENE_XML = str(Path(__file__).parent.parent / "mujoco_menagerie" / "unitree_g1" / "g1_tabletop.xml")

def main():
    print("=" * 60)
    print("  WORLD MODEL — LOCAL PERCEPTION TEST")
    print("=" * 60)

    # --- 1. Boot sim ---
    print("\n[1/6] Loading MuJoCo scene...")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    from motor.unitree_bridge import UnitreeBridge
    bridge = UnitreeBridge(model, data)
    bridge.reset_to_stand()

    for _ in range(500):
        mujoco.mj_step(model, data)

    base_z = bridge.get_base_pos()[2]
    assert base_z > 0.5, f"Robot fell! z={base_z:.3f}"
    print(f"  ✅ Robot standing at z={base_z:.3f}m")

    # Ground truth
    gt = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("obj_"):
            gt[name] = data.xpos[i].copy()
            print(f"  GT {name}: ({data.xpos[i][0]:.3f}, {data.xpos[i][1]:.3f}, {data.xpos[i][2]:.3f})")

    # --- 2. Render cameras ---
    print("\n[2/6] Rendering cameras...")
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

    cam = "overview_cam"
    rgb = render_rgb(cam)
    depth = render_depth(cam)
    print(f"  {cam}: RGB {rgb.shape}, Depth [{depth.min():.2f}, {depth.max():.2f}]")

    # Save a frame
    from PIL import Image
    Image.fromarray(rgb).save("test_overview.png")
    print("  Saved test_overview.png")

    # Intrinsics
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    fovy = model.cam_fovy[cam_id]
    f = 0.5 * 480 / np.tan(fovy * np.pi / 360)
    intrinsics = {"fx": f, "fy": f, "cx": 320, "cy": 240}

    # Extrinsics
    pos = data.cam_xpos[cam_id].copy()
    rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rot @ flip
    extrinsics[:3, 3] = pos

    # Sim body names
    sim_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(model.nbody)
                 if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                 and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i).startswith("obj_")]

    # --- 3. Run perception pipeline ---
    print("\n[3/6] Running perception pipeline...")
    from perception.pipeline import PerceptionPipeline, PipelineConfig

    config = PipelineConfig(
        use_sim_tagger=True,
        depth_fallback=True,
        min_confidence=0.10,
    )
    pipeline = PerceptionPipeline(config=config)
    scene_graph = pipeline.perceive(
        rgb, depth, intrinsics,
        camera_extrinsics=extrinsics,
        sim_body_names=sim_names,
    )

    # --- 4. Validate ---
    print("\n[4/6] Validating against ground truth...")
    n_found = 0
    n_accurate = 0
    for gt_name, gt_pos in gt.items():
        keyword = gt_name.replace("obj_", "").replace("_", " ")
        match = scene_graph.query(keyword)
        if match:
            n_found += 1
            error = np.linalg.norm(np.array(match.position_world) - gt_pos)
            if error < 0.10:
                n_accurate += 1
            status = "✅" if error < 0.10 else "❌"
            print(f"  {gt_name}: error={error:.3f}m {status} (T{match.track_id}: {match.label})")
        else:
            print(f"  {gt_name}: NOT FOUND ❌")

    # --- 5. Multi-frame tracking ---
    print("\n[5/6] Multi-frame tracking (3 frames)...")
    for frame in range(3):
        for _ in range(50):
            mujoco.mj_step(model, data)
        rgb2 = render_rgb(cam)
        depth2 = render_depth(cam)
        pipeline.perceive(rgb2, depth2, intrinsics,
                         camera_extrinsics=extrinsics,
                         sim_body_names=sim_names)
        objs = scene_graph.get_all_objects()
        tracks = sorted([(o.track_id, o.label, o.frames_since_seen) for o in objs])
        print(f"  Frame {scene_graph.frame_count}: {len(objs)} objects, "
              f"tracks={[t[0] for t in tracks]}")

    # --- 6. Export ---
    print("\n[6/6] Exporting...")
    scene_graph.save_json("scene_graph_local_test.json")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  RESULTS:")
    print(f"    Objects found:    {n_found}/{len(gt)}")
    print(f"    Within 10cm:      {n_accurate}/{len(gt)}")
    print(f"    Frames processed: {scene_graph.frame_count}")
    print(f"    Track IDs stable: {sorted([o.track_id for o in scene_graph.get_all_objects()])}")
    if n_found == len(gt):
        print(f"\n  ✅ ALL OBJECTS DETECTED")
    else:
        print(f"\n  ⚠️  {len(gt) - n_found} objects missed (expected in fallback mode)")
    print(f"{'='*60}")

    renderer.close()

if __name__ == "__main__":
    main()
