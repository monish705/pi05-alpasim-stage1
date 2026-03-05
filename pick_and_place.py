"""
pick_and_place.py
=================
VLA + Learned RL Motor Policy → MuJoCo G1 Sim.

    Camera image + task instruction
        → VLA (pi0 / OpenVLA) outputs 7D arm action
        → Learned RL policy (unitree_rl_mjlab ONNX) keeps body balanced
        → MuJoCo steps physics

That's it. No custom controllers.

Usage:
    python pick_and_place.py
    python pick_and_place.py --instruction "pick up the blue cube"
    python pick_and_place.py --render
"""
import sys, os, time, argparse
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import onnxruntime as ort
import yaml

# === Paths ===
_ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_ROOT, "sim", "scenes", "g1_pick_and_place.xml")

_POLICY_DIR = os.path.join(
    _ROOT, "unitree_rl_mjlab", "deploy", "robots", "g1",
    "config", "policy", "velocity", "v0"
)
_ONNX_PATH = os.path.join(_POLICY_DIR, "exported", "policy.onnx")
_YAML_PATH = os.path.join(_POLICY_DIR, "params", "deploy.yaml")

# Right arm actuator indices in G1's 29-joint layout
RIGHT_ARM = list(range(22, 29))


def load_rl_policy():
    """Load Unitree's trained RL locomotion policy (ONNX)."""
    with open(_YAML_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    session = ort.InferenceSession(_ONNX_PATH, providers=['CPUExecutionProvider'])
    print(f"[RL] Loaded locomotion policy: {_ONNX_PATH}")
    return session, cfg


def load_vla():
    """Load VLA model (pi0 or OpenVLA)."""
    vla_path = os.path.join(_ROOT, "..", "vla_humanoid")
    sys.path.insert(0, vla_path)
    from vla_agent.model import VLAPolicy
    vla = VLAPolicy()
    print(f"[VLA] Loaded: {vla.model_type}")
    return vla


def get_gravity_in_body(data, pelvis_id):
    """Project gravity into body frame."""
    rot = data.xmat[pelvis_id].reshape(3, 3)
    return (rot.T @ np.array([0, 0, -1.0])).astype(np.float32)


def build_rl_obs(model, data, cfg, last_action, cmd=(0., 0., 0.)):
    """Build 96-dim observation for the RL locomotion policy."""
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    joint_ids = cfg['joint_ids_map']
    default_pos = np.array(cfg['default_joint_pos'], dtype=np.float32)

    # Angular velocity (from qvel[3:6] in body frame)
    ang_vel = data.qvel[3:6].astype(np.float32)

    # Projected gravity
    grav = get_gravity_in_body(data, pelvis_id)

    # Joint positions relative to default
    joint_pos = np.array([float(data.qpos[model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg['joint_names'][i])
    ]]) for i in range(len(joint_ids))], dtype=np.float32) if 'joint_names' in cfg else \
        data.qpos[7:7+len(joint_ids)].astype(np.float32)

    joint_pos_rel = joint_pos - default_pos

    # Joint velocities
    joint_vel = data.qvel[6:6+len(joint_ids)].astype(np.float32)

    # Command
    cmd_arr = np.array(cmd, dtype=np.float32)

    obs = np.concatenate([ang_vel, grav, cmd_arr, joint_pos_rel, joint_vel, last_action])
    return obs.reshape(1, -1).astype(np.float32)


def run(instruction, max_steps, render):
    print("=" * 50)
    print("  G1 Pick-and-Place")
    print(f"  Task: {instruction}")
    print("=" * 50)

    # 1. Load scene
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    print(f"[SIM] Loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # 2. Load learned RL policy (whole body balance + locomotion)
    rl_session, rl_cfg = load_rl_policy()
    rl_input = rl_session.get_inputs()[0].name
    rl_output = rl_session.get_outputs()[0].name
    n_joints = len(rl_cfg['joint_ids_map'])
    action_scale = np.array(rl_cfg['actions']['JointPositionAction']['scale'], dtype=np.float64)
    action_offset = np.array(rl_cfg['actions']['JointPositionAction']['offset'], dtype=np.float64)
    step_dt = rl_cfg['step_dt']
    sim_dt = model.opt.timestep
    decimation = max(1, int(step_dt / sim_dt))
    last_rl_action = np.zeros(n_joints, dtype=np.float32)

    # Apply training PD gains
    stiffness = np.array(rl_cfg['stiffness'], dtype=np.float64)
    damping = np.array(rl_cfg['damping'], dtype=np.float64)
    for i, jid in enumerate(rl_cfg['joint_ids_map']):
        model.actuator_gainprm[jid, 0] = stiffness[i]
        model.actuator_biasprm[jid, 1] = -stiffness[i]
        model.actuator_biasprm[jid, 2] = -damping[i]

    # 3. Load VLA (learned manipulation policy)
    vla = load_vla()

    # 4. Camera renderer
    renderer = mujoco.Renderer(model, 224, 224)

    # 5. Reset robot
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Settle objects
    for _ in range(500):
        mujoco.mj_step(model, data)

    # Viewer
    viewer = None
    if render:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as e:
            print(f"[WARN] Viewer: {e}")

    # 6. Control loop
    print(f"\n[LOOP] Running {max_steps} steps...")
    for step in range(max_steps):

        # --- Camera → VLA → arm action ---
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="head_cam")
        image = renderer.render()

        vla_action = vla.predict(image, instruction)  # 7D [-1, 1]

        # Map VLA output to right arm actuators
        for i, idx in enumerate(RIGHT_ARM):
            if i < len(vla_action):
                lo = model.actuator_ctrlrange[idx, 0]
                hi = model.actuator_ctrlrange[idx, 1]
                data.ctrl[idx] = lo + (vla_action[i] + 1.0) / 2.0 * (hi - lo)

        # --- RL policy → whole body balance (legs + waist) ---
        obs = build_rl_obs(model, data, rl_cfg, last_rl_action, cmd=(0., 0., 0.))
        rl_action = rl_session.run([rl_output], {rl_input: obs})[0].flatten()
        last_rl_action = rl_action.astype(np.float32)

        targets = rl_action * action_scale + action_offset
        # Apply RL policy to legs + waist only (indices 0-14), skip arms
        for i in range(min(15, len(targets))):
            jid = rl_cfg['joint_ids_map'][i]
            if jid < 22:  # legs + waist only
                data.ctrl[jid] = targets[i]

        # --- Step sim ---
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        if viewer:
            viewer.sync()

        # Log every 50 steps
        if step % 50 == 0:
            base = data.qpos[:3]
            print(f"  [step {step:4d}] base=({base[0]:.2f},{base[1]:.2f},{base[2]:.2f}) "
                  f"vla={np.round(vla_action[:3], 2)}")

    print("\n[DONE]")
    renderer.close()
    if viewer:
        viewer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", default="pick up the red mug and place it on the other table")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    run(args.instruction, args.steps, args.render)


if __name__ == "__main__":
    main()
