"""
motor/locomotion.py
====================
G1 locomotion using Unitree's trained RL velocity policy.

Loads the ONNX checkpoint from unitree_rl_mjlab and runs inference
at 50Hz (step_dt=0.02) to produce walking joint targets.

Policy observation (96-dim):
  base_ang_vel (3) + projected_gravity (3) + velocity_commands (3) +
  joint_pos_rel (29) + joint_vel_rel (29) + last_action (29)

Policy action (29-dim):
  Joint position targets = action * scale + offset
"""
import numpy as np
import mujoco
import os
import yaml
from typing import Optional, Tuple

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from motor.unitree_bridge import UnitreeBridge, G1_JOINT_NAMES


# Paths relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POLICY_DIR = os.path.join(
    _PROJECT_ROOT, "unitree_rl_mjlab", "deploy", "robots", "g1",
    "config", "policy", "velocity", "v0"
)
_ONNX_PATH = os.path.join(_POLICY_DIR, "exported", "policy.onnx")
_YAML_PATH = os.path.join(_POLICY_DIR, "params", "deploy.yaml")


class LocomotionController:
    """
    Locomotion using Unitree's trained RL policy (ONNX).

    Usage:
        loco = LocomotionController(bridge)
        loco.walk_to(target_x, target_y)
    """

    def __init__(self, bridge: UnitreeBridge):
        self.bridge = bridge
        self.model = bridge.model
        self.data = bridge.data

        # Load deploy config
        with open(_YAML_PATH, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.step_dt = self.cfg['step_dt']  # 0.02s = 50Hz policy
        self.sim_dt = self.model.opt.timestep   # usually 0.002s
        self.decimation = max(1, int(self.step_dt / self.sim_dt))

        # Joint mapping
        self.joint_ids = self.cfg['joint_ids_map']  # [0..28]
        self.n_joints = len(self.joint_ids)

        # PD gains from the policy
        self.stiffness = np.array(self.cfg['stiffness'], dtype=np.float64)
        self.damping = np.array(self.cfg['damping'], dtype=np.float64)

        # Default pose & action scaling
        self.default_pos = np.array(self.cfg['default_joint_pos'], dtype=np.float64)
        self.action_scale = np.array(self.cfg['actions']['JointPositionAction']['scale'],
                                      dtype=np.float64)
        self.action_offset = np.array(self.cfg['actions']['JointPositionAction']['offset'],
                                       dtype=np.float64)

        # Velocity command limits
        vel_ranges = self.cfg['commands']['base_velocity']['ranges']
        self.vx_range = vel_ranges['lin_vel_x']  # [-0.5, 1.0]
        self.vy_range = vel_ranges['lin_vel_y']  # [-0.5, 0.5]
        self.wz_range = vel_ranges['ang_vel_z']  # [-1.0, 1.0]

        # State
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)

        # Load ONNX policy
        if HAS_ONNX and os.path.exists(_ONNX_PATH):
            self.session = ort.InferenceSession(
                _ONNX_PATH,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            print(f"[Loco] ✅ Loaded Unitree RL policy: {_ONNX_PATH}")
            print(f"       Input: {input_shape}, Output: {self.session.get_outputs()[0].shape}")
            print(f"       Step dt={self.step_dt}s, decimation={self.decimation}")
            self._has_policy = True

            # Override MuJoCo actuator PD gains to match training config
            self._apply_training_pd_gains()
        else:
            print(f"[Loco] ⚠️ ONNX policy not found at {_ONNX_PATH}")
            print(f"       Using fallback velocity controller")
            self._has_policy = False

    def _apply_training_pd_gains(self):
        """
        Override the MuJoCo model's actuator gains to match the stiffness/damping
        from deploy.yaml. This is critical: the RL policy was trained with these
        specific PD parameters, and sim-to-sim transfer requires matching them.
        """
        for i, jid in enumerate(self.joint_ids):
            kp = self.stiffness[i]
            kd = self.damping[i]
            # MuJoCo position actuator:
            #   gainprm[0] = kp (proportional gain)
            #   biasprm[1] = -kp (position bias)
            #   biasprm[2] = -kd (velocity bias / damping)
            self.model.actuator_gainprm[jid, 0] = kp
            self.model.actuator_biasprm[jid, 1] = -kp
            self.model.actuator_biasprm[jid, 2] = -kd
        print(f"[Loco] Applied training PD gains to {len(self.joint_ids)} actuators")

    # ---- Observation Construction ----

    def _get_projected_gravity(self) -> np.ndarray:
        """Get gravity vector projected into the robot's base frame."""
        # Get pelvis rotation matrix
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        rot_mat = self.data.xmat[pelvis_id].reshape(3, 3)
        # World gravity = [0, 0, -9.81], project into body frame
        gravity_world = np.array([0, 0, -1.0])
        return (rot_mat.T @ gravity_world).astype(np.float32)

    def _get_base_ang_vel(self) -> np.ndarray:
        """Get base angular velocity in body frame."""
        imu = self.bridge.get_imu("pelvis")
        return imu['angular_velocity'].astype(np.float32)

    def _get_joint_pos_rel(self) -> np.ndarray:
        """Get joint positions relative to default pose."""
        all_pos = self.bridge.get_joint_positions()
        pos = np.array([all_pos.get(G1_JOINT_NAMES[i], 0.0)
                        for i in self.joint_ids], dtype=np.float32)
        return pos - self.default_pos.astype(np.float32)

    def _get_joint_vel(self) -> np.ndarray:
        """Get joint velocities."""
        all_vel = self.bridge.get_joint_velocities()
        return np.array([all_vel.get(G1_JOINT_NAMES[i], 0.0)
                         for i in self.joint_ids], dtype=np.float32)

    def _build_obs(self, cmd_vx: float, cmd_vy: float, cmd_wz: float) -> np.ndarray:
        """
        Build the 96-dim observation vector for the policy.
        Order: ang_vel(3) + gravity(3) + cmd(3) + joint_pos(29) + joint_vel(29) + last_action(29)
        """
        obs = np.concatenate([
            self._get_base_ang_vel(),           # 3
            self._get_projected_gravity(),       # 3
            np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float32),  # 3
            self._get_joint_pos_rel(),           # 29
            self._get_joint_vel(),               # 29
            self.last_action,                    # 29
        ])
        return obs.reshape(1, -1).astype(np.float32)

    # ---- Policy Inference ----

    def _policy_step(self, cmd_vx: float, cmd_vy: float, cmd_wz: float):
        """Run one policy inference step and apply actions."""
        obs = self._build_obs(cmd_vx, cmd_vy, cmd_wz)

        # Run ONNX inference
        action = self.session.run(
            [self.output_name],
            {self.input_name: obs}
        )[0].flatten()

        self.last_action = action.astype(np.float32)

        # Convert to joint position targets: target = action * scale + offset
        targets = action * self.action_scale + self.action_offset

        # Apply to actuators
        for i, jid in enumerate(self.joint_ids):
            self.data.ctrl[jid] = targets[i]

        # Simulate for decimation steps
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

    def _fallback_step(self, cmd_vx: float, cmd_vy: float, cmd_wz: float):
        """Fallback: direct velocity injection (for when ONNX isn't available)."""
        cx, cy, cyaw = self._get_base_xytheta()
        cos_yaw = np.cos(cyaw)
        sin_yaw = np.sin(cyaw)
        self.data.qvel[0] = cmd_vx * cos_yaw - cmd_vy * sin_yaw
        self.data.qvel[1] = cmd_vx * sin_yaw + cmd_vy * cos_yaw
        self.data.qvel[2] = 0.0
        self.data.qvel[5] = cmd_wz
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

    # ---- High-Level Commands ----

    def _get_base_xytheta(self) -> Tuple[float, float, float]:
        pos = self.bridge.get_base_pos()
        quat = self.bridge.get_base_quat()
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return pos[0], pos[1], yaw

    def walk_velocity(self, vx: float, vy: float, wz: float, duration_s: float = 1.0):
        """
        Send velocity command for a duration.

        Args:
            vx: forward velocity (m/s), range [-0.5, 1.0]
            vy: lateral velocity (m/s), range [-0.5, 0.5]
            wz: yaw rate (rad/s), range [-1.0, 1.0]
            duration_s: how long to execute (seconds)
        """
        vx = np.clip(vx, self.vx_range[0], self.vx_range[1])
        vy = np.clip(vy, self.vy_range[0], self.vy_range[1])
        wz = np.clip(wz, self.wz_range[0], self.wz_range[1])

        n_steps = int(duration_s / self.step_dt)
        step_fn = self._policy_step if self._has_policy else self._fallback_step

        for _ in range(n_steps):
            step_fn(vx, vy, wz)

    def walk_to(self, target_x: float, target_y: float,
                target_theta: Optional[float] = None,
                tolerance: float = 0.15,
                max_duration_s: float = 15.0) -> dict:
        """
        Walk to a target position using the RL velocity policy.

        Args:
            target_x, target_y: target in world frame
            tolerance: arrival distance (m)
            max_duration_s: timeout (seconds)

        Returns:
            dict with 'success', 'final_distance', 'steps', 'positions'
        """
        target = np.array([target_x, target_y])
        positions = []
        n_max = int(max_duration_s / self.step_dt)
        step_fn = self._policy_step if self._has_policy else self._fallback_step

        kp_lin = 1.5
        kp_ang = 2.0

        for step in range(n_max):
            mujoco.mj_forward(self.model, self.data)
            cx, cy, cyaw = self._get_base_xytheta()
            current = np.array([cx, cy])
            dist = np.linalg.norm(target - current)
            positions.append(current.copy())

            if dist < tolerance:
                # Stop
                step_fn(0.0, 0.0, 0.0)
                print(f"[Loco] ✅ Arrived in {step} policy steps "
                      f"({step * self.step_dt:.1f}s, dist={dist:.3f}m)")
                return {
                    "success": True,
                    "final_distance": dist,
                    "steps": step,
                    "positions": positions,
                }

            # Direction to target in world frame
            dx = target[0] - cx
            dy = target[1] - cy
            angle_to_target = np.arctan2(dy, dx)

            # Angle error
            angle_error = angle_to_target - cyaw
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

            # Velocity commands in robot frame
            if abs(angle_error) > 0.4:
                # Turn in place first
                cmd_vx = 0.0
                cmd_vy = 0.0
                cmd_wz = np.clip(kp_ang * angle_error, *self.wz_range)
            else:
                # Forward with correction
                cmd_vx = np.clip(kp_lin * dist, 0, self.vx_range[1])
                cmd_vy = 0.0
                cmd_wz = np.clip(kp_ang * angle_error, *self.wz_range)

            step_fn(cmd_vx, cmd_vy, cmd_wz)

            # Stability check
            base_z = self.bridge.get_base_pos()[2]
            if base_z < 0.4:
                print(f"[Loco] ❌ Robot fell (z={base_z:.3f}m) at step {step}")
                return {
                    "success": False,
                    "final_distance": dist,
                    "steps": step,
                    "positions": positions,
                }

            if step % 50 == 0:
                print(f"  [step {step}] pos=({cx:.3f},{cy:.3f}) "
                      f"dist={dist:.3f}m cmd=({cmd_vx:.2f},{cmd_vy:.2f},{cmd_wz:.2f})")

        final_dist = np.linalg.norm(target - self.bridge.get_base_pos()[:2])
        print(f"[Loco] ⚠️ Timeout ({max_duration_s}s), dist={final_dist:.3f}m")
        return {
            "success": False,
            "final_distance": final_dist,
            "steps": n_max,
            "positions": positions,
        }

    def stand_still(self, duration_s: float = 2.0):
        """Hold zero velocity for a duration."""
        self.walk_velocity(0.0, 0.0, 0.0, duration_s)
        print(f"[Loco] Standing at {self.bridge.get_base_pos()}")
