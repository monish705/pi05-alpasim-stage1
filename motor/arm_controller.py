"""
motor/arm_controller.py
========================
CoM-aware Jacobian IK arm controller for the G1 robot.

Key improvements over naive IK:
  1. Support polygon check — rejects targets that would shift CoM outside feet
  2. Waist compensation — uses torso joints to lean back and extend reach
  3. Self-collision detection — aborts if arm trajectory causes self-contact
  4. Reachability envelope — pre-filters obviously unreachable targets
  
Inspired by whole-body control patterns from:
  - unitree_mujoco (Unitree's official sim bridge)
  - MuJoCo Menagerie manipulation examples
  - Resolved-rate + null-space humanoid control literature
"""
import numpy as np
import mujoco
from typing import Optional, Dict
from motor.unitree_bridge import (
    UnitreeBridge, RIGHT_ARM_INDICES, LEFT_ARM_INDICES,
    RIGHT_HAND_BODY, LEFT_HAND_BODY, G1_JOINT_NAMES,
    WAIST_INDICES
)


class ArmController:
    """
    Whole-body aware arm controller for G1.
    
    Uses the 3 waist joints (yaw, roll, pitch) together with the 7 arm joints
    to reach targets while keeping the CoM over the support polygon.
    """

    # Approximate foot positions relative to pelvis (from G1 MJCF)
    FOOT_HALF_WIDTH = 0.08   # lateral distance from centre to foot edge
    FOOT_FRONT = 0.15        # forward reach of support polygon
    FOOT_BACK = -0.08        # backward extent

    def __init__(self, bridge: UnitreeBridge, hand: str = "right"):
        self.bridge = bridge
        self.model = bridge.model
        self.data = bridge.data
        self.hand = hand

        # Arm joint indices
        self._arm_indices = RIGHT_ARM_INDICES if hand == "right" else LEFT_ARM_INDICES
        self._arm_joint_names = [G1_JOINT_NAMES[i] for i in self._arm_indices]
        self._ee_body = RIGHT_HAND_BODY if hand == "right" else LEFT_HAND_BODY
        self._ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self._ee_body
        )

        # Waist + arm combined for whole-body IK
        self._wb_indices = list(WAIST_INDICES) + list(self._arm_indices)  # 3 waist + 7 arm = 10 DoF
        self._wb_joint_names = [G1_JOINT_NAMES[i] for i in self._wb_indices]

        # Get qpos and dof addresses
        self._qpos_addrs = []
        self._dof_addrs = []
        for name in self._wb_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._qpos_addrs.append(self.model.jnt_qposadr[jid])
            self._dof_addrs.append(self.model.jnt_dofadr[jid])

        # Joint limits for waist + arm
        self._joint_limits = np.zeros((len(self._wb_indices), 2))
        for i, name in enumerate(self._wb_joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_limits[i] = self.model.jnt_range[jid]

        # Pelvis body ID for CoM reference
        self._pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        # Max reach from shoulder (approximate)
        self._max_reach = 0.65  # metres

        print(f"[ArmCtrl] {hand} arm: {len(self._wb_indices)} DoF "
              f"(3 waist + 7 arm), EE='{self._ee_body}'")

    # ---- Queries ----

    def get_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._ee_body_id].copy()

    def get_ee_mat(self) -> np.ndarray:
        return self.data.xmat[self._ee_body_id].reshape(3, 3).copy()

    def get_com(self) -> np.ndarray:
        """Compute whole-body centre of mass."""
        com = np.zeros(3)
        total_mass = 0.0
        for i in range(self.model.nbody):
            m = self.model.body_mass[i]
            com += m * self.data.xipos[i]
            total_mass += m
        return com / total_mass if total_mass > 0 else com

    def get_support_polygon(self) -> tuple:
        """
        Get the support polygon bounds in world frame.
        Returns (x_min, x_max, y_min, y_max) based on foot positions.
        """
        pelvis_pos = self.data.xpos[self._pelvis_id]

        # Left and right foot sites
        lf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        rf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        if lf_id >= 0 and rf_id >= 0:
            lf_pos = self.data.site_xpos[lf_id]
            rf_pos = self.data.site_xpos[rf_id]
            x_min = min(lf_pos[0], rf_pos[0]) - 0.05
            x_max = max(lf_pos[0], rf_pos[0]) + 0.15
            y_min = min(lf_pos[1], rf_pos[1]) - 0.04
            y_max = max(lf_pos[1], rf_pos[1]) + 0.04
        else:
            # Fallback: estimate from pelvis
            x_min = pelvis_pos[0] + self.FOOT_BACK
            x_max = pelvis_pos[0] + self.FOOT_FRONT
            y_min = pelvis_pos[1] - self.FOOT_HALF_WIDTH
            y_max = pelvis_pos[1] + self.FOOT_HALF_WIDTH

        return x_min, x_max, y_min, y_max

    def is_com_stable(self, margin: float = 0.02) -> bool:
        """Check if CoM XY is within the support polygon."""
        com = self.get_com()
        x_min, x_max, y_min, y_max = self.get_support_polygon()
        return (x_min + margin <= com[0] <= x_max - margin and
                y_min + margin <= com[1] <= y_max - margin)

    def is_reachable(self, target_pos: np.ndarray) -> bool:
        """Quick check if target is within approximate reach envelope."""
        # Shoulder position (approximate: torso_link + shoulder offset)
        shoulder_pos = self.data.xpos[self._pelvis_id].copy()
        shoulder_pos[2] += 0.45  # approximate shoulder height above pelvis
        dist = np.linalg.norm(target_pos - shoulder_pos)
        return dist < self._max_reach

    # ---- Jacobian ----

    def compute_jacobian(self) -> np.ndarray:
        """
        Compute positional Jacobian for the end-effector,
        extracting columns for waist + arm DoFs.
        Returns: (3 x n_wb_joints) Jacobian
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr,
                       self.data.xpos[self._ee_body_id], self._ee_body_id)
        return jacp[:, self._dof_addrs]

    # ---- Main Control ----

    def move_to(self, target_pos: np.ndarray,
                max_steps: int = 500,
                tolerance: float = 0.03,
                step_size: float = 1.0,
                com_aware: bool = True) -> dict:
        """
        Move end-effector to target position using whole-body IK.
        
        Uses waist joints to compensate for CoM shift during arm motion.
        Aborts if CoM leaves the support polygon.
        
        Args:
            target_pos: [x, y, z] target in world frame
            max_steps: maximum iterations
            tolerance: position error threshold (m)
            step_size: IK gain
            com_aware: if True, check CoM stability each step
            
        Returns:
            dict with 'success', 'final_error', 'steps', 'trajectory',
                       'com_stable', 'abort_reason'
        """
        target_pos = np.array(target_pos, dtype=np.float64)
        trajectory = []

        # Pre-check reachability
        if not self.is_reachable(target_pos):
            print(f"[ArmCtrl] ❌ Target unreachable (too far from shoulder)")
            return {
                "success": False, "final_error": float('inf'),
                "steps": 0, "trajectory": [],
                "com_stable": True, "abort_reason": "unreachable"
            }

        for step in range(max_steps):
            mujoco.mj_forward(self.model, self.data)

            current_pos = self.get_ee_pos()
            error_vec = target_pos - current_pos
            error = np.linalg.norm(error_vec)
            trajectory.append(current_pos.copy())

            if error < tolerance:
                com_ok = self.is_com_stable()
                print(f"[ArmCtrl] ✅ Reached target in {step} steps "
                      f"(error={error:.4f}m, CoM={'stable' if com_ok else 'UNSTABLE'})")
                return {
                    "success": True, "final_error": error,
                    "steps": step, "trajectory": trajectory,
                    "com_stable": com_ok, "abort_reason": None
                }

            # CoM stability check
            if com_aware and step > 10 and not self.is_com_stable(margin=0.01):
                print(f"[ArmCtrl] ⚠️ CoM leaving support polygon at step {step}. "
                      f"Backing off...")
                # Try to recover: undo last step
                for i, idx in enumerate(self._wb_indices):
                    self.data.ctrl[idx] -= self._last_dq[i] if hasattr(self, '_last_dq') else 0
                for _ in range(10):
                    mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)

                return {
                    "success": False, "final_error": error,
                    "steps": step, "trajectory": trajectory,
                    "com_stable": False, "abort_reason": "com_unstable"
                }

            # Clamp max step size
            max_dx = 0.015  # 1.5cm per step
            if error > max_dx:
                error_vec = error_vec / error * max_dx

            # Compute whole-body Jacobian
            J = self.compute_jacobian()

            # Damped least squares with waist regularisation
            damping = 0.05
            n_joints = J.shape[1]

            # Weight matrix: penalise waist joints more (prefer arm-only motion)
            W = np.eye(n_joints)
            W[0, 0] = 0.3   # waist_yaw — allow some
            W[1, 1] = 0.2   # waist_roll — less
            W[2, 2] = 0.2   # waist_pitch — less

            # Weighted damped least squares: dq = W J^T (J W J^T + λ²I)^{-1} dx
            JW = J @ W
            JJT = JW @ J.T + damping**2 * np.eye(3)
            dq = step_size * W @ J.T @ np.linalg.solve(JJT, error_vec)

            self._last_dq = dq.copy()

            # Apply to actuators
            for i, idx in enumerate(self._wb_indices):
                new_q = self.data.ctrl[idx] + dq[i]
                new_q = np.clip(new_q, self._joint_limits[i, 0],
                                self._joint_limits[i, 1])
                self.data.ctrl[idx] = new_q

            # Step simulation (multiple sub-steps for PD tracking)
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)

            # Self-collision check
            if self.data.ncon > 30:  # more contacts than expected
                # Check if any are self-collisions (both geoms on robot)
                self_collisions = 0
                for c in range(self.data.ncon):
                    g1 = self.data.contact[c].geom1
                    g2 = self.data.contact[c].geom2
                    b1 = self.model.geom_bodyid[g1]
                    b2 = self.model.geom_bodyid[g2]
                    # Both bodies have parent chain to pelvis = self-collision
                    if b1 > 0 and b2 > 0 and b1 != b2:
                        # Simple heuristic: if both body IDs are < 30 (robot bodies)
                        if b1 < 30 and b2 < 30:
                            self_collisions += 1
                if self_collisions > 5:
                    print(f"[ArmCtrl] ⚠️ Self-collision detected ({self_collisions}), stopping")
                    return {
                        "success": False, "final_error": error,
                        "steps": step, "trajectory": trajectory,
                        "com_stable": True, "abort_reason": "self_collision"
                    }

        # Timeout
        mujoco.mj_forward(self.model, self.data)
        final_error = np.linalg.norm(target_pos - self.get_ee_pos())
        print(f"[ArmCtrl] ⚠️ Did not converge after {max_steps} steps "
              f"(error={final_error:.4f}m)")
        return {
            "success": False, "final_error": final_error,
            "steps": max_steps, "trajectory": trajectory,
            "com_stable": self.is_com_stable(), "abort_reason": "timeout"
        }

    def move_trajectory(self, waypoints: list,
                        tolerance: float = 0.04,
                        max_steps_per_wp: int = 300) -> dict:
        """Follow a sequence of Cartesian waypoints with CoM awareness."""
        total_steps = 0
        for i, wp in enumerate(waypoints):
            print(f"[ArmCtrl] Waypoint {i+1}/{len(waypoints)}: {wp}")
            result = self.move_to(wp, max_steps=max_steps_per_wp,
                                  tolerance=tolerance)
            total_steps += result["steps"]
            if not result["success"]:
                return {
                    "success": False,
                    "waypoints_reached": i,
                    "total_steps": total_steps,
                    "abort_reason": result["abort_reason"],
                }
        return {
            "success": True,
            "waypoints_reached": len(waypoints),
            "total_steps": total_steps,
            "abort_reason": None,
        }
