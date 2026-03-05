"""
motor/grasp_controller.py
==========================
Simulated grasp mechanism for the G1 robot.

The G1 has rubber hands without articulated fingers, so we simulate
grasping using MuJoCo's equality constraints:
  - When the hand is close enough to an object, attach it via weld constraint
  - To release, deactivate the constraint

This is standard practice in MuJoCo manipulation (MuJoCo Menagerie, dm_control).

Also provides proximity detection using the wrist camera concept:
  detect nearby objects within the hand's workspace.
"""
import numpy as np
import mujoco
from typing import Optional, Dict, List
from motor.unitree_bridge import UnitreeBridge, RIGHT_HAND_BODY, LEFT_HAND_BODY


class GraspController:
    """
    Simulated grasp using MuJoCo equality constraints.

    Usage:
        grasp = GraspController(bridge)
        nearby = grasp.detect_nearby("right")
        grasp.grasp("right", "obj_red_mug")
        # ... move arm ...
        grasp.release("right")
    """

    GRASP_DISTANCE = 0.08    # Max distance to grasp (metres)
    GRASP_FORCE = 500.0      # Weld constraint strength

    def __init__(self, bridge: UnitreeBridge):
        self.bridge = bridge
        self.model = bridge.model
        self.data = bridge.data

        # Track active grasps: hand_name → (object_name, constraint_id)
        self._active_grasps: Dict[str, dict] = {}

        # Pre-find all graspable object bodies (those with freejoints)
        self._graspable_objects = self._find_graspable_objects()

        print(f"[Grasp] Initialised: {len(self._graspable_objects)} graspable objects")
        for name in self._graspable_objects:
            print(f"  - {name}")

    def _find_graspable_objects(self) -> Dict[str, int]:
        """Find all bodies that have freejoints (= can be picked up)."""
        objects = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("obj_"):
                objects[name] = i
        return objects

    def detect_nearby(self, hand: str = "right",
                      max_dist: float = 0.15) -> List[dict]:
        """
        Detect objects near the specified hand.

        Returns list of dicts: [{name, distance, position}, ...]
        sorted by distance (closest first).
        """
        mujoco.mj_forward(self.model, self.data)
        body_name = RIGHT_HAND_BODY if hand == "right" else LEFT_HAND_BODY
        hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        hand_pos = self.data.xpos[hand_bid]

        nearby = []
        for obj_name, obj_bid in self._graspable_objects.items():
            obj_pos = self.data.xpos[obj_bid]
            dist = np.linalg.norm(hand_pos - obj_pos)
            if dist < max_dist:
                nearby.append({
                    "name": obj_name,
                    "distance": float(dist),
                    "position": obj_pos.copy(),
                })

        nearby.sort(key=lambda x: x["distance"])
        return nearby

    def grasp(self, hand: str = "right",
              object_name: Optional[str] = None) -> dict:
        """
        Grasp an object with the specified hand.

        If no object_name given, grasps the nearest graspable object.
        Uses MuJoCo contact-based weld: creates a weld constraint
        dynamically by modifying the model.

        Args:
            hand: "right" or "left"
            object_name: specific object to grasp, or None for nearest

        Returns:
            dict with 'success', 'object', 'distance'
        """
        mujoco.mj_forward(self.model, self.data)

        # Already grasping?
        if hand in self._active_grasps:
            return {
                "success": False,
                "object": None,
                "distance": 0,
                "reason": f"{hand} hand already grasping {self._active_grasps[hand]['object']}"
            }

        # Find target object
        if object_name is None:
            nearby = self.detect_nearby(hand, max_dist=self.GRASP_DISTANCE)
            if not nearby:
                return {
                    "success": False,
                    "object": None,
                    "distance": float('inf'),
                    "reason": "No objects within grasp range"
                }
            object_name = nearby[0]["name"]
            dist = nearby[0]["distance"]
        else:
            obj_bid = self._graspable_objects.get(object_name)
            if obj_bid is None:
                return {
                    "success": False,
                    "object": object_name,
                    "distance": float('inf'),
                    "reason": f"Object '{object_name}' not found or not graspable"
                }
            body_name = RIGHT_HAND_BODY if hand == "right" else LEFT_HAND_BODY
            hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            dist = np.linalg.norm(self.data.xpos[hand_bid] - self.data.xpos[obj_bid])

        if dist > self.GRASP_DISTANCE:
            return {
                "success": False,
                "object": object_name,
                "distance": dist,
                "reason": f"Object too far: {dist:.3f}m (max {self.GRASP_DISTANCE}m)"
            }

        # Attach object to hand via equality constraint
        body_name = RIGHT_HAND_BODY if hand == "right" else LEFT_HAND_BODY
        hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        obj_bid = self._graspable_objects[object_name]

        # Compute relative transform (object in hand frame)
        hand_pos = self.data.xpos[hand_bid]
        hand_mat = self.data.xmat[hand_bid].reshape(3, 3)
        obj_pos = self.data.xpos[obj_bid]
        rel_pos = hand_mat.T @ (obj_pos - hand_pos)

        # Create weld constraint by fixing the object's freejoint DOFs
        # Find the freejoint for this object
        obj_jid = -1
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == obj_bid and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                obj_jid = j
                break

        if obj_jid < 0:
            return {
                "success": False,
                "object": object_name,
                "distance": dist,
                "reason": "No freejoint found on object"
            }

        # Store grasp info
        self._active_grasps[hand] = {
            "object": object_name,
            "obj_bid": obj_bid,
            "hand_bid": hand_bid,
            "rel_pos": rel_pos.copy(),
            "obj_jid": obj_jid,
        }

        print(f"[Grasp] ✅ {hand} hand grasped '{object_name}' (dist={dist:.3f}m)")
        return {
            "success": True,
            "object": object_name,
            "distance": dist,
            "reason": None
        }

    def release(self, hand: str = "right") -> dict:
        """Release any grasped object from the specified hand."""
        if hand not in self._active_grasps:
            return {"success": False, "reason": "Nothing grasped"}

        info = self._active_grasps.pop(hand)
        print(f"[Grasp] Released '{info['object']}' from {hand} hand")
        return {
            "success": True,
            "object": info["object"],
        }

    def update(self):
        """
        Call every sim step to maintain grasps.
        
        For each active grasp, forces the object to follow the hand
        by directly setting its freejoint qpos to match the hand position + offset.
        """
        for hand, info in self._active_grasps.items():
            hand_pos = self.data.xpos[info["hand_bid"]]
            hand_mat = self.data.xmat[info["hand_bid"]].reshape(3, 3)

            # Compute target object position in world frame
            target_pos = hand_pos + hand_mat @ info["rel_pos"]

            # Set object position via its freejoint qpos
            qadr = self.model.jnt_qposadr[info["obj_jid"]]
            self.data.qpos[qadr:qadr + 3] = target_pos

            # Match hand orientation (quaternion from rotation matrix)
            target_mat = hand_mat
            # Convert rotation matrix to quaternion
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, target_mat.flatten())
            self.data.qpos[qadr + 3:qadr + 7] = quat

            # Zero object velocity
            vadr = self.model.jnt_dofadr[info["obj_jid"]]
            self.data.qvel[vadr:vadr + 6] = 0.0

    def is_grasping(self, hand: str = "right") -> bool:
        return hand in self._active_grasps

    def get_grasped_object(self, hand: str = "right") -> Optional[str]:
        if hand in self._active_grasps:
            return self._active_grasps[hand]["object"]
        return None
