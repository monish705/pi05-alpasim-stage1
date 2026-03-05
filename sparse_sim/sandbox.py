"""
sparse_sim/sandbox.py
======================
Fast MuJoCo sandbox for testing grasps on novel objects.

Uses the active perception sim as starting point, clones the scene,
strips non-essential objects, and runs N test grasps rapidly.

Only used when the VLM decides direct motor calls aren't enough.
"""
import numpy as np
import mujoco
import tempfile
import os
from typing import Optional, List, Tuple
from perception.scene_graph import SceneGraph, SceneObject


# Template for dynamic objects in MuJoCo XML
_OBJECT_XML_TEMPLATE = '''
  <body name="{name}" pos="{px} {py} {pz}">
    <freejoint name="{name}_joint"/>
    <geom name="{name}_geom" type="{geom_type}" size="{size}"
          rgba="{r} {g} {b} 1" mass="{mass}" friction="{friction} 0.005 0.0001"/>
  </body>
'''


class SparseSandbox:
    """
    Fast simulation sandbox for trajectory planning.
    
    Usage:
        sandbox = SparseSandbox(base_xml_path)
        result = sandbox.plan_grasp(scene_graph, "red_mug")
    """

    N_TRIALS = 50        # Number of random approach angles to test
    SIM_STEPS = 200      # Steps per trial to check stability

    def __init__(self, base_xml_path: str):
        """
        Args:
            base_xml_path: path to the base G1 scene XML (without objects)
        """
        self.base_xml_path = base_xml_path
        print(f"[Sandbox] Initialised with base: {base_xml_path}")

    def build_sparse_scene(self, scene_graph: SceneGraph,
                           target_label: str) -> Tuple[Optional[str], Optional[SceneObject]]:
        """
        Build a minimal MuJoCo XML with just the robot + target object.
        
        Returns:
            (xml_path, target_object) or (None, None) if object not found
        """
        target = scene_graph.query(target_label)
        if target is None:
            print(f"[Sandbox] Object '{target_label}' not found in scene graph")
            return None, None

        # Read base XML and inject just the target object
        with open(self.base_xml_path, 'r') as f:
            base_xml = f.read()

        # Generate object geometry
        dims = target.dimensions_m
        w = dims.get("width", 0.05)
        h = dims.get("height", 0.05)
        d = dims.get("depth", 0.05)
        pos = target.position_world

        shape = target.collision_primitive
        if shape == "cylinder":
            radius = max(w, d) / 2
            half_h = h / 2
            size_str = f"{radius:.4f} {half_h:.4f}"
        elif shape == "sphere":
            radius = max(w, h, d) / 2
            size_str = f"{radius:.4f}"
        else:  # box
            size_str = f"{w/2:.4f} {d/2:.4f} {h/2:.4f}"

        obj_xml = _OBJECT_XML_TEMPLATE.format(
            name=f"sparse_{target.id}",
            px=pos[0], py=pos[1], pz=pos[2],
            geom_type=shape if shape in ("cylinder", "sphere", "box") else "box",
            size=size_str,
            mass=target.mass_kg,
            friction=target.friction if hasattr(target, 'friction') else 0.4,
            r=0.8, g=0.2, b=0.2,
        )

        # Inject before </worldbody>
        sparse_xml = base_xml.replace("</worldbody>", obj_xml + "\n  </worldbody>")

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode='w')
        tmp.write(sparse_xml)
        tmp.close()

        return tmp.name, target

    def plan_grasp(self, scene_graph: SceneGraph,
                   target_label: str) -> dict:
        """
        Plan a grasp trajectory for the target object.
        
        Builds a sparse scene, tries N approach angles, picks the best.
        
        Returns:
            dict with 'success', 'waypoints', 'approach_angle', 'score'
        """
        xml_path, target = self.build_sparse_scene(scene_graph, target_label)
        if xml_path is None:
            return {"success": False, "reason": "Object not found"}

        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)

            # Import here to avoid circular imports
            from motor.unitree_bridge import UnitreeBridge
            from motor.arm_controller import ArmController

            bridge = UnitreeBridge(model, data)
            bridge.reset_to_stand()

            target_pos = np.array(target.position_world)
            best_result = None
            best_score = float('inf')

            for trial in range(self.N_TRIALS):
                # Reset for each trial
                bridge.reset_to_stand()
                mujoco.mj_forward(model, data)

                # Random approach: offset the grasp point slightly
                angle = 2 * np.pi * trial / self.N_TRIALS
                offset = 0.03 * np.array([np.cos(angle), np.sin(angle), 0])
                grasp_point = target_pos + np.array([0, 0, 0.03]) + offset

                # Try arm IK
                arm = ArmController(bridge, hand="right")
                result = arm.move_to(grasp_point, max_steps=150, tolerance=0.04)

                if result["success"] and result["com_stable"]:
                    score = result["final_error"]
                    if score < best_score:
                        best_score = score
                        best_result = {
                            "success": True,
                            "waypoints": [grasp_point.tolist()],
                            "approach_angle": float(angle),
                            "score": float(score),
                            "trial": trial,
                        }

            if best_result:
                print(f"[Sandbox] ✅ Found grasp (trial {best_result['trial']}, "
                      f"score={best_score:.4f})")
                return best_result
            else:
                print(f"[Sandbox] ❌ No stable grasp found in {self.N_TRIALS} trials")
                return {"success": False, "reason": "No stable grasp found"}

        finally:
            os.unlink(xml_path)

    def test_trajectory(self, scene_graph: SceneGraph,
                        waypoints: List[List[float]]) -> dict:
        """
        Test a trajectory in the sandbox before executing on real robot.
        
        Returns:
            dict with 'success', 'final_pos', 'com_stable', 'collisions'
        """
        xml_path, _ = self.build_sparse_scene(scene_graph, "")
        if xml_path is None:
            # No target, just use base scene
            xml_path = self.base_xml_path

        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)

            from motor.unitree_bridge import UnitreeBridge
            from motor.arm_controller import ArmController

            bridge = UnitreeBridge(model, data)
            bridge.reset_to_stand()

            arm = ArmController(bridge, hand="right")
            results = []

            for wp in waypoints:
                target = np.array(wp)
                result = arm.move_to(target, max_steps=200)
                results.append(result)
                if not result["success"]:
                    break

            return {
                "success": all(r["success"] for r in results),
                "results": results,
                "final_pos": bridge.get_ee_pos("right").tolist(),
                "com_stable": arm.is_com_stable(),
            }

        finally:
            if xml_path != self.base_xml_path:
                os.unlink(xml_path)
