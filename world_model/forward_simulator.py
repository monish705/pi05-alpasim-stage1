"""
world_model/forward_simulator.py
==================================
Action-Conditioned Forward Simulator — the "What If" Engine.

This is what makes a world model a WORLD MODEL, not just perception.
Given the current scene graph, this module can:
  1. Build a temporary MuJoCo simulation from perceived objects
  2. Apply a candidate action (push, drop, place)
  3. Simulate N seconds forward in physics time
  4. Return the predicted outcome (object positions after action)
  5. Compare prediction against actual observation after execution

Usage:
    sim = ForwardSimulator()
    
    # "What happens if I push the mug 10cm to the right?"
    prediction = sim.predict_action(
        scene_graph=current_sg,
        action=PushAction(target="red mug", force=[0, 5, 0], duration=0.5)
    )
    print(prediction.final_positions)  # where everything ends up
    
    # After robot actually pushes it:
    actual_sg = perception.perceive()
    error = sim.compare_prediction(prediction, actual_sg)
    print(f"Prediction error: {error.mean_position_error_cm:.1f}cm")
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

from perception.scene_graph import SceneGraph, SceneObject


# ─── Action Definitions ───

@dataclass
class PushAction:
    """Push an object with a force vector."""
    target_label: str
    force_vector: List[float]    # [fx, fy, fz] in Newtons
    contact_point: List[float] = field(default_factory=lambda: [0, 0, 0])  # relative to object center
    duration_s: float = 0.3

@dataclass
class DropAction:
    """Drop an object from its current position."""
    target_label: str
    release_height_m: float = 0.0  # additional height above current position

@dataclass
class PlaceAction:
    """Place an object at a target position."""
    target_label: str
    target_position: List[float]   # [x, y, z] world frame
    release_velocity: List[float] = field(default_factory=lambda: [0, 0, 0])


# ─── Prediction Results ───

@dataclass
class ObjectPrediction:
    """Predicted state of a single object after an action."""
    track_id: int
    label: str
    initial_position: np.ndarray
    final_position: np.ndarray
    displacement: float            # total distance moved
    is_stable: bool               # settled or still moving at end of sim
    contacts: List[str]           # labels of objects it contacted

@dataclass
class ActionPrediction:
    """Full prediction result from a forward simulation."""
    action_description: str
    sim_duration_s: float
    object_predictions: List[ObjectPrediction]
    final_positions: Dict[int, np.ndarray]    # track_id → final pos
    collisions_detected: List[Tuple[str, str]] # pairs of objects that collided
    simulation_stable: bool        # did the scene settle?

@dataclass
class PredictionError:
    """Comparison between prediction and actual observation."""
    mean_position_error_m: float
    max_position_error_m: float
    per_object_errors: Dict[int, float]  # track_id → error
    model_quality: float           # 0-1 score, 1=perfect prediction


class ForwardSimulator:
    """
    Builds temporary MuJoCo simulations from scene graphs and
    runs "what-if" scenarios to predict action outcomes.
    
    The forward simulator is the core of the world model's
    predictive capability. It answers: "If I do X, what happens?"
    """
    
    def __init__(self, sim_timestep: float = 0.002,
                 default_sim_duration: float = 2.0,
                 gravity: List[float] = None):
        """
        Args:
            sim_timestep: MuJoCo simulation timestep (smaller = more accurate)
            default_sim_duration: how long to simulate forward by default
            gravity: gravity vector, default earth gravity
        """
        self.sim_timestep = sim_timestep
        self.default_sim_duration = default_sim_duration
        self.gravity = gravity or [0, 0, -9.81]
    
    def predict_action(self, scene_graph: SceneGraph,
                       action, sim_duration: float = None) -> ActionPrediction:
        """
        Predict the outcome of an action on the current scene.
        
        Args:
            scene_graph: current perceived world state
            action: PushAction, DropAction, or PlaceAction
            sim_duration: override simulation duration
            
        Returns:
            ActionPrediction with predicted final positions
        """
        if not _MUJOCO_AVAILABLE:
            print("[ForwardSim] MuJoCo not available, returning empty prediction")
            return self._empty_prediction(action)
        
        duration = sim_duration or self.default_sim_duration
        
        # Build temporary simulation from scene graph
        xml_str = self._build_sim_xml(scene_graph)
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        
        # Record initial positions
        objects = scene_graph.get_all_objects()
        initial_positions = {}
        body_id_map = {}  # track_id → mujoco body id
        
        for obj in objects:
            body_name = f"obj_{obj.track_id}"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                initial_positions[obj.track_id] = data.xpos[body_id].copy()
                body_id_map[obj.track_id] = body_id
        
        # Apply action
        action_desc = self._apply_action(model, data, action, scene_graph, body_id_map)
        
        # Simulate forward
        n_steps = int(duration / self.sim_timestep)
        contacts_log = []
        
        for step in range(n_steps):
            mujoco.mj_step(model, data)
            
            # Log contacts
            for i in range(data.ncon):
                con = data.contact[i]
                geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
                geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
                if geom1 and geom2:
                    contacts_log.append((geom1, geom2))
        
        # Read final positions
        final_positions = {}
        object_predictions = []
        
        for obj in objects:
            tid = obj.track_id
            if tid in body_id_map:
                bid = body_id_map[tid]
                final_pos = data.xpos[bid].copy()
                final_positions[tid] = final_pos
                
                init_pos = initial_positions.get(tid, np.zeros(3))
                displacement = np.linalg.norm(final_pos - init_pos)
                
                # Check if object is stable (velocity near zero)
                # In MuJoCo, we check qvel for the free joint
                is_stable = True
                joint_id = model.body_jntadr[bid]
                if joint_id >= 0 and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    qvel_start = model.jnt_dofadr[joint_id]
                    vel = data.qvel[qvel_start:qvel_start+3]
                    is_stable = np.linalg.norm(vel) < 0.01
                
                # Find contacts involving this object
                obj_contacts = []
                obj_geom_name = f"geom_{obj.track_id}"
                for g1, g2 in contacts_log:
                    if g1 == obj_geom_name or g2 == obj_geom_name:
                        other = g2 if g1 == obj_geom_name else g1
                        if other not in obj_contacts:
                            obj_contacts.append(other)
                
                pred = ObjectPrediction(
                    track_id=tid,
                    label=obj.label,
                    initial_position=init_pos,
                    final_position=final_pos,
                    displacement=displacement,
                    is_stable=is_stable,
                    contacts=obj_contacts,
                )
                object_predictions.append(pred)
        
        # Collect unique collision pairs
        unique_collisions = list(set(contacts_log))
        
        # Check overall stability
        all_stable = all(p.is_stable for p in object_predictions)
        
        result = ActionPrediction(
            action_description=action_desc,
            sim_duration_s=duration,
            object_predictions=object_predictions,
            final_positions=final_positions,
            collisions_detected=unique_collisions,
            simulation_stable=all_stable,
        )
        
        self._print_prediction(result)
        return result
    
    def compare_prediction(self, prediction: ActionPrediction,
                           actual_scene_graph: SceneGraph) -> PredictionError:
        """
        Compare a prediction against actual observation.
        
        This is how the world model learns its own accuracy.
        After predicting "the cup will be at X" and then observing
        "the cup is actually at Y", the error tells us how good
        the world model is.
        """
        per_object_errors = {}
        
        for obj in actual_scene_graph.get_all_objects():
            tid = obj.track_id
            if tid in prediction.final_positions:
                predicted = prediction.final_positions[tid]
                actual = np.array(obj.position_world)
                error = np.linalg.norm(predicted - actual)
                per_object_errors[tid] = error
        
        if not per_object_errors:
            return PredictionError(
                mean_position_error_m=float('inf'),
                max_position_error_m=float('inf'),
                per_object_errors={},
                model_quality=0.0,
            )
        
        errors = list(per_object_errors.values())
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        
        # Quality score: 1.0 if mean error < 1cm, 0.0 if > 10cm
        quality = max(0.0, min(1.0, 1.0 - (mean_err - 0.01) / 0.09))
        
        result = PredictionError(
            mean_position_error_m=mean_err,
            max_position_error_m=max_err,
            per_object_errors=per_object_errors,
            model_quality=quality,
        )
        
        print(f"\n[ForwardSim] Prediction vs Reality:")
        print(f"  Mean error: {mean_err*100:.1f}cm")
        print(f"  Max error:  {max_err*100:.1f}cm")
        print(f"  Model quality: {quality:.0%}")
        for tid, err in per_object_errors.items():
            emoji = "✅" if err < 0.05 else "⚠️" if err < 0.10 else "❌"
            print(f"  {emoji} T{tid}: {err*100:.1f}cm error")
        
        return result
    
    # ─── Internal Methods ───
    
    def _build_sim_xml(self, scene_graph: SceneGraph) -> str:
        """Build a temporary MuJoCo XML from the scene graph."""
        gx, gy, gz = self.gravity
        
        bodies = ""
        for obj in scene_graph.get_all_objects():
            pos = obj.position_world
            dims = obj.dimensions_m
            w = dims.get("width", 0.05)
            h = dims.get("height", 0.05)
            d = dims.get("depth", 0.05)
            
            shape = obj.collision_primitive or "box"
            mass = max(obj.mass_kg, 0.01)
            friction = f"{obj.friction:.2f} 0.005 0.0001"
            
            if shape == "cylinder":
                geom = f'<geom name="geom_{obj.track_id}" type="cylinder" size="{max(w,d)/2:.4f} {h/2:.4f}" mass="{mass:.3f}" friction="{friction}"/>'
            elif shape == "sphere":
                r = max(w, h, d) / 2
                geom = f'<geom name="geom_{obj.track_id}" type="sphere" size="{r:.4f}" mass="{mass:.3f}" friction="{friction}"/>'
            else:
                geom = f'<geom name="geom_{obj.track_id}" type="box" size="{w/2:.4f} {h/2:.4f} {d/2:.4f}" mass="{mass:.3f}" friction="{friction}"/>'
            
            bodies += f"""
    <body name="obj_{obj.track_id}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}">
      <freejoint name="joint_{obj.track_id}"/>
      {geom}
    </body>"""
        
        xml = f"""<mujoco>
  <option timestep="{self.sim_timestep}" gravity="{gx} {gy} {gz}"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.01" friction="0.8 0.005 0.0001"/>
    <geom name="table" type="box" pos="0.65 0 0.4" size="0.3 0.3 0.02" friction="0.6 0.005 0.0001"/>{bodies}
  </worldbody>
</mujoco>"""
        return xml
    
    def _apply_action(self, model, data, action,
                      scene_graph: SceneGraph,
                      body_id_map: Dict[int, int]) -> str:
        """Apply an action to the simulation. Returns description."""
        
        if isinstance(action, PushAction):
            target = scene_graph.query(action.target_label)
            if target and target.track_id in body_id_map:
                bid = body_id_map[target.track_id]
                # Apply force via xfrc_applied
                force = np.array(action.force_vector)
                data.xfrc_applied[bid, :3] = force
                # Force will be cleared after first step; we keep it for duration
                desc = f"Push {action.target_label} with force {action.force_vector} for {action.duration_s}s"
            else:
                desc = f"Push {action.target_label} — TARGET NOT FOUND"
        
        elif isinstance(action, DropAction):
            target = scene_graph.query(action.target_label)
            if target and target.track_id in body_id_map:
                bid = body_id_map[target.track_id]
                # Just let gravity do its thing — object already in sim
                pos = data.xpos[bid].copy()
                pos[2] += action.release_height_m
                # Reset position higher
                joint_id = model.body_jntadr[bid]
                if joint_id >= 0:
                    qadr = model.jnt_qposadr[joint_id]
                    data.qpos[qadr:qadr+3] = pos
                    mujoco.mj_forward(model, data)
                desc = f"Drop {action.target_label} from {pos[2]:.2f}m"
            else:
                desc = f"Drop {action.target_label} — TARGET NOT FOUND"
        
        elif isinstance(action, PlaceAction):
            target = scene_graph.query(action.target_label)
            if target and target.track_id in body_id_map:
                bid = body_id_map[target.track_id]
                joint_id = model.body_jntadr[bid]
                if joint_id >= 0:
                    qadr = model.jnt_qposadr[joint_id]
                    data.qpos[qadr:qadr+3] = action.target_position
                    vadr = model.jnt_dofadr[joint_id]
                    data.qvel[vadr:vadr+3] = action.release_velocity
                    mujoco.mj_forward(model, data)
                desc = f"Place {action.target_label} at {action.target_position}"
            else:
                desc = f"Place {action.target_label} — TARGET NOT FOUND"
        
        else:
            desc = f"Unknown action type: {type(action)}"
        
        return desc
    
    def _empty_prediction(self, action) -> ActionPrediction:
        """Return an empty prediction when simulation isn't available."""
        return ActionPrediction(
            action_description=f"[NO SIM] {type(action).__name__}",
            sim_duration_s=0,
            object_predictions=[],
            final_positions={},
            collisions_detected=[],
            simulation_stable=False,
        )
    
    def _print_prediction(self, pred: ActionPrediction):
        """Print prediction results."""
        print(f"\n{'='*60}")
        print(f"  FORWARD SIMULATION PREDICTION")
        print(f"{'='*60}")
        print(f"  Action: {pred.action_description}")
        print(f"  Duration: {pred.sim_duration_s:.1f}s")
        print(f"  Stable: {'Yes' if pred.simulation_stable else 'No'}")
        
        for op in pred.object_predictions:
            moved = "📍 MOVED" if op.displacement > 0.01 else "⚪ STATIC"
            print(f"\n  [T{op.track_id}] {op.label} — {moved}")
            print(f"    Displacement: {op.displacement*100:.1f}cm")
            print(f"    Final pos: ({op.final_position[0]:.3f}, {op.final_position[1]:.3f}, {op.final_position[2]:.3f})")
            if op.contacts:
                print(f"    Contacts: {', '.join(op.contacts)}")
        
        if pred.collisions_detected:
            print(f"\n  Collisions: {len(pred.collisions_detected)} detected")
        
        print(f"\n{'='*60}\n")

