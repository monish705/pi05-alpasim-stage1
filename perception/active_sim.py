"""
perception/active_sim.py
=========================
Active Perception — the robot's eyes.

Works IDENTICALLY in simulation and on a real robot:
  - Sim:  renders MuJoCo cameras → perception → scene graph
  - Real: receives camera images over network → perception → scene graph

v2 changes:
  - Multi-camera fusion: renders from multiple cameras, deduplicates across views
  - Camera extrinsics: transforms detections to world frame
  - Sim-mode tagging: reads MuJoCo body names for deterministic fallback
  - Position deduplication across camera views

The scene graph includes:
  - All detected objects (position, shape, mass, label)
  - The robot itself (base position, joint states, hand positions)
  - Spatial relationships ("mug ON table", "robot NEAR table")

This is the SINGLE source of truth about the world.
The VLM reads this to decide what to do.
"""
import numpy as np
import mujoco
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from perception.scene_graph import SceneGraph, SceneObject
from perception.pipeline import PerceptionPipeline, PipelineConfig


@dataclass
class RobotState:
    """Robot's own state — always included in the world model."""
    base_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    base_yaw: float = 0.0
    right_hand_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    left_hand_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    is_grasping: bool = False
    grasped_object: Optional[str] = None
    is_stable: bool = True
    base_height: float = 0.78


class ActivePerception:
    """
    The robot's perception system. Simple and deployable.

    SIM MODE:
        perception = ActivePerception.from_sim(model, data)
        scene = perception.perceive()

    REAL ROBOT MODE:
        perception = ActivePerception.from_real(vlm_url="http://gpu:8000/v1")
        scene = perception.perceive_from_images(rgb, depth, intrinsics)

    Both modes return the same SceneGraph + RobotState.
    """

    # Cameras available in the G1 sim
    DEFAULT_CAMERAS = ["head_cam"]
    ALL_CAMERAS = ["head_cam", "overview_cam", "side_cam"]

    # Distance threshold for deduplicating objects across camera views
    DEDUP_DISTANCE = 0.08  # metres

    def __init__(self, vlm_api_base: str = "http://localhost:8000/v1",
                 vlm_model: str = "Qwen/Qwen3-VL-8B",
                 config: PipelineConfig = None):
        # Perception pipeline (SAM3 + VLM)
        self.pipeline = PerceptionPipeline(
            vlm_api_base=vlm_api_base,
            vlm_model=vlm_model,
            config=config,
        )

        # Robot state (updated every cycle)
        self.robot_state = RobotState()

        # Sim-mode resources (set by from_sim)
        self._model = None
        self._data = None
        self._renderer = None
        self._bridge = None
        self._grasp = None
        self._is_sim = False

        # Frame counter
        self._frame = 0

        print(f"[Perception] Initialised (VLM: {vlm_model})")

    @classmethod
    def from_sim(cls, model, data, bridge=None, grasp=None,
                 vlm_api_base="http://localhost:8000/v1",
                 vlm_model="Qwen/Qwen3-VL-8B",
                 config=None):
        """Create from MuJoCo simulation — renders cameras internally."""
        if config is None:
            # Default to sim-mode tagger and depth fallback
            config = PipelineConfig(use_sim_tagger=True, depth_fallback=True)

        inst = cls(vlm_api_base, vlm_model, config)
        inst._model = model
        inst._data = data
        inst._renderer = mujoco.Renderer(model, 480, 640)
        inst._bridge = bridge
        inst._grasp = grasp
        inst._is_sim = True
        print("[Perception] Mode: SIMULATION (MuJoCo cameras)")
        return inst

    @classmethod
    def from_real(cls, vlm_api_base="http://localhost:8000/v1",
                  vlm_model="Qwen/Qwen3-VL-8B",
                  config=None):
        """Create for real robot — images provided externally."""
        inst = cls(vlm_api_base, vlm_model, config)
        inst._is_sim = False
        print("[Perception] Mode: REAL ROBOT (external cameras)")
        return inst

    # ---- Core API ----

    def perceive(self, cameras: List[str] = None) -> dict:
        """
        Run one perception cycle.

        In sim: renders cameras from MuJoCo, transforms to world frame.
        Returns: {"scene_graph": SceneGraph, "robot": RobotState}
        """
        if not self._is_sim:
            raise RuntimeError("Call perceive_from_images() in real robot mode")

        if cameras is None:
            cameras = self.DEFAULT_CAMERAS

        self._frame += 1

        # Collect sim body names for sim-mode tagging
        sim_body_names = self._get_sim_object_names() if self._is_sim else None

        # Render and perceive each camera
        for cam in cameras:
            rgb, depth = self._render(cam)
            intrinsics = self._get_intrinsics(cam)
            extrinsics = self._get_extrinsics(cam)

            self.pipeline.perceive(
                rgb, depth, intrinsics,
                camera_extrinsics=extrinsics,
                sim_body_names=sim_body_names,
            )

        # Update robot state
        self._update_robot_state()

        return {
            "scene_graph": self.pipeline.scene_graph,
            "robot": self.robot_state,
        }

    def perceive_from_images(self, rgb: np.ndarray, depth: np.ndarray,
                             intrinsics: dict,
                             camera_extrinsics: np.ndarray = None,
                             robot_state: dict = None) -> dict:
        """
        Run perception from externally provided images (real robot).

        Args:
            rgb: (H, W, 3) uint8
            depth: (H, W) float32 metres
            intrinsics: {fx, fy, cx, cy}
            camera_extrinsics: optional (4,4) camera-to-world transform
            robot_state: optional dict with base_pos, hand_pos, etc.
        """
        self._frame += 1
        self.pipeline.perceive(
            rgb, depth, intrinsics,
            camera_extrinsics=camera_extrinsics,
        )

        if robot_state:
            self.robot_state.base_pos = robot_state.get("base_pos", [0, 0, 0])
            self.robot_state.right_hand_pos = robot_state.get("right_hand_pos", [0, 0, 0])
            self.robot_state.is_grasping = robot_state.get("is_grasping", False)

        return {
            "scene_graph": self.pipeline.scene_graph,
            "robot": self.robot_state,
        }

    # ---- Queries (used by VLM) ----

    def get_scene_graph(self) -> SceneGraph:
        return self.pipeline.scene_graph

    def get_scene_graph_json(self) -> str:
        """Full world state as JSON — fed directly to the VLM."""
        sg_json = self.pipeline.scene_graph.to_json()

        # Inject robot state into the JSON
        import json
        data = json.loads(sg_json) if sg_json else {}
        data["robot"] = {
            "base_pos": self.robot_state.base_pos,
            "base_yaw_rad": self.robot_state.base_yaw,
            "right_hand_pos": self.robot_state.right_hand_pos,
            "left_hand_pos": self.robot_state.left_hand_pos,
            "is_grasping": self.robot_state.is_grasping,
            "grasped_object": self.robot_state.grasped_object,
            "is_stable": self.robot_state.is_stable,
        }
        return json.dumps(data, indent=2)

    def query_object(self, label: str) -> Optional[SceneObject]:
        return self.pipeline.scene_graph.query(label)

    def get_graspable_objects(self) -> list:
        return self.pipeline.scene_graph.get_graspable_objects()

    def update_from_cameras(self, cameras=None):
        """Alias for perceive() — backward compat."""
        return self.perceive(cameras)

    def print_state(self):
        r = self.robot_state
        print(f"\n[World] Frame {self._frame}")
        print(f"  Robot: base=({r.base_pos[0]:.2f}, {r.base_pos[1]:.2f}, "
              f"{r.base_pos[2]:.2f}), stable={r.is_stable}")
        print(f"  R.Hand: ({r.right_hand_pos[0]:.2f}, {r.right_hand_pos[1]:.2f}, "
              f"{r.right_hand_pos[2]:.2f}), "
              f"grasping={r.grasped_object or 'nothing'}")
        sg = self.pipeline.scene_graph
        objs = sg.get_all_objects()
        print(f"  Objects: {len(objs)}")
        for obj in objs:
            p = obj.position_world
            stale = f" [stale {obj.frames_since_seen}f]" if obj.frames_since_seen > 0 else ""
            print(f"    [T{obj.track_id}] {obj.label}: "
                  f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) "
                  f"conf={obj.confidence:.2f}{stale}")

    # ---- Internal: Rendering ----

    def _render(self, camera_name: str):
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=camera_name)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self._data, camera=camera_name)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return rgb, depth

    def _get_intrinsics(self, camera_name: str):
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        fovy = self._model.cam_fovy[cam_id]
        f = 0.5 * 480 / np.tan(fovy * np.pi / 360)
        return {"fx": f, "fy": f, "cx": 320, "cy": 240}

    def _get_extrinsics(self, camera_name: str) -> np.ndarray:
        """
        Get the camera-to-world 4x4 transformation matrix.
        MuJoCo stores camera position and orientation in data.cam_xpos/xmat.
        """
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            return np.eye(4)

        pos = self._data.cam_xpos[cam_id].copy()
        rot = self._data.cam_xmat[cam_id].reshape(3, 3).copy()

        # MuJoCo camera convention: -Z forward, +X right, +Y down
        # Convert to OpenCV convention: +Z forward, +X right, +Y down
        # Flip Z and Y axes
        flip = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1],
        ], dtype=np.float64)

        rot_world = rot @ flip

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rot_world
        extrinsics[:3, 3] = pos

        return extrinsics

    def _get_sim_object_names(self) -> List[str]:
        """Get all graspable object body names from the MuJoCo model."""
        names = []
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("obj_"):
                names.append(name)
        return names

    def _update_robot_state(self):
        if self._bridge is None:
            return
        b = self._bridge
        pos = b.get_base_pos()
        quat = b.get_base_quat()
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        self.robot_state.base_pos = pos.tolist()
        self.robot_state.base_yaw = float(yaw)
        self.robot_state.base_height = float(pos[2])
        self.robot_state.right_hand_pos = b.get_ee_pos("right").tolist()
        self.robot_state.left_hand_pos = b.get_ee_pos("left").tolist()
        self.robot_state.is_stable = pos[2] > 0.5

        if self._grasp:
            self.robot_state.is_grasping = self._grasp.is_grasping("right")
            self.robot_state.grasped_object = self._grasp.get_grasped_object("right")
