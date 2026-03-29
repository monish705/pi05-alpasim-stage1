import json
import math
import time
import numpy as np

from collections import namedtuple
from multiprocessing.connection import Connection
from panda3d.core import Vec3

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.scenario.scenario_description import ScenarioDescription as SD

from openpilot.common.realtime import Ratekeeper
from openpilot.tools.sim.lib.common import vec3
from openpilot.tools.sim.lib.camerad import W, H

C3_POSITION = Vec3(0.0, 0, 1.22)
C3_HPR = Vec3(0, 0, 0)

scenarionet_simulation_state = namedtuple("scenarionet_simulation_state", ["running", "done", "done_info"])
scenarionet_vehicle_state = namedtuple(
  "scenarionet_vehicle_state",
  [
    "velocity",
    "position",
    "bearing",
    "steering_angle",
    "scenario_index",
    "scenario_id",
    "route_completion",
    "distance_to_goal",
    "goal_position",
    "goal_reached",
  ],
)


def _to_float_list(values):
  return [float(v) for v in values]


def _extract_goal_position(env: ScenarioEnv) -> np.ndarray:
  scenario = env.engine.data_manager.current_scenario
  sdc_id = str(scenario[SD.METADATA][SD.SDC_ID])
  track_state = scenario[SD.TRACKS][sdc_id]["state"]
  valid_mask = np.asarray(track_state["valid"]).astype(bool)
  positions = np.asarray(track_state["position"])
  valid_positions = positions[valid_mask]
  goal_position = valid_positions[-1] if len(valid_positions) else positions[-1]
  return np.asarray(goal_position[:2], dtype=np.float64)


def _make_done_info(env: ScenarioEnv, scenario_index: int, goal_position: np.ndarray, acceptance_radius: float, info: dict):
  current_position = np.asarray(env.vehicle.position[:2], dtype=np.float64)
  distance_to_goal = float(np.linalg.norm(current_position - goal_position))
  route_completion = float(env.vehicle.navigation.route_completion)
  goal_reached = bool(distance_to_goal <= acceptance_radius or route_completion >= 0.95)
  return {
    "scenario_index": scenario_index,
    "scenario_id": env.engine.data_manager.current_scenario_id,
    "route_completion": route_completion,
    "distance_to_goal_m": distance_to_goal,
    "goal_position_xy": _to_float_list(goal_position),
    "current_position_xy": _to_float_list(current_position),
    "goal_reached": goal_reached,
    "metadrive_info": {
      str(k): (bool(v) if isinstance(v, (np.bool_, bool)) else float(v) if isinstance(v, (np.floating, float, int)) else str(v))
      for k, v in info.items()
    },
  }


def scenarionet_process(
  dual_camera: bool,
  config: dict,
  camera_array,
  wide_camera_array,
  image_lock,
  controls_recv: Connection,
  simulation_state_send: Connection,
  vehicle_state_send: Connection,
  exit_event,
  op_engaged,
  test_duration,
  test_run,
  scenario_index: int,
  acceptance_radius: float,
):
  road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
  if dual_camera:
    wide_road_image = np.frombuffer(wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

  env = ScenarioEnv(config)

  def reset():
    env.reset(seed=scenario_index)
    env.vehicle.config["max_speed_km_h"] = 1000
    goal_position = _extract_goal_position(env)

    simulation_state_send.send(
      scenarionet_simulation_state(
        running=True,
        done=False,
        done_info={
          "scenario_index": scenario_index,
          "scenario_id": env.engine.data_manager.current_scenario_id,
          "goal_position_xy": _to_float_list(goal_position),
        },
      )
    )
    return goal_position

  goal_position = reset()
  start_time = None
  rk = Ratekeeper(100, None)
  steer_ratio = 8
  vc = [0.0, 0.0]

  def get_cam_as_rgb(sensor_name: str):
    cam = env.engine.sensors[sensor_name]
    cam.get_cam().reparentTo(env.vehicle.origin)
    cam.get_cam().setPos(C3_POSITION)
    cam.get_cam().setHpr(C3_HPR)
    img = cam.perceive(to_float=False)
    if not isinstance(img, np.ndarray):
      img = img.get()
    return img

  while not exit_event.is_set():
    current_position = np.asarray(env.vehicle.position[:2], dtype=np.float64)
    distance_to_goal = float(np.linalg.norm(current_position - goal_position))
    route_completion = float(env.vehicle.navigation.route_completion)
    goal_reached = bool(distance_to_goal <= acceptance_radius or route_completion >= 0.95)

    vehicle_state_send.send(
      scenarionet_vehicle_state(
        velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0.0),
        position=current_position,
        bearing=float(math.degrees(env.vehicle.heading_theta)),
        steering_angle=float(env.vehicle.steering * env.vehicle.MAX_STEERING),
        scenario_index=scenario_index,
        scenario_id=env.engine.data_manager.current_scenario_id,
        route_completion=route_completion,
        distance_to_goal=distance_to_goal,
        goal_position=goal_position.copy(),
        goal_reached=goal_reached,
      )
    )

    if controls_recv.poll(0):
      while controls_recv.poll(0):
        steer_angle, gas, should_reset = controls_recv.recv()

      steer_metadrive = steer_angle / (env.vehicle.MAX_STEERING * steer_ratio)
      steer_metadrive = float(np.clip(steer_metadrive, -1.0, 1.0))
      vc = [steer_metadrive, gas]

      if should_reset:
        goal_position = reset()
        start_time = None

    if op_engaged.is_set() and start_time is None:
      start_time = time.monotonic()

    if rk.frame % 5 == 0:
      _, _, terminated, truncated, info = env.step(vc)
      timeout = bool(start_time is not None and time.monotonic() - start_time >= test_duration)
      done_info = _make_done_info(env, scenario_index, goal_position, acceptance_radius, info)
      done_info["timeout"] = timeout

      if terminated or truncated or done_info["goal_reached"] or (timeout and test_run):
        done_info["terminated"] = bool(terminated)
        done_info["truncated"] = bool(truncated)
        simulation_state_send.send(
          scenarionet_simulation_state(
            running=False,
            done=True,
            done_info=done_info,
          )
        )

      if dual_camera:
        wide_road_image[...] = get_cam_as_rgb("rgb_wide")
      road_image[...] = get_cam_as_rgb("rgb_road")
      image_lock.release()

    rk.keep_time()
