import ctypes
import functools
import json
import math
import multiprocessing
import os
import tempfile
import time
import numpy as np

from multiprocessing import Array, Pipe

from openpilot.tools.sim.bridge.common import QueueMessage, QueueMessageType
from openpilot.tools.sim.bridge.scenarionet.scenarionet_process import (
  scenarionet_process,
  scenarionet_simulation_state,
  scenarionet_vehicle_state,
)
from openpilot.tools.sim.lib.common import SimulatorState, World, vec3
from openpilot.tools.sim.lib.camerad import W, H


class ScenarioNetWorld(World):
  def __init__(self, status_q, config, test_duration, test_run, state_path, scenario_index, acceptance_radius,
               dual_camera=False):
    super().__init__(dual_camera)
    self.status_q = status_q
    self.state_path = state_path
    self.scenario_index = scenario_index
    self.acceptance_radius = acceptance_radius
    self.last_state_write = 0.0
    self.last_done_info = None
    self.latest_vehicle_state = None

    self.camera_array = Array(ctypes.c_uint8, W * H * 3)
    self.road_image = np.frombuffer(self.camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

    self.wide_camera_array = None
    if dual_camera:
      self.wide_camera_array = Array(ctypes.c_uint8, W * H * 3)
      self.wide_road_image = np.frombuffer(self.wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

    self.controls_send, self.controls_recv = Pipe()
    self.simulation_state_send, self.simulation_state_recv = Pipe()
    self.vehicle_state_send, self.vehicle_state_recv = Pipe()

    self.exit_event = multiprocessing.Event()
    self.op_engaged = multiprocessing.Event()

    self.test_run = test_run
    self.first_engage = None
    self.last_check_timestamp = 0.0
    self.distance_moved = 0.0

    self.scenario_process = multiprocessing.Process(
      name="scenarionet process",
      target=functools.partial(
        scenarionet_process,
        dual_camera,
        config,
        self.camera_array,
        self.wide_camera_array,
        self.image_lock,
        self.controls_recv,
        self.simulation_state_send,
        self.vehicle_state_send,
        self.exit_event,
        self.op_engaged,
        test_duration,
        self.test_run,
        scenario_index,
        acceptance_radius,
      ),
    )

    self.scenario_process.start()
    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, "starting"))

    print("--------------------------------------------------------------")
    print("---- Spawning ScenarioEnv world, this might take awhile   ----")
    print("--------------------------------------------------------------")

    self.vehicle_last_pos = self.vehicle_state_recv.recv().position
    self.vehicle_last_velocity = None
    self.vehicle_last_bearing = None
    self.vehicle_last_sensor_time = None
    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, "started"))

    self.vc = [0.0, 0.0]
    self.reset_time = 0.0
    self.should_reset = False

  def _atomic_write_state(self, payload):
    os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(self.state_path), encoding="utf-8") as tmp:
      json.dump(payload, tmp, indent=2, sort_keys=True)
      tmp.flush()
      os.fsync(tmp.fileno())
      temp_name = tmp.name
    os.replace(temp_name, self.state_path)

  def _maybe_write_state(self, simulator_state: SimulatorState, force=False):
    if self.latest_vehicle_state is None:
      return

    now = time.time()
    if not force and (now - self.last_state_write) < 0.2:
      return

    velocity = self.latest_vehicle_state.velocity
    speed_mps = float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))
    payload = {
      "updated_at_unix_s": now,
      "bridge_running": not self.exit_event.is_set(),
      "done": self.last_done_info is not None,
      "engaged": bool(simulator_state.is_engaged),
      "scenario_index": int(self.latest_vehicle_state.scenario_index),
      "scenario_id": self.latest_vehicle_state.scenario_id,
      "position_xy": [float(v) for v in self.latest_vehicle_state.position],
      "goal_position_xy": [float(v) for v in self.latest_vehicle_state.goal_position],
      "distance_to_goal_m": float(self.latest_vehicle_state.distance_to_goal),
      "acceptance_radius_m": float(self.acceptance_radius),
      "route_completion": float(self.latest_vehicle_state.route_completion),
      "goal_reached": bool(self.latest_vehicle_state.goal_reached),
      "speed_mps": speed_mps,
      "speed_kph": speed_mps * 3.6,
      "bearing_deg": float(self.latest_vehicle_state.bearing),
    }

    if self.last_done_info is not None:
      payload["done_info"] = self.last_done_info

    self._atomic_write_state(payload)
    self.last_state_write = now

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    if (time.monotonic() - self.reset_time) > 2:
      self.vc[0] = steer_angle
      self.vc[1] = throttle_out if throttle_out else -brake_out
    else:
      self.vc = [0.0, 0.0]

    self.controls_send.send([*self.vc, self.should_reset])
    self.should_reset = False

  def read_state(self):
    while self.simulation_state_recv.poll(0):
      md_state: scenarionet_simulation_state = self.simulation_state_recv.recv()
      if md_state.done:
        self.last_done_info = md_state.done_info
        self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, md_state.done_info))
        self.exit_event.set()

  def read_sensors(self, simulator_state: SimulatorState):
    while self.vehicle_state_recv.poll(0):
      md_vehicle: scenarionet_vehicle_state = self.vehicle_state_recv.recv()
      self.latest_vehicle_state = md_vehicle
      curr_pos = md_vehicle.position
      curr_velocity = md_vehicle.velocity
      curr_bearing = md_vehicle.bearing
      now = time.monotonic()

      yaw_rate = 0.0
      accel_long = 0.0
      accel_lat = 0.0
      if self.vehicle_last_sensor_time is not None and self.vehicle_last_velocity is not None and self.vehicle_last_bearing is not None:
        dt = max(now - self.vehicle_last_sensor_time, 1e-3)

        heading_delta = math.radians(((curr_bearing - self.vehicle_last_bearing + 180.0) % 360.0) - 180.0)
        yaw_rate = heading_delta / dt

        accel_world = np.array([
          (curr_velocity.x - self.vehicle_last_velocity.x) / dt,
          (curr_velocity.y - self.vehicle_last_velocity.y) / dt,
        ])
        heading_rad = math.radians(curr_bearing)
        forward = np.array([math.cos(heading_rad), math.sin(heading_rad)])
        right = np.array([-math.sin(heading_rad), math.cos(heading_rad)])
        accel_long = float(np.dot(accel_world, forward))
        accel_lat = float(np.dot(accel_world, right))

      simulator_state.velocity = md_vehicle.velocity
      simulator_state.bearing = curr_bearing
      simulator_state.imu.bearing = curr_bearing
      simulator_state.imu.gyroscope = vec3(x=-yaw_rate, y=0.0, z=0.0)
      simulator_state.imu.accelerometer = vec3(x=accel_long, y=accel_lat, z=-9.81)
      simulator_state.steering_angle = md_vehicle.steering_angle
      simulator_state.gps.from_xy(curr_pos)
      simulator_state.valid = True

      self.vehicle_last_velocity = curr_velocity
      self.vehicle_last_bearing = curr_bearing
      self.vehicle_last_sensor_time = now

      is_engaged = simulator_state.is_engaged
      if is_engaged and self.first_engage is None:
        self.first_engage = time.monotonic()
        self.op_engaged.set()

      after_engaged_check = is_engaged and self.first_engage is not None and \
        (time.monotonic() - self.first_engage >= 5) and self.test_run

      moved = float(np.linalg.norm(np.asarray(curr_pos) - np.asarray(self.vehicle_last_pos)))
      if moved >= 1.0:
        self.distance_moved += moved
        self.vehicle_last_pos = curr_pos

      current_time = time.monotonic()
      since_last_check = current_time - self.last_check_timestamp
      if since_last_check >= 29:
        if after_engaged_check and self.distance_moved == 0:
          info = {"vehicle_not_moving": True}
          self.last_done_info = info
          self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, info))
          self.exit_event.set()

        self.last_check_timestamp = current_time
        self.distance_moved = 0

      self._maybe_write_state(simulator_state)

    if self.exit_event.is_set():
      self._maybe_write_state(simulator_state, force=True)

  def read_cameras(self):
    pass

  def tick(self):
    pass

  def reset(self):
    self.reset_time = time.monotonic()
    self.vehicle_last_velocity = None
    self.vehicle_last_bearing = None
    self.vehicle_last_sensor_time = None
    self.last_done_info = None
    self.should_reset = True

  def close(self, reason: str):
    self.status_q.put(QueueMessage(QueueMessageType.CLOSE_STATUS, reason))
    self.exit_event.set()
    self.scenario_process.join()
