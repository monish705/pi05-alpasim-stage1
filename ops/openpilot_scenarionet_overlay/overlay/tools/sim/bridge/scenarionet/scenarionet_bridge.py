import math
from multiprocessing import Queue

from metadrive.component.sensors.base_camera import _cuda_enable

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.scenarionet.scenarionet_world import ScenarioNetWorld
from openpilot.tools.sim.lib.camerad import W, H


class ScenarioNetBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, database_path, scenario_index, acceptance_radius, state_path,
               reactive_traffic=False, test_duration=math.inf, test_run=False):
    super().__init__(dual_camera, high_quality)
    self.database_path = database_path
    self.scenario_index = scenario_index
    self.acceptance_radius = acceptance_radius
    self.state_path = state_path
    self.reactive_traffic = reactive_traffic

    self.should_render = True
    self.test_run = test_run
    self.test_duration = test_duration if self.test_run else math.inf

  def spawn_world(self, queue: Queue):
    sensors = {
      "rgb_road": (RGBCameraRoad, W, H),
    }
    if self.dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    config = dict(
      use_render=self.should_render,
      render_pipeline=bool(self.high_quality),
      show_interface=False,
      show_logo=False,
      show_fps=False,
      manual_control=False,
      reactive_traffic=self.reactive_traffic,
      data_directory=self.database_path,
      start_scenario_index=self.scenario_index,
      num_scenarios=1,
      sequential_seed=False,
      sensors=sensors,
      image_on_cuda=_cuda_enable,
      image_observation=True,
      interface_panel=[],
      horizon=100000,
      out_of_route_done=True,
      crash_vehicle_done=True,
      crash_object_done=True,
      vehicle_config=dict(
        enable_reverse=False,
        render_vehicle=True,
        image_source="rgb_road",
        show_navi_mark=True,
        show_line_to_dest=False,
        show_dest_mark=True,
      ),
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME / 100,
      preload_models=False,
      anisotropic_filtering=bool(self.high_quality),
    )

    return ScenarioNetWorld(
      queue,
      config,
      self.test_duration,
      self.test_run,
      state_path=self.state_path,
      scenario_index=self.scenario_index,
      acceptance_radius=self.acceptance_radius,
      dual_camera=self.dual_camera,
    )
