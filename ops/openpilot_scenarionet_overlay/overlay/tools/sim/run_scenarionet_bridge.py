#!/usr/bin/env python3
import argparse
from multiprocessing import Queue
from pathlib import Path

from metadrive.scenario.utils import read_dataset_summary

from openpilot.tools.sim.bridge.scenarionet.scenarionet_bridge import ScenarioNetBridge


def resolve_scenario_index(database_path: str, scenario_index: int | None, scenario_id: str | None) -> int:
  if scenario_index is not None:
    return scenario_index

  if not scenario_id:
    raise ValueError("Either --scenario_index or --scenario_id is required")

  _, summary_lookup, _ = read_dataset_summary(database_path)

  if scenario_id in summary_lookup:
    return summary_lookup.index(scenario_id)

  matches = [i for i, name in enumerate(summary_lookup) if scenario_id in name]
  if len(matches) == 1:
    return matches[0]
  if len(matches) > 1:
    raise ValueError(f"Scenario id '{scenario_id}' is ambiguous. Matches: {[summary_lookup[i] for i in matches[:10]]}")

  raise ValueError(f"Scenario id '{scenario_id}' was not found in {database_path}")


def create_bridge(args: argparse.Namespace):
  queue = Queue()
  bridge = ScenarioNetBridge(
    dual_camera=args.dual_camera,
    high_quality=args.high_quality,
    database_path=args.database_path,
    scenario_index=args.scenario_index,
    acceptance_radius=args.acceptance_radius,
    state_path=args.state_path,
    reactive_traffic=args.reactive_traffic,
  )
  simulator_process = bridge.run(queue)
  return queue, simulator_process, bridge


def parse_args():
  parser = argparse.ArgumentParser(description="Bridge between ScenarioNet/ScenarioEnv and openpilot.")
  parser.add_argument("--database_path", required=True, help="Converted ScenarioNet database path")
  parser.add_argument("--scenario_index", type=int, default=None, help="Scenario index inside the database")
  parser.add_argument("--scenario_id", default=None, help="Scenario file name or unique substring")
  parser.add_argument("--acceptance_radius", type=float, default=10.0, help="Goal success threshold in meters")
  parser.add_argument("--state_path", default=str(Path(__file__).with_name("scenario_bridge_state.json")),
                      help="JSON state file written by the bridge for the monitor")
  parser.add_argument("--joystick", action="store_true")
  parser.add_argument("--high_quality", action="store_true")
  parser.add_argument("--dual_camera", action="store_true")
  parser.add_argument("--reactive_traffic", action="store_true",
                      help="Enable IDM/reactive traffic instead of strict trajectory replay for surrounding actors")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  args.database_path = str(Path(args.database_path).expanduser().resolve())
  args.state_path = str(Path(args.state_path).expanduser().resolve())
  args.scenario_index = resolve_scenario_index(args.database_path, args.scenario_index, args.scenario_id)

  queue, simulator_process, simulator_bridge = create_bridge(args)

  if args.joystick:
    from openpilot.tools.sim.lib.manual_ctrl import wheel_poll_thread

    wheel_poll_thread(queue)
  else:
    from openpilot.tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

    keyboard_poll_thread(queue)

  simulator_bridge.shutdown()
  simulator_process.join()
