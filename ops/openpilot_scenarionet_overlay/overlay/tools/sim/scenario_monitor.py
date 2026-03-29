#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path


def parse_args():
  parser = argparse.ArgumentParser(description="Monitor ScenarioNet A-to-B bridge state.")
  parser.add_argument("--state_path", default=str(Path(__file__).with_name("scenario_bridge_state.json")))
  parser.add_argument("--poll_interval", type=float, default=0.5)
  parser.add_argument("--stale_seconds", type=float, default=5.0)
  return parser.parse_args()


def load_state(state_path: Path):
  if not state_path.exists():
    return None
  with state_path.open("r", encoding="utf-8") as f:
    return json.load(f)


if __name__ == "__main__":
  args = parse_args()
  state_path = Path(args.state_path).expanduser().resolve()

  print(f"Monitoring bridge state at {state_path}")
  while True:
    state = load_state(state_path)
    if state is None:
      print("Waiting for bridge state file...")
      time.sleep(args.poll_interval)
      continue

    age = time.time() - float(state["updated_at_unix_s"])
    if age > args.stale_seconds:
      print(f"Bridge state is stale ({age:.1f}s).")
      sys.exit(1)

    print(
      f"Scenario {state['scenario_index']} | "
      f"Speed {state['speed_kph']:.1f} km/h | "
      f"Engaged {state['engaged']} | "
      f"Route {state['route_completion']:.3f} | "
      f"Dist {state['distance_to_goal_m']:.2f} m",
      end="\r",
      flush=True,
    )

    if state.get("goal_reached"):
      print()
      print("SUCCESS: destination reached")
      print(json.dumps(state, indent=2, sort_keys=True))
      sys.exit(0)

    if state.get("done") and not state.get("goal_reached"):
      print()
      print("FAILURE: bridge terminated before reaching the goal")
      print(json.dumps(state, indent=2, sort_keys=True))
      sys.exit(1)

    time.sleep(args.poll_interval)
