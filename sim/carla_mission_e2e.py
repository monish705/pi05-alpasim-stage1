import argparse
import glob
import json
import math
import os
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.groq_semantic_client import DEFAULT_MODEL, GroqSemanticClient


def add_carla_to_path(carla_root: str) -> None:
    python_api = os.path.join(carla_root, "PythonAPI", "carla")
    egg_pattern = os.path.join(python_api, "dist", "carla-*.egg")
    egg_matches = glob.glob(egg_pattern)
    if egg_matches:
        sys.path.insert(0, egg_matches[0])
    sys.path.insert(0, python_api)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic CARLA A-to-B mission with a Groq-compiled mission spec."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--town", default="current")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=19)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--npc-count", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=150.0)
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", "/workspace"))
    parser.add_argument("--output-dir", default="/workspace/output")
    parser.add_argument(
        "--mission-text",
        default="Drive to the marked destination smoothly and safely without aggressive maneuvers.",
    )
    parser.add_argument("--compile-mission", action="store_true")
    parser.add_argument("--mission-json", default="")
    parser.add_argument("--groq-model", default=DEFAULT_MODEL)
    return parser.parse_args()


def distance(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def wait_for_image(image_queue: "queue.Queue", expected_frame: int, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    latest = None
    while time.time() < deadline:
        remaining = max(0.0, deadline - time.time())
        image = image_queue.get(timeout=remaining)
        latest = image
        if image.frame >= expected_frame:
            return image
    if latest is None:
        raise queue.Empty("No image received before timeout.")
    return latest


def append_trace(trace: list[dict[str, Any]], event: str, **payload: Any) -> None:
    trace.append(
        {
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            **payload,
        }
    )


def load_or_compile_mission(args: argparse.Namespace) -> dict[str, Any]:
    if args.mission_json:
        loaded = json.loads(Path(args.mission_json).read_text(encoding="utf-8"))
        if "parsed" in loaded:
            return loaded
        return {"parsed": loaded, "raw_text": json.dumps(loaded), "model": None, "usage": None}

    if not args.compile_mission:
        default_spec = {
            "mission_title": "Fixed CARLA A-to-B mission",
            "behavior": "normal",
            "target_speed_kmh": 25,
            "completion_radius_m": 3.0,
            "risk_posture": "medium",
            "fallback_if_blocked": "stop_and_wait",
            "notes": ["Mission compiler disabled; using default mission spec."],
        }
        return {
            "parsed": default_spec,
            "raw_text": json.dumps(default_spec),
            "model": None,
            "usage": None,
        }

    client = GroqSemanticClient(model=args.groq_model)
    return client.mission_json(mission_text=args.mission_text)


def encode_video(frames_dir: Path, video_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    args = parse_args()
    add_carla_to_path(args.carla_root)

    import carla
    from agents.navigation.behavior_agent import BehaviorAgent

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("carla_mission_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    mission_path = run_dir / "mission.json"
    trace_path = run_dir / "trace.json"
    summary_path = run_dir / "summary.json"
    video_path = run_dir / "run.mp4"

    trace: list[dict[str, Any]] = []
    mission_result = load_or_compile_mission(args)
    mission_spec = mission_result["parsed"]
    mission_bundle = {
        "mission_text": args.mission_text,
        "model": mission_result["model"],
        "usage": mission_result["usage"],
        "parsed": mission_spec,
        "raw_text": mission_result["raw_text"],
    }
    mission_path.write_text(json.dumps(mission_bundle, indent=2), encoding="utf-8")
    append_trace(
        trace,
        "mission_ready",
        mission_title=mission_spec["mission_title"],
        behavior=mission_spec["behavior"],
        target_speed_kmh=mission_spec["target_speed_kmh"],
    )

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    requested_town = args.town.strip()
    if requested_town and requested_town.lower() != "current":
        current_map_name = world.get_map().name
        if requested_town not in current_map_name:
            client.load_world(requested_town)
            world = client.get_world()
    append_trace(trace, "world_connected", map_name=world.get_map().name)

    original_settings = world.get_settings()
    get_tm = getattr(client, "get_traffic_manager", None)
    if get_tm is None:
        get_tm = getattr(client, "get_trafficmanager")
    traffic_manager = get_tm(args.traffic_manager_port)

    actors_to_destroy = []
    sensor_queue: "queue.Queue" = queue.Queue()

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(args.fps)
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        world.tick()

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if args.start_index >= len(spawn_points) or args.end_index >= len(spawn_points):
            raise IndexError("Configured spawn index is out of range for this map.")
        start_spawn = spawn_points[args.start_index]
        end_spawn = spawn_points[args.end_index]
        route_distance = distance(start_spawn.location, end_spawn.location)
        append_trace(
            trace,
            "route_selected",
            start_index=args.start_index,
            end_index=args.end_index,
            route_distance_m=route_distance,
        )

        vehicle_bp = blueprint_library.filter("vehicle.lincoln.mkz_2020")[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "hero")
        ego = world.try_spawn_actor(vehicle_bp, start_spawn)
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle at the configured start index.")
        actors_to_destroy.append(ego)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(
            carla.Location(x=-7.5, z=3.0),
            carla.Rotation(pitch=-15.0),
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego)
        actors_to_destroy.append(camera)
        camera.listen(sensor_queue.put)
        append_trace(trace, "ego_spawned")

        agent = BehaviorAgent(ego, behavior=mission_spec["behavior"])
        agent.set_target_speed(float(mission_spec["target_speed_kmh"]))
        agent.set_destination(
            end_spawn.location,
            start_location=start_spawn.location,
            clean_queue=True,
        )

        started_at = time.time()
        ticks = 0
        first_frame_path = None
        remaining_distance = route_distance
        last_trace_second = -1
        append_trace(trace, "run_started", max_seconds=args.max_seconds)

        while True:
            frame_id = world.tick()
            image = wait_for_image(sensor_queue, frame_id)
            ticks += 1
            image_path = frames_dir / f"frame_{ticks:06d}.jpg"
            image.save_to_disk(str(image_path))
            if first_frame_path is None:
                first_frame_path = str(image_path)

            control = agent.run_step()
            ego.apply_control(control)
            remaining_distance = distance(ego.get_location(), end_spawn.location)
            elapsed = time.time() - started_at
            current_second = int(elapsed)
            if current_second != last_trace_second:
                velocity = ego.get_velocity()
                speed_kmh = 3.6 * math.sqrt(
                    velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
                )
                append_trace(
                    trace,
                    "progress",
                    elapsed_s=round(elapsed, 1),
                    remaining_distance_m=round(remaining_distance, 2),
                    speed_kmh=round(speed_kmh, 1),
                )
                last_trace_second = current_second

            if agent.done() or remaining_distance <= float(mission_spec["completion_radius_m"]):
                status = "reached_goal"
                break

            if elapsed > args.max_seconds:
                status = "timeout"
                break

        encode_video(frames_dir=frames_dir, video_path=video_path, fps=args.fps)
        append_trace(trace, "run_finished", status=status, remaining_distance_m=remaining_distance)

        summary = {
            "status": status,
            "run_id": run_id,
            "map_name": world.get_map().name,
            "mission_text": args.mission_text,
            "mission_title": mission_spec["mission_title"],
            "behavior": mission_spec["behavior"],
            "target_speed_kmh": mission_spec["target_speed_kmh"],
            "risk_posture": mission_spec["risk_posture"],
            "fallback_if_blocked": mission_spec["fallback_if_blocked"],
            "route_distance_m": route_distance,
            "remaining_distance_m": remaining_distance,
            "completion_radius_m": mission_spec["completion_radius_m"],
            "start_index": args.start_index,
            "end_index": args.end_index,
            "ticks": ticks,
            "fps": args.fps,
            "max_seconds": args.max_seconds,
            "first_frame_path": first_frame_path,
            "video_path": str(video_path),
            "frames_dir": str(frames_dir),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if status == "reached_goal" else 1
    finally:
        for actor in reversed(actors_to_destroy):
            try:
                if actor.is_listening:
                    actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
