import argparse
import glob
import json
import math
import os
import queue
import random
import sys
import time
from pathlib import Path


def add_carla_to_path(carla_root: str) -> None:
    python_api = os.path.join(carla_root, "PythonAPI", "carla")
    egg_pattern = os.path.join(
        python_api,
        "dist",
        "carla-*.egg",
    )
    egg_matches = glob.glob(egg_pattern)
    if egg_matches:
        sys.path.insert(0, egg_matches[0])
    sys.path.insert(0, python_api)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CARLA built-in driving agent from A to B and record a video."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--town", default="current")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-speed", type=float, default=30.0)
    parser.add_argument("--npc-count", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--behavior", default="normal", choices=["cautious", "normal", "aggressive"])
    parser.add_argument("--vehicle-filter", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--output-dir", default="/workspace/output")
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", "/workspace"))
    return parser.parse_args()


def distance(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def choose_route(spawn_points, min_distance: float = 150.0, max_distance: float = 350.0):
    fallback_pair = None
    fallback_distance = -1.0
    for i, start in enumerate(spawn_points):
        for j, end in enumerate(spawn_points):
            if i == j:
                continue
            current = distance(start.location, end.location)
            if min_distance <= current <= max_distance:
                return (i, j, start, end), current
            if current > fallback_distance:
                fallback_distance = current
                fallback_pair = (i, j, start, end)
    if fallback_pair is None:
        raise RuntimeError("Could not find distinct spawn points for the route.")
    return fallback_pair, fallback_distance


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


def main() -> int:
    args = parse_args()
    add_carla_to_path(args.carla_root)

    import carla
    from agents.navigation.behavior_agent import BehaviorAgent

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = output_dir / f"carla_agent_run_{run_stamp}.mp4"
    meta_path = output_dir / f"carla_agent_run_{run_stamp}.json"
    frames_dir = output_dir / f"carla_agent_run_{run_stamp}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    print(f"Connecting to CARLA at {args.host}:{args.port}", flush=True)
    world = client.get_world()
    requested_town = args.town.strip()
    if requested_town and requested_town.lower() != "current":
        current_map_name = world.get_map().name
        if requested_town not in current_map_name:
            print(f"Loading world {requested_town}", flush=True)
            client.load_world(requested_town)
            world = client.get_world()
    world.tick()

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
        traffic_manager.set_random_device_seed(args.seed)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        world.tick()

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            raise RuntimeError("Map does not have enough spawn points.")

        (start_idx, end_idx, start_spawn, end_spawn), route_distance = choose_route(spawn_points)
        print(
            f"Using route start index {start_idx} -> end index {end_idx} "
            f"(approx {route_distance:.1f} meters)",
            flush=True,
        )

        vehicle_blueprints = blueprint_library.filter(args.vehicle_filter)
        if not vehicle_blueprints:
            vehicle_blueprints = blueprint_library.filter("vehicle.*")
        vehicle_bp = vehicle_blueprints[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "hero")
        ego = world.try_spawn_actor(vehicle_bp, start_spawn)
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle at the selected start point.")
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

        npc_candidates = spawn_points.copy()
        random.shuffle(npc_candidates)
        npc_count = 0
        for spawn in npc_candidates:
            if spawn.location.distance(start_spawn.location) < 8.0:
                continue
            if spawn.location.distance(end_spawn.location) < 8.0:
                continue
            npc_bp = random.choice(blueprint_library.filter("vehicle.*"))
            npc = world.try_spawn_actor(npc_bp, spawn)
            if npc is None:
                continue
            npc.set_autopilot(True, traffic_manager.get_port())
            actors_to_destroy.append(npc)
            npc_count += 1
            if npc_count >= args.npc_count:
                break

        agent = BehaviorAgent(ego, behavior=args.behavior)
        agent.set_target_speed(args.target_speed)
        agent.set_destination(end_spawn.location, start_location=start_spawn.location, clean_queue=True)

        started_at = time.time()
        ticks = 0
        last_distance = distance(ego.get_location(), end_spawn.location)

        print(
            f"Ego vehicle spawned. NPC count: {npc_count}. Recording to {video_path}",
            flush=True,
        )

        while True:
            frame_id = world.tick()
            image = wait_for_image(sensor_queue, frame_id)
            ticks += 1
            image_path = frames_dir / f"frame_{ticks:06d}.jpg"
            image.save_to_disk(str(image_path))

            control = agent.run_step()
            ego.apply_control(control)
            current_location = ego.get_location()
            remaining_distance = distance(current_location, end_spawn.location)

            if ticks % args.fps == 0:
                speed = ego.get_velocity()
                speed_kmh = 3.6 * math.sqrt(speed.x * speed.x + speed.y * speed.y + speed.z * speed.z)
                print(
                    f"t={ticks / args.fps:6.1f}s remaining={remaining_distance:7.2f}m speed={speed_kmh:5.1f}km/h",
                    flush=True,
                )

            if remaining_distance < last_distance:
                last_distance = remaining_distance

            if agent.done() or remaining_distance < 3.0:
                status = "reached_goal"
                break

            if time.time() - started_at > args.max_seconds:
                status = "timeout"
                break

        metadata = {
            "status": status,
            "town": args.town,
            "seed": args.seed,
            "behavior": args.behavior,
            "target_speed_kmh": args.target_speed,
            "fps": args.fps,
            "ticks": ticks,
            "max_seconds": args.max_seconds,
            "route_distance_m": route_distance,
            "remaining_distance_m": remaining_distance,
            "start_index": start_idx,
            "end_index": end_idx,
            "start_location": {
                "x": start_spawn.location.x,
                "y": start_spawn.location.y,
                "z": start_spawn.location.z,
            },
            "end_location": {
                "x": end_spawn.location.x,
                "y": end_spawn.location.y,
                "z": end_spawn.location.z,
            },
            "frames_dir": str(frames_dir),
            "video_path": str(video_path),
        }
        meta_path.write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2), flush=True)
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
