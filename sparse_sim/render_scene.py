import argparse
from pathlib import Path

import mujoco
from PIL import Image


def render_scene(xml_path: Path, out_path: Path, camera: str = "main_cam", width: int = 960, height: int = 720) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    Image.fromarray(pixels).save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True, help="MuJoCo XML to render")
    parser.add_argument("--out", required=True, help="PNG output path")
    parser.add_argument("--camera", default="main_cam", help="Camera name")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    xml_path = Path(args.xml)
    out_path = Path(args.out)
    render_scene(xml_path, out_path, camera=args.camera, width=args.width, height=args.height)
    print("OK: rendered", out_path)


if __name__ == "__main__":
    main()
