import argparse
from pathlib import Path
import struct
import random
from typing import List, Tuple

import numpy as np


def _read_ply_vertices(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a binary_little_endian PLY with x,y,z and RGBA per-vertex.
    Returns (points Nx3, colors Nx3 in 0-1).
    """
    with path.open("rb") as f:
        header_lines: List[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: missing end_header")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii", errors="ignore")

        # Parse vertex count
        vertex_count = None
        for line in header.splitlines():
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                break
        if vertex_count is None:
            raise ValueError("PLY header missing vertex count")

        # Fixed layout: x,y,z float32 + r,g,b,a uint8
        stride = 3 * 4 + 4 * 1
        data = f.read(vertex_count * stride)
        if len(data) < vertex_count * stride:
            raise ValueError("PLY file truncated")

        pts = np.zeros((vertex_count, 3), dtype=np.float32)
        cols = np.zeros((vertex_count, 3), dtype=np.float32)
        offset = 0
        for i in range(vertex_count):
            x, y, z = struct.unpack_from("<fff", data, offset)
            offset += 12
            r, g, b, _a = struct.unpack_from("<BBBB", data, offset)
            offset += 4
            pts[i] = (x, y, z)
            cols[i] = (r / 255.0, g / 255.0, b / 255.0)

        return pts, cols


def _normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize points into a centered coordinate frame and scale so the
    largest dimension fits within ~1.5 meters.
    Returns (points_norm, center, scale).
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = maxs - mins
    max_dim = float(np.max(extent))
    if max_dim < 1e-6:
        scale = 1.0
    else:
        scale = 1.5 / max_dim

    points_norm = (points - center) * scale
    # Shift so min z sits on ground (z=0)
    min_z = float(points_norm[:, 2].min())
    points_norm[:, 2] -= min_z
    return points_norm, center, scale


def build_pointcloud_xml(
    points: np.ndarray,
    colors: np.ndarray,
    sample_count: int = 5000,
    point_radius: float = 0.004,
) -> str:
    """
    Create a MuJoCo XML with small spheres representing sampled point cloud.
    """
    if sample_count < len(points):
        indices = random.sample(range(len(points)), sample_count)
        points = points[indices]
        colors = colors[indices]

    # Simple camera looking at the scene center
    center = points.mean(axis=0)
    cam_pos = center + np.array([0.0, -2.5, 1.5])

    geom_lines = []
    for i, (p, c) in enumerate(zip(points, colors)):
        geom_lines.append(
            f'    <geom name="pt_{i}" type="sphere" size="{point_radius:.4f}" '
            f'pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1" />'
        )

    geom_block = "\n".join(geom_lines)
    xml = f"""<mujoco model="sparse_pointcloud">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85" rgb2="0.7 0.7 0.7" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="1 1 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid" rgba="0.9 0.9 0.9 1"/>
    <camera name="main_cam" pos="{cam_pos[0]:.3f} {cam_pos[1]:.3f} {cam_pos[2]:.3f}" quat="0.9239 0.3827 0 0"/>
{geom_block}
  </worldbody>
</mujoco>
"""
    return xml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Path to COLMAP points.ply")
    parser.add_argument("--out", required=True, help="Output MuJoCo XML path")
    parser.add_argument("--sample", type=int, default=5000, help="Number of points to render")
    parser.add_argument("--radius", type=float, default=0.004, help="Sphere radius in meters")
    args = parser.parse_args()

    ply_path = Path(args.ply)
    points, colors = _read_ply_vertices(ply_path)
    points, center, scale = _normalize_points(points)

    xml = build_pointcloud_xml(points, colors, sample_count=args.sample, point_radius=args.radius)
    out_path = Path(args.out)
    out_path.write_text(xml, encoding="utf-8")

    print("OK: wrote", out_path)
    print("Points:", len(points), "sample:", min(args.sample, len(points)))
    print("Center:", center.tolist())
    print("Scale:", scale)


if __name__ == "__main__":
    main()
