import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# gaussian-grouping imports (gg env)
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


def look_at(cam_pos, target, up=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up2 = np.cross(right, forward)
    # camera-to-world rotation (columns are axes)
    R_c2w = np.stack([right, up2, forward], axis=1).astype(np.float32)
    R_w2c = R_c2w.T
    t = -R_w2c @ cam_pos
    return R_c2w, t.astype(np.float32)


def fibonacci_sphere(n):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (2.0 * (i + 0.5) / n)
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points, dtype=np.float32)


def render_index_map(points, R_w2c, t, width, height, fx, fy, cx, cy):
    X = (R_w2c @ points.T).T + t
    z = X[:, 2]
    valid = z > 1e-4
    X = X[valid]
    z = z[valid]
    ids = np.nonzero(valid)[0]

    u = (fx * (X[:, 0] / z) + cx).astype(np.int32)
    v = (fy * (X[:, 1] / z) + cy).astype(np.int32)

    inb = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inb]
    v = v[inb]
    z = z[inb]
    ids = ids[inb]

    idx_map = -np.ones((height, width), dtype=np.int32)
    if u.size == 0:
        return idx_map

    lin = v * width + u
    order = np.argsort(z)
    lin_sorted = lin[order]
    _, first = np.unique(lin_sorted, return_index=True)
    sel = order[first]

    idx_map.reshape(-1)[lin[sel]] = ids[sel].astype(np.int32)
    return idx_map


class Pipe:
    def __init__(self):
        self.debug = False
        self.convert_SHs_python = True
        self.compute_cov3D_python = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_views', type=int, default=40)
    ap.add_argument('--image_size', type=int, default=768)
    ap.add_argument('--fx_scale', type=float, default=0.9)
    ap.add_argument('--white_bg', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / 'images'
    idx_dir = out_dir / 'index_maps'
    img_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    print('Loading Gaussian model...')
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(args.ply)

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    center = xyz.mean(axis=0)
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    radius = float(np.linalg.norm(extent)) * 0.6
    radius = max(radius, 1.0)

    width = height = args.image_size
    fx = fy = args.fx_scale * width
    cx = width / 2.0
    cy = height / 2.0
    fovx = float(focal2fov(fx, width))
    fovy = float(focal2fov(fy, height))

    cam_positions = fibonacci_sphere(args.num_views) * radius + center

    bg_color = torch.tensor([1, 1, 1] if args.white_bg else [0, 0, 0], dtype=torch.float32, device='cuda')
    pipe = Pipe()

    cameras = []
    print('Rendering %d views...' % args.num_views)
    for i, cam_pos in enumerate(cam_positions):
        R_c2w, t = look_at(cam_pos, center)
        R_w2c = R_c2w.T

        world_view = torch.tensor(getWorld2View2(R_c2w, t), dtype=torch.float32, device='cuda').transpose(0, 1)
        proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

        class MiniCam:
            pass
        cam = MiniCam()
        cam.image_width = width
        cam.image_height = height
        cam.FoVx = fovx
        cam.FoVy = fovy
        cam.world_view_transform = world_view
        cam.full_proj_transform = full_proj
        cam.camera_center = torch.inverse(world_view)[3][:3]

        with torch.no_grad():
            results = render(cam, gaussians, pipe, bg_color)
        rendering = results['render']
        img = (rendering.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        img_path = img_dir / ("view_%03d.png" % i)
        Image.fromarray(img).save(img_path)

        idx_map = render_index_map(xyz, R_w2c, t, width, height, fx, fy, cx, cy)
        np.save(idx_dir / ("view_%03d.npy" % i), idx_map)

        cameras.append({
            'image': img_path.name,
            'index_map': ("view_%03d.npy" % i),
            'width': width,
            'height': height,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'R': R_w2c.tolist(),
            't': t.tolist(),
        })

    with open(out_dir / 'cameras.json', 'w') as f:
        json.dump(cameras, f, indent=2)

    print('Render complete:', out_dir)


if __name__ == '__main__':
    main()
