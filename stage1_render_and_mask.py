import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from gsply import plyread
from gsply.formats import SH_C0

import torch
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def look_at(cam_pos, target, up=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up2 = np.cross(right, forward)
    R = np.stack([right, up2, forward], axis=0)
    t = -R @ cam_pos
    return R.astype(np.float32), t.astype(np.float32)


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


def render_view(points, colors, R, t, width, height, fx, fy, cx, cy, splat_radius=1):
    X = (R @ points.T).T + t
    z = X[:, 2]
    valid = z > 1e-4
    X = X[valid]
    z = z[valid]
    cols = colors[valid]
    ids = np.nonzero(valid)[0]

    u = (fx * (X[:, 0] / z) + cx).astype(np.int32)
    v = (fy * (X[:, 1] / z) + cy).astype(np.int32)

    if splat_radius > 0:
        offs = np.array([(dx, dy) for dy in range(-splat_radius, splat_radius + 1)
                         for dx in range(-splat_radius, splat_radius + 1)], dtype=np.int32)
        u = (u[:, None] + offs[None, :, 0]).reshape(-1)
        v = (v[:, None] + offs[None, :, 1]).reshape(-1)
        z = np.repeat(z, offs.shape[0])
        cols = np.repeat(cols, offs.shape[0], axis=0)
        ids = np.repeat(ids, offs.shape[0])

    inb = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inb]
    v = v[inb]
    z = z[inb]
    cols = cols[inb]
    ids = ids[inb]

    if u.size == 0:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        idx_map = -np.ones((height, width), dtype=np.int32)
        return img, idx_map

    lin = v * width + u
    order = np.argsort(z)
    lin_sorted = lin[order]
    _, first = np.unique(lin_sorted, return_index=True)
    sel = order[first]

    img = np.zeros((height, width, 3), dtype=np.uint8)
    idx_map = -np.ones((height, width), dtype=np.int32)

    img_flat = img.reshape(-1, 3)
    img_flat[lin[sel]] = (np.clip(cols[sel], 0.0, 1.0) * 255).astype(np.uint8)
    idx_map.reshape(-1)[lin[sel]] = ids[sel].astype(np.int32)

    return img, idx_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_views', type=int, default=40)
    ap.add_argument('--image_size', type=int, default=640)
    ap.add_argument('--prompt', type=str, default='all objects in the scene')
    ap.add_argument('--confidence', type=float, default=0.35)
    ap.add_argument('--splat_radius', type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / 'images'
    idx_dir = out_dir / 'index_maps'
    mask_dir = out_dir / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    gs = plyread(args.ply)
    means, rots, scales, opacities, sh0, shN = gs.unpack()
    points = means.astype(np.float32)
    colors = sh0.astype(np.float32) * SH_C0 + 0.5

    center = points.mean(axis=0)
    extent = points.max(axis=0) - points.min(axis=0)
    radius = float(np.linalg.norm(extent)) * 0.6
    radius = max(radius, 1.0)

    width = height = args.image_size
    fx = fy = 0.9 * width
    cx = width / 2.0
    cy = height / 2.0

    cam_positions = fibonacci_sphere(args.num_views) * radius + center

    cameras = []

    print(f"Rendering {args.num_views} views...")
    for i, cam_pos in enumerate(cam_positions):
        R, t = look_at(cam_pos, center)
        img, idx_map = render_view(points, colors, R, t, width, height, fx, fy, cx, cy, args.splat_radius)

        img_path = img_dir / f"view_{i:03d}.png"
        idx_path = idx_dir / f"view_{i:03d}.npy"
        Image.fromarray(img).save(img_path)
        np.save(idx_path, idx_map)

        cameras.append({
            'image': img_path.name,
            'index_map': idx_path.name,
            'width': width,
            'height': height,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'R': R.tolist(),
            't': t.tolist(),
        })

    with open(out_dir / 'cameras.json', 'w') as f:
        json.dump(cameras, f, indent=2)

    print("Loading SAM3 model...")
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, load_from_HF=False)
    processor = Sam3Processor(model, device='cuda', confidence_threshold=args.confidence)

    print("Running SAM3 masks...")
    for i, cam in enumerate(cameras):
        img_path = img_dir / cam['image']
        image = Image.open(img_path).convert('RGB')
        state = processor.set_image(image)
        state = processor.set_text_prompt(args.prompt, state)

        masks = state['masks'].squeeze(1).cpu().numpy().astype(np.uint8)
        boxes = state['boxes'].cpu().numpy()
        scores = state['scores'].cpu().numpy()

        out_path = mask_dir / f"view_{i:03d}.npz"
        np.savez_compressed(out_path, masks=masks, boxes=boxes, scores=scores)

    print("Stage1 complete:", out_dir)


if __name__ == '__main__':
    main()