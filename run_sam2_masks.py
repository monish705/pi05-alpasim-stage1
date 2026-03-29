import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def main():
    out_dir = Path('/home/ubuntu/autosim/data/masks')
    img_dir = out_dir / 'images'
    mask_dir = out_dir / 'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    cameras = json.loads((out_dir / 'cameras.json').read_text())

    ckpt = '/home/ubuntu/autosim/sam2/checkpoints/sam2.1_hiera_small.pt'
    cfg = 'configs/sam2.1/sam2.1_hiera_s'

    sam_model = build_sam2(cfg, ckpt, device='cuda')
    mask_gen = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        min_mask_region_area=200,
        output_mode='binary_mask',
    )

    counts = []
    for i, cam in enumerate(cameras):
        img_path = img_dir / cam['image']
        image = np.array(Image.open(img_path).convert('RGB'))
        h, w = image.shape[:2]
        max_area = 0.6 * h * w

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            masks = mask_gen.generate(image)

        # Filter out giant masks that swallow the whole scene
        masks = [m for m in masks if m.get('area', 0) <= max_area]

        if len(masks) == 0:
            m = np.zeros((0, h, w), dtype=np.uint8)
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            m = np.stack([x['segmentation'].astype(np.uint8) for x in masks], axis=0)
            boxes = np.stack([x['bbox'] for x in masks], axis=0).astype(np.float32)
            scores = np.array([x['predicted_iou'] for x in masks], dtype=np.float32)

        np.savez_compressed(mask_dir / ('view_%03d.npz' % i), masks=m, boxes=boxes, scores=scores)
        counts.append(m.shape[0])

    print('views', len(counts), 'nonzero', sum(c>0 for c in counts), 'max', max(counts) if counts else 0)


if __name__ == '__main__':
    main()
