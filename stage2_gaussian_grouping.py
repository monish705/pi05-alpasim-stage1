import argparse
import json
from pathlib import Path

import numpy as np
from gsply import plyread


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min_gaussians', type=int, default=50)
    args = ap.parse_args()

    mask_dir = Path(args.masks) / 'masks'
    idx_dir = Path(args.masks) / 'index_maps'
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gs = plyread(args.ply)
    means, rots, scales, opacities, sh0, shN = gs.unpack()
    num_gaussians = means.shape[0]

    mask_files = sorted(mask_dir.glob('view_*.npz'))
    idx_files = sorted(idx_dir.glob('view_*.npy'))
    assert len(mask_files) == len(idx_files), "Masks and index maps count mismatch"

    # First pass: count total masks
    total_masks = 0
    view_mask_counts = []
    for mf in mask_files:
        data = np.load(mf)
        m = data['masks']
        view_mask_counts.append(m.shape[0])
        total_masks += m.shape[0]

    uf = UnionFind(total_masks)
    gaussian_rep = -np.ones(num_gaussians, dtype=np.int64)

    mask_global_offset = 0
    mask_meta = []

    for mf, idxf, mcount in zip(mask_files, idx_files, view_mask_counts):
        masks = np.load(mf)['masks']  # [K,H,W]
        idx_map = np.load(idxf)       # [H,W] gaussian id or -1
        for mi in range(masks.shape[0]):
            mask = masks[mi].astype(bool)
            gids = idx_map[mask]
            gids = gids[gids >= 0]
            if gids.size < args.min_gaussians:
                mask_meta.append({'keep': False, 'size': int(gids.size)})
                mask_global_offset += 1
                continue
            gids = np.unique(gids)
            mask_id = mask_global_offset
            mask_meta.append({'keep': True, 'size': int(gids.size)})
            for gid in gids:
                rep = gaussian_rep[gid]
                if rep == -1:
                    gaussian_rep[gid] = mask_id
                else:
                    uf.union(mask_id, rep)
            mask_global_offset += 1

    # Assign group id per gaussian
    group_ids = -np.ones(num_gaussians, dtype=np.int64)
    for gid in range(num_gaussians):
        rep = gaussian_rep[gid]
        if rep != -1:
            group_ids[gid] = uf.find(rep)

    # Compress group ids to consecutive
    unique_groups = sorted(set(group_ids[group_ids >= 0].tolist()))
    remap = {g: i for i, g in enumerate(unique_groups)}
    for gid in range(num_gaussians):
        if group_ids[gid] >= 0:
            group_ids[gid] = remap[group_ids[gid]]

    np.save(out_dir / 'group_ids.npy', group_ids)
    with open(out_dir / 'groups.json', 'w') as f:
        json.dump({'num_groups': len(unique_groups)}, f, indent=2)

    print('Stage2 complete:', out_dir, 'groups:', len(unique_groups))


if __name__ == '__main__':
    main()