# scripts/precompute_fps.py
"""
Pre-compute FPS sampling for all ModelNet40 files and cache to disk.
Run once before training — saves ~0.16s per sample per epoch.

Usage:
    python scripts/precompute_fps.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.utils.config import PointNetConfig
from src.data.dataset import farthest_point_sample, normalize_point_cloud

cfg       = PointNetConfig()
data_path = cfg.data_path
cache_dir = Path("data/fps_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

print(f"Pre-computing FPS cache → {cache_dir}")
print(f"num_points = {cfg.num_points}")

all_files = list(data_path.glob("*/*.txt"))
print(f"Total files : {len(all_files)}")

skipped = 0
for path in tqdm(all_files, desc="FPS sampling"):
    # Cache key = relative path with .npy extension
    # e.g. airplane/airplane_0001.txt → fps_cache/airplane/airplane_0001.npy
    rel    = path.relative_to(data_path)
    out    = cache_dir / rel.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        skipped += 1
        continue   # already cached — skip

    # Load raw points
    data   = np.loadtxt(str(path), delimiter=",", dtype=np.float32)
    points = data[:, :3]   # [10000, 3]

    # FPS sample
    sampled = farthest_point_sample(points, cfg.num_points)   # [1024, 3]

    # Normalise
    sampled = normalize_point_cloud(sampled)   # [1024, 3]

    # Save
    np.save(str(out), sampled)

print(f"\nDone. Skipped {skipped} already-cached files.")
print(f"Cache saved at: {cache_dir.resolve()}")