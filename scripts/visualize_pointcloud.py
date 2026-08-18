# scripts/visualize_pointcloud.py
"""
Visualise a ModelNet40 point cloud as a 3D scatter plot.
Saves PNG for README.

Usage:
    python scripts/visualize_pointcloud.py --file data/.../chair_0001.txt --label chair
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from src.data.dataset import farthest_point_sample, normalize_point_cloud


def visualize_pointcloud(
    file_path: str,
    label:     str,
    save_dir:  str = "outputs",
    n_points:  int = 1024,
) -> None:

    # Load
    data   = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
    points = data[:, :3]

    # FPS + normalise
    points = farthest_point_sample(points, n_points)
    points = normalize_point_cloud(points)

    # Colour by height (Z axis) — looks great
    z      = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c      = z_norm,
        cmap   = "plasma",
        s      = 4,
        alpha  = 0.85,
    )

    ax.set_title(f"PointNet Input — {label.upper()}", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Best viewing angle
    ax.view_init(elev=20, azim=45)

    plt.colorbar(scatter, ax=ax, label="Height", shrink=0.5, pad=0.1)
    plt.tight_layout()

    Path(save_dir).mkdir(exist_ok=True)
    out = Path(save_dir) / f"pointcloud_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",  type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    args = parser.parse_args()
    visualize_pointcloud(args.file, args.label)