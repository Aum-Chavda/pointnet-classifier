# src/utils/inference.py
"""
Inference pipeline for PointNet — load checkpoint, predict single point cloud.

OOP  : Facade pattern — hides checkpoint loading, preprocessing, forward pass
       Strategy pattern — reuses same preprocessing as training (FPS + normalise)
DS&A : Softmax + argsort — O(C log C) ranking of class probabilities
       pathlib.Path — safe file handling across OS
Role : Production inference — one .txt file in, predicted class + confidence out
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from src.utils.config import PointNetConfig
from src.models.registry import ModelRegistry
from src.data.dataset import (
    farthest_point_sample,
    normalize_point_cloud,
    MODELNET40_CLASSES,
)


class Inferencer:
    """
    Loads a trained PointNet checkpoint and predicts on a single point cloud.

    OOP  : Facade — single .predict() call hides all complexity
           Strategy — preprocessing reuses dataset.py functions exactly
                      (same FPS + normalise as training = consistent results)
    DS&A : Softmax converts logits to probabilities — O(C)
           Argsort ranks classes by confidence — O(C log C), C=40

    Usage:
        inf = Inferencer(config, checkpoint_dir="checkpoints")
        result = inf.predict("data/modelnet40_normal_resampled/chair/chair_0001.txt")
        print(result["class"])        # "chair"
        print(result["confidence"])   # 0.923
        print(result["top5"])         # [("chair", 0.923), ("sofa", 0.031), ...]
    """

    def __init__(
        self,
        config:         PointNetConfig,
        checkpoint_dir: str | Path = "checkpoints",
        device:         torch.device | None = None,
    ) -> None:
        """
        Load model and checkpoint at init — pay loading cost once,
        then .predict() is fast for every subsequent call.

        OOP  : Constructor does the heavy lifting — model loading, weight restore
               After init, inferencer is stateless (no mutable state per call)
        DS&A : O(P) checkpoint load — P = num parameters ~3.5M

        Args:
            config         : PointNetConfig — must match training config
            checkpoint_dir : directory containing best_model.pth
            device         : defaults to CUDA if available
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Build model architecture — same as training
        self.model = ModelRegistry.build("pointnet", config)

        # Load best checkpoint weights
        checkpoint_path = Path(checkpoint_dir) / "best_model.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}.\n"
                f"Train the model first with: python main.py --mode train"
            )

        # Load checkpoint — map_location handles GPU→CPU transfer if needed
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()   # disables Dropout, BN uses running stats

        # Store training metadata from checkpoint
        self.best_val_acc = checkpoint.get("best",    None)
        self.best_epoch   = checkpoint.get("epoch",   None)
        self.ckpt_metrics = checkpoint.get("metrics", {})

        print(
            f"  [Inferencer] Loaded checkpoint from epoch {self.best_epoch} "
            f"(val_acc={self.best_val_acc:.4f})"
            if self.best_val_acc is not None
            else f"  [Inferencer] Loaded checkpoint from {checkpoint_path}"
        )

    def predict(
        self,
        point_cloud_path: str | Path,
    ) -> dict:
        """
        Predict class of a single point cloud from a .txt file.

        Pipeline:
            1. Load raw point cloud from .txt file    O(N)
            2. FPS — sample num_points from N          O(N*K)
            3. Normalise — centre + unit sphere        O(K)
            4. Forward pass                            O(model)
            5. Softmax + argsort — rank predictions   O(C log C)

        OOP  : Strategy — steps 2+3 use exact same functions as dataset.py
               This guarantees train/inference preprocessing consistency
        DS&A : argsort descending = sort indices by value, highest first
               probs[sorted_idx[0]] = top-1 confidence

        Args:
            point_cloud_path : path to .txt file (ModelNet40 format)

        Returns:
            dict with keys:
                class       : str   — predicted class name
                confidence  : float — probability of predicted class (0-1)
                class_idx   : int   — predicted class index (0-39)
                top5        : list of (class_name, probability) tuples
                all_probs   : [40] numpy array — full probability distribution
                file        : str — input file path
        """
        path = Path(point_cloud_path)

        if not path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {path}")

        # -- Step 1: Load raw point cloud --
        points = self._load_points(path)          # [N, 3]

        # -- Step 2: FPS sampling --
        # MUST use same num_points as training — model expects fixed N
        points = farthest_point_sample(
            points, self.config.num_points
        )                                          # [1024, 3]

        # -- Step 3: Normalise --
        # MUST use same normalisation as training
        points = normalize_point_cloud(points)     # [1024, 3]

        # -- Step 4: Forward pass --
        # Add batch dim: [1024, 3] → [1, 1024, 3]
        # DS&A : unsqueeze is O(1) — metadata change, no data copy
        points_tensor = torch.from_numpy(
            points.astype(np.float32)
        ).unsqueeze(0).to(self.device)             # [1, 1024, 3]

        with torch.no_grad():
            logits, _, _ = self.model(points_tensor)   # logits: [1, 40]

        logits = logits.squeeze(0)   # [40] — remove batch dim

        # -- Step 5: Softmax + rank --
        # DS&A : softmax is O(C) — exp + normalise
        probs = F.softmax(logits, dim=0).cpu().numpy()   # [40] sums to 1.0

        # argsort descending — O(C log C), C=40 negligible
        sorted_idx = np.argsort(probs)[::-1]   # highest probability first

        top1_idx  = int(sorted_idx[0])
        top1_prob = float(probs[top1_idx])
        top1_name = MODELNET40_CLASSES[top1_idx]

        # Top-5 predictions
        top5 = [
            (MODELNET40_CLASSES[int(idx)], float(probs[int(idx)]))
            for idx in sorted_idx[:5]
        ]

        return {
            "class"      : top1_name,
            "confidence" : top1_prob,
            "class_idx"  : top1_idx,
            "top5"       : top5,
            "all_probs"  : probs,
            "file"       : str(path),
        }

    def predict_batch(
        self,
        point_cloud_paths: list[str | Path],
    ) -> list[dict]:
        """
        Predict classes for a list of point cloud files.

        OOP  : delegates to predict() — reuses all preprocessing logic
        DS&A : O(M * N * K) where M = num files, N = raw points, K = num_points

        Args:
            point_cloud_paths : list of .txt file paths

        Returns:
            list of result dicts (same format as predict())
        """
        return [self.predict(p) for p in point_cloud_paths]

    def format_result(self, result: dict) -> str:
        """
        Format prediction result as a human-readable string.

        OOP  : pure method — no side effects, same input → same output
        DS&A : O(5) — formats top-5 predictions

        Args:
            result : dict from predict()

        Returns:
            formatted string for printing or saving
        """
        lines = [
            "=" * 50,
            f"  PointNet Inference Result",
            "=" * 50,
            f"  File       : {result['file']}",
            f"  Prediction : {result['class'].upper()}",
            f"  Confidence : {result['confidence']*100:.1f}%",
            "",
            "  Top-5 Predictions:",
        ]
        for rank, (cls, prob) in enumerate(result["top5"], 1):
            bar   = "#" * int(prob * 30)   # simple ASCII bar chart
            lines.append(f"    {rank}. {cls:<15} {prob*100:5.1f}%  {bar}")

        lines.append("=" * 50)
        return "\n".join(lines)

    @staticmethod
    def _load_points(path: Path) -> np.ndarray:
        """
        Load XYZ from ModelNet40 .txt file.
        Same as dataset.py _load_point_cloud — consistency guaranteed.

        DS&A : O(N) file read, N = 10000 lines
        """
        data = np.loadtxt(str(path), delimiter=",", dtype=np.float32)
        return data[:, :3]   # [N, 3] XYZ only


# ======================================================================
# Convenience function — quick predict without building Inferencer
# ======================================================================

def quick_predict(
    point_cloud_path: str | Path,
    config:           PointNetConfig | None = None,
    checkpoint_dir:   str | Path = "checkpoints",
) -> dict:
    """
    One-shot prediction without manually building Inferencer.

    Usage:
        result = quick_predict("data/.../chair_0001.txt")
        print(result["class"], result["confidence"])
    """
    cfg = config or PointNetConfig()
    inf = Inferencer(cfg, checkpoint_dir=checkpoint_dir)
    return inf.predict(point_cloud_path)