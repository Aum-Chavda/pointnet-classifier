# src/utils/config.py
"""
Configuration dataclass for PointNet classifier.

OOP  : dataclass (frozen=True) — typed record, auto __init__/__repr__, immutable
DS&A : typed record = struct in memory, O(1) attribute access via __dict__ hash map
Role : single source of truth for ALL hyperparameters — never hardcode elsewhere
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PointNetConfig:
    """
    Immutable configuration for the entire PointNet pipeline.

    frozen=True means:
      - No field can be reassigned after construction
      - Raises FrozenInstanceError if you try: cfg.lr = 0.1
      - Safe to pass around — no function can silently mutate it
      - Hashable — can be used as dict key or set member

    Usage:
        cfg = PointNetConfig()                          # all defaults
        cfg = PointNetConfig(batch_size=16, epochs=50)  # override specific fields
    """

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_root: str = "data/modelnet40"
    # Number of points sampled per point cloud via Farthest Point Sampling
    # 1024 is the standard from the original paper
    num_points: int = 1024
    # ModelNet40 has 40 object categories
    num_classes: int = 40
    # Train/val split ratio (0.9 = 90% train, 10% val from training set)
    val_split: float = 0.1

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    # Dimension of the global feature vector after max pooling
    # Paper uses 1024 — this is the "fingerprint" of the whole point cloud
    feature_dim: int = 1024
    # Intermediate dims for the classifier MLP head
    # [B, 1024] -> 512 -> 256 -> 40
    classifier_dims: tuple[int, ...] = (512, 256)
    # Dropout probability in classifier head (paper uses 0.3)
    dropout: float = 0.3
    # Weight for T-Net orthogonality regularization loss
    # L_total = CrossEntropy + reg_weight * ||I - AA^T||_F^2
    # Paper uses 0.001 — small enough not to dominate, large enough to constrain
    reg_weight: float = 0.001

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    batch_size: int = 32
    epochs: int = 50
    # Adam optimizer — paper uses lr=0.001 with decay
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Gradient clipping — prevents exploding gradients in T-Net
    grad_clip: float = 1.0
    # Learning rate step decay — halve LR every 20 epochs
    lr_step: int = 20
    lr_gamma: float = 0.5
    # Early stopping — stop if val accuracy doesn't improve for N epochs
    early_stop_patience: int = 15

    # ------------------------------------------------------------------
    # Data augmentation (training only)
    # ------------------------------------------------------------------
    # Randomly jitter point positions — adds robustness to sensor noise
    jitter_std: float = 0.02
    jitter_clip: float = 0.05
    # Randomly scale the point cloud (simulates different object sizes)
    scale_low: float = 0.8
    scale_high: float = 1.25
    # Random rotation around Y-axis only (objects sit on a table)
    rotate_y: bool = True

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------
    # num_workers=0 mandatory on Windows — multiprocessing spawn fails in scripts
    num_workers: int = 0
    pin_memory: bool = False
    checkpoint_dir: str = "checkpoints"
    log_every_n_steps: int = 10
    # Random seed for reproducibility
    seed: int = 42

    # ------------------------------------------------------------------
    # Derived properties (computed from other fields, no new data)
    # ------------------------------------------------------------------
    @property
    def data_path(self) -> Path:
        """
        DS&A: converts string to Path object — O(1), just a wrapper
        Returns pathlib.Path for safe cross-OS path manipulation.
        Always use this instead of raw data_root string.
        """
        return Path(self.data_root)

    @property
    def checkpoint_path(self) -> Path:
        """Returns checkpoint directory as Path object."""
        return Path(self.checkpoint_dir)

    def summary(self) -> str:
        """
        Human-readable config summary for logging.
        Called at start of every training run so you always know exact settings used.
        """
        lines = [
            "=" * 50,
            "PointNet Configuration",
            "=" * 50,
            f"  Data root     : {self.data_root}",
            f"  Num points    : {self.num_points}",
            f"  Num classes   : {self.num_classes}",
            f"  Feature dim   : {self.feature_dim}",
            f"  Batch size    : {self.batch_size}",
            f"  Epochs        : {self.epochs}",
            f"  LR            : {self.lr}",
            f"  Weight decay  : {self.weight_decay}",
            f"  Reg weight    : {self.reg_weight}",
            f"  Dropout       : {self.dropout}",
            f"  Augmentation  : jitter={self.jitter_std}, scale=[{self.scale_low},{self.scale_high}]",
            "=" * 50,
        ]
        return "\n".join(lines)