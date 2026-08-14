# src/models/pointnet.py
"""
Full PointNet classification network.

OOP  : Composition — PointNet HAS-A TNet, HAS-A SharedMLPs, HAS-A classifier
       Inherits BasePointNetBackbone — fulfils ABC contract
       Template Method — forward() defines fixed algorithm, delegates to sub-modules
DS&A : Pipeline — fixed stage sequence, O(N) per stage
       torch.bmm — batched matrix multiply, O(B*N*K^2) for transform application
Role : Wires all blocks into complete [B,N,3] → [B,40] forward pass
"""

from __future__ import annotations
import torch
import torch.nn as nn
from src.utils.config import PointNetConfig
from src.models.base import BasePointNetBackbone
from src.models.blocks import SharedMLP, TNet


class PointNet(BasePointNetBackbone):
    """
    PointNet classification network (Qi et al., CVPR 2017).

    Forward pass:
        [B, N, 3]
          -> input transform (TNet 3x3)
          -> SharedMLP(3->64)
          -> feature transform (TNet 64x64)
          -> SharedMLP(64->128->1024)
          -> global max pool -> [B, 1024]
          -> classifier MLP  -> [B, num_classes]

    Returns (from forward):
        logits     : [B, num_classes]  — raw class scores (no softmax)
        trans_inp  : [B, 3,  3]        — input transform matrix  (for reg loss)
        trans_feat : [B, 64, 64]       — feature transform matrix (for reg loss)

    Why return transforms?
        The regularization loss ||I - AA^T||_F^2 is computed in trainer.py
        using these matrices. The model just predicts them; the trainer
        decides how to penalise them.
    """

    def __init__(self, config: PointNetConfig) -> None:
        """
        OOP  : calls super().__init__(config) → nn.Module.__init__() via MRO
               Every nn.Module layer assigned to self.* is auto-registered
               in the parameter tree — no manual registration needed
        DS&A : all sub-modules stored as instance attributes = O(1) access
        """
        super().__init__(config)

        # -- Spatial transformers --
        # Input TNet: aligns raw XYZ point cloud (3D space)
        self.tnet_input = TNet(k=3, config=config)
        # Feature TNet: aligns 64-dim feature space
        self.tnet_feat  = TNet(k=64, config=config)

        # -- Per-point feature extraction --
        # Stage 1: 3 → 64 (after input transform)
        self.mlp1 = SharedMLP(3,   64,   batch_norm=True, activation=True)
        # Stage 2: 64 → 128 (after feature transform)
        self.mlp2 = SharedMLP(64,  128,  batch_norm=True, activation=True)
        # Stage 3: 128 → 1024 (global feature)
        self.mlp3 = SharedMLP(128, 1024, batch_norm=True, activation=True)

        # -- Classifier head --
        # Takes global feature [B, 1024] → class scores [B, num_classes]
        # Uses Linear + BN + ReLU + Dropout (paper: dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(config.feature_dim, config.classifier_dims[0]),  # 1024→512
            nn.BatchNorm1d(config.classifier_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),

            nn.Linear(config.classifier_dims[0], config.classifier_dims[1]),  # 512→256
            nn.BatchNorm1d(config.classifier_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),

            nn.Linear(config.classifier_dims[1], config.num_classes),  # 256→40
            # No softmax here — nn.CrossEntropyLoss applies log_softmax internally
            # Adding softmax here would cause double-softmax → wrong gradients
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full PointNet forward pass.

        Args:
            x: [B, N, 3] — batch of point clouds, raw XYZ coordinates

        Returns:
            logits     : [B, num_classes] — raw class scores
            trans_inp  : [B, 3,  3]       — input space transform
            trans_feat : [B, 64, 64]      — feature space transform

        DS&A : every transpose() is O(1) — stride metadata change, no data copy
               every bmm() is O(B*N*K^2) — single parallelised GPU kernel
               global max pool is O(B*C*N) — single scan across N
        """
        # -- Stage 1: Input transform --
        # TNet predicts 3x3 matrix from the raw point cloud
        trans_inp = self.tnet_input(x)          # [B, 3, 3]

        # Apply transform: rotate/align the point cloud
        # torch.bmm: batched matrix multiply
        # x is [B, N, 3], trans_inp is [B, 3, 3]
        # result: [B, N, 3] @ [B, 3, 3] → [B, N, 3]
        x = torch.bmm(x, trans_inp)             # [B, N, 3] aligned

        # -- Stage 2: First SharedMLP (3 → 64) --
        # Conv1d needs channels first: [B, N, 3] → [B, 3, N]
        x = x.transpose(2, 1)                   # [B, 3, N]
        x = self.mlp1(x)                        # [B, 64, N]

        # -- Stage 3: Feature transform --
        # TNet64 operates on [B, N, 64] — transpose back
        x = x.transpose(2, 1)                   # [B, N, 64]
        trans_feat = self.tnet_feat(x)          # [B, 64, 64]

        # Apply 64x64 feature transform
        # [B, N, 64] @ [B, 64, 64] → [B, N, 64]
        x = torch.bmm(x, trans_feat)            # [B, N, 64] aligned features

        # -- Stage 4: Remaining SharedMLPs (64 → 128 → 1024) --
        x = x.transpose(2, 1)                   # [B, 64, N]
        x = self.mlp2(x)                        # [B, 128, N]
        x = self.mlp3(x)                        # [B, 1024, N]

        # -- Stage 5: Global max pool --
        # Collapse N points → single global descriptor
        # torch.max returns (values, indices) — we only need values [0]
        # This is the permutation-invariant operation — core of PointNet
        x = torch.max(x, dim=2)[0]             # [B, 1024]

        # -- Stage 6: Classify --
        logits = self.classifier(x)             # [B, num_classes]

        return logits, trans_inp, trans_feat

    def get_feature_dim(self) -> int:
        """
        ABC contract fulfilment — returns global feature vector dimension.
        DS&A : O(1) — pure attribute lookup, no computation
        """
        return self.config.feature_dim

    def get_transform_loss(
        self,
        trans_inp:  torch.Tensor,
        trans_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes combined orthogonality regularization loss for both T-Nets.

        L_reg = reg_weight * (||I - T_inp @ T_inp^T||_F^2
                            + ||I - T_feat @ T_feat^T||_F^2)

        OOP  : delegates to TNet.regularization_loss() — single responsibility
               PointNet just sums them; TNet knows how to compute each one
        DS&A : O(K^2) per TNet — small constant cost relative to forward pass

        Args:
            trans_inp  : [B, 3,  3] — input transform from forward()
            trans_feat : [B, 64, 64] — feature transform from forward()

        Returns:
            scalar regularization loss weighted by config.reg_weight
        """
        loss_inp  = self.tnet_input.regularization_loss(trans_inp)
        loss_feat = self.tnet_feat.regularization_loss(trans_feat)
        return self.config.reg_weight * (loss_inp + loss_feat)