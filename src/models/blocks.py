# src/models/blocks.py
"""
Reusable building blocks for PointNet.

OOP  : Composition — small focused classes combined into larger structures
       SharedMLP is a brick; TNet is built from SharedMLP bricks
DS&A : Pipeline pattern — nn.Sequential is a stack, O(L) forward pass
Role : SharedMLP (per-point MLP) + TNet (spatial transformer)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from src.utils.config import PointNetConfig


# ======================================================================
# SharedMLP — per-point feature extractor
# ======================================================================

class SharedMLP(nn.Module):
    """
    Applies the same MLP independently to every point in a point cloud.

    OOP  : Composition — wraps nn.Sequential (stack of layers)
           __init__ builds the pipeline; forward just calls it
    DS&A : Pipeline — data flows through layers in O(L) steps
           Conv1d with kernel_size=1 = Linear applied to every position
           in parallel — one GPU call for all N points

    Why Conv1d not Linear?
        Linear : input must be [B, C_in] — one point at a time
        Conv1d : input is  [B, C_in, N] — all N points at once
        kernel_size=1 means no spatial mixing — each position independent
        Mathematically identical to N separate Linear calls, but parallelised

    Input  shape : [B, C_in, N]   (note: N is the LAST dim for Conv1d)
    Output shape : [B, C_out, N]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        activation: bool = True,
    ) -> None:
        """
        Args:
            in_channels  : number of input features per point (e.g. 3 for XYZ)
            out_channels : number of output features per point (e.g. 64)
            batch_norm   : whether to add BatchNorm1d after Conv1d
            activation   : whether to add ReLU after BatchNorm
                           set False for the last layer of TNet (no activation
                           before the reshape to transform matrix)
        """
        super().__init__()

        # DS&A : build layers list, then wrap in Sequential (stack/pipeline)
        layers: list[nn.Module] = []

        # Conv1d(in, out, kernel_size=1) — the core SharedMLP operation
        # bias=False because BatchNorm has its own bias (beta parameter)
        # adding bias here is redundant and wastes parameters
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=not batch_norm))

        if batch_norm:
            # BatchNorm1d normalises across the batch dimension for each channel
            # Input to BN1d must be [B, C, N] — matches Conv1d output perfectly
            # Effect: zero-mean, unit-variance activations → stable training
            layers.append(nn.BatchNorm1d(out_channels))

        if activation:
            # ReLU — element-wise, no parameters, O(1) memory overhead
            # Introduces non-linearity — without this, stacking MLPs = one linear layer
            layers.append(nn.ReLU(inplace=True))
            # inplace=True modifies tensor in-place — saves memory allocation

        # nn.Sequential = ordered dict of modules, forward calls them in order
        # DS&A : this IS a stack — FIFO execution, each layer's output = next layer's input
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, N] — batch of point clouds, channels first

        Returns:
            [B, C_out, N] — transformed features, same N points
        """
        return self.layers(x)


# ======================================================================
# TNet — Spatial Transformer Network
# ======================================================================

class TNet(nn.Module):
    """
    Predicts a KxK spatial transformation matrix from the input point cloud.

    Architecture (mini PointNet):
        [B, N, K]
          → transpose to [B, K, N]                     (Conv1d needs channels first)
          → SharedMLP(K→64) → SharedMLP(64→128) → SharedMLP(128→1024)
          → Global Max Pool [B, 1024]
          → Linear(1024→512) → BN → ReLU
          → Linear(512→256)  → BN → ReLU
          → Linear(256→K*K)                            (no activation — raw matrix values)
          → reshape [B, K, K]
          → add identity matrix                        (initialise as do-nothing transform)

    OOP  : Composition — TNet HAS-A list of SharedMLPs + Linear layers
           Identity initialisation is a specific OOP pattern: initialise to neutral state
    DS&A : Global max pool = set operation, O(N) scan across points
           Matrix reshape: K*K flat vector → [K, K] matrix, O(1) view operation

    Why add identity?
        At init, Linear(256→K*K) weights are near zero → output ≈ 0
        0 + I = I → TNet starts as identity transform (do nothing)
        As training progresses, it learns small corrections away from I
        Without this: random rotation at step 1 → loss explodes → no recovery

    Args:
        k: dimension of the transform (3 for input transform, 64 for feature transform)
    """

    def __init__(self, k: int, config: PointNetConfig) -> None:
        super().__init__()
        self.k = k
        self.config = config

        # -- Point-wise feature extraction (shared across all N points) --
        # Same structure as main PointNet but smaller
        self.mlp1 = SharedMLP(k,    64,   batch_norm=True, activation=True)
        self.mlp2 = SharedMLP(64,   128,  batch_norm=True, activation=True)
        self.mlp3 = SharedMLP(128,  1024, batch_norm=True, activation=True)

        # -- Global feature → transform matrix --
        # After max pool: [B, 1024] → predict K*K values
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512,  256)
        self.fc3 = nn.Linear(256,  k * k)   # output: K*K flat, reshape to [B, K, K]

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)

        # -- Identity matrix initialisation --
        # OOP  : initialise to neutral state — the "do nothing" transform
        # DS&A : register_buffer = stored in state_dict but NOT a parameter
        #        (not optimised by gradient descent — it's a constant)
        #        .to(device) moves it automatically with the model
        self.register_buffer(
            "identity",
            torch.eye(k).unsqueeze(0)   # [1, K, K] — broadcast over batch
        )

        # Initialise fc3 weights and bias to near-zero
        # So that at init: fc3(x) ≈ 0 → 0 + I = I (identity transform)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, K] — point cloud (K=3 for input TNet, K=64 for feature TNet)

        Returns:
            transform: [B, K, K] — transformation matrix to apply to x
        """
        B = x.shape[0]   # batch size

        # Conv1d needs channels first: [B, N, K] → [B, K, N]
        # DS&A : .transpose() is O(1) — just changes stride metadata, no data copy
        x = x.transpose(2, 1)   # [B, K, N]

        # Per-point MLP — same weights applied to all N points
        x = self.mlp1(x)   # [B, 64,   N]
        x = self.mlp2(x)   # [B, 128,  N]
        x = self.mlp3(x)   # [B, 1024, N]

        # Global max pool — permutation invariant aggregation
        # torch.max returns (values, indices) — we only need values
        # dim=2 = collapse the N points dimension → [B, 1024]
        x = torch.max(x, dim=2)[0]   # [B, 1024]

        # MLP to predict transform matrix
        x = self.relu(self.bn1(self.fc1(x)))   # [B, 512]
        x = self.relu(self.bn2(self.fc2(x)))   # [B, 256]
        x = self.fc3(x)                         # [B, K*K]

        # Reshape flat vector to matrix
        # DS&A : .view() is O(1) — reinterprets memory layout, no data copy
        x = x.view(B, self.k, self.k)   # [B, K, K]

        # Add identity — at init this makes TNet a do-nothing transform
        # self.identity is [1, K, K] → broadcasts to [B, K, K]
        transform = x + self.identity   # [B, K, K]

        return transform

    def regularization_loss(self, transform: torch.Tensor) -> torch.Tensor:
        """
        Orthogonality regularization: L = ||I - A @ A^T||_F^2

        Forces the predicted transform to be a rotation matrix (orthogonal).
        A rotation matrix satisfies: A @ A^T = I
        If A is not orthogonal, it can shear/scale the point cloud — bad.

        DS&A : Frobenius norm = sqrt(sum of squared elements)
               torch.norm(..., p='fro') computes this in O(K^2)
               We square it (.pow(2)) so gradient is smooth near zero

        Args:
            transform: [B, K, K] — the predicted transform matrix

        Returns:
            scalar loss — mean over batch
        """
        B = transform.shape[0]

        # A @ A^T — batched matrix multiply
        # [B, K, K] @ [B, K, K]^T → [B, K, K]
        AAt = torch.bmm(transform, transform.transpose(1, 2))

        # Identity matrix — same device as transform
        # self.identity is [1, K, K], expand to [B, K, K] for subtraction
        I = self.identity.expand(B, -1, -1)   # [B, K, K]

        # ||I - AA^T||_F^2 — Frobenius norm squared, mean over batch
        diff = I - AAt                              # [B, K, K]
        loss = torch.norm(diff, p="fro", dim=(1,2)) # [B] — one norm per sample
        return loss.pow(2).mean()                   # scalar