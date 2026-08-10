# src/models/base.py
"""
Abstract base class for all PointNet-style backbone networks.

OOP  : ABC (Abstract Base Class) — enforces interface contract on all subclasses
DS&A : Interface pattern — caller depends on abstraction not implementation
       O(1) method dispatch via Python vtable (C-level __dict__ lookup)
Role : Defines the contract that TNet and PointNet must both fulfil
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class BasePointNetBackbone(ABC, nn.Module):
    """
    Abstract base for all PointNet-style networks.

    Inherits from BOTH:
      - ABC          : enforces abstract method contracts
      - nn.Module    : gives us .parameters(), .to(device), .train()/.eval() for free

    Multiple inheritance order matters (MRO — Method Resolution Order):
      Python resolves method calls left to right: ABC first, then nn.Module.
      ABC.__init__ does nothing; nn.Module.__init__ sets up parameter tracking.
      We call nn.Module.__init__ explicitly via super().__init__().

    Any subclass that does NOT implement all @abstractmethod methods will raise:
      TypeError: Can't instantiate abstract class X with abstract method y
    """

    def __init__(self, config: Any) -> None:
        """
        OOP  : super().__init__() calls nn.Module.__init__() — registers
               this object with PyTorch's parameter tracking system.
               Without this call, .parameters() returns nothing and the
               optimizer has nothing to optimize.
        DS&A : config stored as instance attribute — O(1) access throughout
               subclass without passing it as argument to every method.
        """
        super().__init__()     # nn.Module.__init__() — must always call this
        self.config = config   # store config — subclasses access via self.config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass — all subclasses must implement this.

        Args:
            x: Input point cloud tensor.
               Shape depends on subclass:
               - PointNet backbone : [B, N, 3]   (batch, points, XYZ)
               - TNet              : [B, N, K]   (batch, points, K features)

        Returns:
            Tuple of tensors — subclass decides what to return.
            PointNet returns : (logits [B, C], transform [B, 3, 3], transform [B, 64, 64])
            TNet returns     : (transform_matrix [B, K, K],)

        Why tuple? Different subclasses need to return different numbers of
        tensors. A tuple is the most flexible contract — callers unpack what
        they need.
        """
        ...

    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Returns the dimensionality of the global feature vector.

        PointNet backbone : returns 1024  (after global max pool)
        TNet              : returns K     (the transform dimension)

        DS&A: This is a pure query — O(1), no computation.
        Used by registry and trainer to verify architecture compatibility.
        """
        ...

    def count_parameters(self) -> int:
        """
        Count total trainable parameters in this model.

        DS&A : sum over a generator — O(P) where P = number of parameter tensors
               p.numel() = number of elements in each parameter tensor
               Only counts requires_grad=True params (frozen params excluded)

        Not abstract — concrete utility available to ALL subclasses for free.
        Usage: print(f"Model params: {model.count_parameters():,}")
        """
        return sum(
            p.numel()                    # numel = number of elements in tensor
            for p in self.parameters()  # generator over all nn.Module parameters
            if p.requires_grad           # skip frozen parameters
        )

    def reset_parameters(self) -> None:
        """
        Re-initialise all weights to their default initialisation.

        OOP  : Template method pattern — calls a hook that subclasses can override.
               Base implementation uses PyTorch's default reset_parameters()
               on any child module that has one.
        DS&A : Traverses module tree via self.modules() — O(M) where M = num modules.

        Useful for:
          - Ablation studies (train same architecture from different random inits)
          - Hyperparameter search (reset between trials)
        """
        for module in self.modules():
            # Only reset modules that have their own reset_parameters method
            # (Conv1d, Linear, BatchNorm all have this — our custom modules may not)
            if hasattr(module, "reset_parameters") and module is not self:
                module.reset_parameters()

    def __repr__(self) -> str:
        """
        OOP  : override __repr__ for clean print output
        DS&A : O(1) — just string formatting, no traversal

        Example output:
            PointNet(params=3,476,520, feature_dim=1024, classes=40)
        """
        return (
            f"{self.__class__.__name__}("
            f"params={self.count_parameters():,}, "
            f"feature_dim={self.get_feature_dim()})"
        )