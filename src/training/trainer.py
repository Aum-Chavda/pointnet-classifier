# src/training/trainer.py
"""
Trainer — full training + validation loop for PointNet.

OOP  : Facade pattern — hides PyTorch training mechanics, loss computation,
       metric tracking, and callback firing behind a single .fit() call
DS&A : History dict — append-only log, O(1) per epoch, O(E) total space
       Gradient clipping — L2 norm of gradient vector, O(P) computation
Role : Wires model + data + optimizer + callbacks into complete training run
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.config import PointNetConfig
from src.models.base import BasePointNetBackbone
from src.training.metrics import PerClassAccuracyTracker, AverageMeter
from src.training.callbacks import (
    EarlyStopping, ModelCheckpoint, LRSchedulerCallback, CallbackList
)


class Trainer:
    """
    Manages the complete training lifecycle for PointNet.

    OOP  : Facade — single interface over complex training subsystem
           Dependency injection — model, config, device injected at init
           not hardcoded. Makes testing and swapping components easy.
    DS&A : History dict — append-only log indexed by metric name
           All O(1) per batch operations; O(E*B) total where E=epochs, B=batches

    Usage:
        trainer = Trainer(model, config, device)
        history = trainer.fit(train_loader, val_loader)
        # history["val_acc"] -> list of val accuracy per epoch
    """

    def __init__(
        self,
        model:  BasePointNetBackbone,
        config: PointNetConfig,
        device: torch.device | None = None,
    ) -> None:
        """
        OOP  : Dependency injection — all dependencies passed in, not created here
               Makes unit testing possible (inject mock model/config)
        DS&A : device selection O(1) — just a string check

        Args:
            model  : PointNet model (subclass of BasePointNetBackbone)
            config : PointNetConfig — all hyperparameters
            device : torch.device — defaults to CUDA if available
        """
        self.model  = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move model to device — O(P) memory transfer where P = param count
        self.model.to(self.device)

        # -- Optimizer --
        # Adam: adaptive learning rate per parameter
        # weight_decay: L2 regularization on weights (not biases)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr           = config.lr,
            weight_decay = config.weight_decay,
        )

        # -- Loss function --
        # CrossEntropyLoss = LogSoftmax + NLLLoss combined
        # Numerically more stable than softmax + cross entropy separately
        # reduction="mean" — average loss over batch (not sum)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # -- Metrics --
        self.train_tracker = PerClassAccuracyTracker(config.num_classes)
        self.val_tracker   = PerClassAccuracyTracker(config.num_classes)

        # -- Callbacks --
        self.early_stopping = EarlyStopping(
            monitor  = "val_acc",
            patience = config.early_stop_patience,
            mode     = "max",
        )

        self.checkpoint = ModelCheckpoint(
            config    = config,
            monitor   = "val_acc",
            mode      = "max",
            model     = self.model,
            optimizer = self.optimizer,
        )

        self.lr_scheduler = LRSchedulerCallback(
            optimizer = self.optimizer,
            step_size = config.lr_step,
            gamma     = config.lr_gamma,
        )

        self.callbacks = CallbackList([
            self.early_stopping,
            self.checkpoint,
            self.lr_scheduler,
        ])

        # -- History --
        # DS&A : append-only log — O(1) list.append per epoch
        self.history: dict[str, list[float]] = {
            "train_loss"     : [],
            "val_loss"       : [],
            "train_acc"      : [],
            "val_acc"        : [],
            "train_mean_acc" : [],
            "val_mean_acc"   : [],
            "lr"             : [],
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> dict[str, list[float]]:
        """
        Run the full training loop.

        OOP  : Facade entry point — one call runs everything
        DS&A : O(E * B * N) total — E epochs, B batches, N points per cloud

        Args:
            train_loader : DataLoader for training split
            val_loader   : DataLoader for validation split

        Returns:
            history dict — metric lists indexed by name, one value per epoch
        """
        print(f"\n{'='*60}")
        print(f"  Training PointNet")
        print(f"  Device  : {self.device}")
        print(f"  Epochs  : {self.config.epochs}")
        print(f"  Params  : {self.model.count_parameters():,}")
        print(f"{'='*60}\n")

        self.callbacks.on_train_begin()

        for epoch in range(self.config.epochs):
            self.callbacks.on_epoch_begin(epoch)

            # -- Train one epoch --
            train_metrics = self._train_epoch(epoch, train_loader)

            # -- Validate one epoch --
            val_metrics = self._val_epoch(epoch, val_loader)

            # -- Merge metrics for callbacks --
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics["lr"] = self._get_lr()

            # -- Log to history --
            # DS&A : O(1) append per metric per epoch
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["train_acc"].append(train_metrics["train_acc"])
            self.history["val_acc"].append(val_metrics["val_acc"])
            self.history["train_mean_acc"].append(train_metrics["train_mean_acc"])
            self.history["val_mean_acc"].append(val_metrics["val_mean_acc"])
            self.history["lr"].append(epoch_metrics["lr"])

            # -- Print epoch summary --
            self._print_epoch(epoch, train_metrics, val_metrics)

            # -- Fire callbacks --
            self.callbacks.on_epoch_end(epoch, epoch_metrics)

            # -- Check early stopping --
            if self.callbacks.should_stop:
                print(f"\n  Stopped at epoch {epoch}.")
                break

        self.callbacks.on_train_end()
        print(f"\n{'='*60}")
        print(f"  Training complete.")
        print(f"  Best val_acc : {self.early_stopping.best:.4f} "
              f"at epoch {self.early_stopping.best_epoch}")
        print(f"{'='*60}\n")

        return self.history

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        epoch:        int,
        train_loader: DataLoader,
    ) -> dict[str, float]:
        """
        One full pass over the training set.

        DS&A : O(B * N * C) — B batches, N=1024 points, C=forward pass cost
               AverageMeter: O(1) update per batch
               PerClassAccuracyTracker: O(B_size) update per batch

        Returns:
            dict with train_loss, train_acc, train_mean_acc
        """
        # Set model to train mode — enables Dropout, BatchNorm uses batch stats
        self.model.train()

        loss_meter = AverageMeter("train_loss")
        self.train_tracker.reset()

        n_batches = len(train_loader)

        for batch_idx, (points, labels) in enumerate(train_loader):
            # Move data to device — O(B*N*3) memory transfer
            points = points.to(self.device)   # [B, N, 3]
            labels = labels.to(self.device)   # [B]

            # -- Forward pass --
            # optimizer.step() BEFORE scheduler.step() (called in callback)
            self.optimizer.zero_grad()

            logits, trans_inp, trans_feat = self.model(points)
            # logits: [B, 40]  trans_inp: [B,3,3]  trans_feat: [B,64,64]

            # -- Loss computation --
            # CrossEntropy loss — classification
            cls_loss = self.criterion(logits, labels)

            # T-Net orthogonality regularization
            reg_loss = self.model.get_transform_loss(trans_inp, trans_feat)

            # Total loss
            loss = cls_loss + reg_loss

            # -- Backward pass --
            loss.backward()

            # Gradient clipping — prevents exploding gradients in T-Net
            # DS&A : clips gradient L2 norm to max_norm — O(P) computation
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm = self.config.grad_clip,
            )

            # -- Optimizer step --
            self.optimizer.step()

            # -- Track metrics --
            # AverageMeter: weight by batch size for correct epoch average
            loss_meter.update(loss.item(), n=points.size(0))
            self.train_tracker.update(logits, labels)

            # -- Progress print every N steps --
            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                print(
                    f"  Epoch [{epoch+1:03d}] "
                    f"[{batch_idx+1:03d}/{n_batches:03d}] "
                    f"loss: {loss_meter.avg:.4f}",
                    end="\r",
                )

        # Compute epoch-level metrics
        train_metrics = self.train_tracker.compute()

        return {
            "train_loss"     : loss_meter.avg,
            "train_acc"      : train_metrics["overall_acc"],
            "train_mean_acc" : train_metrics["mean_class_acc"],
        }

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------

    def _val_epoch(
        self,
        epoch:      int,
        val_loader: DataLoader,
    ) -> dict[str, float]:
        """
        One full pass over the validation set.

        Key differences from train:
            - model.eval() — disables Dropout, BatchNorm uses running stats
            - torch.no_grad() — disables gradient computation, saves memory
            - No optimizer.step() — we're just measuring, not updating

        DS&A : O(B * N * C) same as train but ~2x faster (no backward pass)

        Returns:
            dict with val_loss, val_acc, val_mean_acc
        """
        # eval mode — disables Dropout, BN uses running mean/var
        self.model.eval()

        loss_meter = AverageMeter("val_loss")
        self.val_tracker.reset()

        # no_grad — disables autograd graph construction
        # DS&A : saves ~2x memory (no activation storage for backward)
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(self.device)   # [B, N, 3]
                labels = labels.to(self.device)   # [B]

                # Forward pass only
                logits, trans_inp, trans_feat = self.model(points)

                # Loss (for monitoring — not used for gradient)
                cls_loss = self.criterion(logits, labels)
                reg_loss = self.model.get_transform_loss(trans_inp, trans_feat)
                loss     = cls_loss + reg_loss

                loss_meter.update(loss.item(), n=points.size(0))
                self.val_tracker.update(logits, labels)

        val_metrics = self.val_tracker.compute()

        return {
            "val_loss"     : loss_meter.avg,
            "val_acc"      : val_metrics["overall_acc"],
            "val_mean_acc" : val_metrics["mean_class_acc"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_lr(self) -> float:
        """
        Get current learning rate from optimizer.
        DS&A : O(1) — list index into optimizer param groups
        """
        return self.optimizer.param_groups[0]["lr"]

    def _print_epoch(
        self,
        epoch:        int,
        train_metrics: dict,
        val_metrics:   dict,
    ) -> None:
        """Print one-line epoch summary."""
        print(
            f"  Epoch [{epoch+1:03d}/{self.config.epochs:03d}] "
            f"| train_loss: {train_metrics['train_loss']:.4f} "
            f"| train_acc: {train_metrics['train_acc']*100:.2f}% "
            f"| val_loss: {val_metrics['val_loss']:.4f} "
            f"| val_acc: {val_metrics['val_acc']*100:.2f}% "
            f"| lr: {self._get_lr():.6f}"
        )

    def get_val_confusion_matrix(self) -> "np.ndarray":
        """
        Return confusion matrix from last validation epoch.
        Used by visualize.py for confusion matrix plot.
        """
        return self.val_tracker.get_confusion_matrix()