# src/training/callbacks.py
"""
Training callbacks — modular hooks fired by the trainer at epoch boundaries.

OOP  : Observer pattern — trainer fires hooks, callbacks respond independently
       Each callback is a single-responsibility class
       Base class defines the interface; subclasses implement behaviour
DS&A : EarlyStopping — monotonic best-value tracker, O(1) per epoch
       ModelCheckpoint — O(1) comparison + O(P) file write (P = param count)
Role : EarlyStopping, ModelCheckpoint, LRSchedulerCallback
"""

from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path
from src.utils.config import PointNetConfig


# ======================================================================
# Base Callback — defines the hook interface
# ======================================================================

class Callback:
    """
    Abstract base for all callbacks.

    OOP  : Template / Observer base — defines hook methods with no-op defaults
           Subclasses override only the hooks they care about
           Trainer calls all hooks without knowing which callbacks are active

    Hooks available:
        on_train_begin  — called once before training starts
        on_epoch_begin  — called at start of each epoch
        on_epoch_end    — called at end of each epoch with metrics
        on_train_end    — called once after training finishes

    Why no-op defaults?
        If EarlyStopping only needs on_epoch_end, it shouldn't be forced
        to implement on_train_begin. No-op defaults make partial implementation
        clean — override what you need, ignore the rest.
    """

    def on_train_begin(self) -> None:
        """Called once before the first epoch."""
        pass

    def on_epoch_begin(self, epoch: int) -> None:
        """Called at the start of each epoch before train loop."""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        """
        Called at the end of each epoch after val loop.

        Args:
            epoch   : current epoch number (0-indexed)
            metrics : dict with keys like 'val_loss', 'val_acc', 'train_loss'
        """
        pass

    def on_train_end(self) -> None:
        """Called once after the last epoch."""
        pass


# ======================================================================
# EarlyStopping
# ======================================================================

class EarlyStopping(Callback):
    """
    Stops training when a monitored metric stops improving.

    OOP  : Overrides only on_epoch_end — single responsibility
    DS&A : Monotonic tracker — O(1) per epoch
           best_value: running maximum of monitored metric
           counter   : epochs since last improvement
           When counter >= patience -> sets self.should_stop = True

    Real-world analogy: a runner who stops after N laps with no personal best.

    Args:
        monitor  : metric key to watch (e.g. "val_acc", "val_loss")
        patience : epochs to wait without improvement before stopping
        mode     : "max" (higher=better, e.g. accuracy)
                   "min" (lower=better, e.g. loss)
        delta    : minimum change to count as improvement
    """

    def __init__(
        self,
        monitor:  str   = "val_acc",
        patience: int   = 15,
        mode:     str   = "max",
        delta:    float = 1e-4,
    ) -> None:
        self.monitor     = monitor
        self.patience    = patience
        self.mode        = mode
        self.delta       = delta
        self.should_stop = False   # trainer checks this after on_epoch_end

        # DS&A : monotonic best tracker — O(1) comparison each epoch
        if mode == "max":
            self.best    = float("-inf")
            self._is_better = lambda current, best: current > best + delta
        else:  # "min"
            self.best    = float("inf")
            self._is_better = lambda current, best: current < best - delta

        self.counter     = 0   # epochs without improvement
        self.best_epoch  = 0   # epoch where best was achieved

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        """
        Check if monitored metric improved. Update counter or reset.

        DS&A : O(1) — one comparison, one increment or reset
        """
        if self.monitor not in metrics:
            return   # metric not available this epoch — skip

        current = metrics[self.monitor]

        if self._is_better(current, self.best):
            # Improvement found — reset counter, update best
            self.best       = current
            self.best_epoch = epoch
            self.counter    = 0
        else:
            # No improvement — increment counter
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True
                print(
                    f"\n  [EarlyStopping] No improvement in '{self.monitor}' "
                    f"for {self.patience} epochs. "
                    f"Best: {self.best:.4f} at epoch {self.best_epoch}. "
                    f"Stopping."
                )

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(monitor={self.monitor}, patience={self.patience}, "
            f"best={self.best:.4f}, counter={self.counter})"
        )


# ======================================================================
# ModelCheckpoint
# ======================================================================

class ModelCheckpoint(Callback):
    """
    Saves model weights when a monitored metric improves.

    OOP  : Single responsibility — only saves checkpoints, nothing else
           Overrides on_epoch_end and on_train_end
    DS&A : O(1) comparison per epoch
           O(P) file write when improvement found (P = num params ~3.5M)
           Saves: model state_dict, optimizer state, epoch, metrics

    Saves two files:
        best_model.pth   — best checkpoint (overwritten on each improvement)
        last_model.pth   — last epoch checkpoint (for resuming)
    """

    def __init__(
        self,
        config:    PointNetConfig,
        monitor:   str   = "val_acc",
        mode:      str   = "max",
        delta:     float = 1e-4,
        model:     nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.config    = config
        self.monitor   = monitor
        self.mode      = mode
        self.delta     = delta
        self.model     = model
        self.optimizer = optimizer

        # Create checkpoint directory
        # pathlib.Path.mkdir — creates parent dirs too if needed
        self.save_dir = config.checkpoint_path
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if mode == "max":
            self.best       = float("-inf")
            self._is_better = lambda c, b: c > b + delta
        else:
            self.best       = float("inf")
            self._is_better = lambda c, b: c < b - delta

        self.best_epoch = 0

    def set_model(self, model: nn.Module) -> None:
        """Attach model after init (needed because model built after callback)."""
        self.model = model

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Attach optimizer after init."""
        self.optimizer = optimizer

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        """
        Save best checkpoint if monitored metric improved.
        Always save last checkpoint for resuming.

        DS&A : O(1) comparison + O(P) file write on improvement
        """
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]

        # Always save last checkpoint
        self._save(epoch, metrics, filename="last_model.pth")

        if self._is_better(current, self.best):
            self.best       = current
            self.best_epoch = epoch
            self._save(epoch, metrics, filename="best_model.pth")
            print(
                f"  [Checkpoint] New best {self.monitor}: "
                f"{self.best:.4f} -> saved best_model.pth"
            )

    def on_train_end(self) -> None:
        """Print summary of best checkpoint location."""
        print(
            f"\n  [Checkpoint] Best {self.monitor}: {self.best:.4f} "
            f"at epoch {self.best_epoch}. "
            f"Saved at: {self.save_dir / 'best_model.pth'}"
        )

    def _save(self, epoch: int, metrics: dict, filename: str) -> None:
        """
        Write checkpoint to disk.

        Checkpoint contains:
            epoch      : int — current epoch
            model      : state_dict — all parameter tensors
            optimizer  : state_dict — optimizer momentum/variance states
            metrics    : dict — val_acc, val_loss etc. at this epoch
            best       : float — best metric seen so far

        DS&A : torch.save uses pickle under the hood — O(P) serialisation
        """
        if self.model is None:
            return

        checkpoint = {
            "epoch"    : epoch,
            "model"    : self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "metrics"  : metrics,
            "best"     : self.best,
        }

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)

    def load_best(self, model: nn.Module, device: torch.device) -> nn.Module:
        """
        Load best checkpoint weights into model.

        Used by inference.py to restore the trained model.

        Args:
            model  : uninitialised PointNet model (same architecture)
            device : torch.device to load onto

        Returns:
            model with best weights loaded
        """
        best_path = self.save_dir / "best_model.pth"

        if not best_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {best_path}. Train first."
            )

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(
            f"  [Checkpoint] Loaded best model from epoch "
            f"{checkpoint['epoch']} "
            f"({self.monitor}={checkpoint['best']:.4f})"
        )
        return model


# ======================================================================
# LRSchedulerCallback
# ======================================================================

class LRSchedulerCallback(Callback):
    """
    Wraps a PyTorch LR scheduler as a callback.

    OOP  : Adapter pattern — adapts PyTorch scheduler interface to our
           Callback interface. Trainer doesn't need to know about schedulers.
    DS&A : O(1) per epoch — scheduler.step() is O(1) parameter update

    We use StepLR: multiply LR by gamma every step_size epochs.
    Config: lr_step=20, lr_gamma=0.5 -> halve LR every 20 epochs.

    Example LR schedule (lr=0.001, step=20, gamma=0.5):
        epochs  1-20  : lr = 0.001000
        epochs 21-40  : lr = 0.000500
        epochs 41-50  : lr = 0.000250
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int   = 20,
        gamma:     float = 0.5,
    ) -> None:
        # torch.optim.lr_scheduler.StepLR:
        # multiplies LR by gamma every step_size epochs
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = step_size,
            gamma     = gamma,
        )
        self.current_lr = None

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        """
        Step the scheduler — called after val loop each epoch.
        DS&A : O(1) — just updates learning rate scalar
        """
        self.scheduler.step()
        # Get current LR for logging
        self.current_lr = self.scheduler.get_last_lr()[0]

    def get_lr(self) -> float:
        """Return current learning rate."""
        if self.current_lr is None:
            return self.scheduler.get_last_lr()[0]
        return self.current_lr


# ======================================================================
# CallbackList — fires all callbacks in sequence
# ======================================================================

class CallbackList:
    """
    Holds a list of callbacks and fires them all at each hook point.

    OOP  : Composite pattern — treats a list of callbacks as a single callback
           Trainer only needs to call one object, not iterate manually
    DS&A : O(K) per hook where K = number of callbacks — tiny in practice

    Usage:
        callbacks = CallbackList([
            EarlyStopping(patience=15),
            ModelCheckpoint(config),
            LRSchedulerCallback(optimizer),
        ])
        callbacks.on_epoch_end(epoch, metrics)
        if callbacks.should_stop:
            break
    """

    def __init__(self, callbacks: list[Callback]) -> None:
        self.callbacks = callbacks

    @property
    def should_stop(self) -> bool:
        """
        Returns True if any EarlyStopping callback says to stop.
        DS&A : O(K) scan — K callbacks, stop at first True
        """
        return any(
            getattr(cb, "should_stop", False)
            for cb in self.callbacks
        )

    def on_train_begin(self) -> None:
        for cb in self.callbacks:
            cb.on_train_begin()

    def on_epoch_begin(self, epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics)

    def on_train_end(self) -> None:
        for cb in self.callbacks:
            cb.on_train_end()