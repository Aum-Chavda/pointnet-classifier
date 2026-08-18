# src/training/metrics.py
"""
Per-class accuracy tracking for PointNet classification.

OOP  : Accumulator pattern — collect predictions over epoch, compute at end
       update() per batch, compute() per epoch, reset() between epochs
DS&A : Confusion matrix — C×C numpy array, O(B) update, O(C²) compute
       diagonal entries = correct predictions per class
Role : Tracks overall accuracy, per-class accuracy, mean class accuracy
"""

from __future__ import annotations
import numpy as np
import torch


class PerClassAccuracyTracker:
    """
    Tracks per-class classification accuracy over an epoch.

    OOP  : Accumulator pattern — stateful object that collects batch
           results and produces epoch-level summary on demand
           Three-method interface: update / compute / reset

    DS&A : Confusion matrix stored as [C, C] int64 numpy array
           confusion[i, j] = number of times true class i predicted as j
           diagonal confusion[i, i] = correct predictions for class i
           update : O(B) — one increment per sample in batch
           compute: O(C) — one division per class
           reset  : O(C²) — zero-fill the matrix

    Usage:
        tracker = PerClassAccuracyTracker(num_classes=40)
        for batch in dataloader:
            logits, labels = model(batch)
            tracker.update(logits, labels)
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self, num_classes: int = 40) -> None:
        """
        Args:
            num_classes: number of classes (40 for ModelNet40)
        """
        self.num_classes = num_classes

        # DS&A : C×C confusion matrix — int64 to avoid overflow on large datasets
        # initialised to zeros — no predictions seen yet
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Accumulate predictions from one batch into the confusion matrix.

        DS&A : O(B) — one matrix increment per sample in batch
               argmax over class dim to get predicted labels
               numpy indexing for fast matrix update

        Args:
            logits : [B, C] float tensor — raw model output (no softmax needed)
            labels : [B]    int tensor   — ground truth class indices
        """
        # Get predicted class — argmax over class dimension
        # DS&A : argmax is O(C) per sample, O(B*C) total — fast on GPU then CPU
        preds = logits.argmax(dim=1)   # [B] predicted class indices

        # Move to CPU + numpy for matrix indexing
        # DS&A : .cpu().numpy() is O(B) memory copy — small since B=32
        preds_np  = preds.detach().cpu().numpy()    # [B]
        labels_np = labels.detach().cpu().numpy()   # [B]

        # Update confusion matrix — one increment per sample
        # DS&A : numpy fancy indexing — O(B) scatter operation
        # np.add.at prevents race conditions in vectorised update
        np.add.at(self.confusion, (labels_np, preds_np), 1)

    def compute(self) -> dict[str, float | np.ndarray]:
        """
        Compute accuracy metrics from accumulated confusion matrix.

        DS&A : O(C) — one division per class
               diagonal = correct per class
               row sum  = total samples per class

        Returns dict with:
            overall_acc      : float — total correct / total samples
            mean_class_acc   : float — mean of per-class accuracies
            per_class_acc    : [C] array — accuracy for each class (0.0-1.0)
            total_samples    : int — total predictions accumulated
        """
        # Diagonal = correct predictions per class
        # DS&A : np.diag extracts diagonal — O(C)
        correct_per_class = np.diag(self.confusion)           # [C]

        # Row sum = total ground truth samples per class
        total_per_class = self.confusion.sum(axis=1)          # [C]

        # Per-class accuracy — avoid division by zero for unseen classes
        # DS&A : np.where = element-wise conditional — O(C)
        per_class_acc = np.where(
            total_per_class > 0,
            correct_per_class / total_per_class,
            0.0,
        )   # [C] floats in [0, 1]

        # Overall accuracy — total correct / total samples
        total_correct = correct_per_class.sum()
        total_samples = total_per_class.sum()
        overall_acc   = float(total_correct / total_samples) if total_samples > 0 else 0.0

        # Mean class accuracy — fairer metric than overall
        # Gives equal weight to each class regardless of sample count
        # A class with 50 samples counts as much as one with 500
        seen_classes    = total_per_class > 0           # [C] bool mask
        mean_class_acc  = float(per_class_acc[seen_classes].mean()) \
                          if seen_classes.any() else 0.0

        return {
            "overall_acc"    : overall_acc,
            "mean_class_acc" : mean_class_acc,
            "per_class_acc"  : per_class_acc,
            "total_samples"  : int(total_samples),
        }

    def reset(self) -> None:
        """
        Zero out the confusion matrix for the next epoch.

        DS&A : O(C²) fill — C=40 so 1600 elements, negligible cost
        Called at the start of each epoch before accumulating new predictions.
        """
        self.confusion.fill(0)

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Return a copy of the raw confusion matrix.

        DS&A : O(C²) copy — protects internal state from external mutation
        Used by visualize.py to plot the confusion matrix heatmap.

        Returns:
            [C, C] int64 numpy array
        """
        return self.confusion.copy()


class AverageMeter:
    """
    Tracks running average of a scalar (loss, accuracy) over batches.

    OOP  : Accumulator pattern — simpler scalar version of PerClassAccuracyTracker
    DS&A : O(1) update and compute — just addition and division
           Numerically stable: tracks sum + count separately (not rolling average)
           Rolling average: avg = avg + (new - avg)/n  ← loses precision over time
           Our approach:     avg = sum / count          ← always exact

    Usage:
        meter = AverageMeter("loss")
        for batch in dataloader:
            loss = criterion(...)
            meter.update(loss.item(), n=batch_size)
        print(meter.avg)   # epoch average loss
    """

    def __init__(self, name: str = "") -> None:
        self.name  = name
        self.sum   = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """
        Add n observations of value.

        DS&A : O(1) — just addition
        n = batch size because each sample in the batch contributes
        equally to the average (not just the mean of means)
        """
        self.sum   += value * n   # total contribution of this batch
        self.count += n           # total samples seen

    @property
    def avg(self) -> float:
        """
        DS&A : O(1) division
        Property decorator — access as meter.avg not meter.avg()
        """
        return self.sum / self.count if self.count > 0 else 0.0

    def reset(self) -> None:
        """DS&A : O(1) — reset sum and count to zero"""
        self.sum   = 0.0
        self.count = 0

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name}, avg={self.avg:.4f}, n={self.count})"