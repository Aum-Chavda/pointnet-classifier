# main.py
"""
Entry point for PointNet classifier.

Modes:
    train : full training run -> saves checkpoints + history
    eval  : load best checkpoint -> generate all plots
    infer : predict class of a single point cloud file

Usage:
    python main.py --mode train
    python main.py --mode eval
    python main.py --mode infer --file path/to/cloud.txt
    
"""

from __future__ import annotations
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from src.utils.config import PointNetConfig
from src.models.registry import ModelRegistry
from src.data.dataset import build_dataloader, MODELNET40_CLASSES
from src.training.trainer import Trainer
from src.utils.visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_tsne,
    extract_features,
)
#from src.utils.inference import Inferencer


# ======================================================================
# Reproducibility
# ======================================================================

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ======================================================================
# Train
# ======================================================================

def run_train(cfg: PointNetConfig) -> None:
    """Full training run."""
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print(cfg.summary())

    # -- Data --
    print("\n  Loading dataset...")
    train_loader = build_dataloader(cfg, split="train")
    val_loader   = build_dataloader(cfg, split="test")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # -- Model --
    model = ModelRegistry.build("pointnet", cfg)
    print(f"\n  Model : {model}")

    # -- Train --
    trainer = Trainer(model, cfg, device)
    history = trainer.fit(train_loader, val_loader)

    # -- Save history --
    Path("outputs").mkdir(exist_ok=True)
    np.save("outputs/history.npy", history)
    print("  Saved history -> outputs/history.npy")

    # -- Plots --
    print("\n  Generating plots...")
    plot_training_curves(history, save_dir="outputs")

    # Confusion matrix from last val epoch
    cm = trainer.get_val_confusion_matrix()
    plot_confusion_matrix(cm, save_dir="outputs", normalize=True)

    print("\n  Training complete. Run --mode eval for full evaluation.")


# ======================================================================
# Eval
# ======================================================================

def run_eval(cfg: PointNetConfig) -> None:
    """Load best checkpoint, run full evaluation, generate all plots."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Load history if available --
    history_path = Path("outputs/history.npy")
    if history_path.exists():
        history = np.load(str(history_path), allow_pickle=True).item()
        print("  Loaded training history.")
        plot_training_curves(history, save_dir="outputs")
    else:
        print("  No history found — skipping training curves.")

    # -- Load model --
    ckpt_path = Path("checkpoints/best_model.pth")
    if not ckpt_path.exists():
        print("  No checkpoint found. Train first with --mode train")
        return

    model = ModelRegistry.build("pointnet", cfg)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    best_epoch   = checkpoint.get("epoch", "?")
    best_val_acc = checkpoint.get("best",  0.0)
    print(f"\n  Loaded best model — epoch {best_epoch}, val_acc={best_val_acc:.4f}")

    # -- Full test set evaluation --
    print("\n  Running full test set evaluation...")
    from src.training.metrics import PerClassAccuracyTracker, AverageMeter
    import torch.nn as nn

    criterion  = nn.CrossEntropyLoss()
    tracker    = PerClassAccuracyTracker(cfg.num_classes)
    loss_meter = AverageMeter("test_loss")
    test_loader = build_dataloader(cfg, split="test")

    with torch.no_grad():
        for points, labels in test_loader:
            points = points.to(device)
            labels = labels.to(device)
            logits, ti, tf = model(points)
            loss = criterion(logits, labels)
            loss_meter.update(loss.item(), n=points.size(0))
            tracker.update(logits, labels)

    metrics = tracker.compute()
    print(f"\n  -- Test Set Results --")
    print(f"  Overall Accuracy   : {metrics['overall_acc']*100:.2f}%")
    print(f"  Mean Class Accuracy: {metrics['mean_class_acc']*100:.2f}%")
    print(f"  Test Loss          : {loss_meter.avg:.4f}")
    print(f"  Total Samples      : {metrics['total_samples']}")

    # Per-class breakdown
    print(f"\n  -- Per-Class Accuracy --")
    per_class = metrics["per_class_acc"]
    for i, (cls, acc) in enumerate(zip(MODELNET40_CLASSES, per_class)):
        bar = "#" * int(acc * 20)
        print(f"  {cls:<15} {acc*100:5.1f}%  {bar}")

    # -- Confusion matrix --
    cm = tracker.get_confusion_matrix()
    plot_confusion_matrix(cm, save_dir="outputs", normalize=True)

    # -- t-SNE --
    print("\n  Extracting features for t-SNE...")
    features, labels_np = extract_features(
        model, test_loader, device, max_batches=80
    )
    plot_tsne(features, labels_np, save_dir="outputs")

    # -- Save results --
    results_path = Path("outputs/test_results.txt")
    with open(results_path, "w") as f:
        f.write(f"PointNet Test Results — ModelNet40\n")
        f.write(f"{'='*40}\n")
        f.write(f"Checkpoint epoch  : {best_epoch}\n")
        f.write(f"Overall accuracy  : {metrics['overall_acc']*100:.2f}%\n")
        f.write(f"Mean class acc    : {metrics['mean_class_acc']*100:.2f}%\n")
        f.write(f"Test loss         : {loss_meter.avg:.4f}\n")
        f.write(f"Total samples     : {metrics['total_samples']}\n\n")
        f.write(f"Per-class accuracy:\n")
        for cls, acc in zip(MODELNET40_CLASSES, per_class):
            f.write(f"  {cls:<15} {acc*100:.2f}%\n")

    print(f"\n  Results saved -> {results_path}")
    print(f"\n  Outputs:")
    print(f"    outputs/training_curves.png")
    print(f"    outputs/confusion_matrix.png")
    print(f"    outputs/tsne_features.png")
    print(f"    outputs/test_results.txt")


# ======================================================================
# Infer
# ======================================================================

#def run_infer(cfg: PointNetConfig, file_path: str) -> None:
    """Run inference on a single point cloud file."""
    #print(f"\n  Running inference on: {file_path}")

   # inf    = Inferencer(cfg, checkpoint_dir="checkpoints")
   # result = inf.predict(file_path)

   # print(inf.format_result(result))

    # Save result
   #Path("outputs").mkdir(exist_ok=True)
    #out = Path("outputs/inference_result.txt")
   # with open(out, "w") as f:
    #    f.write(inf.format_result(result))
    #print(f"\n  Saved -> {out}")


# ======================================================================
# Entry point
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="PointNet Classifier")
    parser.add_argument(
        "--mode",
        type    = str,
        default = "train",
        choices = ["train", "eval", "infer"],
        help    = "train | eval | infer",
    )
    parser.add_argument(
        "--file",
        type    = str,
        default = None,
        help    = "Path to .txt point cloud file (required for --mode infer)",
    )
    args = parser.parse_args()

    cfg = PointNetConfig()

    if args.mode == "train":
        run_train(cfg)

    elif args.mode == "eval":
        run_eval(cfg)

    elif args.mode == "infer":
        print("  Inference not yet available — train first.")


if __name__ == "__main__":
    main()