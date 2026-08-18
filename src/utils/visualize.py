# src/utils/visualize.py
"""
Visualisation utilities for PointNet training results.

OOP  : Builder pattern — Visualizer builds plots step by step, save() finalises
DS&A : Dimensionality reduction — PCA(1024->50) + t-SNE(50->2) for feature viz
       O(N*D*K) PCA, O(N^2) t-SNE where N=samples, D=1024, K=50
Role : Training curves, confusion matrix, t-SNE feature visualisation
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from src.data.dataset import MODELNET40_CLASSES


# ======================================================================
# Training Curves
# ======================================================================

def plot_training_curves(
    history:  dict[str, list[float]],
    save_dir: str | Path = "outputs",
) -> Path:
    """
    Plot loss and accuracy curves for train and validation.

    OOP  : Builder pattern — builds figure panel by panel
    DS&A : O(E) where E = number of epochs — simple line plot

    Layout:
        Left  : Train loss vs Val loss
        Right : Train acc vs Val acc (overall + mean class)

    Args:
        history  : dict from Trainer.fit() — lists of per-epoch metrics
        save_dir : directory to save the plot

    Returns:
        Path to saved PNG file
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PointNet Training History — ModelNet40", fontsize=14, fontweight="bold")

    # -- Left panel: Loss --
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss",
            color="#2196F3", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",
            color="#F44336", linewidth=2, linestyle="--")

    # Mark LR decay points
    if "lr" in history:
        lr_array = np.array(history["lr"])
        decay_epochs = np.where(np.diff(lr_array) < 0)[0] + 2  # +2 for 1-indexed
        for ep in decay_epochs:
            ax.axvline(x=ep, color="gray", linestyle=":", alpha=0.6, linewidth=1)
        if len(decay_epochs) > 0:
            ax.axvline(x=decay_epochs[0], color="gray", linestyle=":",
                       alpha=0.6, linewidth=1, label="LR decay")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))

    # -- Right panel: Accuracy --
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]],
            label="Train Acc (overall)", color="#2196F3", linewidth=2)
    ax.plot(epochs, [a * 100 for a in history["val_acc"]],
            label="Val Acc (overall)",   color="#F44336",
            linewidth=2, linestyle="--")

    if "train_mean_acc" in history:
        ax.plot(epochs, [a * 100 for a in history["train_mean_acc"]],
                label="Train Mean Class Acc", color="#4CAF50",
                linewidth=1.5, linestyle="-.")
    if "val_mean_acc" in history:
        ax.plot(epochs, [a * 100 for a in history["val_mean_acc"]],
                label="Val Mean Class Acc",   color="#FF9800",
                linewidth=1.5, linestyle=":")

    # Annotate best val acc
    best_val_acc = max(history["val_acc"])
    best_epoch   = history["val_acc"].index(best_val_acc) + 1
    ax.annotate(
        f"Best: {best_val_acc*100:.1f}%\n(epoch {best_epoch})",
        xy        = (best_epoch, best_val_acc * 100),
        xytext    = (best_epoch + max(1, len(epochs)//10), best_val_acc * 100 - 5),
        fontsize  = 9,
        color     = "#F44336",
        arrowprops= dict(arrowstyle="->", color="#F44336", lw=1.2),
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out = save_path / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Visualize] Saved training curves -> {out}")
    return out


# ======================================================================
# Confusion Matrix
# ======================================================================

def plot_confusion_matrix(
    confusion:   "np.ndarray",
    save_dir:    str | Path = "outputs",
    normalize:   bool = True,
) -> Path:
    """
    Plot 40×40 confusion matrix as a heatmap.

    OOP  : Builder — builds figure, adds heatmap, labels, colorbar
    DS&A : O(C²) render — C=40, so 1600 cells
           Normalisation: row-wise divide by class total -> values in [0,1]

    Args:
        confusion  : [40, 40] int64 confusion matrix from PerClassAccuracyTracker
        save_dir   : directory to save the plot
        normalize  : if True, show proportions; if False, show raw counts

    Returns:
        Path to saved PNG file
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    C = confusion.shape[0]

    if normalize:
        # Row-wise normalisation — each row sums to 1.0
        # DS&A : O(C²) element-wise divide
        row_sums = confusion.sum(axis=1, keepdims=True)
        # Avoid division by zero for unseen classes
        cm = np.where(row_sums > 0, confusion / row_sums.astype(float), 0.0)
        fmt_str  = ".2f"
        vmin, vmax = 0.0, 1.0
        cb_label = "Proportion"
    else:
        cm       = confusion.astype(float)
        fmt_str  = "d"
        vmin, vmax = 0, confusion.max()
        cb_label = "Count"

    # Short class labels for readability (first 6 chars)
    short_labels = [cls[:6] for cls in MODELNET40_CLASSES]

    fig, ax = plt.subplots(figsize=(18, 16))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues",
                   vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cb_label, fontsize=11)

    # Axis labels
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Class",      fontsize=12, labelpad=10)
    ax.set_title("PointNet Confusion Matrix — ModelNet40 (Test Set)",
                 fontsize=13, fontweight="bold", pad=15)

    # Annotate cells with values — only diagonal for readability
    # DS&A : O(C) diagonal annotation — only 40 cells, not 1600
    for i in range(C):
        val = cm[i, i]
        txt = f"{val:.2f}" if normalize else f"{int(val)}"
        # white text on dark cells, dark text on light cells
        color = "white" if val > 0.5 else "black"
        ax.text(i, i, txt, ha="center", va="center",
                fontsize=6, color=color, fontweight="bold")

    plt.tight_layout()
    out = save_path / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Visualize] Saved confusion matrix -> {out}")
    return out


# ======================================================================
# t-SNE Feature Visualisation
# ======================================================================

def plot_tsne(
    features:  "np.ndarray",
    labels:    "np.ndarray",
    save_dir:  str | Path = "outputs",
    max_samples: int = 2000,
) -> Path:
    """
    t-SNE 2D projection of global feature vectors coloured by class.

    DS&A : PCA first (O(N*D*K), D=1024, K=50) then t-SNE (O(N²))
           PCA reduces dimensions before t-SNE for speed + stability
           Subsample to max_samples if N > max_samples (t-SNE is O(N²))

    Args:
        features    : [N, 1024] global feature vectors (after max pool)
        labels      : [N] int class indices
        save_dir    : directory to save the plot
        max_samples : cap for t-SNE — prevents O(N²) explosion

    Returns:
        Path to saved PNG file
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    N = features.shape[0]

    # Subsample if too many points — t-SNE is O(N²)
    if N > max_samples:
        # DS&A : random index selection — O(N) sample without replacement
        idx      = np.random.choice(N, max_samples, replace=False)
        features = features[idx]
        labels   = labels[idx]
        N        = max_samples

    print(f"  [Visualize] Running t-SNE on {N} samples...")

    # Step 1: PCA 1024 -> 50 dims
    # DS&A : O(N * 1024 * 50) — much faster than raw t-SNE on 1024 dims
    pca      = PCA(n_components=min(50, features.shape[1]), random_state=42)
    features_pca = pca.fit_transform(features)   # [N, 50]

    # Step 2: t-SNE 50 -> 2 dims
    # DS&A : O(N²) — expensive, hence the subsample above
    # perplexity: balance between local and global structure
    # n_iter: more iterations = better layout but slower
    tsne     = TSNE(
        n_components = 2,
        perplexity   = min(30, N // 4),
        max_iter     = 1000,
        random_state = 42,
        verbose      = 0,
    )
    features_2d = tsne.fit_transform(features_pca)   # [N, 2]

    # -- Plot --
    fig, ax = plt.subplots(figsize=(14, 12))

    # 40 colours — use tab20 × 2 (matplotlib's 20-colour qualitative map)
    from matplotlib import colormaps
    cmap   = colormaps["tab20"]
    colors = [cmap(i % 20) for i in range(40)]

    # Plot each class separately so we can label them
    unique_labels = np.unique(labels)
    for cls_idx in unique_labels:
        mask = labels == cls_idx
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c      = [colors[cls_idx % 20]],
            label  = MODELNET40_CLASSES[cls_idx],
            s      = 8,
            alpha  = 0.7,
        )

    ax.set_title(
        f"t-SNE of PointNet Global Features — ModelNet40 ({N} samples)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.grid(True, alpha=0.2)

    # Legend — two columns to fit 40 classes
    ax.legend(
        fontsize    = 7,
        ncol        = 2,
        loc         = "upper right",
        markerscale = 2,
        framealpha  = 0.8,
    )

    plt.tight_layout()
    out = save_path / "tsne_features.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Visualize] Saved t-SNE plot -> {out}")
    return out


# ======================================================================
# Feature Extractor — extracts global features for t-SNE
# ======================================================================

def extract_features(
    model:      "nn.Module",
    dataloader: "DataLoader",
    device:     "torch.device",
    max_batches: int = 80,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Run model forward pass and collect global features + labels.

    DS&A : O(B * N * C) — same as val loop
           Stores [B, 1024] features per batch -> concat at end O(N_total)

    Args:
        model       : trained PointNet model
        dataloader  : test DataLoader
        device      : torch.device
        max_batches : cap to avoid OOM (80 × 32 = 2560 samples)

    Returns:
        features : [N, 1024] numpy array — global feature vectors
        labels   : [N] numpy array — class indices
    """
    import torch

    model.eval()
    all_features = []
    all_labels   = []

    with torch.no_grad():
        for batch_idx, (points, labels) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            points = points.to(device)   # [B, N, 3]

            # We need the global feature vector [B, 1024] — not logits
            # Hook into the model: run forward but capture after max pool
            # Trick: run full forward, then re-extract features via hook

            # Register a forward hook on the classifier's first layer
            # The input to classifier[0] IS the global feature [B, 1024]
            features_batch = []

            def hook_fn(module, input, output):
                # input[0] is the tensor fed into classifier[0] = [B, 1024]
                features_batch.append(input[0].cpu().numpy())

            # Register hook on first layer of classifier
            handle = model.classifier[0].register_forward_hook(hook_fn)

            # Forward pass — hook fires and captures features
            model(points)

            handle.remove()   # always remove hooks after use

            if features_batch:
                all_features.append(features_batch[0])    # [B, 1024]
            all_labels.append(labels.numpy())              # [B]

    features = np.concatenate(all_features, axis=0)   # [N, 1024]
    labels   = np.concatenate(all_labels,   axis=0)   # [N]

    return features, labels
