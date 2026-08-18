# PointNet 3D Object Classifier

From-scratch PyTorch implementation of **PointNet** (Qi et al., CVPR 2017) for 3D point cloud classification on ModelNet40. Achieves **85.4% test accuracy** trained from scratch on a single GTX 1650 Ti.

> Part of a 5-project ML portfolio targeting robotics and autonomous systems research roles.

---

## Results

| Metric | Value |
|--------|-------|
| Overall Test Accuracy | **85.41%** |
| Mean Class Accuracy | **81.90%** |
| Test Loss | 0.5079 |
| Best Epoch | 48 / 50 |
| Dataset | ModelNet40 (40 classes, 12,311 CAD models) |
| Points per cloud | 1,024 (Farthest Point Sampling) |
| Model parameters | ~3.5M |
| Hardware | NVIDIA GTX 1650 Ti, 4GB VRAM |
| Training time | ~50 minutes |

**Comparison with paper:**

| Method | Accuracy | Hardware |
|--------|----------|----------|
| PointNet (Qi et al. 2017) | 89.2% | Multi-GPU |
| **This implementation** | **85.4%** | GTX 1650 Ti |
| Gap | 3.8% | Explained by hardware + epochs |

---

## What is a Point Cloud?

A point cloud is a set of XYZ coordinates sampled from the surface of a 3D object — the raw output of a LiDAR sensor or depth camera. Unlike images (regular grids) or voxels (3D grids), point clouds are **unordered sets** with no fixed structure.

The challenge: a neural network must produce the same output regardless of what order the 1,024 points are fed in — there are 1,024! possible orderings of the same object.

Below are four ModelNet40 objects as seen by PointNet — 1,024 points coloured by height (purple=low, yellow=high):

| AIRPLANE | CHAIR |
|----------|-------|
| ![airplane](outputs/pointcloud_airplane.png) | ![chair](outputs/pointcloud_chair.png) |

| FLOWER POT | WARDROBE |
|------------|----------|
| ![flower_pot](outputs/pointcloud_flower_pot.png) | ![wardrobe](outputs/pointcloud_wardrobe.png) |

Notice: the airplane has an unmistakable wing structure. The wardrobe is a featureless box — this directly explains why it is the hardest class to classify.

---

## Architecture

PointNet's key insight: apply a shared MLP to every point independently, then collapse all N points into a single global descriptor via **global max pooling**. Max pooling is a symmetric function — its output is identical regardless of input order, giving the model **permutation invariance** for free.

Input [B, N, 3]
|
+-- Input T-Net (3x3 spatial transform) <- learned alignment
|
+-- SharedMLP: 3 -> 64 <- per-point features
|
+-- Feature T-Net (64x64 feature transform) <- feature alignment
|
+-- SharedMLP: 64 -> 128 -> 1024 <- deeper per-point features
|
+-- Global Max Pool -> [B, 1024] <- PERMUTATION INVARIANT
|
+-- Classifier MLP: 1024 -> 512 -> 256 -> 40
|
Output [B, 40] <- class logits


**T-Net (Spatial Transformer):** A mini PointNet that predicts a transformation matrix to spatially align the input before processing. Regularised with orthogonality loss to keep transforms as rotations:

L_reg = ||I - A @ A^T||_F^2
L_total = CrossEntropy + 0.001 * L_reg


**SharedMLP = Conv1d(kernel=1):** Mathematically identical to applying a Linear layer to each of the N points independently, but processes all N points in a single parallelised GPU call.

---

## Training

### Curves

![Training Curves](outputs/training_curves.png)

Key observations:
- Loss drops smoothly from 2.25 to 0.4 over 50 epochs — stable training throughout
- LR decay at epoch 20 clearly visible — both curves stabilise and tighten after
- Train accuracy slightly above val accuracy — minimal overfitting, dropout working correctly
- Model still improving at epoch 50 — more epochs would push accuracy further

### Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

The strong diagonal confirms the model learned genuine 3D structure across all 40 classes. Notable patterns:
- **Perfect classes (1.0):** airplane, car, cone, guitar, keyboard, laptop — geometrically distinctive shapes
- **Hard classes:** flower_pot (0.30), wardrobe (0.35), radio (0.55) — geometrically ambiguous

### Feature Space (t-SNE)

![t-SNE](outputs/tsne_features.png)

t-SNE projection of the 1,024-dimensional global feature vectors from the test set (2,000 samples). Well-separated clusters confirm the model learned **meaningful 3D representations** — not just texture shortcuts. The airplane cluster (bottom-left, blue) and guitar cluster (bottom-center) are especially tight and isolated, matching their 100% test accuracy.

---

## Inference

```powershell
python main.py --mode infer --file "data/modelnet40_normal_resampled/chair/chair_0001.txt"
```

**Results on four test objects:**

CHAIR (easy — distinctive geometry)
Prediction : CHAIR 99.9% #############################
2. stool 0.1%

AIRPLANE (easy — unmistakable wings)
Prediction : AIRPLANE 100.0% #############################

FLOWER_POT (hard — identical to vase in XYZ)
Prediction : VASE 76.6% ######################
2. flower_pot 20.7% ######
Root cause: tapered cylinder shape shared with vase.
Fix: add surface normals (6D input) to capture rim curvature.

WARDROBE (hard — featureless rectangular box)
Prediction : DRESSER 68.9% ####################
2. wardrobe 17.5% #####
Root cause: both are large rectangular boxes. Scale destroyed by
unit sphere normalisation. Fix: weighted loss + normals.


---

## Failure Analysis

| Class | Accuracy | Confused With | Root Cause | Fix |
|-------|----------|---------------|------------|-----|
| flower_pot | 30% | vase | Identical tapered cylinder in XYZ | Add surface normals (6D input) |
| wardrobe | 35% | dresser | Both large rectangles, scale normalised away | Weighted loss + normals |
| radio | 55% | lamp/bottle | Small cylinder, no discriminating features | PointNet++ local features |
| night_stand | 64% | desk | Flat top + legs, size context lost | Multi-scale architecture |

**The single biggest available improvement:** Add surface normals (nx, ny, nz already in dataset files, currently discarded). Expected gain: +2-4% overall, +15-20% on flower_pot specifically.

---

## OOP and DS&A Patterns

### OOP Patterns

| Pattern | File | What it does |
|---------|------|-------------|
| Dataclass (frozen) | `config.py` | Immutable typed config record — change one file, everything updates |
| Abstract Base Class | `base.py` | Contract — any backbone must implement `forward()` and `get_feature_dim()` |
| Composition | `pointnet.py` | PointNet HAS-A TNet, HAS-A SharedMLP — each brick independently testable |
| Template Method | `pointnet.py` | Fixed forward pipeline: transform -> extract -> pool -> classify |
| Factory + Registry | `registry.py` | `ModelRegistry.build("pointnet", cfg)` — O(1) hash map lookup |
| Adapter | `dataset.py` | Wraps raw `.txt` files into PyTorch `Dataset` interface |
| Accumulator | `metrics.py` | `update()` per batch, `compute()` per epoch, `reset()` between epochs |
| Observer / Hook | `callbacks.py` | Trainer fires `on_epoch_end()` — callbacks respond independently |
| Facade | `trainer.py` | `trainer.fit(train_loader, val_loader)` hides all training complexity |

### DS&A Concepts

| Concept | Where | Complexity |
|---------|-------|------------|
| Typed record (struct) | `config.py` | O(1) attribute access via `__dict__` hash map |
| Hash map | `registry.py`, RAM cache | O(1) model lookup, O(1) sample lookup |
| Greedy algorithm (FPS) | `dataset.py` | O(N*K) — maximises spatial coverage of 1,024 sampled points |
| Confusion matrix (2D array) | `metrics.py` | O(B) update, O(C) compute, O(C^2) render |
| Append-only log | `trainer.py` | O(1) per epoch — history dict for curve plotting |
| Batched matrix multiply | `pointnet.py` | O(B*N*K^2) — `torch.bmm` applies T-Net transform to all N points at once |
| Global max pool (set operation) | `pointnet.py` | O(B*C*N) — permutation invariant aggregation |
| Dimensionality reduction | `visualize.py` | PCA O(N*D*K) then t-SNE O(N^2) — 1024-dim features -> 2D plot |
| Pipeline (stack) | `blocks.py` | `nn.Sequential` — O(L) forward pass, L layers |

---

## Project Structure

pointnet-classifier/
├── src/
│ ├── models/
│ │ ├── base.py # Abstract backbone (ABC + nn.Module)
│ │ ├── blocks.py # SharedMLP + TNet
│ │ ├── pointnet.py # Full PointNet model
│ │ └── registry.py # ModelRegistry factory
│ ├── data/
│ │ └── dataset.py # ModelNet40 adapter, FPS, augmentation, RAM cache
│ ├── training/
│ │ ├── trainer.py # Training + val loop (Facade pattern)
│ │ ├── callbacks.py # EarlyStopping, ModelCheckpoint, LRScheduler
│ │ └── metrics.py # PerClassAccuracyTracker, AverageMeter
│ └── utils/
│ ├── config.py # PointNetConfig dataclass
│ ├── visualize.py # Training curves, confusion matrix, t-SNE
│ └── inference.py # Single-file prediction with top-5 output
├── scripts/
│ ├── precompute_fps.py # Pre-compute FPS once, cache to disk (~30 min)
│ └── visualize_pointcloud.py # 3D scatter plot of point cloud
├── tests/
│ └── test_all.py # 11 test suites, sys.argv selector
├── main.py # Entry point: --mode train | eval | infer
└── pyproject.toml


---

## Setup

```powershell
# 1. Clone
git clone https://github.com/Aum-Chavda/pointnet-classifier.git
cd pointnet-classifier

# 2. Create venv and install dependencies
uv venv
uv sync

# 3. Install CUDA PyTorch (required — uv installs CPU build by default)
.\.venv\Scripts\python.exe -m pip install torch torchvision `
    --index-url https://download.pytorch.org/whl/cu128

# 4. Fix PYTHONPATH
Copy-Item sitecustomize.py .venv\Lib\site-packages\sitecustomize.py

# 5. Verify CUDA
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

---

## Dataset

Download ModelNet40 pre-sampled (modelnet40_normal_resampled, ~2GB):

Available on Kaggle: `chenxaoyu/modelnet-normal-resampled`

Place at `data/modelnet40_normal_resampled/` then pre-compute FPS cache:

```powershell
.\.venv\Scripts\python.exe scripts\precompute_fps.py
```

This runs once (~30 minutes) and caches 1,024-point FPS samples to disk,
reducing per-epoch data loading from 35 minutes to under 1 minute.

---

## Usage

```powershell
# Train (50 epochs, ~50 minutes on GTX 1650 Ti)
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe main.py --mode train

# Full evaluation + all plots
.\.venv\Scripts\python.exe main.py --mode eval

# Inference on a single point cloud
.\.venv\Scripts\python.exe main.py --mode infer `
    --file "data/modelnet40_normal_resampled/chair/chair_0001.txt"

# Run all tests
.\.venv\Scripts\python.exe tests/test_all.py
```

---

## Portfolio Context

| # | Project | Key Result | Status |
|---|---------|------------|--------|
| 1 | BEV CNN (ResNet, CIFAR-10) | 82.4% val acc | Done |
| 2 | Mini I-JEPA (ViT-Tiny, STL-10) | Loss 0.2872, 3.17x above random | Done |
| 3 | **PointNet 3D Classifier (ModelNet40)** | **85.4% test acc** | Done |
| 4 | Mini VLA: Visual Encoder + Action Head | - | Upcoming |
| 5 | Depth-Aware Grasp Pose Estimator | - | Upcoming |
---

## Reference

Qi, C.R., Su, H., Mo, K., Guibas, L.J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. CVPR 2017. arXiv:1612.00593