# PointNet 3D Object Classifier

From-scratch PyTorch implementation of PointNet (Qi et al., CVPR 2017) for
3D point cloud classification on ModelNet40.

![Architecture](docs/architecture.png)

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | ModelNet40 (40 classes) |
| Points per cloud | 1024 (Farthest Point Sampling) |
| Model params | ~3.5M |
| Val accuracy | TBD (training in progress) |
| GPU | NVIDIA GTX 1650 Ti, 4GB VRAM |
| Training time | TBD |

---

## Architecture

PointNet processes raw 3D point clouds — unordered sets of XYZ coordinates —
directly, with no voxelization or convolution grids.

Input [B, N, 3]
│
├─ Input T-Net (3×3 spatial transform)
├─ SharedMLP: 3 → 64
├─ Feature T-Net (64×64 feature transform)
├─ SharedMLP: 64 → 128 → 1024
├─ Global Max Pool → [B, 1024] ← permutation invariant
└─ Classifier MLP → [B, 40]


**Key insight:** Global max pooling makes the model permutation-invariant —
the same point cloud in any ordering produces identical output.

**T-Net:** A mini PointNet that predicts a transformation matrix to spatially
align the input before processing. Regularised with orthogonality loss:
`L_reg = ||I - AA^T||_F²`

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
│ │ └── dataset.py # ModelNet40 dataset + FPS sampling [Phase 2]
│ ├── training/
│ │ ├── trainer.py # Training loop + per-class accuracy [Phase 3]
│ │ ├── callbacks.py # EarlyStopping, Checkpoint [Phase 3]
│ │ └── metrics.py # PerClassAccuracyTracker [Phase 3]
│ └── utils/
│ ├── config.py # PointNetConfig dataclass
│ └── visualize.py # Confusion matrix + t-SNE [Phase 4]
├── tests/
│ └── test_all.py # All tests, sys.argv selector
├── configs/
│ └── default.yaml
├── main.py
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

Download ModelNet40 from the official source and place it at `data/modelnet40/`:

data/modelnet40/
├── airplane/
│ ├── train/
│ │ ├── airplane_0001.off
│ │ └── ...
│ └── test/
├── chair/
└── ...


---

## Training

```powershell
.\.venv\Scripts\python.exe main.py
```

---

## Key Concepts Implemented

| Concept | Implementation |
|---------|---------------|
| Permutation invariance | Global max pooling across N points |
| Spatial alignment | T-Net: learned 3×3 + 64×64 transforms |
| Shared weights | Conv1d(kernel=1) applied to all N points |
| Orthogonality loss | `||I - AA^T||_F²` on both T-Nets |
| FPS sampling | Farthest Point Sampling from mesh faces |

---

## OOP & DS&A Patterns

**OOP:** dataclass (frozen config), ABC (backbone contract),
composition (PointNet HAS-A TNet), factory + registry (ModelRegistry),
template method (fixed forward pipeline)

**DS&A:** hash map (registry O(1) lookup), pipeline (nn.Sequential),
batched matrix multiply (torch.bmm), global max pool (set operation),
typed record (config dataclass)

---

## Portfolio Context

Part of a 5-project ML portfolio targeting robotics and autonomous systems roles.

| # | Project | Status |
|---|---------|--------|
| 1 | BEV CNN Feature Extractor (ResNet, CIFAR-10, 82.4% acc) | ✅ |
| 2 | Mini I-JEPA World Model (ViT-Tiny, STL-10, loss=0.2872) | ✅ |
| 3 | PointNet 3D Object Classifier (ModelNet40) | 🔄 In Progress |
| 4 | Mini VLA: Visual Encoder + Action Head | ⬜ |
| 5 | Depth-Aware Grasp Pose Estimator | ⬜ |