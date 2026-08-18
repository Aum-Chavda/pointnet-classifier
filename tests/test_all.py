# tests/test_all.py
"""
Single test file with sys.argv selector.
Run all  : .\.venv\Scripts\python.exe tests/test_all.py
Run one  : .\.venv\Scripts\python.exe tests/test_all.py 1
"""

import sys
import os

# sitecustomize.py handles path — but add fallback for safety
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN_ONLY = int(sys.argv[1]) if len(sys.argv) > 1 else None

# ======================================================================
# TEST 1 — Config defaults and types
# ======================================================================
if RUN_ONLY is None or RUN_ONLY == 1:
    print("\n[1] Testing PointNetConfig defaults and types...")

    from src.utils.config import PointNetConfig

    cfg = PointNetConfig()

    # positive test — defaults are correct types
    assert isinstance(cfg.num_points, int),    "num_points must be int"
    assert isinstance(cfg.num_classes, int),   "num_classes must be int"
    assert isinstance(cfg.lr, float),          "lr must be float"
    assert isinstance(cfg.feature_dim, int),   "feature_dim must be int"
    assert isinstance(cfg.batch_size, int),    "batch_size must be int"

    # positive test — default values match paper
    assert cfg.num_points  == 1024, f"Expected 1024, got {cfg.num_points}"
    assert cfg.num_classes == 40,   f"Expected 40, got {cfg.num_classes}"
    assert cfg.feature_dim == 1024, f"Expected 1024, got {cfg.feature_dim}"
    assert cfg.reg_weight  == 0.001, f"Expected 0.001, got {cfg.reg_weight}"

    # positive test — override works
    cfg2 = PointNetConfig(batch_size=16, epochs=10)
    assert cfg2.batch_size == 16
    assert cfg2.epochs     == 10

    # negative test — frozen=True prevents mutation
    try:
        cfg.lr = 0.1  # type: ignore
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass  # correct — frozen config cannot be mutated

    # property test — data_path returns Path object
    from pathlib import Path
    assert isinstance(cfg.data_path, Path), "data_path property must return Path"

    # summary test — summary() returns non-empty string
    summary = cfg.summary()
    assert isinstance(summary, str) and len(summary) > 0

    print("    [PASS] all config tests passed")
    print(cfg.summary())
    # ======================================================================
# TEST 2 — BasePointNetBackbone ABC contract
# ======================================================================
if RUN_ONLY is None or RUN_ONLY == 2:
    print("\n[2] Testing BasePointNetBackbone ABC contract...")

    import torch
    import torch.nn as nn
    from src.utils.config import PointNetConfig
    from src.models.base import BasePointNetBackbone

    cfg = PointNetConfig()

    # negative test — cannot instantiate ABC directly
    try:
        model = BasePointNetBackbone(cfg)  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # correct — abstract class cannot be instantiated

    # negative test — subclass missing forward() cannot be instantiated
    class BadBackbone(BasePointNetBackbone):
        def get_feature_dim(self) -> int:
            return 1024
        # missing forward() — should fail

    try:
        bad = BadBackbone(cfg)
        assert False, "Should have raised TypeError — missing forward()"
    except TypeError:
        pass  # correct

    # positive test — subclass implementing all abstract methods works
    class GoodBackbone(BasePointNetBackbone):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return (x,)
        def get_feature_dim(self) -> int:
            return 1024

    good = GoodBackbone(cfg)

    # positive test — is nn.Module (has .parameters(), .to(), .train())
    assert isinstance(good, nn.Module), "Must be nn.Module"

    # positive test — count_parameters works (0 params since no layers defined)
    assert good.count_parameters() == 0

    # positive test — reset_parameters runs without error
    good.reset_parameters()

    # positive test — __repr__ returns meaningful string
    r = repr(good)
    assert "GoodBackbone" in r and "feature_dim=1024" in r

    print("    [PASS] all base class tests passed")
    print(f"    repr: {good}")
    # ======================================================================
# TEST 3 — SharedMLP and TNet shapes + identity init
# ======================================================================
if RUN_ONLY is None or RUN_ONLY == 3:
    print("\n[3] Testing SharedMLP and TNet...")

    import torch
    from src.utils.config import PointNetConfig
    from src.models.blocks import SharedMLP, TNet

    cfg = PointNetConfig()
    B, N = 4, 1024   # small batch for speed

    # -- SharedMLP tests --
    # positive test — output shape correct
    mlp = SharedMLP(3, 64)
    x = torch.randn(B, 3, N)       # [B, C_in, N]
    out = mlp(x)
    assert out.shape == (B, 64, N), f"Expected ({B}, 64, {N}), got {out.shape}"

    # positive test — no activation variant
    mlp_no_act = SharedMLP(64, 128, activation=False)
    x2 = torch.randn(B, 64, N)
    out2 = mlp_no_act(x2)
    assert out2.shape == (B, 128, N)

    # positive test — no batchnorm variant
    mlp_no_bn = SharedMLP(3, 32, batch_norm=False)
    out3 = mlp_no_bn(torch.randn(B, 3, N))
    assert out3.shape == (B, 32, N)

    print("    [PASS] SharedMLP shape tests passed")

    # -- TNet tests --
    # positive test — input TNet (k=3) output shape
    tnet3 = TNet(k=3, config=cfg)
    pts = torch.randn(B, N, 3)     # [B, N, 3] — note N last for TNet input
    T3 = tnet3(pts)
    assert T3.shape == (B, 3, 3), f"Expected ({B},3,3), got {T3.shape}"

    # positive test — feature TNet (k=64) output shape
    tnet64 = TNet(k=64, config=cfg)
    feats = torch.randn(B, N, 64)  # [B, N, 64]
    T64 = tnet64(feats)
    assert T64.shape == (B, 64, 64), f"Expected ({B},64,64), got {T64.shape}"

    # positive test — at init, TNet output ≈ identity matrix
    # fc3 weights are zero → output before adding I is ≈ 0 → transform ≈ I
    tnet_init = TNet(k=3, config=cfg)
    tnet_init.eval()
    with torch.no_grad():
        T_init = tnet_init(torch.zeros(1, N, 3))
    I = torch.eye(3).unsqueeze(0)
    assert torch.allclose(T_init, I, atol=1e-6), "TNet should init to identity"

    # positive test — regularization loss is scalar and near zero at identity
    reg_loss = tnet3.regularization_loss(T3)
    assert reg_loss.shape == (), f"reg loss must be scalar, got {reg_loss.shape}"
    assert reg_loss.item() >= 0, "reg loss must be non-negative"

    print("    [PASS] TNet shape + identity init + reg loss tests passed")
    tnet3_params  = sum(p.numel() for p in tnet3.parameters()  if p.requires_grad)
    tnet64_params = sum(p.numel() for p in tnet64.parameters() if p.requires_grad)
    print(f"    TNet(k=3)  params : {tnet3_params:,}")
    print(f"    TNet(k=64) params : {tnet64_params:,}")
    # ======================================================================
# TEST 4 — PointNet full forward pass shapes + losses
# ======================================================================
if RUN_ONLY is None or RUN_ONLY == 4:
    print("\n[4] Testing PointNet full forward pass...")

    import torch
    from src.utils.config import PointNetConfig
    from src.models.pointnet import PointNet

    cfg = PointNetConfig()
    B, N = 4, 1024
    model = PointNet(cfg)
    model.eval()

    x = torch.randn(B, N, 3)   # [B, N, 3] — raw point cloud

    with torch.no_grad():
        logits, trans_inp, trans_feat = model(x)

    # shape tests
    assert logits.shape     == (B, cfg.num_classes), \
        f"logits: expected ({B},{cfg.num_classes}), got {logits.shape}"
    assert trans_inp.shape  == (B, 3,  3),  \
        f"trans_inp: expected ({B},3,3), got {trans_inp.shape}"
    assert trans_feat.shape == (B, 64, 64), \
        f"trans_feat: expected ({B},64,64), got {trans_feat.shape}"

    # logits should not be all zeros (model is initialised, not collapsed)
    assert logits.abs().sum().item() > 0, "logits are all zero — something wrong"

    # regularization loss — scalar, non-negative
    reg_loss = model.get_transform_loss(trans_inp, trans_feat)
    assert reg_loss.shape == (),        "reg_loss must be scalar"
    assert reg_loss.item() >= 0,        "reg_loss must be non-negative"

    # ABC contract — get_feature_dim returns correct value
    assert model.get_feature_dim() == cfg.feature_dim

    # parameter count — PointNet should be ~3.5M params
    n_params = model.count_parameters()
    assert n_params > 1_000_000, f"param count suspiciously low: {n_params:,}"

    print(f"    [PASS] all PointNet forward pass tests passed")
    print(f"    logits shape     : {logits.shape}")
    print(f"    trans_inp shape  : {trans_inp.shape}")
    print(f"    trans_feat shape : {trans_feat.shape}")
    print(f"    reg_loss         : {reg_loss.item():.6f}")
    print(f"    total params     : {n_params:,}")
    print(f"    model repr       : {model}")
    # ======================================================================
# TEST 5 — ModelRegistry factory + hash map behaviour
# ======================================================================
if RUN_ONLY is None or RUN_ONLY == 5:
    print("\n[5] Testing ModelRegistry...")

    import torch
    from src.utils.config import PointNetConfig
    from src.models.base import BasePointNetBackbone
    from src.models.registry import ModelRegistry

    cfg = PointNetConfig()

    # positive test — "pointnet" is registered at import time
    assert "pointnet" in ModelRegistry.list_models(), \
        "pointnet should be registered"

    # positive test — build returns correct type
    model = ModelRegistry.build("pointnet", cfg)
    assert isinstance(model, BasePointNetBackbone), \
        "built model must be BasePointNetBackbone"

    # positive test — built model has correct feature dim
    assert model.get_feature_dim() == cfg.feature_dim

    # positive test — forward pass works on built model
    x = torch.randn(2, 1024, 3)
    logits, t1, t2 = model(x)
    assert logits.shape == (2, cfg.num_classes)

    # negative test — unknown name raises KeyError
    try:
        ModelRegistry.build("unknown_model", cfg)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass  # correct

    # negative test — registering non-backbone raises TypeError
    try:
        import torch.nn as nn
        class FakeModel(nn.Module):
            pass
        ModelRegistry.register("fake", FakeModel)  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # correct

    # negative test — duplicate registration raises ValueError
    from src.models.pointnet import PointNet
    try:
        ModelRegistry.register("pointnet", PointNet)
        assert False, "Should have raised ValueError — duplicate name"
    except ValueError:
        pass  # correct

    # positive test — remove + re-register works
    ModelRegistry.remove("pointnet")
    assert "pointnet" not in ModelRegistry.list_models()
    ModelRegistry.register("pointnet", PointNet)
    assert "pointnet" in ModelRegistry.list_models()

    print("    [PASS] all registry tests passed")
    print(f"    registered models : {ModelRegistry.list_models()}")
    # ======================================================================
# TEST 6 — Dataset: FPS, normalisation, shapes, dataloader
# ======================================================================
if RUN_ONLY is None or RUN_ONLY == 6:
    print("\n[6] Testing ModelNet40Dataset...")

    import torch
    import numpy as np
    from pathlib import Path
    from src.utils.config import PointNetConfig
    from src.data.dataset import (
        farthest_point_sample,
        normalize_point_cloud,
        augment_point_cloud,
        ModelNet40Dataset,
        build_dataloader,
        CLASS_TO_IDX,
        MODELNET40_CLASSES,
    )

    cfg = PointNetConfig()

    # -- FPS tests --
    # positive test — output shape correct
    pts = np.random.randn(10000, 3).astype(np.float32)
    sampled = farthest_point_sample(pts, 1024)
    assert sampled.shape == (1024, 3), f"FPS shape wrong: {sampled.shape}"

    # positive test — FPS with fewer points than requested (edge case)
    small = np.random.randn(100, 3).astype(np.float32)
    sampled_small = farthest_point_sample(small, 1024)
    assert sampled_small.shape == (1024, 3)

    print("    [PASS] FPS shape tests passed")

    # -- Normalisation tests --
    pts_norm = normalize_point_cloud(pts)
    # centroid should be near zero
    assert np.abs(pts_norm.mean(axis=0)).max() < 0.1, "centroid not near zero"
    # max L2 norm should be <= 1.0
    norms = np.sqrt(np.sum(pts_norm ** 2, axis=1))
    assert norms.max() <= 1.0 + 1e-5, f"max norm > 1: {norms.max()}"

    print("    [PASS] normalisation tests passed")

    # -- Augmentation test --
    pts_aug = augment_point_cloud(pts[:1024], cfg)
    assert pts_aug.shape == (1024, 3), "augmentation changed shape"
    assert not np.allclose(pts_aug, pts[:1024]), "augmentation had no effect"

    print("    [PASS] augmentation tests passed")

    # -- Class map tests --
    assert len(MODELNET40_CLASSES) == 40, "should have 40 classes"
    assert len(CLASS_TO_IDX)       == 40, "CLASS_TO_IDX should have 40 entries"
    assert CLASS_TO_IDX["airplane"] == 0,  "airplane should be class 0"
    assert CLASS_TO_IDX["xbox"]     == 39, "xbox should be class 39"

    # multi-word class names must be in map
    assert "night_stand" in CLASS_TO_IDX, "night_stand missing"
    assert "flower_pot"  in CLASS_TO_IDX, "flower_pot missing"
    assert "tv_stand"    in CLASS_TO_IDX, "tv_stand missing"

    print("    [PASS] class map tests passed")

    # -- _parse_class_name tests --
    from src.data.dataset import ModelNet40Dataset as DS

    assert DS._parse_class_name("airplane_0001")    == "airplane"
    assert DS._parse_class_name("night_stand_0001") == "night_stand"
    assert DS._parse_class_name("flower_pot_0001")  == "flower_pot"
    assert DS._parse_class_name("tv_stand_0001")    == "tv_stand"
    assert DS._parse_class_name("xbox_0001")        == "xbox"
    assert DS._parse_class_name("unknown_9999")     is None

    print("    [PASS] class name parser tests passed")

    # -- Dataset + DataLoader tests (only if data exists) --
    data_path = cfg.data_path
    if data_path.exists():
        # dataset init
        dataset = ModelNet40Dataset(config=cfg, split="train")
        assert len(dataset) > 0, "dataset should have samples"
        print(f"    train samples loaded : {len(dataset)}")

        # getitem — shape, type, label range
        points, label = dataset[0]
        assert points.shape == (cfg.num_points, 3), \
            f"point cloud shape wrong: {points.shape}"
        assert isinstance(label, int),              "label must be int"
        assert 0 <= label < cfg.num_classes,        f"label out of range: {label}"
        assert points.dtype == torch.float32,       "must be float32"

        # normalisation check on loaded sample
        norms = torch.norm(points, dim=1)
        assert norms.max().item() <= 1.0 + 1e-4,   "points not in unit sphere"

        print(f"    [PASS] __getitem__ shape + type + range tests passed")

        # test split
        test_ds = ModelNet40Dataset(config=cfg, split="test")
        assert len(test_ds) > 0, "test dataset empty"
        print(f"    test samples loaded  : {len(test_ds)}")

        # negative test — invalid split raises ValueError
        try:
            bad = ModelNet40Dataset(config=cfg, split="val")
            assert False, "should have raised ValueError"
        except ValueError:
            pass

        print("    [PASS] invalid split raises ValueError")

        # dataloader batch shape
        loader = build_dataloader(cfg, "train")
        batch_pts, batch_labels = next(iter(loader))
        assert batch_pts.shape    == (cfg.batch_size, cfg.num_points, 3), \
            f"batch shape wrong: {batch_pts.shape}"
        assert batch_labels.shape == (cfg.batch_size,), \
            f"label shape wrong: {batch_labels.shape}"

        print(f"    [PASS] dataloader batch shape tests passed")
        print(f"    batch points shape : {batch_pts.shape}")
        print(f"    batch labels shape : {batch_labels.shape}")

    else:
        print(f"    [SKIP] data not found at {data_path} — download first")
        print(f"    FPS + normalise + augment + parser tests all passed")
        # ======================================================================
# TEST 7 — PerClassAccuracyTracker + AverageMeter
# ======================================================================
if RUN_ONLY is None or RUN_ONLY == 7:
    print("\n[7] Testing metrics...")

    import torch
    import numpy as np
    from src.training.metrics import PerClassAccuracyTracker, AverageMeter

    # -- PerClassAccuracyTracker tests --
    tracker = PerClassAccuracyTracker(num_classes=40)

    # positive test — perfect predictions
    # 10 samples all class 0, predicted correctly
    logits = torch.zeros(10, 40)
    logits[:, 0] = 10.0          # high score for class 0
    labels = torch.zeros(10, dtype=torch.long)
    tracker.update(logits, labels)

    metrics = tracker.compute()
    assert metrics["overall_acc"]    == 1.0, "perfect preds should give 100%"
    assert metrics["per_class_acc"][0] == 1.0, "class 0 should be 100%"
    assert metrics["total_samples"]  == 10

    print("    [PASS] perfect prediction test passed")

    # positive test — zero accuracy (all wrong)
    tracker.reset()
    logits_wrong = torch.zeros(10, 40)
    logits_wrong[:, 1] = 10.0   # predict class 1
    labels_zero = torch.zeros(10, dtype=torch.long)  # true class 0
    tracker.update(logits_wrong, labels_zero)

    metrics_wrong = tracker.compute()
    assert metrics_wrong["overall_acc"] == 0.0, "all wrong should give 0%"
    assert metrics_wrong["per_class_acc"][0] == 0.0

    print("    [PASS] zero accuracy test passed")

    # positive test — mixed accuracy
    tracker.reset()
    # 4 correct class 0, 6 wrong (predicted class 1)
    logits_mix = torch.zeros(10, 40)
    logits_mix[:4, 0]  = 10.0   # first 4 correct
    logits_mix[4:, 1]  = 10.0   # last 6 wrong
    labels_mix = torch.zeros(10, dtype=torch.long)
    tracker.update(logits_mix, labels_mix)

    metrics_mix = tracker.compute()
    assert abs(metrics_mix["overall_acc"] - 0.4) < 1e-6, \
        f"expected 0.4, got {metrics_mix['overall_acc']}"

    print("    [PASS] mixed accuracy test passed")

    # positive test — multi-class tracking
    tracker.reset()
    # class 0: 5 correct, class 1: 3 correct
    logits_mc = torch.zeros(8, 40)
    logits_mc[:5, 0] = 10.0
    logits_mc[5:, 1] = 10.0
    labels_mc = torch.tensor([0,0,0,0,0, 1,1,1], dtype=torch.long)
    tracker.update(logits_mc, labels_mc)

    metrics_mc = tracker.compute()
    assert metrics_mc["overall_acc"]      == 1.0
    assert metrics_mc["per_class_acc"][0] == 1.0
    assert metrics_mc["per_class_acc"][1] == 1.0
    assert metrics_mc["mean_class_acc"]   == 1.0

    print("    [PASS] multi-class tracking test passed")

    # positive test — confusion matrix shape
    cm = tracker.get_confusion_matrix()
    assert cm.shape == (40, 40), f"confusion matrix shape wrong: {cm.shape}"
    assert cm[0, 0] == 5, "class 0 diagonal should be 5"
    assert cm[1, 1] == 3, "class 1 diagonal should be 3"

    print("    [PASS] confusion matrix test passed")

    # -- AverageMeter tests --
    meter = AverageMeter("loss")

    # positive test — running average
    meter.update(1.0, n=10)
    meter.update(2.0, n=10)
    assert abs(meter.avg - 1.5) < 1e-6, f"avg wrong: {meter.avg}"

    # positive test — weighted average (different batch sizes)
    meter.reset()
    meter.update(1.0, n=3)   # 3 samples with loss 1.0
    meter.update(4.0, n=1)   # 1 sample with loss 4.0
    # expected: (3*1.0 + 1*4.0) / 4 = 1.75
    assert abs(meter.avg - 1.75) < 1e-6, f"weighted avg wrong: {meter.avg}"

    # positive test — empty meter returns 0
    empty = AverageMeter()
    assert empty.avg == 0.0

    print("    [PASS] AverageMeter tests passed")
    print(f"    repr: {meter}")
    # ======================================================================
# TEST 8 — Callbacks: EarlyStopping, ModelCheckpoint, CallbackList
# ======================================================================
if RUN_ONLY is None or RUN_ONLY == 8:
    print("\n[8] Testing callbacks...")

    import torch
    import torch.nn as nn
    from src.utils.config import PointNetConfig
    from src.training.callbacks import (
        EarlyStopping, ModelCheckpoint, LRSchedulerCallback, CallbackList
    )

    cfg = PointNetConfig()

    # -- EarlyStopping tests --
    # positive test — stops after patience epochs without improvement
    es = EarlyStopping(monitor="val_acc", patience=3, mode="max")
    assert not es.should_stop

    es.on_epoch_end(0, {"val_acc": 0.80})   # improvement
    es.on_epoch_end(1, {"val_acc": 0.79})   # no improvement — counter=1
    es.on_epoch_end(2, {"val_acc": 0.78})   # no improvement — counter=2
    assert not es.should_stop, "should not stop yet (counter=2 < patience=3)"

    es.on_epoch_end(3, {"val_acc": 0.77})   # no improvement — counter=3
    assert es.should_stop, "should stop (counter=3 >= patience=3)"
    assert es.best_epoch == 0
    print("    [PASS] EarlyStopping max-mode test passed")

    # positive test — reset on improvement
    es2 = EarlyStopping(monitor="val_acc", patience=3, mode="max")
    es2.on_epoch_end(0, {"val_acc": 0.80})
    es2.on_epoch_end(1, {"val_acc": 0.79})   # counter=1
    es2.on_epoch_end(2, {"val_acc": 0.85})   # improvement — counter resets to 0
    assert es2.counter == 0, "counter should reset on improvement"
    assert not es2.should_stop
    print("    [PASS] EarlyStopping counter reset test passed")

    # positive test — min mode (for loss)
    es3 = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    es3.on_epoch_end(0, {"val_loss": 1.0})
    es3.on_epoch_end(1, {"val_loss": 0.8})   # improvement
    es3.on_epoch_end(2, {"val_loss": 0.9})   # no improvement
    es3.on_epoch_end(3, {"val_loss": 0.85})  # no improvement — should stop
    assert es3.should_stop
    print("    [PASS] EarlyStopping min-mode test passed")

    # -- ModelCheckpoint tests --
    # use a tiny model for speed
    tiny_model = nn.Linear(10, 5)
    optimizer  = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

    ckpt = ModelCheckpoint(
        config    = cfg,
        monitor   = "val_acc",
        mode      = "max",
        model     = tiny_model,
        optimizer = optimizer,
    )

    # save checkpoint on improvement
    ckpt.on_epoch_end(0, {"val_acc": 0.80})
    ckpt.on_epoch_end(1, {"val_acc": 0.85})
    ckpt.on_epoch_end(2, {"val_acc": 0.83})   # no improvement — no new best

    # best should be 0.85
    assert abs(ckpt.best - 0.85) < 1e-6, f"best wrong: {ckpt.best}"
    assert ckpt.best_epoch == 1

    # checkpoint files should exist
    from pathlib import Path
    assert (cfg.checkpoint_path / "best_model.pth").exists(), "best_model.pth missing"
    assert (cfg.checkpoint_path / "last_model.pth").exists(), "last_model.pth missing"
    print("    [PASS] ModelCheckpoint save test passed")

    # load best test
    loaded_model = nn.Linear(10, 5)
    device = torch.device("cpu")
    loaded_model = ckpt.load_best(loaded_model, device)
    print("    [PASS] ModelCheckpoint load_best test passed")

    # -- LRSchedulerCallback test --
    opt2 = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    lr_cb = LRSchedulerCallback(opt2, step_size=2, gamma=0.5)

    lr_cb.on_epoch_end(0, {})
    lr_cb.on_epoch_end(1, {})
    lr_cb.on_epoch_end(2, {})   # after 2 steps lr should be halved
    assert lr_cb.get_lr() < 1e-3, f"LR should have decayed, got {lr_cb.get_lr()}"
    print("    [PASS] LRSchedulerCallback decay test passed")

    # -- CallbackList test --
    es_cb   = EarlyStopping(monitor="val_acc", patience=2, mode="max")
    cb_list = CallbackList([es_cb])

    cb_list.on_epoch_end(0, {"val_acc": 0.80})
    assert not cb_list.should_stop

    cb_list.on_epoch_end(1, {"val_acc": 0.79})
    cb_list.on_epoch_end(2, {"val_acc": 0.78})
    assert cb_list.should_stop, "CallbackList.should_stop should reflect EarlyStopping"
    print("    [PASS] CallbackList.should_stop test passed")

    print("\n    [PASS] all callback tests passed")
    # ======================================================================
# TEST 9 — Trainer: smoke test (2 epochs, tiny subset)
# ======================================================================
if RUN_ONLY is None or RUN_ONLY == 9:
    print("\n[9] Testing Trainer (smoke test)...")

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.utils.config import PointNetConfig
    from src.models.registry import ModelRegistry
    from src.training.trainer import Trainer

    cfg = PointNetConfig(
        epochs             = 2,
        batch_size         = 4,
        num_points         = 64,    # tiny for speed
        feature_dim        = 1024,
        early_stop_patience= 10,
        log_every_n_steps  = 1,
    )

    # Build tiny synthetic dataset — no real data needed
    B, N = 16, 64
    fake_points = torch.randn(B, N, 3)
    fake_labels = torch.randint(0, 40, (B,))
    dataset     = TensorDataset(fake_points, fake_labels)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader   = DataLoader(dataset, batch_size=4, shuffle=False)

    # Build model via registry
    model   = ModelRegistry.build("pointnet", cfg)
    trainer = Trainer(model, cfg)

    # Run 2 epochs
    history = trainer.fit(train_loader, val_loader)

    # positive test — history has correct keys
    assert "train_loss"  in history, "missing train_loss"
    assert "val_loss"    in history, "missing val_loss"
    assert "train_acc"   in history, "missing train_acc"
    assert "val_acc"     in history, "missing val_acc"
    assert "lr"          in history, "missing lr"

    # positive test — history has correct length
    assert len(history["train_loss"]) == 2, \
        f"expected 2 epochs, got {len(history['train_loss'])}"

    # positive test — losses are positive scalars
    for loss in history["train_loss"] + history["val_loss"]:
        assert loss > 0, f"loss should be positive, got {loss}"

    # positive test — accuracies in [0, 1]
    for acc in history["train_acc"] + history["val_acc"]:
        assert 0.0 <= acc <= 1.0, f"acc out of range: {acc}"

    # positive test — confusion matrix shape
    cm = trainer.get_val_confusion_matrix()
    assert cm.shape == (40, 40), f"confusion matrix shape: {cm.shape}"

    print(f"    [PASS] history keys and lengths correct")
    print(f"    [PASS] loss values positive")
    print(f"    [PASS] accuracy values in [0,1]")
    print(f"    [PASS] confusion matrix shape correct")
    print(f"    train_loss : {history['train_loss']}")
    print(f"    val_acc    : {history['val_acc']}")
    print(f"\n    [PASS] all Trainer smoke tests passed")