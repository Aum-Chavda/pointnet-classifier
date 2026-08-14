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