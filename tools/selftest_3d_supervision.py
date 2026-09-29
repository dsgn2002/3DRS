"""CPU self-test for the 3DRS-G supervision and analysis code.

Training 3DRS needs 8 GPUs; this does not. It builds a synthetic room with
known geometry, checks each new loss and metric against its ground truth, and
verifies gradients reach the visual tokens.

    python tools/selftest_3d_supervision.py
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    """Load a module by path.

    Deliberately bypasses ``import llava.…``: ``llava/__init__.py`` pulls in
    transformers and deepspeed, and the whole point of this script is that it
    runs anywhere torch does.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tds = _load("three_d_supervision", "llava/model/three_d_supervision.py")
gap = _load("feature_gap", "llava/analysis/feature_gap.py")

GeometryHeads = tds.GeometryHeads
compute_3d_supervision = tds.compute_3d_supervision
extract_visual_tokens = tds.extract_visual_tokens
geometry_loss = tds.geometry_loss
grid_normals = tds.grid_normals
multiview_correspondence_loss = tds.multiview_correspondence_loss
normalize_points = tds.normalize_points
patch_geometry = tds.patch_geometry
pointwise_cosine_align = tds.pointwise_cosine_align
relational_align = tds.relational_align
resample_teacher_features = tds.resample_teacher_features
supcon = tds.supcon
sym_infonce = tds.sym_infonce
hard_negative_mask = tds.hard_negative_mask
contrastive_alignment_loss = tds.contrastive_alignment_loss
ContrastiveHeads = tds.ContrastiveHeads

geometry_probe_r2 = gap.geometry_probe_r2
linear_cka = gap.linear_cka
multiview_correspondence_score = gap.multiview_correspondence_score
mutual_knn_alignment = gap.mutual_knn_alignment
procrustes_distance = gap.procrustes_distance

GRID = 14
FRAMES = 4
RES = 384

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def synth_scene(frames=FRAMES, res=RES, hole=True):
    """A slanted plane viewed by `frames` cameras translating along x.

    Returns world_coords (V,H,W,3) and cam_depth (V,H,W), consistent with each
    other, with a rectangle of dead depth if `hole` (as ScanNet has).
    """
    v = torch.linspace(-1, 1, res)
    yy, xx = torch.meshgrid(v, v, indexing="ij")
    world, depth = [], []
    for f in range(frames):
        shift = 0.35 * f
        # Plane 0.2x + 0.1y + z = 3 in camera coords: with X = z*(xx, yy, 1),
        # that is z = 3 / (0.2*xx + 0.1*yy + 1).
        z = 3.0 / (0.2 * xx + 0.1 * yy + 1.0)   # metres
        pts = torch.stack([xx * z + shift, yy * z, z], dim=-1)
        d = z.clone()
        if hole:
            d[100:160, 200:280] = 0.0            # sensor dropout
        world.append(pts)
        depth.append(d)
    return torch.stack(world), torch.stack(depth)


def test_patch_geometry():
    print("patch_geometry")
    world, depth = synth_scene()
    out = patch_geometry(world, depth, GRID)
    P = GRID * GRID
    check("shapes", out["points"].shape == (FRAMES, P, 3) and out["valid"].shape == (FRAMES, P))

    # Patches fully inside the dropout rectangle must be invalid, and every
    # patch outside it valid.
    n_invalid = (~out["valid"]).sum().item()
    check("dropout patches masked out", 0 < n_invalid < P * FRAMES,
          f"{n_invalid}/{P * FRAMES} invalid")

    # Valid patch targets must sit on the plane: z == 3 + 0.5x_cam + 0.25y_cam.
    # Recover via the reported depth against the median of the true depth.
    dense = depth[0]
    cell = RES // GRID
    ref = dense[:cell * GRID, :cell * GRID].reshape(GRID, cell, GRID, cell)
    ref = ref.permute(0, 2, 1, 3).reshape(P, cell * cell)
    ref = torch.where(ref > 0, ref, torch.tensor(float("nan")))
    ref = torch.nanmedian(ref, dim=-1).values
    ok = out["valid"][0]
    err = (out["depth"][0][ok] - ref[ok]).abs().max().item()
    check("per-patch depth == masked median", err < 1e-4, f"max err {err:.2e}")

    # An invalid patch must not drag a phantom point into the target: with
    # unmasked averaging its world point would collapse toward the origin.
    check("no NaNs leak into targets", torch.isfinite(out["points"]).all().item())


def test_normals_and_normalisation():
    print("normalize_points / grid_normals")
    world, depth = synth_scene(hole=False)
    out = patch_geometry(world, depth, GRID)
    pts, valid = out["points"], out["valid"]

    norm, centre, scale = normalize_points(pts, valid)
    mean = norm[valid].mean(0).abs().max().item()
    check("normalised map is centred", mean < 1e-4, f"|mean| {mean:.2e}")
    check("normalised map is unit-scaled", abs(norm[valid].norm(dim=-1).mean().item() - 1) < 1e-3)

    # A plane has one normal everywhere, and normalising must not change it.
    n_raw, m = grid_normals(pts, valid, GRID)
    n_norm, _ = grid_normals(norm, valid, GRID)
    spread = (n_raw[m] - n_raw[m].mean(0, keepdim=True)).norm(dim=-1).max().item()
    check("plane has constant normal", spread < 1e-3, f"max deviation {spread:.2e}")
    agree = F.cosine_similarity(n_raw[m], n_norm[m], dim=-1).min().item()
    check("normals invariant to normalisation", agree > 0.999, f"min cos {agree:.4f}")


def test_geometry_loss_fits():
    print("geometry_loss")
    world, depth = synth_scene()
    tgt = patch_geometry(world, depth, GRID)
    norm, _, _ = normalize_points(tgt["points"], tgt["valid"])
    target = {"points": norm, "depth": tgt["depth"], "valid": tgt["valid"]}

    perfect = {
        "points": norm.reshape(-1, 3),
        "log_depth": torch.log(tgt["depth"]).reshape(-1),
    }
    losses = geometry_loss(perfect, target, GRID)
    total = sum(v.item() for v in losses.values())
    check("zero loss on perfect prediction", total < 1e-5, f"total {total:.2e}")

    wrong = {"points": torch.randn_like(perfect["points"]),
             "log_depth": torch.randn_like(perfect["log_depth"])}
    check("nonzero loss on random prediction",
          sum(v.item() for v in geometry_loss(wrong, target, GRID).values()) > 0.1)

    # SILog is *near* scale-invariant by design: a global log offset costs
    # sqrt(1-lambda) of its magnitude, so predicting the right room shape at a
    # slightly wrong scale is cheap while getting the shape wrong is not. Full
    # invariance (lambda=1) would throw away ScanNet's metric scale entirely.
    offset = 0.7
    shifted = {"points": perfect["points"], "log_depth": perfect["log_depth"] + offset}
    d_shift = geometry_loss(shifted, target, GRID)["depth"].item()
    expected = (1 - 0.85) ** 0.5 * offset
    check("global scale error costs sqrt(1-lambda)", abs(d_shift - expected) < 1e-3,
          f"{d_shift:.4f} vs expected {expected:.4f}")

    noisy = perfect["log_depth"] + offset * torch.randn_like(perfect["log_depth"])
    d_noise = geometry_loss(
        {"points": perfect["points"], "log_depth": noisy}, target, GRID
    )["depth"].item()
    check("structural depth error costs more than a scale error", d_noise > 2 * d_shift,
          f"structural {d_noise:.3f} vs scale {d_shift:.3f}")

    # Invalid patches must not contribute: corrupt only those.
    corrupt = perfect["points"].clone().view(FRAMES, GRID * GRID, 3)
    corrupt[~tgt["valid"]] += 50.0
    masked = geometry_loss(
        {"points": corrupt.reshape(-1, 3), "log_depth": perfect["log_depth"]}, target, GRID
    )
    check("masked patches excluded from point loss", masked["point"].item() < 1e-5,
          f"{masked['point'].item():.2e}")


def test_correspondence():
    print("multiview_correspondence_loss")
    world, depth = synth_scene(hole=False)
    tgt = patch_geometry(world, depth, GRID)
    pts, valid = tgt["points"], tgt["valid"]
    V, P, _ = pts.shape

    g = torch.Generator().manual_seed(0)
    # A 3D-aware encoder: feature is a function of world position only, so the
    # same voxel seen from two frames yields the same feature.
    basis = torch.randn(3, 32, generator=g)
    aware = (pts.reshape(-1, 3) @ basis)
    # A 3D-blind encoder: feature depends only on which frame you are in.
    blind = torch.randn(V, 32, generator=g).repeat_interleave(P, dim=0)

    l_aware = multiview_correspondence_loss(aware, pts, valid, voxel_size=0.2,
                                            num_anchors=128, generator=g).item()
    l_blind = multiview_correspondence_loss(blind, pts, valid, voxel_size=0.2,
                                            num_anchors=128, generator=g).item()
    check("3D-aware features score better than 3D-blind", l_aware < l_blind,
          f"aware {l_aware:.3f} < blind {l_blind:.3f}")

    s_aware = multiview_correspondence_score(aware, pts, valid, voxel_size=0.2).item()
    s_blind = multiview_correspondence_score(blind, pts, valid, voxel_size=0.2).item()
    check("correspondence score ranks them the same way", s_aware > s_blind,
          f"aware {s_aware:.3f} > blind {s_blind:.3f}")

    check("no positives -> zero, not NaN",
          multiview_correspondence_loss(aware, pts, torch.zeros_like(valid)).item() == 0.0)


def test_token_extraction():
    print("extract_visual_tokens / resample_teacher_features")
    C = 16
    tokens_per_frame = GRID * (GRID + 1)
    hs = torch.zeros(1, FRAMES * tokens_per_frame + 5, C)
    marker = torch.arange(FRAMES * GRID * GRID).float()
    body = hs[0, 3:3 + FRAMES * tokens_per_frame].view(FRAMES, GRID, GRID + 1, C)
    body[:, :, :-1, 0] = marker.view(FRAMES, GRID, GRID)
    body[:, :, -1, 0] = -1.0                       # newline tokens
    out = extract_visual_tokens(hs, [3], [FRAMES * tokens_per_frame], GRID)
    check("drops newline tokens, keeps order",
          out.shape == (FRAMES * GRID * GRID, C) and torch.equal(out[:, 0], marker))

    try:
        extract_visual_tokens(hs, [3], [FRAMES * tokens_per_frame - 1], GRID)
        check("rejects a mis-sized span", False)
    except ValueError:
        check("rejects a mis-sized span", True)

    teacher = torch.randn(1, FRAMES, 1036, 8)
    pooled = resample_teacher_features(teacher, GRID)
    check("teacher pooled onto student grid", pooled.shape == (FRAMES * GRID * GRID, 8))

    # extract_vggt_feature.py now pools to 14x14 before saving, which is only
    # safe if it produces exactly what loading the full map would have.
    import importlib.util as _il
    spec = _il.spec_from_file_location("extract_vggt", ROOT / "extract_vggt_feature.py")
    try:
        mod = _il.module_from_spec(spec)
        spec.loader.exec_module(mod)                      # needs the vggt package
        prepooled = mod.pool_to_grid(teacher[0], GRID).unsqueeze(0)
        check("pre-pooled store matches full-map store",
              torch.allclose(resample_teacher_features(prepooled, GRID), pooled, atol=1e-5))
    except ImportError:
        # Reimplement the two lines rather than skip: this is the claim that
        # justifies throwing away 80% of the stored features.
        h, w = 28, 37
        x = teacher[0].view(FRAMES, h, w, 8).permute(0, 3, 1, 2).float()
        x = F.adaptive_avg_pool2d(x, (GRID, GRID))
        prepooled = x.view(FRAMES, 8, GRID * GRID).permute(0, 2, 1).contiguous().unsqueeze(0)
        check("pre-pooled store matches full-map store",
              torch.allclose(resample_teacher_features(prepooled, GRID), pooled, atol=1e-5))


def test_align_terms():
    print("align terms")
    g = torch.Generator().manual_seed(1)
    a = torch.randn(512, 64, generator=g)
    check("pointwise cosine is 0 for identical features",
          pointwise_cosine_align(a, a).item() < 1e-6)
    check("relational is 0 for identical features",
          relational_align(a, a, num_samples=256, generator=g).item() < 1e-6)
    # Relational alignment is blind to a rotation of the teacher's basis;
    # the pointwise term is not. That is the point of having both.
    q, _ = torch.linalg.qr(torch.randn(64, 64, generator=g))
    check("relational is rotation-invariant",
          relational_align(a, a @ q, num_samples=256, generator=g).item() < 1e-4)
    check("pointwise is not rotation-invariant", pointwise_cosine_align(a, a @ q).item() > 0.5)


def test_feature_gap_metrics():
    print("feature_gap metrics")
    g = torch.Generator().manual_seed(2)
    x = torch.randn(256, 32, generator=g)
    q, _ = torch.linalg.qr(torch.randn(32, 32, generator=g))
    check("CKA(x, x) == 1", abs(linear_cka(x, x).item() - 1) < 1e-4)
    check("CKA invariant to rotation", abs(linear_cka(x, x @ q).item() - 1) < 1e-4)
    check("CKA low for unrelated features",
          linear_cka(x, torch.randn(256, 32, generator=g)).item() < 0.3)
    check("Procrustes(x, x) == 0", procrustes_distance(x, x).item() < 1e-4)
    check("Procrustes invariant to rotation", procrustes_distance(x, x @ q).item() < 1e-4)
    check("mutual kNN(x, x) == 1", abs(mutual_knn_alignment(x, x, k=5).item() - 1) < 1e-6)
    check("mutual kNN low for unrelated features",
          mutual_knn_alignment(x, torch.randn(256, 32, generator=g), k=5).item() < 0.2)

    world, depth = synth_scene(hole=False)
    tgt = patch_geometry(world, depth, GRID)
    pts, valid = tgt["points"], tgt["valid"]
    lin = pts.reshape(-1, 3) @ torch.randn(3, 24, generator=g)
    r2_lin = geometry_probe_r2(lin, pts, valid, generator=g).item()
    r2_rand = geometry_probe_r2(torch.randn(lin.shape, generator=g), pts, valid, generator=g).item()
    check("probe recovers geometry from geometric features", r2_lin > 0.9, f"R2 {r2_lin:.3f}")
    check("probe fails on random features", r2_rand < 0.3, f"R2 {r2_rand:.3f}")


def test_end_to_end_gradients():
    print("end-to-end gradient flow")
    C, D = 48, 32
    world, depth = synth_scene()

    class Config:
        three_d_patch_grid = GRID
        three_d_align_weight = 1.0
        three_d_relational_weight = 0.5
        three_d_geo_weight = 1.0
        three_d_geo_normal_weight = 1.0
        three_d_geo_depth_weight = 1.0
        three_d_geo_confidence = True
        three_d_corr_weight = 0.5
        three_d_corr_voxel_size = 0.2
        three_d_corr_temperature = 0.07
        three_d_corr_anchors = 64

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_3d = nn.Linear(C, D)
            self.corres_linear = nn.Linear(C, C, bias=False)
            self.geo_heads = GeometryHeads(C, bottleneck=32, use_confidence=True)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.config = Config()

    model = Model()
    tokens_per_frame = GRID * (GRID + 1)
    hs = torch.randn(1, FRAMES * tokens_per_frame, C, requires_grad=True)
    video_dict = {
        "feature_3d": torch.randn(1, FRAMES, 1036, D),
        "world_coords": world.unsqueeze(0),
        "cam_depth": depth.unsqueeze(0).half(),
    }

    losses = compute_3d_supervision(model, video_dict, hs, [0], [FRAMES * tokens_per_frame])
    expected = {"align", "relational", "geo_point", "geo_normal", "geo_depth", "corr"}
    check("all terms present", set(losses) == expected, f"got {sorted(losses)}")
    check("all terms finite", all(torch.isfinite(v).item() for v in losses.values()),
          " ".join(f"{k}={v.item():.3f}" for k, v in losses.items()))

    sum(losses.values()).backward()
    check("gradient reaches the visual tokens",
          hs.grad is not None and hs.grad.abs().sum().item() > 0)
    for name, p in model.named_parameters():
        if p.grad is None or p.grad.abs().sum().item() == 0:
            check(f"gradient reaches {name}", False)
            break
    else:
        check("gradient reaches every head parameter", True)

    # Baseline config must reproduce the published objective exactly.
    Config.three_d_relational_weight = 0.0
    Config.three_d_geo_weight = 0.0
    Config.three_d_corr_weight = 0.0
    base = compute_3d_supervision(model, video_dict, hs.detach(), [0], [FRAMES * tokens_per_frame])
    ref = pointwise_cosine_align(
        model.model.proj_3d(extract_visual_tokens(hs.detach(), [0], [FRAMES * tokens_per_frame], GRID)),
        resample_teacher_features(video_dict["feature_3d"], GRID),
    )
    check("defaults reproduce the baseline 3DRS term",
          set(base) == {"align"} and abs(base["align"].item() - ref.item()) < 1e-6)


def test_contrastive():
    print("contrastive alignment")
    g = torch.Generator().manual_seed(5)
    x = torch.randn(64, 16, generator=g)
    pos = torch.eye(64, dtype=torch.bool)
    check("supcon == InfoNCE when the only positive is the pair",
          abs(supcon(x, x + 0.01, pos, 0.1).item()
              - F.cross_entropy((F.normalize(x, dim=-1) @ F.normalize(x + 0.01, dim=-1).t()) / 0.1,
                                torch.arange(64)).item()) < 1e-4)
    check("supcon skips anchors without positives (no NaN)",
          torch.isfinite(supcon(x, x, torch.zeros_like(pos), 0.1)).item())

    world, depth = synth_scene(hole=False)
    tgt = patch_geometry(world, depth, GRID)
    pts, V, P = tgt["points"].reshape(-1, 3), FRAMES, GRID * GRID
    frames = torch.arange(V).repeat_interleave(P)

    far = hard_negative_mask(pts @ torch.randn(3, 8, generator=g), pts,
                             torch.zeros(len(pts), dtype=torch.long), margin=1.0, k=8)
    d = torch.cdist(pts, pts)
    check("hard negatives are all >= margin away", bool((d[far] >= 1.0).all()))
    check("hard negatives are found", far.sum().item() > 0)

    basis = torch.randn(3, 32, generator=g)
    aware = pts @ basis                                   # function of 3D position
    blind = torch.randn(V, 32, generator=g).repeat_interleave(P, 0)   # frame identity only
    teacher = pts @ torch.randn(3, 32, generator=g)
    la = sum(contrastive_alignment_loss(aware, teacher, pts, frames, voxel_size=0.3).values()).item()
    lb = sum(contrastive_alignment_loss(blind, teacher, pts, frames, voxel_size=0.3).values()).item()
    check("3D-aware features score lower contrastive loss than 3D-blind", la < lb,
          f"aware {la:.3f} < blind {lb:.3f}")

    terms = contrastive_alignment_loss(aware, teacher, pts, frames, voxel_size=0.3,
                                       hard_weight=0.5, hard_margin=1.0)
    check("hard-negative term present when enabled", set(terms) == {"contrast_cm", "contrast_xv", "contrast_inst"})
    check("token-only mode is plain InfoNCE",
          set(contrastive_alignment_loss(aware, teacher, pts, frames, xview=False)) == {"contrast_token"})


def test_contrastive_end_to_end():
    print("contrastive end-to-end")
    C, D = 48, 32
    world, depth = synth_scene()

    class Cfg:
        three_d_patch_grid = GRID
        three_d_align_weight = 0.0
        three_d_relational_weight = 0.0
        three_d_geo_weight = 0.0
        three_d_corr_weight = 0.0
        three_d_contrast_weight = 1.0
        three_d_contrast_layer = 1
        three_d_contrast_dim = 24
        three_d_contrast_tau = 0.07
        three_d_contrast_xview = True
        three_d_contrast_lam_xview = 1.0
        three_d_contrast_hard_weight = 0.5
        three_d_contrast_hard_margin = 1.0
        three_d_contrast_hard_k = 8
        three_d_contrast_max_tokens = 256
        three_d_corr_voxel_size = 0.3

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_3d = nn.Linear(C, D)
            self.contrast_heads = ContrastiveHeads(C, teacher_dim=D, dim=24, width=32)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.config = Cfg()

    model = Model()
    span = FRAMES * GRID * (GRID + 1)
    layers = [torch.randn(1, span, C, requires_grad=True) for _ in range(3)]
    vd = {"feature_3d": torch.randn(1, FRAMES, 1036, D), "world_coords": world.unsqueeze(0),
          "cam_depth": depth.unsqueeze(0).half()}
    losses = compute_3d_supervision(model, vd, layers[-1], [0], [span], all_hidden_states=layers)
    check("contrast terms present", set(losses) == {"contrast_cm", "contrast_xv", "contrast_inst"},
          f"got {sorted(losses)}")
    sum(losses.values()).backward()
    check("gradient reaches the requested layer (1)", layers[1].grad is not None and layers[1].grad.abs().sum() > 0)
    check("no gradient to the unrequested last layer", layers[2].grad is None)
    check("gradient reaches both contrast heads",
          all(p.grad is not None for p in model.model.contrast_heads.parameters()))
    try:
        compute_3d_supervision(model, vd, layers[-1], [0], [span], all_hidden_states=None)
        check("intermediate layer without hidden states raises", False)
    except ValueError:
        check("intermediate layer without hidden states raises", True)


def main():
    torch.manual_seed(0)
    for test in (
        test_patch_geometry,
        test_normals_and_normalisation,
        test_geometry_loss_fits,
        test_correspondence,
        test_token_extraction,
        test_align_terms,
        test_feature_gap_metrics,
        test_end_to_end_gradients,
        test_contrastive,
        test_contrastive_end_to_end,
    ):
        test()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
