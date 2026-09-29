"""Explicit 3D supervision for MLLM visual tokens (3DRS-G).

Baseline 3DRS supervises the MLLM's visual hidden states with a single pointwise
cosine-distillation term against frozen VGGT features. Every bit of geometry the
student sees therefore arrives second-hand, which the authors themselves flag as
the ceiling of the method: performance "is upper-bounded by the quality of the
teacher 3D foundation model".

This module adds signal that does not pass through the teacher at all:

  * ``GeometryHeads`` decode metric geometry (point map, camera depth, local
    surface orientation) straight out of the visual tokens, supervised by the
    ScanNet sensor depth and poses that ``video_utils.calculate_world_coords``
    already unprojects for the world-position embedding.
  * ``multiview_correspondence_loss`` pulls together patches from different
    frames that fall in the same voxel -- the exact quantity the paper uses to
    *measure* 3D awareness, turned into a training signal.

and it widens the MLLM<->3DFM comparison itself from one cosine term to a
pointwise term plus a memory-safe relational (Gram-matrix) term.

Every loss here is masked by depth validity: ScanNet depth is 0 where the sensor
returned nothing, and ``unproject`` maps those pixels to the camera centre, so
an unmasked loss would fit a large blob of phantom geometry.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Visual tokens are laid out as a PATCH_GRID x PATCH_GRID grid per frame, with
# one trailing newline token per row (``--mm_newline_position grid``).
PATCH_GRID = 14

# Token count -> (H, W) for the teacher's feature map. VGGT resizes the long
# side to 518, so the grid depends on the source aspect ratio.
TEACHER_GRID_SHAPES = {
    1036: (28, 37),
    768: (24, 32),
    256: (16, 16),
    # Already pooled at extraction time by extract_vggt_feature.py; pooling a
    # 14x14 map to 14x14 is the identity, so both storage formats load the same.
    PATCH_GRID * PATCH_GRID: (PATCH_GRID, PATCH_GRID),
}


# --------------------------------------------------------------------------- #
# Token extraction
# --------------------------------------------------------------------------- #
def infer_patch_layout(span_len: int, num_frames: int) -> Tuple[int, bool]:
    """Recover (grid, newline_per_row) from the length of the visual span.

    The 3DRS training recipe yields a 14x14 grid with one newline token per row
    (``--mm_newline_position grid``, pool stride 2), but a stock LLaVA-Video
    config need not, and guessing wrong silently pairs every visual token with
    the wrong patch of geometry. Deriving the layout makes that impossible.
    """
    if num_frames <= 0 or span_len % num_frames:
        raise ValueError(f"visual span {span_len} is not divisible by {num_frames} frames")
    per_frame = span_len // num_frames
    g = int(round(((1 + 4 * per_frame) ** 0.5 - 1) / 2))   # g*(g+1) == per_frame
    if g > 0 and g * (g + 1) == per_frame:
        return g, True
    g = int(round(per_frame ** 0.5))                        # g*g == per_frame
    if g > 0 and g * g == per_frame:
        return g, False
    raise ValueError(
        f"{per_frame} tokens/frame is neither g*g nor g*(g+1) for integer g"
    )


def extract_visual_tokens(
    hidden_states: torch.Tensor,
    img_pos_list,
    img_length_list,
    grid: int = PATCH_GRID,
    newline_per_row: bool = True,
) -> torch.Tensor:
    """Pull the per-frame patch tokens out of the LLM hidden states.

    Replaces the hardcoded ``.view(32, 14, 15, C)`` in the release: the frame
    count is inferred from the span length so 8/16/64-frame runs work too, and
    grids without a trailing newline token are handled.

    Returns (num_frames * grid * grid, C), newline tokens dropped.
    """
    C = hidden_states.shape[-1]
    span = hidden_states[:, img_pos_list[0] : img_pos_list[0] + img_length_list[0], :]
    tokens_per_frame = grid * (grid + 1) if newline_per_row else grid * grid
    num_frames, remainder = divmod(span.shape[1], tokens_per_frame)
    if remainder != 0:
        raise ValueError(
            f"visual span of {span.shape[1]} tokens is not a multiple of "
            f"{tokens_per_frame} (grid={grid}"
            + (" plus one newline per row)" if newline_per_row else ")")
        )
    if not newline_per_row:
        return span.reshape(-1, C)
    return span.view(num_frames, grid, grid + 1, C)[:, :, :-1, :].contiguous().view(-1, C)


def resample_teacher_features(
    feature_3d: torch.Tensor,
    grid: int = PATCH_GRID,
    grid_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Pool frozen 3DFM tokens onto the student's grid -> (S*grid*grid, D)."""
    feature_3d = feature_3d.squeeze()
    S, L, D = feature_3d.shape
    if grid_shape is None:
        if L not in TEACHER_GRID_SHAPES:
            raise NotImplementedError(
                f"unknown teacher token count L={L}; add it to TEACHER_GRID_SHAPES "
                f"or pass grid_shape explicitly"
            )
        grid_shape = TEACHER_GRID_SHAPES[L]
    h, w = grid_shape
    feature_3d = feature_3d.view(S, h, w, D).permute(0, 3, 1, 2).contiguous()
    feature_3d = F.adaptive_avg_pool2d(feature_3d, (grid, grid))
    return feature_3d.view(S, D, grid * grid).permute(0, 2, 1).contiguous().view(-1, D)


# --------------------------------------------------------------------------- #
# Ground-truth geometry from sensor depth
# --------------------------------------------------------------------------- #
def patch_geometry(
    world_coords: torch.Tensor,
    cam_depth: torch.Tensor,
    grid: int = PATCH_GRID,
    min_valid_frac: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Reduce dense per-pixel geometry to one target per visual patch.

    Args:
        world_coords: (V, H, W, 3) axis-aligned world XYZ from ``unproject``.
        cam_depth: (V, H, W) camera depth in metres; <= 0 marks a dead pixel.

    Uses a per-patch *median* rather than a mean: a patch straddling a depth
    discontinuity would otherwise be supervised toward a point floating in
    empty space between the two surfaces.

    Returns points (V, P, 3), depth (V, P), valid (V, P) with P = grid*grid.
    """
    V, H, W, _ = world_coords.shape
    cell = min(H, W) // grid
    side = cell * grid
    coords = world_coords[:, :side, :side, :].float()
    depth = cam_depth[:, :side, :side].float()

    # (V, grid, grid, cell*cell, ...)
    coords = coords.view(V, grid, cell, grid, cell, 3).permute(0, 1, 3, 2, 4, 5)
    coords = coords.reshape(V, grid * grid, cell * cell, 3)
    depth = depth.view(V, grid, cell, grid, cell).permute(0, 1, 3, 2, 4)
    depth = depth.reshape(V, grid * grid, cell * cell)

    pix_valid = depth > 0
    frac = pix_valid.float().mean(dim=-1)

    # Median over valid pixels only, via NaN masking.
    nan = torch.tensor(float("nan"), device=coords.device, dtype=coords.dtype)
    coords = torch.where(pix_valid.unsqueeze(-1), coords, nan)
    depth = torch.where(pix_valid, depth, nan)

    points = torch.nanmedian(coords, dim=2).values
    depth = torch.nanmedian(depth, dim=2).values

    valid = (frac >= min_valid_frac) & torch.isfinite(points).all(dim=-1) & torch.isfinite(depth)
    points = torch.nan_to_num(points, nan=0.0)
    depth = torch.nan_to_num(depth, nan=1.0).clamp_min(1e-3)
    return {"points": points, "depth": depth, "valid": valid}


def normalize_points(
    points: torch.Tensor, valid: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centre and scale a point map by its own valid extent (DUSt3R-style).

    The world frame is an arbitrary per-scene axis alignment, so regressing raw
    coordinates would make the loss scale with how far the scene origin happens
    to sit from the furniture. Normalising makes the target the scene's *shape*.
    """
    mask = valid.unsqueeze(-1).float()
    count = mask.sum().clamp_min(1.0)
    centre = (points * mask).sum(dim=(0, 1)) / count
    centred = points - centre
    scale = ((centred * mask).norm(dim=-1).sum() / count).clamp_min(1e-3)
    return centred / scale, centre, scale


def grid_normals(
    points: torch.Tensor, valid: torch.Tensor, grid: int = PATCH_GRID
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Surface normals by central differences on the patch grid.

    Returns normals (V, grid-2, grid-2, 3) and their validity mask.
    """
    V = points.shape[0]
    p = points.view(V, grid, grid, 3)
    m = valid.view(V, grid, grid)
    du = p[:, 1:-1, 2:, :] - p[:, 1:-1, :-2, :]
    dv = p[:, 2:, 1:-1, :] - p[:, :-2, 1:-1, :]
    normals = F.normalize(torch.cross(du, dv, dim=-1), dim=-1, eps=1e-6)
    mask = (
        m[:, 1:-1, 2:] & m[:, 1:-1, :-2] & m[:, 2:, 1:-1] & m[:, :-2, 1:-1] & m[:, 1:-1, 1:-1]
    )
    return normals, mask


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
class GeometryHeads(nn.Module):
    """Decode metric geometry directly from visual tokens.

    Deliberately shallow: the point of the probe is to force geometry into the
    *backbone* features, so the head must not be expressive enough to synthesise
    it on its own.
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck: Optional[int] = None,
        predict_depth: bool = True,
        use_confidence: bool = False,
    ):
        super().__init__()
        bottleneck = bottleneck or max(hidden_size // 8, 64)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.LayerNorm(bottleneck),
        )
        self.point_head = nn.Linear(bottleneck, 3)
        self.depth_head = nn.Linear(bottleneck, 1) if predict_depth else None
        self.conf_head = nn.Linear(bottleneck, 1) if use_confidence else None

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(tokens)
        out = {"points": self.point_head(h)}
        if self.depth_head is not None:
            out["log_depth"] = self.depth_head(h).squeeze(-1)
        if self.conf_head is not None:
            # softplus keeps confidence positive; +1 keeps the log term tame.
            out["conf"] = F.softplus(self.conf_head(h).squeeze(-1)) + 1.0
        return out


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def geometry_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    grid: int = PATCH_GRID,
    normal_weight: float = 1.0,
    depth_weight: float = 1.0,
    huber_delta: float = 0.1,
    silog_lambda: float = 0.85,
) -> Dict[str, torch.Tensor]:
    """Point-map + normal + scale-invariant depth losses, all validity-masked."""
    points_gt, valid = target["points"], target["valid"]
    V, P, _ = points_gt.shape
    device, dtype = points_gt.device, points_gt.dtype
    zero = torch.zeros((), device=device, dtype=dtype)

    pred_points = pred["points"].view(V, P, 3).float()
    losses: Dict[str, torch.Tensor] = {}

    if valid.any():
        err = F.huber_loss(pred_points, points_gt, reduction="none", delta=huber_delta).sum(-1)
        if "conf" in pred:
            # DUSt3R-style: the model may declare a patch ambiguous, but pays
            # log(conf) for the privilege.
            conf = pred["conf"].view(V, P).float()
            err = conf * err - 0.2 * torch.log(conf)
        losses["point"] = err[valid].mean()
    else:
        losses["point"] = zero

    if normal_weight > 0:
        n_pred, _ = grid_normals(pred_points, valid, grid)
        n_gt, n_mask = grid_normals(points_gt, valid, grid)
        if n_mask.any():
            cos = (n_pred * n_gt).sum(-1)[n_mask]
            losses["normal"] = normal_weight * (1.0 - cos).mean()
        else:
            losses["normal"] = zero

    # Scale-invariant log depth (Eigen et al.). lambda < 1 leaves a weak metric
    # anchor: ScanNet depth is real metres and the MLLM has no other source of
    # absolute scale, so discarding it entirely (lambda = 1) would be wasteful.
    if depth_weight > 0 and "log_depth" in pred:
        d = pred["log_depth"].view(V, P).float() - torch.log(target["depth"])
        if valid.any():
            d = d[valid]
            losses["depth"] = depth_weight * (
                (d.pow(2).mean() - silog_lambda * d.mean().pow(2)).clamp_min(0.0).sqrt()
            )
        else:
            losses["depth"] = zero

    return losses


def pointwise_cosine_align(feature_proj: torch.Tensor, feature_3d: torch.Tensor) -> torch.Tensor:
    """Baseline 3DRS term: squared distance between L2-normalised features.

    Identical to ``2 - 2*cos`` up to a constant, i.e. the paper's ``L_align``.
    """
    a = F.normalize(feature_proj.float(), dim=-1)
    b = F.normalize(feature_3d.float(), dim=-1)
    return ((a - b.detach()) ** 2).sum(dim=-1).mean()


def relational_align(
    feature_proj: torch.Tensor,
    feature_3d: torch.Tensor,
    num_samples: int = 2048,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Match the *pairwise structure* of student and teacher features.

    The pointwise term asks the student to land on the teacher's vector; this
    asks the weaker and more transferable question of which patches the teacher
    considers alike. The release contains this idea in ``feature_3d_similarity``
    but never calls it, and as written it materialises a 6272x6272 matrix in
    both directions -- here it is estimated on a random subset instead.
    """
    n = feature_proj.shape[0]
    if n > num_samples:
        idx = torch.randperm(n, device=feature_proj.device, generator=generator)[:num_samples]
        feature_proj, feature_3d = feature_proj[idx], feature_3d[idx]
    a = F.normalize(feature_proj.float(), dim=-1)
    b = F.normalize(feature_3d.float(), dim=-1)
    return F.l1_loss(a @ a.t(), (b @ b.t()).detach())


def multiview_correspondence_loss(
    features: torch.Tensor,
    points: torch.Tensor,
    valid: torch.Tensor,
    voxel_size: float = 0.2,
    temperature: float = 0.07,
    num_anchors: int = 256,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """InfoNCE over patches that observe the same voxel from different frames.

    The paper measures 3D awareness as the mean cosine similarity of exactly
    these pairs, and reports that the score correlates with downstream accuracy;
    optimising it directly is a supervision signal the teacher cannot provide.
    Same-voxel/same-frame pairs are excluded from both roles -- they are
    trivially similar neighbours and would only dilute the objective.

    Args:
        features: (V*P, C) student features for the same tokens as ``points``.
        points: (V, P, 3) world coordinates; ``valid`` masks dead depth.
    """
    V, P, _ = points.shape
    device = features.device
    zero = torch.zeros((), device=device, dtype=torch.float32)

    flat_valid = valid.view(-1)
    if flat_valid.sum() < 2:
        return zero

    frames = torch.arange(V, device=device).repeat_interleave(P)[flat_valid]
    feats = F.normalize(features[flat_valid].float(), dim=-1)
    voxels = torch.floor(points.view(-1, 3)[flat_valid] / voxel_size).long()
    _, group = torch.unique(voxels, dim=0, return_inverse=True)

    n = feats.shape[0]
    perm = torch.randperm(n, device=device, generator=generator)[: min(num_anchors, n)]

    same_voxel = group[perm].unsqueeze(1) == group.unsqueeze(0)
    cross_frame = frames[perm].unsqueeze(1) != frames.unsqueeze(0)
    positives = same_voxel & cross_frame
    has_pos = positives.any(dim=1)
    if not has_pos.any():
        return zero

    positives = positives[has_pos]
    # Candidates = positives plus everything in a different voxel. Same-voxel
    # same-frame neighbours are dropped from the denominator entirely.
    candidates = positives | ~same_voxel[has_pos]

    logits = (feats[perm][has_pos] @ feats.t()) / temperature
    logits = logits.masked_fill(~candidates, float("-inf"))
    log_denom = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~positives, float("-inf")), dim=1)
    return (log_denom - log_num).mean()


# --------------------------------------------------------------------------- #
# Contrastive alignment (shared by the frozen Stage 2 study and LoRA training)
# --------------------------------------------------------------------------- #
def sym_infonce(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    """CLIP-style symmetric InfoNCE; row i of `a` pairs with row i of `b`."""
    logits = F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).t() / tau
    y = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y))


def supcon(
    anchor: torch.Tensor,
    cand: torch.Tensor,
    pos: torch.Tensor,
    tau: float,
    allowed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-positive SupCon, L_out form (log outside the positive average).

    `allowed` restricts the denominator (positives are always admitted); None
    admits every candidate. Anchors without a positive are skipped rather than
    contributing a degenerate term."""
    logits = F.normalize(anchor.float(), dim=-1) @ F.normalize(cand.float(), dim=-1).t() / tau
    if allowed is not None:
        logits = logits.masked_fill(~(allowed | pos), float("-inf"))
    logprob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    has = pos.any(1)
    if not has.any():
        return anchor.sum() * 0.0
    lp = logprob.masked_fill(~pos, 0.0).sum(1)
    return -(lp[has] / pos.sum(1)[has]).mean()


def hard_negative_mask(
    feats: torch.Tensor, points: torch.Tensor, scene_ids: torch.Tensor, margin: float, k: int
) -> torch.Tensor:
    """Per anchor, the k same-scene tokens most similar in `feats` among those at
    least `margin` metres away: things that look alike but are elsewhere."""
    fn = F.normalize(feats.float(), dim=-1)
    sim = fn @ fn.t()
    far = (scene_ids[:, None] == scene_ids[None, :]) & (torch.cdist(points, points) >= margin)
    sim = sim.masked_fill(~far, float("-inf"))
    top = sim.topk(min(k, sim.shape[1]), dim=1)
    mask = torch.zeros_like(far)
    mask.scatter_(1, top.indices, torch.isfinite(top.values))
    return mask


class ContrastiveHeads(nn.Module):
    """g: VLM tokens -> shared space; h: 3D-foundation-model tokens -> same space.

    Only g is used at inference; the teacher is a training-time target. The
    LayerNorm at each input stands in for the fixed standardisation used in the
    frozen study: Qwen2 hidden states carry massive outlier channels that would
    otherwise dominate every cosine."""

    def __init__(self, hidden_size: int, teacher_dim: int, dim: int = 2048, width: int = 1024):
        super().__init__()
        self.g = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width),
                               nn.GELU(), nn.Linear(width, dim))
        self.h = nn.Sequential(nn.LayerNorm(teacher_dim), nn.Linear(teacher_dim, width),
                               nn.GELU(), nn.Linear(width, dim))


def contrastive_alignment_loss(
    g_tokens: torch.Tensor,
    h_tokens: torch.Tensor,
    points: torch.Tensor,
    frame_ids: torch.Tensor,
    tau: float = 0.07,
    voxel_size: float = 0.2,
    lam_xview: float = 1.0,
    xview: bool = True,
    hard_weight: float = 0.0,
    hard_source: Optional[torch.Tensor] = None,
    hard_margin: float = 1.0,
    hard_k: int = 16,
) -> Dict[str, torch.Tensor]:
    """Cross-modal + cross-view (+ optional hard-negative) alignment for one scene.

    Positives are every token observing the same voxel, in any frame and either
    modality. Without that, plain token InfoNCE treats a second view of the same
    surface as a negative -- the frozen study showed it then learns patch
    identity (retrieval@1 0.47) while cross-view consistency stays at the raw
    level. `hard_source` supplies the similarity used to mine hard negatives
    (detached features); it defaults to the VLM tokens being aligned.
    """
    if not xview:
        return {"contrast_token": sym_infonce(g_tokens, h_tokens, tau)}

    zeros = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
    grp = torch.unique(torch.floor(points / voxel_size).long(), dim=0, return_inverse=True)[1]
    same = grp[:, None] == grp[None, :]
    eye = torch.eye(len(grp), dtype=torch.bool, device=points.device)
    cross_view = same & (frame_ids[:, None] != frame_ids[None, :])

    out = {
        "contrast_cm": 0.5 * (supcon(g_tokens, h_tokens, same, tau)
                              + supcon(h_tokens, g_tokens, same, tau)),
        "contrast_xv": lam_xview * supcon(g_tokens, g_tokens, cross_view, tau, allowed=~eye),
    }
    if hard_weight > 0:
        src = (hard_source if hard_source is not None else g_tokens).detach()
        hard = hard_negative_mask(src, points, zeros, hard_margin, hard_k) & ~eye
        out["contrast_inst"] = hard_weight * supcon(g_tokens, g_tokens, cross_view, tau, allowed=hard)
    return out


# --------------------------------------------------------------------------- #
# Entry point used by the model
# --------------------------------------------------------------------------- #
_DEFAULTS = {
    "three_d_patch_grid": PATCH_GRID,
    "three_d_align_weight": 1.0,        # baseline 3DRS distillation
    "three_d_relational_weight": 0.0,   # structure matching against the teacher
    "three_d_geo_weight": 0.0,          # explicit depth/point-map supervision
    "three_d_geo_normal_weight": 1.0,
    "three_d_geo_depth_weight": 1.0,
    "three_d_geo_confidence": False,
    "three_d_corr_weight": 0.0,         # multi-view correspondence InfoNCE
    "three_d_corr_voxel_size": 0.2,
    "three_d_corr_temperature": 0.07,
    "three_d_corr_anchors": 256,
    # Contrastive VLM <-> 3D-foundation-model alignment (see Stage 2 study).
    "three_d_contrast_weight": 0.0,
    "three_d_contrast_layer": -1,          # decoder hidden_states index; -1 = last
    "three_d_contrast_dim": 2048,
    "three_d_contrast_tau": 0.07,
    "three_d_contrast_xview": True,
    "three_d_contrast_lam_xview": 1.0,
    "three_d_contrast_hard_weight": 0.0,
    "three_d_contrast_hard_margin": 1.0,
    "three_d_contrast_hard_k": 16,
    "three_d_contrast_max_tokens": 3136,   # per-sample subsample; N^2 similarity
}


def supervision_setting(config, name):
    return getattr(config, name, _DEFAULTS[name])


def geometry_supervision_enabled(config) -> bool:
    return supervision_setting(config, "three_d_geo_weight") > 0


def contrastive_supervision_enabled(config) -> bool:
    return supervision_setting(config, "three_d_contrast_weight") > 0


def compute_3d_supervision(
    model,
    video_dict,
    hidden_states: torch.Tensor,
    img_pos_list,
    img_length_list,
    all_hidden_states=None,
) -> Dict[str, torch.Tensor]:
    """Every 3D-supervision term for one step, keyed by name.

    ``model`` is the ``*ForCausalLM``; the submodules live on ``model.model``
    (``proj_3d`` from the release, plus ``geo_heads`` and the previously unused
    ``corres_linear``). With default settings this returns exactly the paper's
    single ``align`` term, so an unflagged run reproduces the baseline.
    """
    config = model.config
    grid = supervision_setting(config, "three_d_patch_grid")
    losses: Dict[str, torch.Tensor] = {}

    tokens = extract_visual_tokens(hidden_states, img_pos_list, img_length_list, grid)

    # --- distillation from the frozen 3D foundation model ------------------- #
    align_w = supervision_setting(config, "three_d_align_weight")
    relational_w = supervision_setting(config, "three_d_relational_weight")
    if (align_w > 0 or relational_w > 0) and video_dict.get("feature_3d", None) is not None:
        feature_proj = model.model.proj_3d(tokens)
        feature_3d = resample_teacher_features(
            video_dict["feature_3d"].to(device=tokens.device, dtype=tokens.dtype), grid
        )
        if feature_proj.shape[-1] != feature_3d.shape[-1]:
            raise ValueError(
                f"proj_3d outputs {feature_proj.shape[-1]}-d but the teacher is "
                f"{feature_3d.shape[-1]}-d"
            )
        if align_w > 0:
            losses["align"] = align_w * pointwise_cosine_align(feature_proj, feature_3d)
        if relational_w > 0:
            losses["relational"] = relational_w * relational_align(feature_proj, feature_3d)

    # --- explicit geometry from the depth sensor ---------------------------- #
    geo_w = supervision_setting(config, "three_d_geo_weight")
    corr_w = supervision_setting(config, "three_d_corr_weight")
    contrast_w = supervision_setting(config, "three_d_contrast_weight")
    needs_geometry = geo_w > 0 or corr_w > 0 or contrast_w > 0
    if needs_geometry:
        if "cam_depth" not in video_dict or video_dict["cam_depth"] is None:
            raise ValueError(
                "geometry/correspondence supervision needs 'cam_depth' in video_dict; "
                "re-run with the updated llava/video_utils.py"
            )
        target = patch_geometry(
            video_dict["world_coords"][0].to(tokens.device),
            video_dict["cam_depth"][0].to(tokens.device),
            grid,
        )
        num_frames = tokens.shape[0] // (grid * grid)
        if target["points"].shape[0] != num_frames:
            raise ValueError(
                f"{target['points'].shape[0]} depth frames vs {num_frames} visual frames"
            )

        if geo_w > 0:
            losses.update(_geometry_terms(model, config, tokens, target, grid, geo_w))
        if contrast_w > 0:
            losses.update(_contrastive_terms(
                model, config, video_dict, tokens, target, grid, contrast_w,
                all_hidden_states, img_pos_list, img_length_list,
            ))
        if corr_w > 0:
            losses["corr"] = corr_w * multiview_correspondence_loss(
                model.model.corres_linear(tokens),
                target["points"],
                target["valid"],
                voxel_size=supervision_setting(config, "three_d_corr_voxel_size"),
                temperature=supervision_setting(config, "three_d_corr_temperature"),
                num_anchors=supervision_setting(config, "three_d_corr_anchors"),
            )

    return losses


def _contrastive_terms(model, config, video_dict, tokens, target, grid, weight,
                       all_hidden_states, img_pos_list, img_length_list):
    layer = supervision_setting(config, "three_d_contrast_layer")
    if layer != -1:
        if all_hidden_states is None:
            raise ValueError(
                f"three_d_contrast_layer={layer} needs the decoder's hidden states; "
                "the caller must run the model with output_hidden_states=True"
            )
        tokens = extract_visual_tokens(all_hidden_states[layer], img_pos_list, img_length_list, grid)
    if model.model.contrast_heads is None:
        raise ValueError("three_d_contrast_weight > 0 but the model was built without contrast heads")
    if video_dict.get("feature_3d", None) is None:
        raise ValueError("contrastive supervision needs 'feature_3d' (VGGT tokens) in video_dict")

    heads = model.model.contrast_heads
    teacher = resample_teacher_features(
        video_dict["feature_3d"].to(device=tokens.device, dtype=tokens.dtype), grid
    )
    V, P = target["valid"].shape
    idx = torch.nonzero(target["valid"].reshape(-1)).flatten()
    cap = supervision_setting(config, "three_d_contrast_max_tokens")
    if idx.numel() > cap:
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:cap]]
    frame_ids = torch.div(idx, P, rounding_mode="floor")

    terms = contrastive_alignment_loss(
        heads.g(tokens[idx]), heads.h(teacher[idx]),
        target["points"].reshape(-1, 3)[idx], frame_ids,
        tau=supervision_setting(config, "three_d_contrast_tau"),
        voxel_size=supervision_setting(config, "three_d_corr_voxel_size"),
        lam_xview=supervision_setting(config, "three_d_contrast_lam_xview"),
        xview=supervision_setting(config, "three_d_contrast_xview"),
        hard_weight=supervision_setting(config, "three_d_contrast_hard_weight"),
        hard_source=tokens[idx],
        hard_margin=supervision_setting(config, "three_d_contrast_hard_margin"),
        hard_k=supervision_setting(config, "three_d_contrast_hard_k"),
    )
    return {name: weight * value for name, value in terms.items()}


def _geometry_terms(model, config, tokens, target, grid, geo_w):
    normalized, _, _ = normalize_points(target["points"], target["valid"])
    if model.model.geo_heads is None:
        raise ValueError(
            "three_d_geo_weight > 0 but the model has no geometry heads; the "
            "config used to build the model must set it too"
        )
    geo = geometry_loss(
        model.model.geo_heads(tokens),
        {"points": normalized, "depth": target["depth"], "valid": target["valid"]},
        grid=grid,
        normal_weight=supervision_setting(config, "three_d_geo_normal_weight"),
        depth_weight=supervision_setting(config, "three_d_geo_depth_weight"),
    )
    return {f"geo_{name}": geo_w * value for name, value in geo.items()}
