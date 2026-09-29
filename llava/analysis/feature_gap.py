"""Quantify the gap between MLLM visual features and 3D foundation model features.

3DRS reduces the student/teacher relationship to one number (the cosine
distillation loss), which conflates two very different failures: features that
encode the same geometry in a rotated basis, and features that do not encode it
at all. A projection layer fixes the first and nothing fixes the second, so the
loss value alone cannot say whether the MLLM has become more 3D-aware.

These metrics separate the two:

  * ``linear_cka`` / ``procrustes_distance`` -- representational similarity that
    is invariant to rotations and isotropic scaling (NOT to arbitrary invertible
    linear maps), so it compares how two spaces organise the same tokens rather
    than their raw coordinates. Dominated by high-variance directions: compute
    it on standardised features when outlier channels are present.
  * ``mutual_knn_alignment`` -- do student and teacher agree on which patches
    are neighbours? Local, non-parametric, and unaffected by global scaling.
  * ``multiview_correspondence_score`` -- the paper's own 3D-awareness proxy,
    computed here so training curves and the headline metric use one code path.
  * ``geometry_probe_r2`` -- a closed-form ridge probe from features to world
    coordinates. Answers the question the other metrics cannot: is metric
    geometry linearly *decodable* from these features at all?

Everything is torch-only, batched, and runs on CPU for offline dumps.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _center_gram(k: torch.Tensor) -> torch.Tensor:
    n = k.shape[0]
    unit = torch.ones(n, n, device=k.device, dtype=k.dtype) / n
    return k - unit @ k - k @ unit + unit @ k @ unit


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear centred kernel alignment in [0, 1] (Kornblith et al., 2019).

    Invariant to orthogonal transforms and isotropic scaling, but not to general
    invertible linear maps (anisotropic rescaling changes it)."""
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    # Computed on the feature-space Grams (d x d), which is far cheaper than the
    # n x n form whenever there are more patches than channels.
    xty = (x.t() @ y).norm(p="fro") ** 2
    xtx = (x.t() @ x).norm(p="fro")
    yty = (y.t() @ y).norm(p="fro")
    return xty / (xtx * yty).clamp_min(1e-12)


def procrustes_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Normalised orthogonal Procrustes distance in [0, 2]; 0 = same up to rotation."""
    x = F.normalize(x.float() - x.float().mean(0, keepdim=True), dim=None)
    y = F.normalize(y.float() - y.float().mean(0, keepdim=True), dim=None)
    nuclear = torch.linalg.svdvals(x.t() @ y).sum()
    return (x.pow(2).sum() + y.pow(2).sum() - 2 * nuclear).clamp_min(0.0)


def mutual_knn_alignment(x: torch.Tensor, y: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Mean overlap of k-nearest-neighbour sets between two feature spaces."""
    n = x.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return torch.zeros((), device=x.device)

    def neighbours(f: torch.Tensor) -> torch.Tensor:
        sim = F.normalize(f.float(), dim=-1) @ F.normalize(f.float(), dim=-1).t()
        sim.fill_diagonal_(float("-inf"))
        return sim.topk(k, dim=-1).indices

    nx, ny = neighbours(x), neighbours(y)
    # Overlap via a boolean membership matrix rather than a per-row isin loop:
    # Stage 2 calls this once per scene per variant, so the loop dominated.
    member = torch.zeros(n, n, dtype=torch.bool, device=x.device)
    member.scatter_(1, ny, True)
    hits = member.gather(1, nx).sum(dim=1).float()
    return (hits / k).mean()


def random_pair_similarity(
    features: torch.Tensor,
    num_pairs: int = 100_000,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Mean cosine between random token pairs -- the floor for any similarity.

    LLM hidden states are strongly anisotropic: every pair of tokens tends to
    look similar, so a raw correspondence score of 0.6 can be entirely
    explained by that offset. Only the *gap* above this floor is evidence of
    3D awareness, which is why the paper's absolute score is hard to read.
    """
    n = features.shape[0]
    if n < 2:
        return torch.zeros((), device=features.device)
    f = F.normalize(features.float(), dim=-1)
    i = torch.randint(0, n, (num_pairs,), device=features.device, generator=generator)
    j = torch.randint(0, n, (num_pairs,), device=features.device, generator=generator)
    keep = i != j
    return (f[i[keep]] * f[j[keep]]).sum(-1).mean()


def multiview_correspondence_score(
    features: torch.Tensor,
    points: torch.Tensor,
    valid: torch.Tensor,
    voxel_size: float = 0.2,
    max_pairs: int = 100_000,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Mean cosine similarity of cross-frame patch pairs sharing a voxel.

    This is the paper's 3D-awareness proxy: S = mean over correspondence pairs
    of cos(f_a, f_b), where a pair is two views of the same voxel. Read it
    against ``random_pair_similarity`` -- the absolute value alone is not
    interpretable.
    """
    V, P, _ = points.shape
    device = features.device
    flat_valid = valid.view(-1)
    if flat_valid.sum() < 2:
        return torch.zeros((), device=device)

    feats = F.normalize(features[flat_valid].float(), dim=-1)
    frames = torch.arange(V, device=device).repeat_interleave(P)[flat_valid]
    voxels = torch.floor(points.view(-1, 3)[flat_valid] / voxel_size).long()
    _, group = torch.unique(voxels, dim=0, return_inverse=True)

    order = torch.argsort(group)
    group, feats, frames = group[order], feats[order], frames[order]
    boundaries = torch.nonzero(group[1:] != group[:-1]).flatten() + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device), boundaries])
    ends = torch.cat([boundaries, torch.tensor([len(group)], device=device)])

    sims, pairs = [], 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s < 2 or pairs >= max_pairs:
            continue
        f, fr = feats[s:e], frames[s:e]
        cross = fr.unsqueeze(1) != fr.unsqueeze(0)
        if not cross.any():
            continue
        sim = f @ f.t()
        sims.append(sim[torch.triu(cross, diagonal=1)])
        pairs += int(sims[-1].numel())
    if not sims:
        return torch.zeros((), device=device)
    return torch.cat(sims).mean()


def geometry_probe_r2(
    features: torch.Tensor,
    points: torch.Tensor,
    valid: torch.Tensor,
    ridge: float = 1e-3,
    train_frac: float = 0.7,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """R^2 of a held-out closed-form ridge probe from features to world XYZ."""
    x = features.view(-1, features.shape[-1]).float()[valid.view(-1)]
    y = points.view(-1, 3).float()[valid.view(-1)]
    n = x.shape[0]
    if n < 16:
        return torch.zeros((), device=x.device)

    y = (y - y.mean(0, keepdim=True)) / y.std(0, keepdim=True).clamp_min(1e-6)
    x = torch.cat([F.normalize(x, dim=-1), torch.ones(n, 1, device=x.device)], dim=-1)

    perm = torch.randperm(n, device=x.device, generator=generator)
    cut = max(int(n * train_frac), 8)
    tr, te = perm[:cut], perm[cut:]
    if te.numel() < 4:
        return torch.zeros((), device=x.device)

    xtr, ytr = x[tr], y[tr]
    gram = xtr.t() @ xtr + ridge * xtr.shape[0] * torch.eye(x.shape[1], device=x.device)
    w = torch.linalg.solve(gram, xtr.t() @ ytr)

    resid = ((x[te] @ w - y[te]) ** 2).sum()
    total = ((y[te] - y[te].mean(0, keepdim=True)) ** 2).sum().clamp_min(1e-12)
    return 1.0 - resid / total


@torch.no_grad()
def feature_gap_report(
    student: torch.Tensor,
    teacher: torch.Tensor,
    points: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    max_tokens: int = 4096,
    knn: int = 10,
    voxel_size: float = 0.2,
) -> Dict[str, float]:
    """All of the above for one scene. Student/teacher are (N, C) and (N, D)."""
    n = student.shape[0]
    if n > max_tokens:
        idx = torch.randperm(n, device=student.device)[:max_tokens]
        sub_student, sub_teacher = student[idx], teacher[idx]
    else:
        sub_student, sub_teacher = student, teacher

    report = {
        "cosine": F.cosine_similarity(
            sub_student.float(), sub_teacher.float(), dim=-1
        ).mean().item(),
        "linear_cka": linear_cka(sub_student, sub_teacher).item(),
        "procrustes": procrustes_distance(sub_student, sub_teacher).item(),
        "mutual_knn": mutual_knn_alignment(sub_student, sub_teacher, k=knn).item(),
    }
    if points is not None and valid is not None:
        V, P, _ = points.shape
        for name, feats in (("student", student), ("teacher", teacher)):
            report[f"corr_score_{name}"] = multiview_correspondence_score(
                feats.view(V * P, -1), points, valid, voxel_size=voxel_size
            ).item()
            report[f"probe_r2_{name}"] = geometry_probe_r2(
                feats.view(V * P, -1), points, valid
            ).item()
    return report
