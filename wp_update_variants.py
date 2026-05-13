from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F


@dataclass
class CoordAdaptiveStats:
    mode: str
    loss_base: float
    tol: float
    num_coords_used: int
    num_coords_total: int
    num_escalated: int
    num_failed: int
    m_hist_all: Dict[int, int]
    m_hist_success: Dict[int, int]
    m_hist_failed: Dict[int, int]
    per_param: Dict[str, dict]


@dataclass
class FlatMeta:
    name: str
    shape: torch.Size
    start: int
    end: int


def build_flat_metas_from_named(
    named_params: List[Tuple[str, torch.nn.Parameter]],
) -> Tuple[List[FlatMeta], int, List[str]]:
    metas: List[FlatMeta] = []
    keys: List[str] = []
    offset = 0
    for name, p in named_params:
        n = p.numel()
        metas.append(FlatMeta(name=name, shape=p.shape, start=offset, end=offset + n))
        keys.append(name)
        offset += n
    return metas, offset, keys


def make_empty_param_stats(num_coords_total: int) -> dict:
    return {
        "num_coords_used": 0,
        "num_coords_total": int(num_coords_total),
        "num_escalated": 0,
        "num_failed": 0,
        "m_hist_all": {},
        "m_hist_success": {},
        "m_hist_failed": {},
    }


def bump_hist(hist: Dict[int, int], key: int) -> None:
    hist[int(key)] = hist.get(int(key), 0) + 1


def update_stats_for_coordinate(stats_dict: dict, chosen_m: int, failed: bool) -> None:
    """
    Update one stats dict (global or per-param) after one coordinate estimate.
    """
    stats_dict["num_coords_used"] += 1
    bump_hist(stats_dict["m_hist_all"], chosen_m)

    if failed:
        stats_dict["num_failed"] += 1
        bump_hist(stats_dict["m_hist_failed"], chosen_m)
    else:
        bump_hist(stats_dict["m_hist_success"], chosen_m)
        if chosen_m > 1:
            stats_dict["num_escalated"] += 1


@torch.no_grad()
def adaptive_two_sided_coordinate_fd(
    model: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    param: torch.nn.Parameter,
    flat_idx: int,
    h: float,
    adaptive: bool = True,
    adaptive_max_mult: int = 4,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-8,
    loss_base: Optional[float] = None,
) -> Tuple[float, int, bool]:
    """
    Estimate one coordinate derivative using two-sided finite differences.

    Returns:
        grad_estimate, chosen_multiplier_m, failed_flag
    """
    if loss_base is None:
        logits_base = model.forward_logits(xb, record=False)
        loss_base = float(F.cross_entropy(logits_base, yb).item())

    tol = max(float(abs_tol), float(rel_tol) * abs(float(loss_base)))

    pflat = param.reshape(-1)
    original = float(pflat[flat_idx].item())

    max_m = int(adaptive_max_mult) if adaptive else 1
    chosen_m = 1
    diff = 0.0

    for m in range(1, max_m + 1):
        step = float(m) * float(h)

        pflat[flat_idx] = original + step
        loss_plus = float(F.cross_entropy(model.forward_logits(xb, record=False), yb).item())

        pflat[flat_idx] = original - step
        loss_minus = float(F.cross_entropy(model.forward_logits(xb, record=False), yb).item())

        pflat[flat_idx] = original

        diff = loss_plus - loss_minus
        chosen_m = m

        if abs(diff) > tol:
            break

    grad_est = diff / (2.0 * float(chosen_m) * float(h))
    failed = abs(diff) <= tol
    return grad_est, chosen_m, failed


@torch.no_grad()
def estimate_wp_coordwise_full(
    model_wp: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    include_fn: Callable[[str], bool],
    h: float,
    adaptive: bool = True,
    adaptive_max_mult: int = 4,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-8,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float, List[str], dict]:
    """
    Full coordinate-wise two-sided WP estimator.
    Perturbs every included scalar parameter.
    """
    device = xb.device
    named = [(n, p) for (n, p) in model_wp.named_parameters() if include_fn(n)]
    keys = [n for (n, _) in named]

    if len(named) == 0:
        empty = torch.empty(0, device=device)
        stats = CoordAdaptiveStats(
            mode="full",
            loss_base=float("nan"),
            tol=float("nan"),
            num_coords_used=0,
            num_coords_total=0,
            num_escalated=0,
            num_failed=0,
            m_hist_all={},
            m_hist_success={},
            m_hist_failed={},
            per_param={},
        )
        return {}, empty, float("nan"), [], asdict(stats)

    logits_base = model_wp.forward_logits(xb, record=False)
    loss_base = float(F.cross_entropy(logits_base, yb).item())
    tol = max(float(abs_tol), float(rel_tol) * abs(loss_base))

    metas, total_len, _ = build_flat_metas_from_named(named)
    dflat = torch.zeros(total_len, device=device)
    ddict: Dict[str, torch.Tensor] = {}

    global_stats = {
        "num_coords_used": 0,
        "num_coords_total": sum(int(p.numel()) for _, p in named),
        "num_escalated": 0,
        "num_failed": 0,
        "m_hist_all": {},
        "m_hist_success": {},
        "m_hist_failed": {},
    }

    per_param_stats: Dict[str, dict] = {}

    for (name, p), meta in zip(named, metas):
        g = torch.zeros_like(p).reshape(-1)

        param_stats = make_empty_param_stats(num_coords_total=int(p.numel()))

        for j in range(p.numel()):
            grad_est, chosen_m, failed = adaptive_two_sided_coordinate_fd(
                model=model_wp,
                xb=xb,
                yb=yb,
                param=p,
                flat_idx=j,
                h=h,
                adaptive=adaptive,
                adaptive_max_mult=adaptive_max_mult,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                loss_base=loss_base,
            )

            g[j] = grad_est

            update_stats_for_coordinate(global_stats, chosen_m, failed)
            update_stats_for_coordinate(param_stats, chosen_m, failed)

        per_param_stats[name] = param_stats

        upd = (-g.view_as(p)).clone()
        ddict[name] = upd
        dflat[meta.start:meta.end] = upd.reshape(-1)

    stats = CoordAdaptiveStats(
        mode="full",
        loss_base=loss_base,
        tol=tol,
        num_coords_used=global_stats["num_coords_used"],
        num_coords_total=global_stats["num_coords_total"],
        num_escalated=global_stats["num_escalated"],
        num_failed=global_stats["num_failed"],
        m_hist_all=global_stats["m_hist_all"],
        m_hist_success=global_stats["m_hist_success"],
        m_hist_failed=global_stats["m_hist_failed"],
        per_param=per_param_stats,
    )

    return ddict, dflat, loss_base, keys, asdict(stats)


@torch.no_grad()
def estimate_wp_coordwise_sampled(
    model_wp: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    include_fn: Callable[[str], bool],
    h: float,
    max_coords: int,
    adaptive: bool = True,
    adaptive_max_mult: int = 4,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-8,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float, List[str], dict]:
    """
    Sampled coordinate-wise two-sided WP estimator.
    Perturbs only a random subset of included scalar parameters.
    Non-sampled coordinates receive zero update.
    """
    device = xb.device
    named = [(n, p) for (n, p) in model_wp.named_parameters() if include_fn(n)]
    keys = [n for (n, _) in named]

    if len(named) == 0:
        empty = torch.empty(0, device=device)
        stats = CoordAdaptiveStats(
            mode="sampled",
            loss_base=float("nan"),
            tol=float("nan"),
            num_coords_used=0,
            num_coords_total=0,
            num_escalated=0,
            num_failed=0,
            m_hist_all={},
            m_hist_success={},
            m_hist_failed={},
            per_param={},
        )
        return {}, empty, float("nan"), [], asdict(stats)

    logits_base = model_wp.forward_logits(xb, record=False)
    loss_base = float(F.cross_entropy(logits_base, yb).item())
    tol = max(float(abs_tol), float(rel_tol) * abs(loss_base))

    metas, total_len, _ = build_flat_metas_from_named(named)
    dflat = torch.zeros(total_len, device=device)
    ddict: Dict[str, torch.Tensor] = {}

    total_available = sum(int(p.numel()) for _, p in named)
    budget = min(int(max_coords), total_available)
    per_param = max(1, budget // max(1, len(named)))

    selected_per_param: Dict[str, torch.Tensor] = {}
    used = 0

    for name, p in named:
        n = int(p.numel())
        k = min(per_param, n)
        idx = torch.randperm(n, device=device)[:k]
        selected_per_param[name] = idx
        used += int(idx.numel())

    remaining = max(0, budget - used)
    if remaining > 0:
        for name, p in named:
            if remaining <= 0:
                break
            n = int(p.numel())
            chosen = selected_per_param[name]
            if int(chosen.numel()) >= n:
                continue

            chosen_mask = torch.zeros(n, device=device, dtype=torch.bool)
            chosen_mask[chosen] = True
            leftover = (~chosen_mask).nonzero(as_tuple=False).flatten()

            take = min(int(remaining), int(leftover.numel()))
            if take > 0:
                extra = leftover[torch.randperm(int(leftover.numel()), device=device)[:take]]
                selected_per_param[name] = torch.cat([chosen, extra], dim=0)
                remaining -= take

    global_stats = {
        "num_coords_used": 0,
        "num_coords_total": total_available,
        "num_escalated": 0,
        "num_failed": 0,
        "m_hist_all": {},
        "m_hist_success": {},
        "m_hist_failed": {},
    }

    per_param_stats: Dict[str, dict] = {}

    for (name, p), meta in zip(named, metas):
        g = torch.zeros_like(p).reshape(-1)
        idx_flat = selected_per_param[name]
        param_stats = make_empty_param_stats(num_coords_total=int(p.numel()))
        param_stats["used_idx"] = [int(x) for x in idx_flat.detach().cpu().tolist()]
        for j in idx_flat.tolist():
            grad_est, chosen_m, failed = adaptive_two_sided_coordinate_fd(
                model=model_wp,
                xb=xb,
                yb=yb,
                param=p,
                flat_idx=j,
                h=h,
                adaptive=adaptive,
                adaptive_max_mult=adaptive_max_mult,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                loss_base=loss_base,
            )

            g[j] = grad_est

            update_stats_for_coordinate(global_stats, chosen_m, failed)
            update_stats_for_coordinate(param_stats, chosen_m, failed)

        per_param_stats[name] = param_stats

        upd = (-g.view_as(p)).clone()
        ddict[name] = upd
        dflat[meta.start:meta.end] = upd.reshape(-1)

    stats = CoordAdaptiveStats(
        mode="sampled",
        loss_base=loss_base,
        tol=tol,
        num_coords_used=global_stats["num_coords_used"],
        num_coords_total=global_stats["num_coords_total"],
        num_escalated=global_stats["num_escalated"],
        num_failed=global_stats["num_failed"],
        m_hist_all=global_stats["m_hist_all"],
        m_hist_success=global_stats["m_hist_success"],
        m_hist_failed=global_stats["m_hist_failed"],
        per_param=per_param_stats,
    )

    return ddict, dflat, loss_base, keys, asdict(stats)


def estimate_wp_coordwise_adaptive(
    model_wp: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    include_fn: Callable[[str], bool],
    h: float,
    mode: str = "sampled",  # "sampled" or "full"
    max_coords: int = 2000,
    adaptive: bool = True,
    adaptive_max_mult: int = 4,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-8,
):
    if mode == "sampled":
        return estimate_wp_coordwise_sampled(
            model_wp=model_wp,
            xb=xb,
            yb=yb,
            include_fn=include_fn,
            h=h,
            max_coords=max_coords,
            adaptive=adaptive,
            adaptive_max_mult=adaptive_max_mult,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )

    if mode == "full":
        return estimate_wp_coordwise_full(
            model_wp=model_wp,
            xb=xb,
            yb=yb,
            include_fn=include_fn,
            h=h,
            adaptive=adaptive,
            adaptive_max_mult=adaptive_max_mult,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )

    raise ValueError(f"unknown mode: {mode}")