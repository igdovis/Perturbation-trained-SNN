import argparse
import json
from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from snn_models import DualSNN
from data_utils import get_dataset
from wp_sgd_metrics import cosine_similarity_wp_sgd_orthogonal, plot_per_layer_cosine
from wp_sgd_metrics import compute_all_metrics, per_layer_cosines, analyze_zero_gradients
from analysis_firing import measure_firing_stats
from wp_update_variants import estimate_wp_coordwise_adaptive
from experiment import (
    analyze_spiking_activity,
    analyze_surrogate_for_nonfiring,
    plot_spiking_activity_violin,
    plot_membrane_voltages,
    plot_surrogate_gradient_function,
    plot_zero_gradient_analysis,
    plot_nonfiring_gradient_analysis
)
from wp_adaptive_plots import plot_wp_adaptive_suite
from train_snn import estimateWpUpdateDirection

def compute_sgd_probe_direction(model, xb, yb, include_fn=None, analyze_details=False):
    device = xb.device

    named_wp = list(model.wp.named_parameters())
    named_sgd = list(model.sgd.named_parameters())

    def included(name):
        if include_fn is not None:
            return include_fn(name)
        return True

    # sync sgd branch to wp branch
    with torch.no_grad():
        for (_, ps), (_, pw) in zip(named_sgd, named_wp):
            ps.copy_(pw)

    for _, ps in named_sgd:
        if ps.grad is not None:
            ps.grad = None

    with torch.enable_grad():
        if analyze_details:
            logits, traces = model.forward_sgd(xb, record=True)
        else:
            logits = model.forward_sgd(xb, record=False)
            traces = None
        loss = F.cross_entropy(logits, yb)
        loss.backward()

    dSGD_dict = {}
    for name, p in named_sgd:
        if included(name):
            if p.grad is not None:
                dSGD_dict[name] = -p.grad.clone()
            else:
                dSGD_dict[name] = torch.zeros_like(p)

    return dSGD_dict, traces

def build_probe_result(dWP_dict, dSGD_dict, device, traces=None, wp_stats=None):
    w_cos, b_cos, comb_cos = per_layer_cosines(dWP_dict, dSGD_dict, exclude_zeros=False)
    w_cos_wp, b_cos_wp, comb_cos_wp = per_layer_cosines(dWP_dict, dSGD_dict, exclude_zeros="wp")
    w_cos_both, b_cos_both, comb_cos_both = per_layer_cosines(dWP_dict, dSGD_dict, exclude_zeros="both")

    dWP_chunks = [dWP_dict[n].reshape(-1) for n in dWP_dict.keys()]
    dSGD_chunks = [dSGD_dict[n].reshape(-1) for n in dSGD_dict.keys()]
    dWP_vec = torch.cat(dWP_chunks) if dWP_chunks else torch.empty(0, device=device)
    dSGD_vec = torch.cat(dSGD_chunks) if dSGD_chunks else torch.empty(0, device=device)

    metrics = compute_all_metrics(dWP_vec, dSGD_vec, exclude_zeros=False)
    metrics_both = compute_all_metrics(dWP_vec, dSGD_vec, exclude_zeros="both")
    metrics_wp = compute_all_metrics(dWP_vec, dSGD_vec, exclude_zeros="wp")
    zero_analysis = analyze_zero_gradients(dWP_dict, dSGD_dict)

    result = {
        "global_metrics": metrics,
        "global_metrics_both": metrics_both,
        "global_metrics_wp": metrics_wp,
        "per_param_zero_analysis": zero_analysis,
        "dWP_norm": float(dWP_vec.norm()),
        "dSGD_norm": float(dSGD_vec.norm()),
        "w_cos": w_cos,
        "b_cos": b_cos,
        "comb_cos": comb_cos,
        "w_cos_wp": w_cos_wp,
        "b_cos_wp": b_cos_wp,
        "comb_cos_wp": comb_cos_wp,
        "w_cos_both": w_cos_both,
        "b_cos_both": b_cos_both,
        "comb_cos_both": comb_cos_both,
        "dWP_dict": dWP_dict,
        "dSGD_dict": dSGD_dict,
    }

    if traces is not None:
        result["traces"] = traces
    if wp_stats is not None:
        result["wpStats"] = wp_stats

    return result

def probe_wp_vs_sgd(model, xb, yb, h, args, device, include_fn, analyze_details=False):
    xb = xb.to(device=device, dtype=torch.float32)
    yb = torch.as_tensor(yb, device=device)
    if yb.dtype != torch.long:
        yb = (yb.argmax(dim=-1) if yb.ndim > 1 else yb).long()

    if args.wpProbeEstimator == "fixed":
        return cosine_similarity_wp_sgd_orthogonal(
            model, xb, yb,
            h=h,
            include_layers=None,
            device=device,
            analyze_details=analyze_details,
            include_bias=args.include_bias,
        )

    if args.wpProbeEstimator == "random":
        include_noise_fn = include_fn if args.wpNoiseScope == "diag" else (lambda _name: True)

        dWP_dict, _, _, keys = estimateWpUpdateDirection(
            modelWp=model.wp,
            xb=xb,
            yb=yb,
            includeReturnFn=include_fn,
            includeNoiseFn=include_noise_fn,
            h=float(h),
            kSamples=int(args.wpK),
            noise=str(args.wpNoise),
            sampling=str(args.wpSampling),
        )

        wp_stats = {
            "mode": "random",
            "loss_base": np.nan,
            "tol": np.nan,
            "num_coords_used": np.nan,
            "num_coords_total": np.nan,
            "num_escalated": np.nan,
            "num_failed": np.nan,
            "m_hist_all": {},
            "m_hist_success": {},
            "m_hist_failed": {},
            "per_param": {},
            "k": int(args.wpK),
            "noise": str(args.wpNoise),
            "sampling": str(args.wpSampling),
            "noise_scope": str(args.wpNoiseScope),
        }

    else:
        dWP_dict, _, _, keys, wp_stats = estimate_wp_coordwise_adaptive(
            model_wp=model.wp,
            xb=xb,
            yb=yb,
            include_fn=include_fn,
            h=float(h),
            mode="sampled" if args.wpProbeEstimator == "coord_sampled" else "full",
            max_coords=int(args.wpCoordMaxCoords),
            adaptive=bool(args.wpCoordAdaptive),
            adaptive_max_mult=int(args.wpCoordAdaptiveMaxMult),
            abs_tol=float(args.wpCoordAbsTol),
            rel_tol=float(args.wpCoordRelTol),
        )

    dSGD_dict, traces = compute_sgd_probe_direction(
        model=model,
        xb=xb,
        yb=yb,
        include_fn=include_fn,
        analyze_details=analyze_details,
    )

    # keep same param order between wp and sgd
    dWP_dict = {k: dWP_dict[k] for k in keys if k in dWP_dict and k in dSGD_dict}
    dSGD_dict = {k: dSGD_dict[k] for k in dWP_dict.keys()}

    return build_probe_result(
        dWP_dict=dWP_dict,
        dSGD_dict=dSGD_dict,
        device=device,
        traces=traces,
        wp_stats=wp_stats,
    )

def parse_args():
    p = argparse.ArgumentParser()
    # wp probe mode
    p.add_argument("--wpProbeEstimator", type=str, default="fixed", choices=["random", "fixed", "coord_sampled", "coord_full"])
    # for random aka standard WP estimator
    p.add_argument("--wpK", type=int, default=16)
    p.add_argument("--wpNoise", type=str, default="gaussian", choices=["rademacher", "gaussian"])
    p.add_argument("--wpSampling", type=str, default="two_sided", choices=["two_sided", "orthogonal"])
    p.add_argument(
        "--wpNoiseScope",
        type=str,
        default="diag",
        choices=["diag", "all"],
        help="diag = perturb only included diagnostic parameters, all = perturb full parameter space",
    )
    # adaptive coordinate-wise probing options
    p.add_argument("--wpCoordMaxCoords", type=int, default=2000)
    p.add_argument("--wpCoordAdaptive", action="store_true")
    p.add_argument("--wpCoordAdaptiveMaxMult", type=int, default=4)
    p.add_argument("--wpCoordAbsTol", type=float, default=1e-12)
    p.add_argument("--wpCoordRelTol", type=float, default=1e-8)

    p.add_argument("--dataset", type=str, default="randman", choices=["randman", "shd"])
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--nb_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--dt", type=float, default=1000)
    
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--experiment_name", type=str, default=None)

    # network
    p.add_argument("--depth_min", type=int, default=1)
    p.add_argument("--depth_max", type=int, default=3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.95)

    # surrogate 
    p.add_argument("--surrogate", type=str, default="fast_sigmoid",
                   choices=["fast_sigmoid", "sigmoid", "atan", "triangular"])
    p.add_argument("--slope", type=float, default=25.0)
    # sweep knobs
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--eq31_alphas", nargs="+", type=float, default=[1.0, 1.75, 2.0, 2.5])
    p.add_argument("--eq31", action="store_true")

    # firing measurement
    p.add_argument("--firing_batches", type=int, default=3)
    p.add_argument("--near_delta", type=float, default=0.1)
    p.add_argument("--rare_k", type=int, default=2)


    # post update loss prediction experiment args
    p.add_argument("--run_post_update_prediction", action="store_true")
    p.add_argument("--postUpdateDirections", nargs="+", type=str, default=["wp", "sg", "noise"], choices=["sg", "wp", "noise"]) 
    p.add_argument("--postUpdateStepModes", nargs="+", type=str, default=["unit", "relative_param_norm"], choices=["raw", "unit", "relative_param_norm"])
    p.add_argument("--postUpdateStepSizes", nargs="+", type=float, default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    p.add_argument("--postUpdateMaxBatchesPerRegime", type=int, default=2)

    # shortlist criteria 
    p.add_argument("--shortlist_min_frac_firing", type=float, default=0.1)
    p.add_argument("--shortlist_max_frac_firing", type=float, default=0.9)
    p.add_argument("--shortlist_min_mean_rate", type=float, default=1e-4)
    p.add_argument("--shortlist_max_layer_std", type=float, default=0.05)

    p.add_argument("--candidate_n_promising", type=int, default=4)
    p.add_argument("--candidate_n_borderline", type=int, default=4)
    p.add_argument("--candidate_n_bad", type=int, default=4)

    # cosine
    p.add_argument("--h_values", nargs="+", type=float, default=[0.005, 0.01, 0.03])
    p.add_argument("--cosine_batches", type=int, default=2)
    p.add_argument("--include_bias", action="store_true")

    # plots
    p.add_argument("--heatmap_depth", type=int, default=2) #? no

    #diagnostics
    p.add_argument("--diagLayerIdxs", nargs="+", type=int, default=None)
    p.add_argument("--diagLastN", type=int, default=0)
    p.add_argument("--diagBiasOnly", action="store_true")
    p.add_argument("--diagBiasLastN", type=int, default=0)
    p.add_argument("--diagScope", type=str, default="all", choices=["all", "subset"])

    return p.parse_args()

def setup(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    out = Path(args.output_dir) / args.experiment_name
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)


    with open(out / "config.json", "w") as f: json.dump(vars(args), f, indent=2)

    return device, out

from snntorch import surrogate
def get_surrogate_function(name, slope):
    if name == "fast_sigmoid":
        return surrogate.fast_sigmoid(slope=slope)
    if name == "sigmoid":
        return surrogate.sigmoid(slope=slope)
    if name == "atan":
        return surrogate.atan(alpha=slope)
    if name == "triangular":
        return surrogate.triangular(threshold=slope)
    raise ValueError(name)

def plot_scatter_firing_vs_cosine(df, out_dir):
    # x = meanRateAvg, y = cosine_wp 
    fig, ax = plt.subplots(figsize=(8, 6))
    for d in sorted(df["depth"].unique()):
        sub = df[df["depth"] == d]
        ax.scatter(sub["meanRateAvg"], sub["cosine_wp"], label=f"depth={d}", alpha=0.8)

    ax.set_xlabel("mean spike rate avg (across layers)")
    ax.set_ylabel("cosine similarity (wp non-zero only)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "scatter_firing_vs_cosine.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_heatmap_thr_alpha(df, out_dir, depth, value_col, title, fname):
    sub = df[df["depth"] == depth].copy()
    if len(sub) == 0:
        return

    piv = sub.pivot_table(
        index="threshold",
        columns="eq31_alpha",
        values=value_col,
        aggfunc="mean"
    )

    vals = piv.values
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(vals, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_xticklabels([str(c) for c in piv.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(r) for r in piv.index])

    ax.set_xlabel("eq31 alpha")
    ax.set_ylabel("threshold")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=value_col)

    plt.tight_layout()
    plt.savefig(out_dir / "plots" / fname, dpi=150, bbox_inches="tight")
    plt.close()

def make_regime_dir(out_dir, depth, thr, alpha):
    tag = f"d{depth}_thr{thr:.3f}_a{alpha:.3f}".replace(".", "p")
    reg = out_dir / "regimes" / tag
    (reg / "plots").mkdir(parents=True, exist_ok=True)
    (reg / "analysis").mkdir(parents=True, exist_ok=True)
    (reg / "data").mkdir(parents=True, exist_ok=True)
    return reg, tag

def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def summarize_zero_analysis(z):
    out = {
        "wp_zero_frac": np.nan,
        "sgd_zero_frac": np.nan,
        "both_zero_frac": np.nan,
        "total_elems": np.nan,
    }
    if not isinstance(z, dict):
        return out

    total = z.get("total_elements", z.get("total_elems", z.get("total", None)))
    if total is not None:
        out["total_elems"] = to_float(total)
        if total > 0:
            if "wp_zero" in z:
                out["wp_zero_frac"] = to_float(z["wp_zero"]) / total
            elif "wp_zero_fraction" in z:
                out["wp_zero_frac"] = to_float(z["wp_zero_fraction"])

            if "sgd_zero" in z:
                out["sgd_zero_frac"] = to_float(z["sgd_zero"]) / total
            elif "sgd_zero_fraction" in z:
                out["sgd_zero_frac"] = to_float(z["sgd_zero_fraction"])

            if "both_zero" in z:
                out["both_zero_frac"] = to_float(z["both_zero"]) / total
            elif "both_zero_fraction" in z:
                out["both_zero_frac"] = to_float(z["both_zero_fraction"])
        return out

    tot = 0.0
    wp0 = 0.0
    sgd0 = 0.0
    both0 = 0.0
    for v in z.values():
        if not isinstance(v, dict):
            continue

        n = v.get("n", v.get("total", v.get("total_elements", None)))
        if n is None:
            continue
        n = float(n)
        if n <= 0:
            continue

        tot += n

        if "wp_zero_count" in v:
            wp0 += float(v["wp_zero_count"])
        elif "wp_zero" in v:
            wp0 += float(v["wp_zero"])
        elif "wp_zero_fraction" in v:
            wp0 += float(v["wp_zero_fraction"]) * n

        if "sgd_zero_count" in v:
            sgd0 += float(v["sgd_zero_count"])
        elif "sgd_zero" in v:
            sgd0 += float(v["sgd_zero"])
        elif "sgd_zero_fraction" in v:
            sgd0 += float(v["sgd_zero_fraction"]) * n

        if "both_zero_count" in v:
            both0 += float(v["both_zero_count"])
        elif "both_zero" in v:
            both0 += float(v["both_zero"])
        elif "both_zero_fraction" in v:
            both0 += float(v["both_zero_fraction"]) * n

    if tot > 0:
        out["total_elems"] = tot
        out["wp_zero_frac"] = wp0 / tot
        out["sgd_zero_frac"] = sgd0 / tot
        out["both_zero_frac"] = both0 / tot
    return out


def build_candidate_shortlist(firing_df: pd.DataFrame, args) -> pd.DataFrame:
    df = firing_df.copy()
    # loose viability gate
    df["passes_min_frac"] = df["fracFiringMin"] >= args.shortlist_min_frac_firing
    df["passes_min_rate"] = df["meanRateAvg"] >= args.shortlist_min_mean_rate
    df["passes_max_std"] = df["meanRateStdAcrossLayers"] <= args.shortlist_max_layer_std
    df["passes_max_frac"] = df["fracFiringMin"] <= args.shortlist_max_frac_firing

    # interpret it only as a candidate ranking score
    df["candidate_score"] = (-df["meanRateStdAcrossLayers"]) + (0.2 * df["nearThrAvg"])

    # categorize regimes
    conditions = []

    too_silent = (
        df["fracFiringMin"] < args.shortlist_min_frac_firing
    ) | (
        df["meanRateAvg"] < args.shortlist_min_mean_rate
    )

    too_unstable = (
        df["meanRateStdAcrossLayers"] > args.shortlist_max_layer_std
    )

    too_saturated = (
        df["fracFiringAvg"] > args.shortlist_max_frac_firing
    )
    
    # promising: not silent, not too unstable, not saturated
    promising_mask = (
        (df["fracFiringMin"] >= 0.10)
        & (df["meanRateAvg"] >= 1e-4)
        & (df["meanRateAvg"] <= 0.30)
        & (df["fracFiringAvg"] <= 0.95)
        & (df["meanRateStdAcrossLayers"] <= 0.05)
        & (df["nearThrAvg"] >= 0.02)
    )

    bad_silent = (
        (df["fracFiringMin"] < 0.02)
        | (df["meanRateAvg"] < 1e-5)
    )

    bad_saturated = (
        (df["meanRateAvg"] > 0.40)
        | (df["fracFiringAvg"] > 0.98)
    )

    bad_unstable = (
        df["meanRateStdAcrossLayers"] > 0.12
    )

    bad_mask = bad_silent | bad_saturated | bad_unstable

    borderline_mask = ~(promising_mask | bad_mask)

    df["candidate_category"] = "uncategorized"
    df.loc[promising_mask, "candidate_category"] = "promising"
    df.loc[borderline_mask, "candidate_category"] = "borderline"
    df.loc[bad_mask, "candidate_category"] = "bad_control"

    # rate bins inside each category to still get diversity
    unique_rates = df["meanRateAvg"].nunique()
    if unique_rates >= 2:
        q = min(3, unique_rates)
        df["rateBin"] = pd.qcut(df["meanRateAvg"], q=q, duplicates="drop")
    else:
        df["rateBin"] = "all"

    def pick_from_category(sub_df: pd.DataFrame, n_pick: int) -> pd.DataFrame:
        if len(sub_df) == 0 or n_pick <= 0:
            return sub_df.head(0).copy()

        n_groups = sub_df.groupby(["depth", "rateBin"]).ngroups
        per_group = max(1, n_pick // max(1, n_groups))

        picked = (
            sub_df.sort_values("candidate_score", ascending=False)
                  .groupby(["depth", "rateBin"], as_index=False)
                  .head(per_group)
        )

        if len(picked) < n_pick:
            remaining = (
                sub_df.sort_values("candidate_score", ascending=False)
                      .loc[~sub_df.index.isin(picked.index)]
                      .head(n_pick - len(picked))
            )
            picked = pd.concat([picked, remaining], ignore_index=False)

        return picked.head(n_pick).copy()

    promising = pick_from_category(
        df[df["candidate_category"] == "promising"],
        args.candidate_n_promising,
    )
    borderline = pick_from_category(
        df[df["candidate_category"] == "borderline"],
        args.candidate_n_borderline,
    )
    bad = pick_from_category(
        df[df["candidate_category"] == "bad_control"],
        args.candidate_n_bad,
    )

    candidates = pd.concat([promising, borderline, bad], ignore_index=False)
    candidate_topk = args.candidate_n_promising + args.candidate_n_borderline + args.candidate_n_bad
    # fallback if categories are sparse
    if len(candidates) < candidate_topk:
        remaining = (
            df.sort_values("candidate_score", ascending=False)
              .loc[~df.index.isin(candidates.index)]
              .head(candidate_topk- len(candidates))
        )
        candidates = pd.concat([candidates, remaining], ignore_index=False)

    candidates = (
        candidates
        .drop_duplicates(subset=["depth", "threshold", "eq31_alpha"])
        .sort_values(["candidate_category", "candidate_score"], ascending=[True, False])
        .head(candidate_topk)
        .copy()
    )

    return candidates

def compute_near_thr_mass_from_traces(traces, thr, delta):
    # fraction of mem values within [thr-delta, thr+delta], averaged across layers
    if not isinstance(traces, dict) or traces.get("mem", None) is None:
        return np.nan
    mem_list = traces["mem"]
    if not isinstance(mem_list, (list, tuple)) or len(mem_list) == 0:
        return np.nan

    vals = []
    for mem in mem_list:
        if not torch.is_tensor(mem):
            continue
        m = mem.detach()
        mask = (m >= (thr - delta)) & (m <= (thr + delta))
        vals.append(mask.float().mean().item())
    return float(np.mean(vals)) if len(vals) else np.nan

def summarize_spike_derivative_stats(stats_list):
    # stats_list = list[dict] (per layer)
    if not isinstance(stats_list, list) or len(stats_list) == 0:
        return {"spike_deriv_nonzero_frac_avg": np.nan}
    key = "nonzero_frac"
    vals = [s.get(key) for s in stats_list if isinstance(s, dict) and s.get(key) is not None]
    return {"spike_deriv_nonzero_frac_avg": float(np.mean(vals))} if vals else {"spike_deriv_nonzero_frac_avg": np.nan}


def plot_findings(diag_df, out_dir):
    if diag_df is None or len(diag_df) == 0:
        return

    # cosine_wp vs meanRateAvg, colored by stability
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(diag_df["meanRateAvg"], diag_df["cosine_wp"], c=diag_df["meanRateStdAcrossLayers"])
    ax.set_xlabel("meanRateAvg")
    ax.set_ylabel("cosine_wp")
    ax.set_title("cosine_wp vs firing rate (color=layer std)")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="meanRateStdAcrossLayers")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "finding_cosine_vs_rate_colored_by_std.png", dpi=150, bbox_inches="tight")
    plt.close()

    # cosine_wp vs nearThrMass
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(diag_df["nearThrMass"], diag_df["cosine_wp"])
    ax.set_xlabel("nearThrMass (from mem)")
    ax.set_ylabel("cosine_wp")
    ax.set_title("cosine_wp vs near-threshold membrane mass")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "finding_cosine_vs_nearThrMass.png", dpi=150, bbox_inches="tight")
    plt.close()

    # cosine_wp vs both_zero_frac
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(diag_df["both_zero_frac"], diag_df["cosine_wp"])
    ax.set_xlabel("both_zero_frac")
    ax.set_ylabel("cosine_wp")
    ax.set_title("cosine_wp vs fraction of params zero in both")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "finding_cosine_vs_bothZeroFrac.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_global_cosine_bars_from_res(res, reg_out, tag):
    gm = res.get("global_metrics", {})
    gm_wp = res.get("global_metrics_wp", {})
    gm_both = res.get("global_metrics_both", {})

    vals = [
        gm.get("cosine_similarity", np.nan),
        gm_wp.get("cosine_similarity", np.nan),
        gm_both.get("cosine_similarity", np.nan),
    ]
    labels = ["all", "wp-only", "both-nonzero"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals)
    ax.set_ylabel("cosine similarity")
    ax.set_title(f"global cosine _{tag}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(reg_out / "plots" / f"global_cosine_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_cosine_by_candidate_category(diag_df, out_dir):
    if diag_df is None or len(diag_df) == 0:
        return
    if "candidate_category" not in diag_df.columns:
        return

    cats = ["promising", "borderline", "bad_control"]
    data = [
        diag_df.loc[diag_df["candidate_category"] == c, "cosine_wp"].dropna().values
        for c in cats
    ]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.boxplot(data, labels=cats)
    ax.set_ylabel("cosine_wp")
    ax.set_title("cosine_wp by candidate category")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "cosine_by_candidate_category.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_cosine_vs_h_for_regime(cosine_df, reg_out, depth, thr, alpha, tag):
    sub = cosine_df[
        (cosine_df["depth"] == depth) &
        (np.isclose(cosine_df["threshold"], thr)) &
        (np.isclose(cosine_df["eq31_alpha"], alpha))
    ].copy()

    if len(sub) == 0:
        return

    agg = sub.groupby("h", as_index=False).agg(
        cosine_wp_mean=("cosine_wp", "mean"),
        cosine_wp_std=("cosine_wp", "std"),
        cosine_all_mean=("cosine_all", "mean"),
        cosine_all_std=("cosine_all", "std"),
        cosine_both_mean=("cosine_both", "mean"),
        cosine_both_std=("cosine_both", "std"),
    ).sort_values("h")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(agg["h"], agg["cosine_wp_mean"], yerr=agg["cosine_wp_std"], marker="o", capsize=3, label="wp-only")
    ax.errorbar(agg["h"], agg["cosine_all_mean"], yerr=agg["cosine_all_std"], marker="o", capsize=3, label="all")
    ax.errorbar(agg["h"], agg["cosine_both_mean"], yerr=agg["cosine_both_std"], marker="o", capsize=3, label="both")
    ax.set_xlabel("h")
    ax.set_ylabel("cosine similarity")
    ax.set_title(f"cosine vs h _{tag}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(reg_out / "plots" / f"cosine_vs_h_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

_hiddenPat = re.compile(r"(?:^|.*\.)fcs\.(\d+)\.(weight|bias)$")
_outPat = re.compile(r"(?:^|.*\.)fc_out\.(weight|bias)$")

def paramLayerIndex(name: str, depthHidden: int):
    m = _hiddenPat.match(name)
    if m:
        return int(m.group(1))
    if _outPat.match(name):
        return int(depthHidden)
    return None
# function to determine what is included in the cosine computations
# based on the various args for probe scope (trainLayerIdxs, trainLastN, trainBiasOnly, trainBiasLastN, includeBias, includeOutput)
def buildIncludeFn(
    depthHidden: int,
    trainLayerIdxs,
    trainLastN,
    trainBiasOnly,
    trainBiasLastN,
    includeBias,
    includeOutput: bool = True,
):
    maxLayer = depthHidden + (1 if includeOutput else 0)

    def include(name: str) -> bool:
        li = paramLayerIndex(name, depthHidden)
        if li is None:
            return False
        if (not includeOutput) and li == depthHidden:
            return False

        kind = "bias" if name.endswith(".bias") else "weight"
        if (not includeBias) and kind == "bias":
            return False

        if trainBiasOnly and kind != "bias":
            return False

        if trainBiasLastN is not None and trainBiasLastN > 0:
            if kind != "bias":
                return False
            if li < (maxLayer - trainBiasLastN):
                return False

        if trainLayerIdxs is not None and len(trainLayerIdxs) > 0:
            return li in trainLayerIdxs

        if trainLastN is not None and trainLastN > 0:
            return li >= (maxLayer - trainLastN)

        return True

    return include

def loss_on_wp_branch(model, xb, yb, device):
    xb = xb.to(device=device, dtype=torch.float32)
    yb = torch.as_tensor(yb, device=device)
    if yb.dtype != torch.long:
        yb = (yb.argmax(dim=-1) if yb.ndim > 1 else yb).long()
    with torch.no_grad():
        logits = model.wp.forward_logits(xb, record=False)
        loss = F.cross_entropy(logits, yb)
    return float(loss.item())

def direction_norm(direction_dict, keys):
    sq = 0.0
    for k in keys:
        if k in direction_dict:
            sq += float((direction_dict[k].detach() ** 2).sum().item())
    return float(np.sqrt(sq))

def dot_direction(d1, d2, keys):
    total = 0.0
    for k in keys:
        if k in d1 and k in d2:
            total += float((d1[k].detach() * d2[k].detach()).sum().item())
    return total

def param_norm(model_wp, include_fn):
    sq = 0.0
    with torch.no_grad():
        for name, p in model_wp.named_parameters():
            if include_fn(name):
                sq += float((p.detach() ** 2).sum().item())
    return float(np.sqrt(sq))

def make_random_direction_like(reference_dict):
    return {k: torch.randn_like(v) for k, v in reference_dict.items()}    

def make_delta_direction(direction_dict, keys, step_size, step_mode, param_norm_value):
    dir_norm = direction_norm(direction_dict, keys)
    if dir_norm <= 1e-12 or not np.isfinite(dir_norm):
        return None, np.nan, np.nan, np.nan
    if step_mode == "raw":
        scale = float(step_size)
    elif step_mode == "unit": # step size is l2 norm
        scale = float(step_size) / dir_norm
    elif step_mode == "relative_param_norm": # step size is norm(delta theta) / norm(theta) 
        scale = (float(step_size) * float(param_norm_value)) / dir_norm
    else:
        raise ValueError(f"Invalid step mode: {step_mode}")
    
    delta_dict = {k: scale * direction_dict[k] for k in keys if k in direction_dict}
    delta_norm = direction_norm(delta_dict, keys)
    relative_delta_norm = delta_norm / max(float(param_norm_value), 1e-12)
    return delta_dict, float(scale), float(delta_norm), float(relative_delta_norm)

def apply_delta_to_wp(model, delta_dict, sign=1.0):
    with torch.no_grad():
        for name, p in model.wp.named_parameters():
            if name in delta_dict:
                p.add_(float(sign) * delta_dict[name].to(p.device))

def run_post_update_prediction_for_probe(model, xb, yb, res, args, device, include_fn, h, batch_idx):
    """
    Compare whether SG or WP better predict the actual loss.
    dSGD_dict and dWP_dict are update direction vectors.
    For an applied parameter displacement delta_theta, the first-order predicted loss change is:
            pred_delta = - <update_direction, delta_theta>
    """
    if "dWP_dict" not in res or "dSGD_dict" not in res:
        return []
    
    dWP = res["dWP_dict"]
    dSGD = res["dSGD_dict"]
    keys = [k for k in dWP.keys() if k in dSGD]
    if len(keys) == 0:
        return []
    
    loss0 = loss_on_wp_branch(model, xb, yb, device)
    param_norm_value = param_norm(model.wp, include_fn)
    
    directions = {"sg": dSGD, "wp": dWP, "noise": make_random_direction_like(dWP)}
    gm = res.get("global_metrics", {})
    gm_wp = res.get("global_metrics_wp", {})
    gm_both = res.get("global_metrics_both", {})
    wp_stats = res.get("wpStats", {})
    
    rows = []
    for direction_name in args.postUpdateDirections:
        direction_dict = directions[direction_name]
        applied_dir_norm = direction_norm(direction_dict, keys)

        for step_mode in args.postUpdateStepModes:
            for step_size in args.postUpdateStepSizes:
                delta_dict, scale, delta_norm, rel_delta_norm = make_delta_direction(
                    direction_dict=direction_dict,
                    keys=keys,
                    step_size=float(step_size),
                    step_mode=step_mode,
                    param_norm_value=param_norm_value,
                )
                if delta_dict is None:
                    continue

                pred_delta_sg = -dot_direction(dSGD, delta_dict, keys)
                pred_delta_wp = -dot_direction(dWP, delta_dict, keys)
                pred_loss_sg = loss0 + pred_delta_sg
                pred_loss_wp = loss0 + pred_delta_wp

                apply_delta_to_wp(model, delta_dict, sign=+1.0)
                loss_actual = loss_on_wp_branch(model, xb, yb, device)
                apply_delta_to_wp(model, delta_dict, sign=-1.0)

                actual_delta = loss_actual - loss0
                abs_err_sg = abs(pred_loss_sg - loss_actual)
                abs_err_wp = abs(pred_loss_wp - loss_actual)
                sq_err_sg = (pred_loss_sg - loss_actual) ** 2
                sq_err_wp = (pred_loss_wp - loss_actual) ** 2

                rows.append({
                    "h": float(h),
                    "batch": int(batch_idx),
                    "loss0": float(loss0),
                    "directionApplied": direction_name,
                    "stepMode": step_mode,
                    "stepSize": float(step_size),
                    "appliedScale": float(scale),
                    "appliedDirectionNorm": float(applied_dir_norm),
                    "deltaNorm": float(delta_norm),
                    "relativeDeltaNorm": float(rel_delta_norm),
                    "paramNorm": float(param_norm_value),

                    "lossActual": float(loss_actual),
                    "actualDelta": float(actual_delta),

                    "predLossSg": float(pred_loss_sg),
                    "predDeltaSg": float(pred_delta_sg),
                    "absErrSg": float(abs_err_sg),
                    "sqErrSg": float(sq_err_sg),
                    "predImprovesSg": bool(pred_delta_sg < 0.0),
                    "signCorrectSg": bool((pred_delta_sg < 0.0) == (actual_delta < 0.0)),

                    "predLossWp": float(pred_loss_wp),
                    "predDeltaWp": float(pred_delta_wp),
                    "absErrWp": float(abs_err_wp),
                    "sqErrWp": float(sq_err_wp),
                    "predImprovesWp": bool(pred_delta_wp < 0.0),
                    "signCorrectWp": bool((pred_delta_wp < 0.0) == (actual_delta < 0.0)),

                    "betterPredictor": "sg" if abs_err_sg < abs_err_wp else ("wp" if abs_err_wp < abs_err_sg else "tie"),

                    "cosine_all": to_float(gm.get("cosine_similarity", np.nan)),
                    "cosine_wp": to_float(gm_wp.get("cosine_similarity", np.nan)),
                    "cosine_both": to_float(gm_both.get("cosine_similarity", np.nan)),
                    "dWP_norm": to_float(res.get("dWP_norm", np.nan)),
                    "dSGD_norm": to_float(res.get("dSGD_norm", np.nan)),
                    "wpProbeEstimator": args.wpProbeEstimator,
                    "wpNumCoordsUsed": to_float(wp_stats.get("num_coords_used", np.nan)),
                    "wpNumCoordsTotal": to_float(wp_stats.get("num_coords_total", np.nan)),
                    "wpNumEscalated": to_float(wp_stats.get("num_escalated", np.nan)),
                    "wpNumFailed": to_float(wp_stats.get("num_failed", np.nan)),
                })
    return rows


def save_post_update_outputs(post_update_rows, out_dir):
    if len(post_update_rows) == 0:
        return

    df = pd.DataFrame(post_update_rows)
    df.to_csv(out_dir / "data" / "post_update_prediction.csv", index=False)

    group_cols = [
        "dataset", "depth", "threshold", "eq31_alpha", "h",
        "directionApplied", "stepMode", "stepSize",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    summary = (
        df.groupby(group_cols, as_index=False)
          .agg(
              n=("absErrSg", "size"),
              absErrSgMean=("absErrSg", "mean"),
              absErrSgStd=("absErrSg", "std"),
              absErrWpMean=("absErrWp", "mean"),
              absErrWpStd=("absErrWp", "std"),
              signCorrectSgMean=("signCorrectSg", "mean"),
              signCorrectWpMean=("signCorrectWp", "mean"),
              actualDeltaMean=("actualDelta", "mean"),
              cosineWpMean=("cosine_wp", "mean"),
              wpFailedMean=("wpNumFailed", "mean"),
              wpEscalatedMean=("wpNumEscalated", "mean"),
          )
    )
    summary.to_csv(out_dir / "data" / "post_update_prediction_summary.csv", index=False)

    # plot mean abs error vs step size for each direction and step mode
    for step_mode in sorted(df["stepMode"].dropna().unique()):
        for direction_name in sorted(df["directionApplied"].dropna().unique()):
            sub = df[(df["stepMode"] == step_mode) & (df["directionApplied"] == direction_name)]
            if len(sub) == 0:
                continue
            agg = sub.groupby("stepSize", as_index=False).agg(
                absErrSgMean=("absErrSg", "mean"),
                absErrWpMean=("absErrWp", "mean"),
            ).sort_values("stepSize")

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(agg["stepSize"], agg["absErrSgMean"], marker="o", label="sg predictor")
            ax.plot(agg["stepSize"], agg["absErrWpMean"], marker="o", label="wp predictor")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("step size")
            ax.set_ylabel("mean absolute prediction error")
            ax.set_title(f"post-update prediction: {direction_name}, {step_mode}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            fname = f"post_update_abs_error_{direction_name}_{step_mode}.png"
            plt.savefig(out_dir / "plots" / fname, dpi=150, bbox_inches="tight")
            plt.close()

def main():
    args = parse_args()
    device, out_dir = setup(args)

    # dataset / loader
    datasets_list, input_dim, output_dim = get_dataset(args)
    ds_train, ds_valid, ds_test = datasets_list

    loader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    surrogate_fn = get_surrogate_function(args.surrogate, args.slope)

    firing_rows = []
    alpha_list = args.eq31_alphas if args.eq31 else [1.0]
    for depth in range(args.depth_min, args.depth_max + 1):
        for thr in args.thresholds:
            for alpha in alpha_list:
                print(f"Measuring firing stats for depth={depth}, thr={thr}, eq31_alpha={alpha}...")
                # new init per config 
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

                model = DualSNN(
                    inDim=input_dim,
                    hidden=args.hidden,
                    nClass=output_dim,
                    beta=args.beta,
                    thr_wp=thr,
                    thr_sgd=thr,
                    eq31=args.eq31,
                    depth=depth,
                    surrogate_fn=surrogate_fn,
                    eq31_alpha=alpha,  
                )
                model.wp.to(device)
                model.sgd.to(device)
                per_layer, summary = measure_firing_stats(
                    model,
                    loader,
                    device=device,
                    num_batches=args.firing_batches,
                    thr=thr,
                    near_delta=args.near_delta,
                    rare_k=args.rare_k,
                )

                firing_rows.append({
                    "depth": depth,
                    "threshold": thr,
                    "eq31_alpha": alpha,
                    **summary,
                })

    firing_df = pd.DataFrame(firing_rows)
    firing_df.to_csv(out_dir / "data" / "firing_sweep.csv", index=False)

    # phasediagram heatmaps from a full sweep
    for d in range(args.depth_min, args.depth_max + 1):
        plot_heatmap_thr_alpha(
            firing_df, out_dir, depth=d,
            value_col="meanRateAvg",
            title=f"meanRateAvg heatmap depth={d}",
            fname=f"phase_meanRateAvg_depth{d}.png"
        )
        plot_heatmap_thr_alpha(
            firing_df, out_dir, depth=d,
            value_col="meanRateStdAcrossLayers",
            title=f"meanRateStdAcrossLayers heatmap depth={d}",
            fname=f"phase_meanRateStdAcrossLayers_depth{d}.png"
        )
        plot_heatmap_thr_alpha(
            firing_df, out_dir, depth=d,
            value_col="fracFiringMin",
            title=f"fracFiringMin heatmap depth={d}",
            fname=f"phase_fracFiringMin_depth{d}.png"
        )
        plot_heatmap_thr_alpha(
            firing_df, out_dir, depth=d,
            value_col="nearThrAvg",
            title=f"nearThrAvg heatmap depth={d}",
            fname=f"phase_nearThrAvg_depth{d}.png"
        )

    # shortlist candidates based on criteria
    candidate_df = build_candidate_shortlist(firing_df, args)
    candidate_df.to_csv(out_dir / "data" / "candidate_shortlist.csv", index=False)
    reg_dirs = {} 
    regime_diag_rows = []
    cosine_rows = []    
    post_update_rows = []
    
    for _, row in candidate_df.iterrows():
        depth = int(row["depth"])
        if args.diagScope == "subset":
            include_fn = buildIncludeFn(
                depthHidden=depth,
                trainLayerIdxs=args.diagLayerIdxs,
                trainLastN=(args.diagLastN if args.diagLastN > 0 else None),
                trainBiasOnly=bool(args.diagBiasOnly),
                trainBiasLastN=(args.diagBiasLastN if args.diagBiasLastN > 0 else None),
                includeBias=bool(args.include_bias),
                includeOutput=True,
            )
        else:
            include_fn = lambda name: (args.include_bias or (not name.endswith(".bias")))
        thr = float(row["threshold"])
        alpha = float(row["eq31_alpha"])
        tag = f"d{depth}_thr{thr:.3f}_a{alpha:.3f}".replace(".", "p")
        print(f"Probing regime for shortlist candidate: depth={depth}, thr={thr}, eq31_alpha={alpha} (tag={tag})...")
        model = DualSNN(
            inDim=input_dim,
            hidden=args.hidden,
            nClass=output_dim,
            beta=args.beta,
            thr_wp=thr,
            thr_sgd=thr,
            eq31=args.eq31,
            depth=depth,
            surrogate_fn=surrogate_fn,
            eq31_alpha=alpha,
        )
        model.wp.to(device)
        model.sgd.to(device)
        it = iter(loader)
        pre_sampled_batches = []
        for bi in range(args.cosine_batches):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            pre_sampled_batches.append((xb, yb))
        for h in args.h_values:
            for bi, (xb, yb) in enumerate(pre_sampled_batches):
                do_details = (bi == 0 and h == args.h_values[0])
                res = probe_wp_vs_sgd(
                    model=model,
                    xb=xb,
                    yb=yb,
                    h=h,
                    args=args,
                    device=device,
                    include_fn=include_fn,
                    analyze_details=do_details,
                )
                gm = res.get("global_metrics", {})
                gm_wp = res.get("global_metrics_wp", {})
                gm_both = res.get("global_metrics_both", {})
                
                if args.run_post_update_prediction and bi < args.postUpdateMaxBatchesPerRegime:
                    pu_rows = run_post_update_prediction_for_probe(
                        model=model,
                        xb=xb,
                        yb=yb,
                        res=res,
                        args=args,
                        device=device,
                        include_fn=include_fn,
                        h=h,
                        batch_idx=bi,
                    )
                    for pu in pu_rows:
                        pu.update({
                            "tag": tag,
                            "dataset": args.dataset,
                            "surrogate": args.surrogate,
                            "slope": args.slope,
                            "eq31": int(args.eq31),
                            "depth": depth,
                            "threshold": thr,
                            "eq31_alpha": alpha,
                            "meanRateAvg": float(row["meanRateAvg"]),
                            "meanRateStdAcrossLayers": float(row["meanRateStdAcrossLayers"]),
                            "fracFiringMin": float(row["fracFiringMin"]),
                            "nearThrAvg": float(row["nearThrAvg"]),
                            "candidate_category": row.get("candidate_category", "uncategorized"),
                            "candidate_score": float(row.get("candidate_score", np.nan)),
                        })
                    post_update_rows.extend(pu_rows)
    
                
                if do_details:
                    reg_out, tag = make_regime_dir(out_dir, depth, thr, alpha)
                    reg_dirs[(depth, thr, alpha)] = (reg_out, tag)
                    plot_global_cosine_bars_from_res(res, reg_out, tag)
                    # per layer cosine bars (all / wp-only / both)
                    plot_per_layer_cosine([res], reg_out, suffix=f"_{tag}") 

                    # zero gradient analysis
                    if "per_param_zero_analysis" in res:
                        plot_zero_gradient_analysis(res["per_param_zero_analysis"], reg_out, suffix=f"_{tag}") 
                    # membrane plots 
                    if "traces" in res:
                        plot_membrane_voltages(res["traces"], reg_out, depth=depth, thr=thr)

                    # firing rate violin plots from analyze_spiking_activity 
                    spike_stats = analyze_spiking_activity(model, loader, device, num_batches=5) 
                    plot_spiking_activity_violin(spike_stats, reg_out, depth=depth) 

                    # non firing / rare /firing gradient analysis (surrogate pushes silent neurons?)
                    nonfire = analyze_surrogate_for_nonfiring(
                        model, loader, device,
                        num_batches=args.firing_batches,
                        rare_k=args.rare_k
                    )
                    if nonfire:
                        plot_nonfiring_gradient_analysis(nonfire, reg_out, depth=depth)

                    # surrogate gradient function plot
                    plot_surrogate_gradient_function(args.surrogate, args.slope, reg_out, thr=thr)

                    # adaptive wp plots
                    if "wpStats" in res:
                        plot_wp_adaptive_suite(
                            wp_stats=res["wpStats"],
                            out_dir=reg_out / "plots/adaptiveWP",
                            prefix=f"{tag}_",
                        )
                    # regime diag summary row
                    zsum = summarize_zero_analysis(res.get("per_param_zero_analysis", None))
                    near_mass = np.nan
                    if "traces" in res:
                        near_mass = compute_near_thr_mass_from_traces(res["traces"], thr, args.near_delta)

                    deriv_sum = summarize_spike_derivative_stats(res.get("spike_derivative_stats", None))
                    regime_diag_rows.append({
                        "tag": tag,
                        "dataset": args.dataset,
                        "surrogate": args.surrogate,
                        "slope": args.slope,
                        "eq31": int(args.eq31),

                        "depth": depth,
                        "threshold": thr,
                        "eq31_alpha": alpha,

                        # firing regime summary from shortlist row
                        "meanRateAvg": float(row["meanRateAvg"]),
                        "meanRateStdAcrossLayers": float(row["meanRateStdAcrossLayers"]),
                        "fracFiringMin": float(row["fracFiringMin"]),
                        "nearThrAvg": float(row["nearThrAvg"]),

                        # cosine (use the same batch/h used for diagnostics)
                        "cosine_all": to_float(gm.get("cosine_similarity", np.nan)),
                        "cosine_wp": to_float(gm_wp.get("cosine_similarity", np.nan)),
                        "cosine_both": to_float(gm_both.get("cosine_similarity", np.nan)),

                        "dWP_norm": to_float(res.get("dWP_norm", np.nan)),
                        "dSGD_norm": to_float(res.get("dSGD_norm", np.nan)),

                        "pearson_all": to_float(gm.get("pearson_correlation", np.nan)),
                        "sign_all": to_float(gm.get("sign_agreement", np.nan)),

                        # zero gradient summary
                        **zsum,

                        # membrane-derived “near threshold mass”
                        "nearThrMass": near_mass,

                        # spike derivative summary
                        **deriv_sum,

                        "wpProbeEstimator": args.wpProbeEstimator,
                        "wpNumCoordsUsed": to_float(res.get("wpStats", {}).get("num_coords_used", np.nan)),
                        "wpNumCoordsTotal": to_float(res.get("wpStats", {}).get("num_coords_total", np.nan)),
                        "wpNumEscalated": to_float(res.get("wpStats", {}).get("num_escalated", np.nan)),
                        "wpNumFailed": to_float(res.get("wpStats", {}).get("num_failed", np.nan)),

                        "candidate_category": row.get("candidate_category", "uncategorized"),
                        "candidate_score": float(row.get("candidate_score", np.nan)),
                        "wpK": int(args.wpK) if args.wpProbeEstimator == "random" else np.nan,
                        "wpNoise": args.wpNoise if args.wpProbeEstimator == "random" else "",
                        "wpSampling": args.wpSampling if args.wpProbeEstimator == "random" else "",
                        "wpNoiseScope": args.wpNoiseScope if args.wpProbeEstimator == "random" else "",
                    })
                    res_to_save = dict(res)
                    res_to_save.pop("traces", None)  # avoiding dumping full tensors
                    res_to_save.pop("dWP_dict", None)
                    res_to_save.pop("dSGD_dict", None)
                    with open(reg_out / "data" / "cosine_details.json", "w") as f:
                        json.dump(res_to_save, f, indent=2, default=str)
                    
                cosine_rows.append({
                    "depth": depth,
                    "threshold": thr,
                    "eq31_alpha": alpha,
                    "h": float(h), 
                    "batch": int(bi),
                    "tag": tag,
                    "surrogate": args.surrogate,
                    "slope": args.slope,

                    "meanRateAvg": float(row["meanRateAvg"]),
                    "meanRateStdAcrossLayers": float(row["meanRateStdAcrossLayers"]),
                    "fracFiringMin": float(row["fracFiringMin"]),
                    "nearThrAvg": float(row["nearThrAvg"]),

                    # main y targets
                    "cosine_all": float(res["global_metrics"]["cosine_similarity"]),
                    "cosine_wp": float(res["global_metrics_wp"]["cosine_similarity"]),
                    "cosine_both": float(res["global_metrics_both"]["cosine_similarity"]),

                    "dWP_norm": float(res.get("dWP_norm", np.nan)),
                    "dSGD_norm": float(res.get("dSGD_norm", np.nan)),
                    "pearson_all": float(gm.get("pearson_correlation", np.nan)) if gm.get("pearson_correlation", None) is not None else np.nan,
                    "sign_all": float(gm.get("sign_agreement", np.nan)) if gm.get("sign_agreement", None) is not None else np.nan, 
                
                    "wpProbeEstimator": args.wpProbeEstimator,
                    "wpNumCoordsUsed": float(res.get("wpStats", {}).get("num_coords_used", np.nan)),
                    "wpNumCoordsTotal": float(res.get("wpStats", {}).get("num_coords_total", np.nan)),
                    "wpNumEscalated": float(res.get("wpStats", {}).get("num_escalated", np.nan)),
                    "wpNumFailed": float(res.get("wpStats", {}).get("num_failed", np.nan)),
                    "candidate_category": row.get("candidate_category", "uncategorized"),
                    "candidate_score": float(row.get("candidate_score", np.nan)),
                    "wpK": int(args.wpK) if args.wpProbeEstimator == "random" else np.nan,
                    "wpNoise": args.wpNoise if args.wpProbeEstimator == "random" else "",
                    "wpSampling": args.wpSampling if args.wpProbeEstimator == "random" else "",
                    "wpNoiseScope": args.wpNoiseScope if args.wpProbeEstimator == "random" else "",
                })

    cosine_df = pd.DataFrame(cosine_rows)
    cosine_df.to_csv(out_dir / "data" / "cosine_shortlist.csv", index=False)
    if len(regime_diag_rows) > 0:
        diag_df = pd.DataFrame(regime_diag_rows)
        diag_df.to_csv(out_dir / "data" / "regime_diagnostics.csv", index=False)
        plot_findings(diag_df, out_dir)
        plot_cosine_by_candidate_category(diag_df, out_dir)
        for (d, t, a), (reg_out, tag) in reg_dirs.items():
            plot_cosine_vs_h_for_regime(cosine_df, reg_out, d, t, a, tag)
    
    if args.run_post_update_prediction:
        save_post_update_outputs(post_update_rows, out_dir)

    
    # aggregate over h and batches (for plotting)
    agg = cosine_df.groupby(["depth", "threshold", "eq31_alpha"], as_index=False).agg({
        "meanRateAvg": "mean",
        "meanRateStdAcrossLayers": "mean",
        "fracFiringMin": "mean",
        "nearThrAvg": "mean",
        "cosine_wp": "mean",
        "cosine_all": "mean",
        "cosine_both": "mean",
    })

    plot_scatter_firing_vs_cosine(agg, out_dir)

    print(f"done. results in {out_dir}")


if __name__ == "__main__":
    main()
