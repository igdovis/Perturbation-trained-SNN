import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from snn_models import DualSNN
from data_utils import get_dataset
from wp_sgd_metrics import cosine_similarity_wp_sgd_orthogonal
from analysis_firing import measure_firing_stats

from experiment import (
    analyze_spiking_activity,
    analyze_surrogate_for_nonfiring,
    plot_spiking_activity_violin,
    plot_membrane_voltages,
    plot_surrogate_gradient_function,
    plot_zero_gradient_analysis,
    plot_nonfiring_gradient_analysis,
    plot_per_layer_cosine,
)


def parse_args():
    p = argparse.ArgumentParser()

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

    # shortlist criteria 
    p.add_argument("--shortlist_topk", type=int, default=10)
    p.add_argument("--shortlist_min_frac_firing", type=float, default=0.1)
    p.add_argument("--shortlist_max_frac_firing", type=float, default=0.9)
    p.add_argument("--shortlist_min_mean_rate", type=float, default=1e-4)
    p.add_argument("--shortlist_max_layer_std", type=float, default=0.05)

    # cosine
    p.add_argument("--h_values", nargs="+", type=float, default=[0.005, 0.01, 0.03])
    p.add_argument("--cosine_batches", type=int, default=2)
    p.add_argument("--include_bias", action="store_true")

    # plots
    p.add_argument("--heatmap_depth", type=int, default=2) #? no

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
        for k_src, k_dst in [
            ("wp_zero", "wp_zero_frac"),
            ("sgd_zero", "sgd_zero_frac"),
            ("both_zero", "both_zero_frac"),
        ]:
            if k_src in z and out["total_elems"] and out["total_elems"] > 0:
                out[k_dst] = to_float(z[k_src]) / out["total_elems"]
        return out

    tot = 0.0
    wp0 = 0.0
    sgd0 = 0.0
    both0 = 0.0
    for v in z.values():
        if not isinstance(v, dict):
            continue
        t = v.get("total", v.get("total_elements", v.get("n", None)))
        if t is None:
            continue
        t = float(t)
        tot += t
        wp0 += float(v.get("wp_zero", 0.0))
        sgd0 += float(v.get("sgd_zero", 0.0))
        both0 += float(v.get("both_zero", 0.0))

    if tot > 0:
        out["total_elems"] = tot
        out["wp_zero_frac"] = wp0 / tot
        out["sgd_zero_frac"] = sgd0 / tot
        out["both_zero_frac"] = both0 / tot
    return out

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
    filt = firing_df[
        (firing_df["fracFiringMin"] >= args.shortlist_min_frac_firing) &
        # (firing_df["fracFiringMin"] <= args.shortlist_max_frac_firing) &
        (firing_df["meanRateAvg"] >= args.shortlist_min_mean_rate) &
        (firing_df["meanRateStdAcrossLayers"] <= args.shortlist_max_layer_std)
    ].copy()

    if len(filt) == 0:
        filt = firing_df.copy()

    # score: need to refine later
    filt["score"] = (-filt["meanRateStdAcrossLayers"]) + (0.2 * filt["nearThrAvg"])
    # bin by meanRateAvg into quantiles eg bins low/med/high
    num_bins = 3
    unique_rates = filt["meanRateAvg"].nunique()

    if unique_rates >= 2:
        q = min(num_bins, unique_rates)  # to avoid error if not enough unique values
        filt["rateBin"] = pd.qcut(filt["meanRateAvg"], q=q, duplicates="drop")
    else:
        filt["rateBin"] = "all"

    # how many per (depth, rateBin)
    # spread across bins+depths but never exceed shortlist_topk overall
    n_groups = filt.groupby(["depth", "rateBin"]).ngroups
    per_group = max(1, args.shortlist_topk // max(1, n_groups))

    shortlist = (
        filt.sort_values("score", ascending=False)
            .groupby(["depth", "rateBin"], as_index=False)
            .head(per_group)
    )

    # if we didnt reach topk, fill remaining with best overall not already included
    if len(shortlist) < args.shortlist_topk:
        remaining = (
            filt.sort_values("score", ascending=False)
                .loc[~filt.index.isin(shortlist.index)]
                .head(args.shortlist_topk - len(shortlist))
        )
        shortlist = pd.concat([shortlist, remaining], ignore_index=False)

    shortlist = shortlist.sort_values("score", ascending=False).head(args.shortlist_topk).copy()

    shortlist.to_csv(out_dir / "data" / "shortlist.csv", index=False)
    reg_dirs = {} 
    regime_diag_rows = []
    cosine_rows = []
    for _, row in shortlist.iterrows():
        depth = int(row["depth"])
        thr = float(row["threshold"])
        alpha = float(row["eq31_alpha"])
        tag = f"d{depth}_thr{thr:.3f}_a{alpha:.3f}".replace(".", "p")
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

        it = iter(loader)
        for h in args.h_values:
            for bi in range(args.cosine_batches):
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(loader)
                    xb, yb = next(it)
                do_details = (bi == 0 and h == args.h_values[0])
                res = cosine_similarity_wp_sgd_orthogonal(
                    model, xb, yb,
                    h=h,
                    include_layers=None,
                    device=device,
                    analyze_details=do_details,
                    include_bias=args.include_bias,
                )
                gm = res.get("global_metrics", {})
                gm_wp = res.get("global_metrics_wp", {})
                gm_both = res.get("global_metrics_both", {})
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
                    })
                    res_to_save = dict(res)
                    res_to_save.pop("traces", None)  # avoiding dumping full tensors
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
                })

    cosine_df = pd.DataFrame(cosine_rows)
    cosine_df.to_csv(out_dir / "data" / "cosine_shortlist.csv", index=False)
    if len(regime_diag_rows) > 0:
        diag_df = pd.DataFrame(regime_diag_rows)
        diag_df.to_csv(out_dir / "data" / "regime_diagnostics.csv", index=False)
        plot_findings(diag_df, out_dir)
        for (d, t, a), (reg_out, tag) in reg_dirs.items():
            plot_cosine_vs_h_for_regime(cosine_df, reg_out, d, t, a, tag)
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
