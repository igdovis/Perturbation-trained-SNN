import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def compute_all_metrics(v1, v2, exclude_zeros=False, eps=1e-10):
    """
    v1 = WP vector (x)
    v2 = SGD vector (y)
    exclude_zeros:
      - False: no masking
      - "both": drop coords where BOTH x and y are ~0
      - "wp": keep only coords where W P (x) is ~nonzero
    """
    x = v1.float().reshape(-1)
    y = v2.float().reshape(-1)
    eps = 1e-10
    if exclude_zeros == "both":
        mask = ~((x.abs() < eps) & (y.abs() < eps))
    elif exclude_zeros == "wp":
        rel_eps = max(eps, float(x.abs().max()) * 1e-6)
        mask = x.abs() >= rel_eps
    elif exclude_zeros in (False, None, "none"):
        mask = None
    else:
        raise ValueError("exclude_zeros must be False, 'both', or 'wp'")

    if mask is not None:
        x = x[mask]
        y = y[mask]
        active_frac = float(mask.float().mean())
        active_count = int(mask.sum().item())
    else:
        active_frac = 1.0
        active_count = int(x.numel())

    if x.numel() == 0:
        return {
            "cosine_similarity": np.nan,
            "pearson_correlation": np.nan,
            "sign_agreement": np.nan,
            "relative_error": np.nan,
            "norm_ratio": np.nan,
            "best_fit_scale": np.nan,
            "best_fit_residual": np.nan,
            "active_frac": active_frac,
            "active_count": active_count,
            "error": "No active coordinates after masking",
    }

    # cosine similarity
    cos_sim = float((x @ y) / (x.norm().clamp_min(1e-12) * y.norm().clamp_min(1e-12)))

    # pearson correlation
    x0 = x - x.mean()
    y0 = y - y.mean()
    pearson = float((x0 @ y0) / (x0.norm().clamp_min(1e-12) * y0.norm().clamp_min(1e-12)))

    # sign agreement (counts 0==0 as agreement if zeros remain)
    sign_agree = float((torch.sign(x) == torch.sign(y)).float().mean())

    # relative error (treat y as reference)
    rel_error = float(torch.norm(x - y) / y.norm().clamp_min(1e-12))

    # norm ratio
    norm_ratio = float(x.norm() / y.norm().clamp_min(1e-12))

    # best-fit scalar a for x ≈ a y
    denom = (y @ y).clamp_min(1e-12)
    a = float((x @ y) / denom)
    residual = float(torch.norm(x - a * y))

    return {
        "cosine_similarity": cos_sim,
        "pearson_correlation": pearson,
        "sign_agreement": sign_agree,
        "relative_error": rel_error,
        "norm_ratio": norm_ratio,
        "best_fit_scale": abs(a),
        "best_fit_residual": residual,
        "active_frac": active_frac,
        "active_count": active_count,
    }

### Helpers for perparam similarity analysis
def split_layer_name(param_name: str):
    m = re.match(r"^(.*)\.(weight|bias)$", param_name)
    if not m:
        return param_name, ""
    return m.group(1), m.group(2)

def cos_from(vec_a: torch.Tensor, vec_b: torch.Tensor, exclude_zeros=False, eps=1e-10) -> float:
    """
    vec_a: WP vector
    vec_b: SGD vector
    """
    a = vec_a.reshape(-1)
    b = vec_b.reshape(-1)
    if exclude_zeros == "both":
        mask = ~((a.abs() < eps) & (b.abs() < eps))
        a = a[mask]
        b = b[mask]
    elif exclude_zeros == "wp":
        # mask = a.abs() >= eps
        # a = a[mask]
        #b = b[mask]
        # relative eps instead of a fixed one. consistent with compute_all_metrics
        rel_eps = max(eps, float(a.abs().max()) * 1e-6)
        mask = a.abs() >= rel_eps
        a = a[mask]
        b = b[mask]
    elif exclude_zeros in (False, None, "none"):
        pass
    else:
        raise ValueError("exclude_zeros must be False, 'both', or 'wp'")

    # If everything got masked away, cos is 0
    if a.numel() == 0:
        return 0.0

    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float(torch.dot(a, b) / denom)

# helper to get the mask of used coordinates for a given param, based on wp_stats
def _used_mask_for_param(name, flat_tensor, wp_stats=None, only_used=False):
    if not only_used:
        return torch.ones_like(flat_tensor, dtype=torch.bool)

    if not isinstance(wp_stats, dict):
        return torch.ones_like(flat_tensor, dtype=torch.bool)

    per_param = wp_stats.get("per_param", {})
    st = per_param.get(name, {})

    used_idx = st.get("used_idx", None)

    # coord_full or old stats -  assume all returned coords were used
    if used_idx is None:
        return torch.ones_like(flat_tensor, dtype=torch.bool)

    mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
    if len(used_idx) > 0:
        idx = torch.as_tensor(used_idx, device=flat_tensor.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < flat_tensor.numel())]
        mask[idx] = True

    return mask

def _used_mask_for_param(name, flat_tensor, wp_stats=None, only_used=False):
    if not only_used:
        return torch.ones_like(flat_tensor, dtype=torch.bool)

    if not isinstance(wp_stats, dict):
        return torch.ones_like(flat_tensor, dtype=torch.bool)

    per_param = wp_stats.get("per_param", {})
    st = per_param.get(name, {})

    used_idx = st.get("used_idx", None)

    # coord_full or old stats: assume all returned coords were used
    if used_idx is None:
        return torch.ones_like(flat_tensor, dtype=torch.bool)

    mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
    if len(used_idx) > 0:
        idx = torch.as_tensor(used_idx, device=flat_tensor.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < flat_tensor.numel())]
        mask[idx] = True

    return mask


def analyze_zero_gradients(dWP_dict, dSGD_dict, wp_stats=None, only_used=False):
    """Analyze zero updates, optionally only on coordinates actually sampled by coordinate WP."""
    analysis = {}

    for name in dWP_dict.keys():
        wp_grad = dWP_dict[name].flatten()
        sgd_grad = dSGD_dict[name].flatten()

        mask = _used_mask_for_param(
            name=name,
            flat_tensor=wp_grad,
            wp_stats=wp_stats,
            only_used=only_used,
        )

        if mask.sum().item() == 0:
            analysis[name] = {
                "wp_zero_fraction": float("nan"),
                "sgd_zero_fraction": float("nan"),
                "both_zero_fraction": float("nan"),
                "wp_mean_abs": float("nan"),
                "sgd_mean_abs": float("nan"),
                "active_count": 0,
                "active_frac": 0.0,
            }
            continue

        wp_used = wp_grad[mask]
        sgd_used = sgd_grad[mask]

        wp_zero = (wp_used.abs() < 1e-10).float().mean().item()
        sgd_zero = (sgd_used.abs() < 1e-10).float().mean().item()
        both_zero = ((wp_used.abs() < 1e-10) & (sgd_used.abs() < 1e-10)).float().mean().item()

        analysis[name] = {
            "wp_zero_fraction": wp_zero,
            "sgd_zero_fraction": sgd_zero,
            "both_zero_fraction": both_zero,
            "wp_mean_abs": wp_used.abs().mean().item(),
            "sgd_mean_abs": sgd_used.abs().mean().item(),
            "active_count": int(mask.sum().item()),
            "active_frac": float(mask.float().mean().item()),
        }

    return analysis

# some helpers for per_layer_cosines
# ideally this is also used for compute all metrics TODO this
def _masked_pair(a: torch.Tensor, b: torch.Tensor, exclude_zeros=False, eps=1e-10):
    a = a.reshape(-1)
    b = b.reshape(-1)

    if exclude_zeros == "both":
        mask = ~((a.abs() < eps) & (b.abs() < eps))
        a = a[mask]
        b = b[mask]
    elif exclude_zeros == "wp":
        rel_eps = max(eps, float(a.abs().max()) * 1e-6) if a.numel() > 0 else eps
        mask = a.abs() >= rel_eps
        a = a[mask]
        b = b[mask]
    elif exclude_zeros in (False, None, "none"):
        pass
    else:
        raise ValueError("exclude_zeros must be False, 'both', or 'wp'")

    return a, b

def _per_layer_decomposition(dWP_dict, dSGD_dict, exclude_zeros=False, eps=1e-12):
    groups = {}
    for name in dWP_dict.keys():
        base, kind = split_layer_name(name)
        if kind not in ("weight", "bias"):
            continue
        groups.setdefault(base, {})[kind] = name

    rows = []
    for base, kinds in groups.items():
        row = {"layer": base}
        # weight part
        if "weight" in kinds:
            nW = kinds["weight"]
            wWp, wSgd = _masked_pair(dWP_dict[nW], dSGD_dict[nW], exclude_zeros=exclude_zeros)
            if wWp.numel() > 0:
                dotW = float(torch.dot(wWp, wSgd))
                normProdW = float(wWp.norm() * wSgd.norm())
            else:
                dotW = 0.0
                normProdW = 0.0
        else:
            dotW = np.nan
            normProdW = np.nan

        # bias part
        if "bias" in kinds:
            nB = kinds["bias"]
            bWp, bSgd = _masked_pair(dWP_dict[nB], dSGD_dict[nB], exclude_zeros=exclude_zeros)
            if bWp.numel() > 0:
                dotB = float(torch.dot(bWp, bSgd))
                normProdB = float(bWp.norm() * bSgd.norm())
            else:
                dotB = 0.0
                normProdB = 0.0
        else:
            dotB = np.nan
            normProdB = np.nan

        row["dotW"] = dotW
        row["dotB"] = dotB
        row["normProdW"] = normProdW
        row["normProdB"] = normProdB

        hasW = not np.isnan(dotW)
        hasB = not np.isnan(dotB)

        if hasW and hasB:
            row["dotTotal"] = dotW + dotB
            denom = max(
                (0.0 if np.isnan(normProdW) else normProdW) +
                (0.0 if np.isnan(normProdB) else normProdB),
                eps,
            )
            row["weightShare"] = (0.0 if np.isnan(normProdW) else normProdW) / denom
            row["biasShare"] = (0.0 if np.isnan(normProdB) else normProdB) / denom

        elif hasW:
            row["dotTotal"] = dotW
            row["weightShare"] = 1.0
            row["biasShare"] = 0.0

        elif hasB:
            row["dotTotal"] = dotB
            row["weightShare"] = 0.0
            row["biasShare"] = 1.0

        else:
            row["dotTotal"] = np.nan
            row["weightShare"] = np.nan
            row["biasShare"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def plot_per_layer_cosine(results, output_dir, suffix=""):
    """Plot per-layer cosine similarities with three variants: all, WP-only, and both"""
    def nat_key(layer):
        m = re.match(r"fcs\.(\d+)$", layer)
        return (0, int(m.group(1))) if m else (1, 0)
    
    res = results[-1] if isinstance(results, list) else results
    dWP_dict = res.get("dWP_dict", None)
    dSGD_dict = res.get("dSGD_dict", None)
    # Extract all three variants
    wCos = res.get("w_cos", {})
    bCos = res.get("b_cos", {})
    cCos = res.get("comb_cos", {})
    wCos_wp = res.get("w_cos_wp", {})
    bCos_wp = res.get("b_cos_wp", {})
    cCos_wp = res.get("comb_cos_wp", {})
    wCos_both = res.get("w_cos_both", {})
    bCos_both = res.get("b_cos_both", {})
    cCos_both = res.get("comb_cos_both", {})

    layers = sorted(set(wCos) | set(bCos) | set(cCos), key=nat_key)
    
    # Prepare data for all variants
    w_vals = np.array([wCos.get(L, np.nan) for L in layers], dtype=float)
    b_vals = np.array([bCos.get(L, np.nan) for L in layers], dtype=float)
    c_vals = np.array([cCos.get(L, np.nan) for L in layers], dtype=float)
    w_vals_wp = np.array([wCos_wp.get(L, np.nan) for L in layers], dtype=float)
    b_vals_wp = np.array([bCos_wp.get(L, np.nan) for L in layers], dtype=float)
    c_vals_wp = np.array([cCos_wp.get(L, np.nan) for L in layers], dtype=float)
    w_vals_both = np.array([wCos_both.get(L, np.nan) for L in layers], dtype=float)
    b_vals_both = np.array([bCos_both.get(L, np.nan) for L in layers], dtype=float)
    c_vals_both = np.array([cCos_both.get(L, np.nan) for L in layers], dtype=float)

    x = np.arange(len(layers), dtype=float)
    width = 0.2 

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex="col")
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # this is the top row of the plot
    ax1.bar(x - width, w_vals, width, label="weight", alpha=0.8, color="steelblue")
    ax1.bar(x, b_vals, width, label="bias", alpha=0.8, color="skyblue")
    ax1.bar(x + width, c_vals, width, label="weight + bias", alpha=0.8, color="lightblue")
    ax1.set_ylabel("Cosine similarity", fontsize=11)
    ax1.set_title(f"All Gradients (No Masking) {suffix}", fontsize=12)
    ax1.set_ylim([-0.3, 1.05])
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.axhline(0.0, color="black", linewidth=1)

    ax2.bar(x - width, w_vals_wp, width, label="weight", alpha=0.8, color="darkgreen")
    ax2.bar(x, b_vals_wp, width, label="bias", alpha=0.8, color="green")
    ax2.bar(x + width, c_vals_wp, width, label="weight + bias", alpha=0.8, color="lightgreen")
    ax2.set_ylabel("Cosine similarity", fontsize=11)
    ax2.set_title(f"WP Only non zero gradient {suffix}", fontsize=12, fontweight="bold")
    ax2.set_ylim([-0.3, 1.05])
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(0.0, color="black", linewidth=1)

    ax3.bar(x - width, w_vals_both, width, label="weight", alpha=0.8, color="coral")
    ax3.bar(x, b_vals_both, width, label="bias", alpha=0.8, color="lightsalmon")
    ax3.bar(x + width, c_vals_both, width, label="weight + bias", alpha=0.8, color="peachpuff")
    ax3.set_ylabel("Cosine similarity", fontsize=11)
    ax3.set_title(f"WP and SGD non zero gradient {suffix}", fontsize=12)
    ax3.set_ylim([-0.3, 1.05])
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.axhline(0.0, color="black", linewidth=1)

    for ax in (ax1, ax2, ax3):
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha="right")

    # this is the bottom row of the plot
    def _plot_decomp(ax, exclude_mode, title, color_w, color_b, color_t):
        df = _per_layer_decomposition(dWP_dict, dSGD_dict, exclude_zeros=exclude_mode)
        if df.empty:
            ax.text(0.5, 0.5, "no decomposition data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.axis("off")
            return

        df = df.set_index("layer").reindex(layers).reset_index()

        dotW = df["dotW"].to_numpy(dtype=float)
        dotB = df["dotB"].to_numpy(dtype=float)
        dotT = df["dotTotal"].to_numpy(dtype=float)

        normW = df["normProdW"].to_numpy(dtype=float)
        normB = df["normProdB"].to_numpy(dtype=float)
        wShare = df["weightShare"].to_numpy(dtype=float)

        x = np.arange(len(layers), dtype=float)
        w = 0.22

        # only dot bars
        ax.bar(x - w, dotW, w, label="weight dot", alpha=0.85, color=color_w)
        ax.bar(x,     dotB, w, label="bias dot",   alpha=0.85, color=color_b)
        ax.bar(x + w, dotT, w, label="total dot",  alpha=0.85, color=color_t)

        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel("dot product", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha="right")

        # choose cosine map for annotations
        if exclude_mode in (False, None):
            c_map = cCos
        elif exclude_mode == "wp":
            c_map = cCos_wp
        else:
            c_map = cCos_both

        y0, y1 = ax.get_ylim()
        yr = y1 - y0

        for i, layer in enumerate(layers):
            cs = c_map.get(layer, np.nan)

            # top annotation: weight share
            if not np.isnan(wShare[i]):
                ax.text(
                    x[i], y1 - 0.08 * yr,
                    f"wShare:{wShare[i]:.2f}",
                    ha="center", va="top", fontsize=8
                )
            # bottom annotation: cosine
            if not np.isnan(cs):
                ax.text(
                    x[i], y0 + 0.05 * yr,
                    f"cos:{cs:.2f}",
                    ha="center", va="bottom", fontsize=8
                )

    _plot_decomp(ax4, False, "dot decomposition (all)", "steelblue", "skyblue", "midnightblue")
    _plot_decomp(ax5, "wp", "dot decomposition (wpgrad nonzero)", "darkgreen", "green", "forestgreen")
    _plot_decomp(ax6, "both", "dot decomposition (both grad nonzero)", "coral", "lightsalmon", "orangered")

    handles, labels = ax6.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=True)
    fig.subplots_adjust(top=0.88)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_dir / "plots" / f"per_layer_cosine{suffix}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

# perlayer cosine similarities
def per_layer_cosines(dWP_dict, dSGD_dict, exclude_zeros=False):
    groups = {}
    for name in dWP_dict.keys():
        base, kind = split_layer_name(name)
        if kind not in ("weight", "bias"):
            continue
        groups.setdefault(base, {})[kind] = name

    weights_only = {}
    bias_only = {}
    combined = {}

    for base, kinds in groups.items():
        has_w = "weight" in kinds
        has_b = "bias" in kinds

        # weightonly cosine for this layer
        if has_w:
            nW = kinds["weight"]
            weights_only[base] = cos_from(
                dWP_dict[nW].reshape(-1),
                dSGD_dict[nW].reshape(-1),
                exclude_zeros=exclude_zeros,
            )

        # bias only cosine for this layer
        if has_b:
            nB = kinds["bias"]
            bias_only[base] = cos_from(
                dWP_dict[nB].reshape(-1),
                dSGD_dict[nB].reshape(-1),
                exclude_zeros=exclude_zeros,
            )

        # combined weight+bias cosine for this layer
        if has_w and has_b:
            nW = kinds["weight"]
            nB = kinds["bias"]

            vwp = torch.cat([
                dWP_dict[nW].reshape(-1),
                dWP_dict[nB].reshape(-1),
            ])
            vsgd = torch.cat([
                dSGD_dict[nW].reshape(-1),
                dSGD_dict[nB].reshape(-1),
            ])

            combined[base] = cos_from(vwp, vsgd, exclude_zeros=exclude_zeros)

        elif has_w:
            combined[base] = weights_only[base]

        elif has_b:
            combined[base] = bias_only[base]

    return weights_only, bias_only, combined

# main function to compute cosine similarity using orthogonal perturbation
def cosine_similarity_wp_sgd_orthogonal(model, xb, yb, h=0.03, include_layers=None, 
                                        include_bias=True, device=None, analyze_details=False,
                                        adaptive_wp=False, adaptive_max_mult=5, adaptive_abs_tol=1e-12, adaptive_rel_tol=1e-8):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(device)
    model.wp.to(device)
    model.sgd.to(device)

    xb = xb.to(device=device, dtype=torch.float32)
    yb = torch.as_tensor(yb, device=device)
    if yb.dtype != torch.long:
        yb = (yb.argmax(dim=-1) if yb.ndim > 1 else yb).long()

    named_wp = list(model.wp.named_parameters())
    named_sgd = list(model.sgd.named_parameters())

    def included(name):
        if (not include_bias) and name.endswith(".bias"):
            return False
        return (include_layers is None) or any(k in name for k in include_layers)

    with torch.no_grad():
        logits_base = model.wp.forward_logits(xb, record=False)
        E_base = F.cross_entropy(logits_base, yb)

    dWP_dict = {}
    param_count = 0

    with torch.no_grad():
        for name, p in named_wp:
            if not included(name):
                continue
            grad_wp = torch.zeros_like(p)
            adaptive_stats = {"m_hist": {}, "num_escalated": 0, "num_failed": 0}
            for idx in np.ndindex(p.shape):
                original_val = p[idx].item()
                p[idx] = original_val + h
                logits_p = model.wp.forward_logits(xb, record=False)
                E_plus = F.cross_entropy(logits_p, yb)

                p[idx] = original_val - h
                logits_m = model.wp.forward_logits(xb, record=False)
                E_minus = F.cross_entropy(logits_m, yb)

                p[idx] = original_val
                grad_wp[idx] = (E_plus - E_minus) / (2.0 * h)
                param_count += 1

            dWP_dict[name] = -grad_wp

    # Get SGD gradients
    with torch.no_grad():
        for (ns, ps), (nw, pw) in zip(named_sgd, named_wp):
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

    w_cos, b_cos, comb_cos = per_layer_cosines(dWP_dict, dSGD_dict, exclude_zeros=False)
    w_cos_wp, b_cos_wp, comb_cos_wp = per_layer_cosines(dWP_dict, dSGD_dict, exclude_zeros="wp")
    w_cos_both, b_cos_both, comb_cos_both = per_layer_cosines(dWP_dict, dSGD_dict, exclude_zeros="both")

    dWP_chunks = [dWP_dict[n].reshape(-1) for n in dWP_dict.keys()]
    dSGD_chunks = [dSGD_dict[n].reshape(-1) for n in dSGD_dict.keys()]
    dWP_vec = torch.cat(dWP_chunks) if dWP_chunks else torch.empty(0, device=device)
    dSGD_vec = torch.cat(dSGD_chunks) if dSGD_chunks else torch.empty(0, device=device)

    metrics = compute_all_metrics(dWP_vec, dSGD_vec, exclude_zeros=False) #wp should go first
    metrics_both = compute_all_metrics(dWP_vec, dSGD_vec, exclude_zeros='both')
    metrics_wp = compute_all_metrics(dWP_vec, dSGD_vec, exclude_zeros='wp')
    zero_analysis = analyze_zero_gradients(dWP_dict, dSGD_dict)

    """
    per_param_metrics = {}
    for name in dWP_dict.keys():
        per_param_metrics[name] = compute_all_metrics(
            dWP_dict[name].flatten(),
            dSGD_dict[name].flatten()
        )
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    result = {
        "method": "orthogonal",
        "num_params_perturbed": param_count,
        "per_param_zero_analysis": zero_analysis,
        "global_metrics": metrics,
        "global_metrics_both": metrics_both,
        "global_metrics_wp": metrics_wp,
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

    if analyze_details and traces is not None and model.sgd.depth > 0:
        spike_derivative_stats = []
        for layer_idx in range(model.sgd.depth):
            stats = compute_actual_spike_derivative(
                traces['mem'][layer_idx],
                traces['spk'][layer_idx],
                model.sgd.threshold
            )
            spike_derivative_stats.append(stats)
        result['spike_derivative_stats'] = spike_derivative_stats
        result['traces'] = traces

    return result


def compute_actual_spike_derivative(mem_trace, spk_trace, threshold=1.0):
    """
    Compute the actual derivative of spikes wrt membrane potential.
    where do spikes occur and what was the membrane potential?
    """
    # mem_trace: (B, T, H)
    # spk_trace: (B, T, H)
    
    spike_events = spk_trace > 0.5  # Boolean mask of spike times
    
    # Get membrane potentials right before spike
    pre_spike_mems = mem_trace[spike_events]
    
    # Distribution of membrane potentials at spike time
    if len(pre_spike_mems) > 0:
        stats = {
            'mean': pre_spike_mems.mean().item(),
            'std': pre_spike_mems.std().item(),
            'min': pre_spike_mems.min().item(),
            'max': pre_spike_mems.max().item(),
            'count': len(pre_spike_mems),
        }
    else:
        stats = {
            'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
        }
    
    # Also membrane potentials when NO spike occurs
    no_spike = ~spike_events
    non_spike_mems = mem_trace[no_spike]
    
    if len(non_spike_mems) > 0:
        stats['no_spike_mean'] = non_spike_mems.mean().item()
        stats['no_spike_std'] = non_spike_mems.std().item()
    else:
        stats['no_spike_mean'] = 0
        stats['no_spike_std'] = 0
    
    return stats
