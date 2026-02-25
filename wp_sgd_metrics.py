import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

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
        mask = a.abs() >= eps
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


def analyze_zero_gradients(dWP_dict, dSGD_dict):
    """Analyze the distribution of zero gradients"""
    analysis = {}
    
    for name in dWP_dict.keys():
        wp_grad = dWP_dict[name].flatten()
        sgd_grad = dSGD_dict[name].flatten()
        
        wp_zero = (wp_grad.abs() < 1e-10).float().mean().item()
        sgd_zero = (sgd_grad.abs() < 1e-10).float().mean().item()
        both_zero = ((wp_grad.abs() < 1e-10) & (sgd_grad.abs() < 1e-10)).float().mean().item()
        
        analysis[name] = {
            'wp_zero_fraction': wp_zero,
            'sgd_zero_fraction': sgd_zero,
            'both_zero_fraction': both_zero,
            'wp_mean_abs': wp_grad.abs().mean().item(),
            'sgd_mean_abs': sgd_grad.abs().mean().item(),
        }
    
    return analysis

# perlayer cosine similarities
def per_layer_cosines(dWP_dict, dSGD_dict, exclude_zeros=False):
    groups = {}
    for name in dWP_dict.keys():
        base, kind = split_layer_name(name)
        if kind not in ("weight", "bias"):
            continue
        groups.setdefault(base, {})[kind] = name

    weights_only, bias_only, combined = {}, {}, {}

    for base, kinds in groups.items():
        if "weight" in kinds:
            n = kinds["weight"]
            weights_only[base] = cos_from(dWP_dict[n].flatten(), dSGD_dict[n].flatten(), exclude_zeros)
        if "bias" in kinds:
            n = kinds["bias"]
            bias_only[base] = cos_from(dWP_dict[n].flatten(), dSGD_dict[n].flatten(), exclude_zeros)
        if "weight" in kinds and "bias" in kinds:
            nW, nB = kinds["weight"], kinds["bias"]
            vwp = torch.cat([dWP_dict[nW].reshape(-1), dWP_dict[nB].reshape(-1)])
            vsgd = torch.cat([dSGD_dict[nW].reshape(-1), dSGD_dict[nB].reshape(-1)])
            combined[base] = cos_from(vwp, vsgd)

    return weights_only, bias_only, combined

# main function to compute cosine similarity using orthogonal perturbation
def cosine_similarity_wp_sgd_orthogonal(model, xb, yb, h=0.01, include_layers=None, 
                                        include_bias=True, device=None, analyze_details=False):
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
