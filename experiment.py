import argparse
import os
import json
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import stork.datasets as datasets
import snntorch as snn
from snntorch import surrogate
from snntorch.spikevision import spikedata
import wandb
# importing shd dataset
import tonic
import tonic.transforms as ttf
# transform shd event data to tensor format
from torchvision import transforms as tvt

# TODO clean this up
def parse_args():
    parser = argparse.ArgumentParser(
        description='SNN WP vs SGD depth cosine experiment for Randman dataset')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='randman', 
                        choices=['randman', 'shd'], help='Dataset to use')
    
    # Experiment parameters
    parser.add_argument('--depth_min', type=int, default=1, help='Minimum network depth')
    parser.add_argument('--depth_max', type=int, default=7, help='Maximum network depth')
    parser.add_argument('--h_values', nargs='+', type=float, default=[0.05, 0.01, 0.03], 
                        help='Perturbation sizes to test')
    parser.add_argument('--batches', type=int, default=8, help='Number of batches per experiment')
    
    # Network parameters
    parser.add_argument('--hidden', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--beta', type=float, default=0.95, help='Leaky integrate factor')
    parser.add_argument('--threshold', type=float, default=1.0, help='Spike threshold')
    parser.add_argument('--eq31', action='store_true', help='Use eq31 initialization')
    parser.add_argument('--include_bias', action='store_true', help='include bias parameters in WP–SGD comparison (default: exclude)')
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--nb_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of time steps (shd)')
    parser.add_argument('--dt', type=float, default=1000, help='dt in microseconds (shd)')
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None, 
                        help='Experiment name (default: timestamp)')
    
    # Computation
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--include_bias', action='store_true', help='include bias parameters in WP–SGD comparison (default: exclude)')
    return parser.parse_args()


def generate_randman(
    dim_manifold=1,
    nb_classes=10,
    nb_inputs=20,
    nb_time_steps=100,
    step_frac=0.5,
    nb_samples=1000,
    nb_spikes=1,
    alpha=1,
    randmanseed=42,
    dt=2e-3,
    plot=True,
):
    duration = nb_time_steps * dt

    data, labels = datasets.make_tempo_randman(
        dim_manifold=dim_manifold,
        nb_classes=nb_classes,
        nb_units=nb_inputs,
        nb_steps=nb_time_steps,
        step_frac=step_frac,
        nb_samples=nb_samples,
        nb_spikes=nb_spikes,
        alpha=alpha,
        seed=randmanseed,
    )

    ds_kwargs = dict(nb_steps=nb_time_steps, nb_units=nb_inputs, time_scale=1.0)

    # Split into train, test and validation set
    datasets_split = datasets.split_dataset(
        data, labels, splits=[0.8, 0.1, 0.1], shuffle=False
    )
    datasets_ras = [
        datasets.RasDataset(ds, **ds_kwargs)
        for ds in datasets_split
    ]
    return datasets_ras

def load_shd_dataset(args):
    """Load Spiking Heidelberg Digits dataset via tonic and return [train, valid, test], input_dim, output_dim.
    Each sample: x shape (T, 700), y in [0..19].
    """
    print("Loading SHD dataset via tonic .. . . . ..")

    # Where tonic stores/creates shd_train.h5, shd_test.h5
    save_to = os.path.join(args.data_dir, "shd")

    # (W, H, P) – for SHD this is typically (700, 1, 1)
    sensor_size = tonic.datasets.SHD.sensor_size
    input_dim = sensor_size[0] * sensor_size[1] * sensor_size[2] 
    output_dim = 20  # SHD has 20 classes

    # Transform: events -> frames -> torch tensor -> (T, input_dim)
    event_transform = ttf.Compose([
        ttf.ToFrame(sensor_size=sensor_size, n_time_bins=args.num_steps),
        tvt.Lambda(
            lambda frames: torch.from_numpy(frames).float()
                            .view(frames.shape[0], -1)  # (T, input_dim)
        ),
    ])
    print("input dim:", input_dim, "output dim:", output_dim)
    # tonic will download the dataset the first time, if needed
    train_ds = tonic.datasets.SHD(
        save_to=save_to,
        train=True,
        transform=event_transform,
        target_transform=None,
    )
    test_full = tonic.datasets.SHD(
        save_to=save_to,
        train=False,
        transform=event_transform,
        target_transform=None,
    )

    # Split test into validation + test (like you did before)
    test_size = len(test_full)
    valid_size = test_size // 2
    test_size  = test_size - valid_size

    valid_ds, test_ds = torch.utils.data.random_split(
        test_full,
        [valid_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    return [train_ds, valid_ds, test_ds], input_dim, output_dim

def get_dataset(args):
    """Get dataset based on args"""
    if args.dataset == 'randman':
        datasets_list = generate_randman()
        input_dim = 20  
        output_dim = 10  
        return datasets_list, input_dim, output_dim
    elif args.dataset == 'shd':
        return load_shd_dataset(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def setup_experiment(args):
    """Setup directories, device, and logging"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'plots').mkdir(exist_ok=True)
    (output_dir / 'data').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return device, output_dir


# Will likely not use this init
def init_eq31(fc, alpha=1.9, std=0.05, freeze_bias=False):
    n = fc.in_features
    mu = ((1.0 - 1.0 / math.sqrt(n)) / n) * alpha
    with torch.no_grad():
        fc.weight.fill_(mu)
        torch.nn.init.normal_(fc.weight, mu, std)
        if fc.bias is not None:
            fc.bias.zero_()
            fc.bias.requires_grad_(not freeze_bias)

class SimpleSNN(nn.Module):
    # Feedforward SNN with Leaky LIF neurons
    # depth: number of hidden layers
    def __init__(self, inDim=20, hidden=128, nClass=10, beta=0.95, 
                 thr=1.0, spikeGrad=None, eq31=False, depth=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inDim = inDim
        self.depth = depth
        self.fcs = nn.ModuleList()
        self.lifs = nn.ModuleList()
        self.hidden = hidden
        self.nClass = nClass
        
        prev = inDim
        for _ in range(depth):
            self.fcs.append(nn.Linear(prev, hidden, bias=True))
            self.lifs.append(snn.Leaky(
                beta=beta, threshold=thr, reset_mechanism="subtract",
                spike_grad=spikeGrad
            ))
            prev = hidden

        self.fc_out = nn.Linear(prev, nClass, bias=True)

        with torch.no_grad():
            if eq31:
                for fc in self.fcs:
                    init_eq31(fc, freeze_bias=False)
                init_eq31(self.fc_out, freeze_bias=False)
            else:
                for fc in self.fcs:
                    nn.init.kaiming_normal_(fc.weight, nonlinearity="linear")
                    nn.init.zeros_(fc.bias)
                    fc.bias.requires_grad_(True)
                nn.init.kaiming_normal_(self.fc_out.weight, nonlinearity="linear")
                nn.init.zeros_(self.fc_out.bias)
                self.fc_out.bias.requires_grad_(True)

    def forward_logits(self, x, record=False):
        B, T, _ = x.shape
        x = x.to(dtype=torch.float32, device=self.device)
        mems = [torch.zeros(B, self.hidden, device=self.device) for _ in range(self.depth)]
        logitsAcc = torch.zeros(B, self.nClass, device=self.device)

        if record and self.depth > 0:
            # Record ALL layers, not just last
            memTraces = [[] for _ in range(self.depth)]
            spkTraces = [[] for _ in range(self.depth)]
            preTraces = [[] for _ in range(self.depth)]

        for t in range(T):
            h = x[:, t, :]
            for i in range(self.depth):
                cur = self.fcs[i](h)
                spk, mems[i] = self.lifs[i](cur, mems[i])
                h = spk
                
                if record:
                    preTraces[i].append(cur.detach())
                    spkTraces[i].append(spk.detach())
                    memTraces[i].append(mems[i].detach())

            logitsAcc = logitsAcc + self.fc_out(h)

        logits = logitsAcc / T
        
        if not record:
            return logits

        if self.depth == 0:
            traces = {"mem": None, "spk": None, "pre": None}
        else:
            # Stack traces for each layer
            traces = {
                "mem": [torch.stack(memTraces[i], dim=1) for i in range(self.depth)],
                "spk": [torch.stack(spkTraces[i], dim=1) for i in range(self.depth)],
                "pre": [torch.stack(preTraces[i], dim=1) for i in range(self.depth)]
            }
        return logits, traces

class DualSNN(nn.Module):
    """Initialize WP and SGD networks with same weights"""
    def __init__(self, inDim=20, hidden=128, nClass=10, beta=0.95, 
                 thr_wp=1.0, thr_sgd=1.0, eq31=False, depth=1):
        super().__init__()
        import copy
        base = SimpleSNN(inDim, hidden, nClass, beta, thr_wp, 
                        spikeGrad=None, eq31=eq31, depth=depth)
        state0 = copy.deepcopy(base.state_dict())
        self.wp = base
        self.sgd = SimpleSNN(inDim, hidden, nClass, beta, thr_sgd,
                            spikeGrad=surrogate.fast_sigmoid(slope=25.0), 
                            eq31=eq31, depth=depth)
        self.sgd.load_state_dict(state0)
        for p in self.wp.parameters():
            p.requires_grad_(False)
        for p in self.sgd.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def forward_wp(self, x, record=False):
        return self.wp.forward_logits(x, record=record)

    def forward_sgd(self, x, record=False):
        return self.sgd.forward_logits(x, record=record)
    

def compute_all_metrics(v1, v2):
    """Compute comparison metrics between two gradient vectors."""
    x = v1.float()
    y = v2.float()

    cos_sim = float((x @ y) / (x.norm().clamp_min(1e-12) * y.norm().clamp_min(1e-12)))
    
    xm, ym = x.mean(), y.mean()
    x0, y0 = x - xm, y - ym
    pearson = float((x0 * y0).sum() / (x0.norm().clamp_min(1e-12) * y0.norm().clamp_min(1e-12)))
    
    sign_agree = float((torch.sign(x) == torch.sign(y)).float().mean())
    rel_error = float(torch.norm(x - y) / y.norm().clamp_min(1e-12))
    norm_ratio = float(x.norm() / y.norm().clamp_min(1e-12))
    
    a = float((x @ y) / (y @ y).clamp_min(1e-12))
    residual = float(torch.norm(x - a * y))
    a = abs(a)

    return {
        "cosine_similarity": cos_sim,
        "pearson_correlation": pearson,
        "sign_agreement": sign_agree,
        "relative_error": rel_error,
        "norm_ratio": norm_ratio,
        "best_fit_scale": a,
        "best_fit_residual": residual,
    }

### Helpers for perparam similarity analysis
def split_layer_name(param_name: str):
    m = re.match(r"^(.*)\.(weight|bias)$", param_name)
    if not m:
        return param_name, ""
    return m.group(1), m.group(2)

def cos_from(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    denom = (vec_a.norm() * vec_b.norm()).clamp_min(1e-12)
    return float(torch.dot(vec_a, vec_b) / denom)

# perlayer cosine similarities
def per_layer_cosines(dWP_dict, dSGD_dict):
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
            weights_only[base] = cos_from(dWP_dict[n].flatten(), dSGD_dict[n].flatten())
        if "bias" in kinds:
            n = kinds["bias"]
            bias_only[base] = cos_from(dWP_dict[n].flatten(), dSGD_dict[n].flatten())
        if "weight" in kinds and "bias" in kinds:
            nW, nB = kinds["weight"], kinds["bias"]
            vwp = torch.cat([dWP_dict[nW].reshape(-1), dWP_dict[nB].reshape(-1)])
            vsgd = torch.cat([dSGD_dict[nW].reshape(-1), dSGD_dict[nB].reshape(-1)])
            combined[base] = cos_from(vwp, vsgd)

    return weights_only, bias_only, combined

# main function to compute cosine similarity using orthogonal perturbation
def cosine_similarity_wp_sgd_orthogonal(model, xb, yb, h=0.01, include_layers=None, 
                                        include_bias=True, device=None):
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
        logits = model.forward_sgd(xb, record=False)
        loss = F.cross_entropy(logits, yb)
        loss.backward()

    dSGD_dict = {}
    for name, p in named_sgd:
        if included(name):
            if p.grad is not None:
                dSGD_dict[name] = -p.grad.clone()
            else:
                dSGD_dict[name] = torch.zeros_like(p)

    w_cos, b_cos, comb_cos = per_layer_cosines(dWP_dict, dSGD_dict)

    dWP_chunks = [dWP_dict[n].reshape(-1) for n in dWP_dict.keys()]
    dSGD_chunks = [dSGD_dict[n].reshape(-1) for n in dSGD_dict.keys()]
    dWP_vec = torch.cat(dWP_chunks) if dWP_chunks else torch.empty(0, device=device)
    dSGD_vec = torch.cat(dSGD_chunks) if dSGD_chunks else torch.empty(0, device=device)

    metrics = compute_all_metrics(dWP_vec, dSGD_vec)
    per_param_metrics = {}
    for name in dWP_dict.keys():
        per_param_metrics[name] = compute_all_metrics(
            dWP_dict[name].flatten(),
            dSGD_dict[name].flatten()
        )

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    result = {
        "method": "orthogonal",
        "num_params_perturbed": param_count,
        "global_metrics": metrics,
        "per_param_metrics": per_param_metrics,
        "dWP_norm": float(dWP_vec.norm()),
        "dSGD_norm": float(dSGD_vec.norm()),
        "w_cos": w_cos,
        "b_cos": b_cos,
        "comb_cos": comb_cos
    }

    return result

def analyze_spiking_activity(model, loader, device, num_batches=4):
    """Analyze spiking activity across layers"""
    model.sgd.to(device)
    model.sgd.eval()
    
    all_spike_rates = [[] for _ in range(model.sgd.depth)]
    all_mem_stats = [[] for _ in range(model.sgd.depth)]
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx >= num_batches:
                break
            
            if isinstance(batch_data, (tuple, list)):
                xb, yb = batch_data[0], batch_data[1]
            else:
                xb, yb = batch_data, None
            
            xb = xb.to(device)
            logits, traces = model.forward_sgd(xb, record=True)
            
            # Analyze each layer
            for layer_idx in range(model.sgd.depth):
                spk = traces['spk'][layer_idx]  # (B, T, H)
                mem = traces['mem'][layer_idx]
                
                # Spike rate per neuron (averaged over batch and time)
                spike_rate = spk.float().mean(dim=(0, 1))  # (H,)
                all_spike_rates[layer_idx].append(spike_rate.cpu())
                
                # Membrane potential statistics
                mem_mean = mem.mean(dim=(0, 1))  # (H,)
                mem_std = mem.std(dim=(0, 1))
                all_mem_stats[layer_idx].append({
                    'mean': mem_mean.cpu(),
                    'std': mem_std.cpu()
                })
    
    # Aggregate statistics
    spike_stats = []
    for layer_idx in range(model.sgd.depth):
        rates = torch.stack(all_spike_rates[layer_idx])  # (num_batches, H)
        spike_stats.append({
            'mean_rate': rates.mean(dim=0).numpy(),  # (H,)
            'std_rate': rates.std(dim=0).numpy(),
            'overall_mean': float(rates.mean()),
            'overall_std': float(rates.std())
        })
    
    return spike_stats

def plot_metrics_vs_h(results, output_dir):
    """Plot all metrics vs h"""
    df = pd.DataFrame([
        {'h': r['h'], **r['global_metrics']}
        for r in results
    ])

    summary = df.groupby('h').agg(['mean', 'std']).reset_index()
    metrics = [col for col in df.columns if col != 'h']

    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        h_vals = summary['h'].values
        means = summary[(metric, 'mean')].values
        stds = summary[(metric, 'std')].values
        ax.errorbar(h_vals, means, yerr=stds, marker='o', capsize=5, linewidth=2)
        ax.set_xlabel('h (perturbation size)', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3)
        if 'error' in metric or 'ratio' in metric or 'residual' in metric:
            ax.set_yscale('log')

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'metrics_vs_h.png', dpi=150, bbox_inches='tight')
    plt.close()
    
def plot_per_layer_cosine(results, output_dir, suffix=""):
    """Plot perlayer cosine similarities"""
    def nat_key(layer):
        m = re.match(r"fcs\.(\d+)$", layer)
        return (0, int(m.group(1))) if m else (1, 0)
    
    res = results[-1] if isinstance(results, list) else results

    wCos = res.get("w_cos", {})
    bCos = res.get("b_cos", {})
    cCos = res.get("comb_cos", {})

    layers = sorted(set(wCos) | set(bCos) | set(cCos), key=nat_key)
    w_vals = np.array([wCos.get(L, np.nan) for L in layers], dtype=float)
    b_vals = np.array([bCos.get(L, np.nan) for L in layers], dtype=float)
    c_vals = np.array([cCos.get(L, np.nan) for L in layers], dtype=float)

    x = np.arange(len(layers), dtype=float)
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(8, 0.9*len(layers)), 3.2))
    ax.bar(x - width, w_vals, width, label="weight")
    ax.bar(x, b_vals, width, label="bias")
    ax.bar(x + width, c_vals, width, label="weight + bias")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_ylabel("cosine similarity")
    ax.set_title(f"Cosine Similarity per Layer {suffix}")
    ax.legend(frameon=False, ncol=3)
    ax.set_ylim([-0.05, 1.05])

    fig.tight_layout()
    plt.savefig(output_dir / 'plots' / f'per_layer_cosine{suffix}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
def plot_spiking_activity(spike_stats, output_dir, depth):
    """Plot spiking activity statistics"""
    num_layers = len(spike_stats)
    
    # Overall spike rates per layer
    fig, ax = plt.subplots(figsize=(10, 5))
    layer_names = [f'Layer {i+1}' for i in range(num_layers)]
    mean_rates = [stats['overall_mean'] for stats in spike_stats]
    std_rates = [stats['overall_std'] for stats in spike_stats]
    
    ax.bar(layer_names, mean_rates, yerr=std_rates, capsize=5, alpha=0.7)
    ax.set_ylabel('Mean spiking rate')
    ax.set_title(f'Average spiking rate per layer (depth={depth})')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / f'spike_rates_depth{depth}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Spike rate distributions per layer
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    for i, (ax, stats) in enumerate(zip(axes, spike_stats)):
        ax.hist(stats['mean_rate'], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Spike Rate')
        ax.set_ylabel('Number of Neurons')
        ax.set_title(f'Layer {i+1}')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / f'spike_distributions_depth{depth}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Heatmap of neuron activity across layers
    fig, ax = plt.subplots(figsize=(12, max(4, num_layers)))
    
    # Sample up to 50 neurons per layer for visualization
    max_neurons = 50
    data = []
    for stats in spike_stats:
        rates = stats['mean_rate']
        if len(rates) > max_neurons:
            indices = np.linspace(0, len(rates)-1, max_neurons, dtype=int)
            rates = rates[indices]
        data.append(rates)
    
    # Pad arrays to same length for heatmap
    max_len = max(len(d) for d in data)
    padded_data = np.array([np.pad(d, (0, max_len - len(d)), constant_values=np.nan) 
                            for d in data])
    
    im = ax.imshow(padded_data, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Neuron index')
    ax.set_ylabel('Layer')
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f'Layer {i+1}' for i in range(num_layers)])
    ax.set_title(f'spike rate heatmap (depth={depth})')
    plt.colorbar(im, ax=ax, label='spike rate')
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / f'spike_heatmap_depth{depth}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def plot_depth_comparison(all_results, output_dir):
    """Compare metrics across different depths"""
    depths = sorted(all_results.keys())
    
    # Extract cosine similarities for each depth
    cos_sims = {}
    for depth in depths:
        results = all_results[depth]
        cos_vals = [r['global_metrics']['cosine_similarity'] for r in results]
        cos_sims[depth] = (np.mean(cos_vals), np.std(cos_vals))
    
    # Plot cosine similarity vs depth
    fig, ax = plt.subplots(figsize=(10, 6))
    depths_list = list(cos_sims.keys())
    means = [cos_sims[d][0] for d in depths_list]
    stds = [cos_sims[d][1] for d in depths_list]
    
    ax.errorbar(depths_list, means, yerr=stds, marker='o', markersize=8, 
                linewidth=2, capsize=5, label='WP vs SGD')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect alignment')
    ax.set_xlabel('network depth', fontsize=12)
    ax.set_ylabel('Cosine similarity', fontsize=12)
    ax.set_title('WP-SGD Alignment vs Network Depth', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    # ax.set_ylim([0.9, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'cosine_vs_depth.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot all metrics vs depth
    metric_names = ['pearson_correlation', 'sign_agreement', 'relative_error', 'norm_ratio']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        metric_vals = {}
        for depth in depths:
            results = all_results[depth]
            vals = [r['global_metrics'][metric] for r in results]
            metric_vals[depth] = (np.mean(vals), np.std(vals))
        
        depths_list = list(metric_vals.keys())
        means = [metric_vals[d][0] for d in depths_list]
        stds = [metric_vals[d][1] for d in depths_list]
        
        ax.errorbar(depths_list, means, yerr=stds, marker='o', 
                   markersize=6, linewidth=2, capsize=5)
        ax.set_xlabel('network depth', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if 'error' in metric:
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'all_metrics_vs_depth.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def plot_spike_rates_vs_depth(all_spike_stats, output_dir):
    """Plot how spike rates change with depth"""
    depths = sorted(all_spike_stats.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for depth in depths:
        spike_stats = all_spike_stats[depth]
        layer_indices = range(1, len(spike_stats) + 1)
        mean_rates = [stats['overall_mean'] for stats in spike_stats]
        std_rates = [stats['overall_std'] for stats in spike_stats]
        
        ax.errorbar(layer_indices, mean_rates, yerr=std_rates, 
                   marker='o', label=f'Depth {depth}', capsize=3, linewidth=2)
    
    ax.set_xlabel('Layer index', fontsize=12)
    ax.set_ylabel('Mean spike rate', fontsize=12)
    ax.set_title('Spike rate by layer for different network depths (Randman)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'spike_rates_by_depth.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
def run_experiment_for_depth(depth, args, loader, output_dir, input_dim, output_dim, device):
    """Run the experiment for a single depth"""
    print(f"\n{'='*60}")
    print(f"Running experiment for depth = {depth}")
    print(f"{'='*60}")
    
    # Create model
    model = DualSNN(
        inDim=input_dim, 
        hidden=args.hidden, 
        nClass=output_dim, 
        beta=args.beta, 
        thr_wp=args.threshold, 
        thr_sgd=args.threshold, 
        eq31=args.eq31, 
        depth=depth
    )
    # Run orthogonal noise 
    results = []
    num_params = sum(p.numel() for p in model.wp.parameters())
    for h in args.h_values:
        print(f"Testing h={h} . . . . . .")
        it = iter(loader)
        
        for batch_idx in range(args.batches):
            print(f"    Batch {batch_idx}/{args.batches}")
            
            batch_data = next(it)
            
            if isinstance(batch_data, (tuple, list)):
                xb, yb = batch_data[0], batch_data[1]
            else:
                xb, yb = batch_data, torch.zeros(xb.shape[0])
                    
            result = cosine_similarity_wp_sgd_orthogonal(
                model, xb, yb, h=h, include_layers=None, device=device
            )
            result['h'] = h
            result['batch'] = batch_idx
            result['depth'] = depth
            results.append(result)
    
    # Analyze spiking activity
    print(f"  Analyzing spiking activity...")
    spike_stats = analyze_spiking_activity(model, loader, device, num_batches=5)
    
    # Save results
    results_file = output_dir / 'data' / f'results_depth{depth}.json'
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in results:
            r_copy = r.copy()
            if 'per_param_metrics' in r_copy:
                del r_copy['per_param_metrics']  # Too large for JSON
            json_results.append(r_copy)
        json.dump(json_results, f, indent=2)
    
    # Generate plots for this depth
    print(f"  Generating plots...")
    plot_per_layer_cosine(results, output_dir, suffix=f"_depth{depth}")
    plot_spiking_activity(spike_stats, output_dir, depth)
    
    return results, spike_stats

def main():
    args = parse_args()
    device, output_dir = setup_experiment(args)

    print(f"Experiment: {args.experiment_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Depth range: {args.depth_min} to {args.depth_max}")
    print(f"h values: {args.h_values}")
    print(f"eq31 is {args.eq31}")
    # Generate dataset
    print("\nGenerating dataset...")
    
    datasets_list, input_dim, output_dim = get_dataset(args)
    ds_train, ds_valid, ds_test = datasets_list
    
    train_loader = torch.utils.data.DataLoader(
        ds_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
    )
    
    print("intializing wandb")
    run = wandb.init(
        entity="igdovis-radboud-university",
        project="snn-wp-vs-sgd-depth-randman",
        config={
            "depth": args.depth_max,
        }
    )
    # Run experiments for each depth
    all_results = {}
    all_spike_stats = {}
    
    for depth in range(args.depth_min, args.depth_max + 1):
        run.log({"depth": depth})
        results, spike_stats = run_experiment_for_depth(
            depth, args, train_loader, output_dir, input_dim, output_dim, device
        )
        all_results[depth] = results
        all_spike_stats[depth] = spike_stats
    run.finish()
    # Generate comparison plots
    print("\nGenerating comparison plots. . . . . . .. . ....  . .")
    plot_depth_comparison(all_results, output_dir)
    plot_spike_rates_vs_depth(all_spike_stats, output_dir)
    
    # Plot metrics vs h for the deepest network
    max_depth = args.depth_max
    plot_metrics_vs_h(all_results[max_depth], output_dir)
    
    # Save summary statistics
    summary = {}
    for depth in all_results.keys():
        results = all_results[depth]
        cos_vals = [r['global_metrics']['cosine_similarity'] for r in results]
        summary[f'depth_{depth}'] = {
            'cosine_mean': float(np.mean(cos_vals)),
            'cosine_std': float(np.std(cos_vals)),
            'num_results': len(results)
        }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    for depth in sorted(all_results.keys()):
        info = summary[f'depth_{depth}']
        print(f"  Depth {depth}: Cosine = {info['cosine_mean']:.4f} ± {info['cosine_std']:.4f}")


if __name__ == "__main__":
    main()




