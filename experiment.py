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

from snn_models import DualSNN
from data_utils import get_dataset
from wp_sgd_metrics import cosine_similarity_wp_sgd_orthogonal, plot_per_layer_cosine


# TODO clean this up
def parse_args():
    parser = argparse.ArgumentParser(
        description='SNN WP vs SGD depth cosine experiment for Randman dataset')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='randman', 
                        choices=['randman', 'shd'], help='Dataset to use')
    

    # Surrogate gradient parameters
    parser.add_argument('--surrogate', type=str, default='fast_sigmoid',
                        choices=['fast_sigmoid', 'sigmoid', 'atan', 'triangular'],
                        help='Surrogate gradient function to use')
    parser.add_argument('--slope', type=float, default=25.0, help='Slope for surrogate gradient')

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
    
    # Analysis options
    parser.add_argument('--analyze_zero_grads', action='store_true', 
                        help='Analyze effect of zero gradients')
    parser.add_argument('--plot_membrane_voltages', action='store_true',
                        help='Plot membrane voltage distributions and traces')
    parser.add_argument('--analyze_surrogate', action='store_true',
                        help='Analyze surrogate gradient behavior')
    
    return parser.parse_args()



def setup_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / 'plots').mkdir(exist_ok=True)
    (output_dir / 'data').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'analysis').mkdir(exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return device, output_dir

def get_surrogate_function(surrogate_name, slope):
    if surrogate_name == 'fast_sigmoid':
        return surrogate.fast_sigmoid(slope=slope)
    elif surrogate_name == 'sigmoid':
        return surrogate.sigmoid(slope=slope)
    elif surrogate_name == 'atan':
        return surrogate.atan(alpha=slope)
    elif surrogate_name == 'triangular':
        return surrogate.triangular(slope=slope)
    else:
        raise ValueError(f"Unknown surrogate: {surrogate_name}")


def analyze_spiking_activity(model, loader, device, num_batches=4):
    """Enhanced spiking activity analysis"""
    model.sgd.to(device)
    model.sgd.eval()
    
    all_spike_rates = [[] for _ in range(model.sgd.depth)]
    all_mem_stats = [[] for _ in range(model.sgd.depth)]
    all_firing_neurons = [[] for _ in range(model.sgd.depth)]
    
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
            
            for layer_idx in range(model.sgd.depth):
                spk = traces['spk'][layer_idx]  # (B, T, H)
                mem = traces['mem'][layer_idx]
                
                spike_rate = spk.float().mean(dim=(0, 1))  # (H,)
                all_spike_rates[layer_idx].append(spike_rate.cpu())
                
                # Count neurons that fire at least once
                fired_mask = spk.sum(dim=(0, 1)) > 0  # (H,)
                all_firing_neurons[layer_idx].append(fired_mask.float().cpu())
                
                mem_mean = mem.mean(dim=(0, 1))
                mem_std = mem.std(dim=(0, 1))
                all_mem_stats[layer_idx].append({
                    'mean': mem_mean.cpu(),
                    'std': mem_std.cpu()
                })
    
    spike_stats = []
    for layer_idx in range(model.sgd.depth):
        rates = torch.stack(all_spike_rates[layer_idx])
        firing_masks = torch.stack(all_firing_neurons[layer_idx])
        
        spike_stats.append({
            'mean_rate': rates.mean(dim=0).numpy(),
            'std_rate': rates.std(dim=0).numpy(),
            'overall_mean': float(rates.mean()),
            'overall_std': float(rates.std()),
            'fraction_firing': float(firing_masks.mean()),
            'rate_distribution': rates.flatten().numpy(),  #  violin plots
        })

    return spike_stats
def analyze_surrogate_for_nonfiring(model, loader, device, num_batches=2, rare_k=2):
    """
    Analyze what gradients are assigned to neurons that never fire.
    We try to understand if the surrogate gradient is trying to increase firing rate.
    Split neurons into:
      - non:   0 spikes across num_batches
      - rare:  1..rare_k spikes across num_batches
      - firing: >rare_k spikes across num_batches
    """
    model.sgd.to(device)
    model.sgd.train()
    spike_counts = [None] * model.sgd.depth

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx >= num_batches:
                break

            xb, yb = batch_data[0], batch_data[1]
            xb = xb.to(device)

            logits, traces = model.forward_sgd(xb, record=True)

            for layer_idx in range(model.sgd.depth):
                spk = traces['spk'][layer_idx]               # (B, T, H)
                counts = spk.sum(dim=(0, 1)).to(torch.long)  # (H,)

                if spike_counts[layer_idx] is None:
                    spike_counts[layer_idx] = counts.clone()
                else:
                    spike_counts[layer_idx] += counts

    # compute gradients and summarize by group
    analysis_per_layer = []

    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= 1:   # one gradient batch is enough for this diagnostic???
            break

        xb, yb = batch_data[0], batch_data[1]
        xb = xb.to(device)
        yb = torch.as_tensor(yb, device=device).long()

        model.sgd.zero_grad(set_to_none=True)
        logits = model.forward_sgd(xb, record=False)
        loss = F.cross_entropy(logits, yb)
        loss.backward()

        for layer_idx in range(model.sgd.depth):
            if spike_counts[layer_idx] is None:
                continue

            counts = spike_counts[layer_idx].detach().cpu()  # (H,)

            non_mask  = (counts == 0)
            rare_mask = (counts > 0) & (counts <= rare_k)
            fire_mask = (counts > rare_k)

            weight_grad = model.sgd.fcs[layer_idx].weight.grad  # (H, prev)
            if weight_grad is None:
                continue

            wg = weight_grad.detach().cpu().abs()

            def masked_mean(mask):
                if mask.sum().item() == 0:
                    return float('nan')
                return float(wg[mask].mean().item())

            analysis_per_layer.append({
                'layer': layer_idx,
                'rare_k': int(rare_k),

                'non_grad_mean':  masked_mean(non_mask),
                'rare_grad_mean': masked_mean(rare_mask),
                'fire_grad_mean': masked_mean(fire_mask),

                'num_non':  int(non_mask.sum().item()),
                'num_rare': int(rare_mask.sum().item()),
                'num_fire': int(fire_mask.sum().item()),
            })

    return analysis_per_layer
    
def plot_spiking_activity_violin(spike_stats, output_dir, depth):
    """Enhanced spiking activity plots with violin plots"""
    num_layers = len(spike_stats)
    
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5), squeeze=False)
    axes = axes[0]
    
    for i, stats in enumerate(spike_stats):
        rates = stats['rate_distribution']
        parts = axes[i].violinplot([rates], positions=[0], widths=0.7,
                                    showmeans=True, showextrema=True)
        
        # Color the violin plots
        for pc in parts['bodies']:
            pc.set_facecolor('skyblue')
            pc.set_alpha(0.7)
        
        axes[i].set_ylabel('Spike Rate')
        axes[i].set_title(f'Layer {i+1}\n(μ={stats["overall_mean"]:.3f})')
        axes[i].set_xticks([])
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add text with fraction firing
        axes[i].text(0, axes[i].get_ylim()[1]*0.9, 
                    f'{stats["fraction_firing"]*100:.1f}% fire',
                    ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'Firing rate distributions (depth={depth})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / f'spike_violin_depth{depth}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # summary statistics across layers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    layer_names = [f'L{i+1}' for i in range(num_layers)]
    mean_rates = [stats['overall_mean'] for stats in spike_stats]
    std_rates = [stats['overall_std'] for stats in spike_stats]
    fraction_firing = [stats['fraction_firing'] for stats in spike_stats]
    
    # Mean firing rates
    ax1.bar(layer_names, mean_rates, yerr=std_rates, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_ylabel('Mean firing rate')
    ax1.set_title(f'Average firing rate per layer (depth={depth})')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Fraction of neurons that fire
    ax2.bar(layer_names, fraction_firing, alpha=0.7, color='coral')
    ax2.set_ylabel('Fraction of Neurons That Fire')
    ax2.set_title(f'Active Neurons per Layer (depth={depth})')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / f'spike_summary_depth{depth}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_membrane_voltages(traces, output_dir, depth, num_neurons=5, thr=1.0):
    """Plot membrane voltage traces and distributions"""
    if traces is None or traces['mem'] is None:
        return
    if traces['mem'] is None:
        return
    num_layers = len(traces['mem'])
    if num_layers == 0:
        return
    # Sample membrane voltage traces for a few neurons
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 3*num_layers), squeeze=False)
    axes = axes.flatten()

    for layer_idx in range(num_layers):
        mem = traces['mem'][layer_idx]  # (B, T, H)
        spk = traces['spk'][layer_idx]

        mem_sample = mem[0, :, :num_neurons].detach().cpu().numpy()
        spk_sample = spk[0, :, :num_neurons].detach().cpu().numpy()

        ax = axes[layer_idx]
        for neuron_idx in range(mem_sample.shape[1]):
            ax.plot(mem_sample[:, neuron_idx], alpha=0.7, label=f'Neuron {neuron_idx}')
            spike_times = np.where(spk_sample[:, neuron_idx] > 0.5)[0]
            if spike_times.size > 0:
                ax.scatter(spike_times, mem_sample[spike_times, neuron_idx],
                           marker='v', s=40, alpha=0.8)

        ax.set_ylabel('Membrane potential')
        ax.set_xlabel('Time step')
        ax.set_title(f'Layer {layer_idx+1} - mem voltage traces')
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(num_neurons, 5), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / f'membrane_traces_depth{depth}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # distributions (spike vs non-spike)
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5), squeeze=False)
    axes = axes[0]

    for layer_idx in range(num_layers):
        mem = traces['mem'][layer_idx].detach().cpu().numpy()
        spk = traces['spk'][layer_idx].detach().cpu().numpy()

        all_mems = mem.reshape(-1)
        spike_mask = (spk.reshape(-1) > 0.5)

        spike_mems = all_mems[spike_mask]
        non_spike_mems = all_mems[~spike_mask]

        ax = axes[layer_idx]
        ax.hist(non_spike_mems, bins=50, alpha=0.6, label='No spike', density=True)

        if spike_mems.size > 0:
            ax.hist(spike_mems, bins=50, alpha=0.6, label='Spike', density=True)

        ax.axvline(thr, linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Membrane potential')
        ax.set_ylabel('Density')
        ax.set_title(f'Layer {layer_idx+1}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f'Membrane potential distributions (depth={depth})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / f'membrane_distributions_depth{depth}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

def plot_surrogate_gradient_function(surrogate_name, slope, output_dir, thr=1.0):
    u = np.linspace(-2, 4, 1000)
    u_tensor = torch.tensor(u, dtype=torch.float32, requires_grad=True)

    surrogate_fn = get_surrogate_function(surrogate_name, slope)

    # forward gives spikes (step-like), backward gives surrogate gradient
    spk = surrogate_fn(u_tensor - thr)           # shape (1000,)
    spk.sum().backward()                         # accumulate grads into u_tensor.grad
    spike_grad_vals = u_tensor.grad.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # True spike function (for reference)
    spike_vals = (u >= thr).astype(float)
    ax1.plot(u, spike_vals, linewidth=2, label='True spike function')
    ax1.axvline(thr, linestyle='--', alpha=0.6, label='Threshold')
    ax1.set_xlabel('Membrane Potential (u)')
    ax1.set_ylabel('Spike Output')
    ax1.set_title('Spike Function (Heaviside)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Actual surrogate gradient (from backward pass)
    ax2.plot(u, spike_grad_vals, linewidth=2, label=f'{surrogate_name} grad (slope={slope})')
    ax2.axvline(thr, linestyle='--', alpha=0.6, label='Threshold')
    ax2.set_xlabel('Membrane Potential (u)')
    ax2.set_ylabel('d spike / d u')
    ax2.set_title(f'Surrogate Gradient')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'analysis' / f'surrogate_function_{surrogate_name}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_zero_gradient_analysis(zero_analysis, output_dir, suffix=""):
    """Plot analysis of zero gradients"""
    layers = sorted(zero_analysis.keys())
    
    wp_zeros = [zero_analysis[l]['wp_zero_fraction'] for l in layers]
    sgd_zeros = [zero_analysis[l]['sgd_zero_fraction'] for l in layers]
    both_zeros = [zero_analysis[l]['both_zero_fraction'] for l in layers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(layers))
    width = 0.25
    
    # Zero gradient fractions
    ax1.bar(x - width, wp_zeros, width, label='WP', alpha=0.8)
    ax1.bar(x, sgd_zeros, width, label='SGD', alpha=0.8)
    ax1.bar(x + width, both_zeros, width, label='Both', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, rotation=45, ha="right")
    ax1.set_ylabel('Fraction of Zero Gradients')
    ax1.set_title(f'Zero Gradient Analysis {suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Mean absolute gradient values
    wp_means = [zero_analysis[l]['wp_mean_abs'] for l in layers]
    sgd_means = [zero_analysis[l]['sgd_mean_abs'] for l in layers]
    
    ax2.bar(x - width/2, wp_means, width, label='WP', alpha=0.8)
    ax2.bar(x + width/2, sgd_means, width, label='SGD', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, rotation=45, ha="right")
    ax2.set_ylabel('Mean |Gradient|')
    ax2.set_title(f'Gradient Magnitudes {suffix}')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / f'zero_gradient_analysis{suffix}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

def plot_nonfiring_gradient_analysis(analysis, output_dir, depth):
    if not analysis:
        return

    layers = [a['layer'] for a in analysis]

    non_g = [a['non_grad_mean']  for a in analysis]
    rare_g = [a['rare_grad_mean'] for a in analysis]
    fire_g = [a['fire_grad_mean'] for a in analysis]

    non_n = [a['num_non']  for a in analysis]
    rare_n = [a['num_rare'] for a in analysis]
    fire_n = [a['num_fire'] for a in analysis]

    x = np.arange(len(layers))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # gradient magnitudes
    ax1.bar(x - width, non_g, width, label='non (0 spikes)', alpha=0.8)
    ax1.bar(x,         rare_g, width, label='rare', alpha=0.8)
    ax1.bar(x + width, fire_g, width, label='firing', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L{l+1}' for l in layers])
    ax1.set_ylabel('mean |gradient|')
    ax1.set_title(f'gradient magnitudes by spike activity (depth={depth})')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    # neuron counts
    ax2.bar(x - width, non_n, width, label='non (0 spikes)', alpha=0.8)
    ax2.bar(x,         rare_n, width, label='rare', alpha=0.8)
    ax2.bar(x + width, fire_n, width, label='firing', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{l+1}' for l in layers])
    ax2.set_ylabel('number of neurons')
    ax2.set_title(f'neuron counts by spike activity (depth={depth})')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'analysis' / f'nonfiring_gradients_depth{depth}.png',
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
    """Compare metrics across different depths with all three masking strategies"""
    depths = sorted(all_results.keys())
    
    # Extract cosine similarities for each depth (all three variants)
    cos_sims = {}
    cos_sims_wp = {}
    cos_sims_both = {}
    
    for depth in depths:
        results = all_results[depth]
        cos_vals = [r['global_metrics']['cosine_similarity'] for r in results]
        cos_vals_wp = [r['global_metrics_wp']['cosine_similarity'] for r in results]
        cos_vals_both = [r['global_metrics_both']['cosine_similarity'] for r in results]
        
        cos_sims[depth] = (np.mean(cos_vals), np.std(cos_vals))
        cos_sims_wp[depth] = (np.mean(cos_vals_wp), np.std(cos_vals_wp))
        cos_sims_both[depth] = (np.mean(cos_vals_both), np.std(cos_vals_both))
    
    # Create plot with all three variants
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    depths_list = list(cos_sims.keys())
    
    # Plot 1: All gradients (no masking)
    means = [cos_sims[d][0] for d in depths_list]
    stds = [cos_sims[d][1] for d in depths_list]
    ax1.errorbar(depths_list, means, yerr=stds, marker='o', markersize=8, 
                linewidth=2, capsize=5, label='WP vs SGD', color='steelblue')
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect alignment')
    ax1.set_xlabel('Network Depth', fontsize=12)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_title('All Gradients (No Masking)', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    means_wp = [cos_sims_wp[d][0] for d in depths_list]
    stds_wp = [cos_sims_wp[d][1] for d in depths_list]
    ax2.errorbar(depths_list, means_wp, yerr=stds_wp, marker='o', markersize=8,
                linewidth=2, capsize=5, label='WP vs SGD', color='darkgreen')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect alignment')
    ax2.set_xlabel('Network Depth', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('WP Non-Zero Only', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Both non-zero
    means_both = [cos_sims_both[d][0] for d in depths_list]
    stds_both = [cos_sims_both[d][1] for d in depths_list]
    ax3.errorbar(depths_list, means_both, yerr=stds_both, marker='o', markersize=8,
                linewidth=2, capsize=5, label='WP vs SGD', color='coral')
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect alignment')
    ax3.set_xlabel('Network Depth', fontsize=12)
    ax3.set_ylabel('Cosine Similarity', fontsize=12)
    ax3.set_title('Both Non-Zero', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'cosine_vs_depth_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional plot: All three on the same graph for easy comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.errorbar(depths_list, means, yerr=stds, marker='o', markersize=8,
               linewidth=2.5, capsize=5, label='All gradients', color='steelblue', alpha=0.7)
    ax.errorbar(depths_list, means_wp, yerr=stds_wp, marker='s', markersize=8,
               linewidth=2.5, capsize=5, label='WP non-zero only ⭐', color='darkgreen', alpha=0.9)
    ax.errorbar(depths_list, means_both, yerr=stds_both, marker='^', markersize=8,
               linewidth=2.5, capsize=5, label='Both non-zero', color='coral', alpha=0.7)
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Perfect alignment')
    ax.set_xlabel('Network Depth', fontsize=14)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_title('WP-SGD Alignment vs Network Depth\n(Comparison of Masking Strategies)', fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'cosine_vs_depth_combined.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def plot_spike_rates_vs_depth(all_spike_stats, output_dir):
    """Plot how spike rates change with depth"""
    depths = sorted(all_spike_stats.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Mean firing rates by layer
    for depth in depths:
        spike_stats = all_spike_stats[depth]
        layer_indices = range(1, len(spike_stats) + 1)
        mean_rates = [stats['overall_mean'] for stats in spike_stats]
        
        ax1.plot(layer_indices, mean_rates, marker='o', label=f'Depth {depth}', linewidth=2)
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Mean Spike Rate', fontsize=12)
    ax1.set_title('Spike Rate by Layer for Different Depths', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Fraction of neurons firing by layer
    for depth in depths:
        spike_stats = all_spike_stats[depth]
        layer_indices = range(1, len(spike_stats) + 1)
        frac_firing = [stats['fraction_firing'] for stats in spike_stats]
        
        ax2.plot(layer_indices, frac_firing, marker='o', label=f'Depth {depth}', linewidth=2)
    
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Fraction of Neurons Firing', fontsize=12)
    ax2.set_title('Active Neurons by Layer for Different Depths', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plots' / 'spike_analysis_by_depth.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    

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

def run_experiment_for_depth(depth, args, loader, output_dir, input_dim, output_dim, device):
    """Run the experiment for a single depth"""
    print(f"\n{'='*60}")
    print(f"Running experiment for depth = {depth}")
    print(f"{'='*60}")
    
    surrogate_fn = get_surrogate_function(args.surrogate, args.slope)

    # Create model
    model = DualSNN(
        inDim=input_dim, 
        hidden=args.hidden, 
        nClass=output_dim, 
        beta=args.beta, 
        thr_wp=args.threshold, 
        thr_sgd=args.threshold, 
        eq31=args.eq31, 
        depth=depth,
        surrogate_fn=surrogate_fn
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
            analyze_details = (batch_idx == 0)
            result = cosine_similarity_wp_sgd_orthogonal(
                model, xb, yb, h=h, include_layers=None, device=device,
                analyze_details = analyze_details
            )
            result['h'] = h
            result['batch'] = batch_idx
            result['depth'] = depth
            results.append(result)
    
    # Analyze spiking activity
    print(f"  Analyzing spiking activity...")
    spike_stats = analyze_spiking_activity(model, loader, device, num_batches=5)
    
    if args.analyze_surrogate:
        print(f"Analyzing surrogate gradients for non-firing neurons...")
        nonfiring_analysis = analyze_surrogate_for_nonfiring(model, loader, device, num_batches=args.batches)
    else:
        nonfiring_analysis = None
    
    # Generate plots for this depth
    print(f"  Generating plots...")
    plot_per_layer_cosine(results, output_dir, suffix=f"_depth{depth}")
    plot_spiking_activity(spike_stats, output_dir, depth)
    plot_spiking_activity_violin(spike_stats, output_dir, depth)
    # Plot zero gradient analysis
    if 'per_param_zero_analysis' in results[0]:
        plot_zero_gradient_analysis(results[0]['per_param_zero_analysis'], 
                                    output_dir, suffix=f"_depth{depth}")
    
    # Plot membrane voltages if requested
    if args.plot_membrane_voltages and 'traces' in results[0]:
        plot_membrane_voltages(results[0]['traces'], output_dir, depth)
    
    # Plot non-firing gradient analysis
    if nonfiring_analysis:
        plot_nonfiring_gradient_analysis(nonfiring_analysis, output_dir, depth)
    
    # Save results
    results_file = output_dir / 'data' / f'results_depth{depth}.json'
    with open(results_file, 'w') as f:
        json_results = []
        for r in results:
            r_copy = r.copy()
            # Remove large objects
            if 'per_param_metrics' in r_copy:
                del r_copy['per_param_metrics']
            if 'traces' in r_copy:
                del r_copy['traces']
            json_results.append(r_copy)
        json.dump(json_results, f, indent=2)
    
    return results, spike_stats, nonfiring_analysis

def main():
    args = parse_args()
    device, output_dir = setup_experiment(args)

    print(f"Experiment: {args.experiment_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Depth range: {args.depth_min} to {args.depth_max}")
    print(f"h values: {args.h_values}")
    print(f"eq31 is {args.eq31}")
    print(f"Surrogate: {args.surrogate} (slope={args.slope})")

    if args.analyze_surrogate:
        print("\nPlotting surrogate gradient function........")
        plot_surrogate_gradient_function(args.surrogate, args.slope, output_dir)
    
    
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
    all_nonfiring_analysis = {}
    
    for depth in range(args.depth_min, args.depth_max + 1):
        run.log({"depth": depth})
        results, spike_stats, nonfiring_analysis = run_experiment_for_depth(
            depth, args, train_loader, output_dir, input_dim, output_dim, device
        )
        all_results[depth] = results
        all_spike_stats[depth] = spike_stats
        if nonfiring_analysis:
            all_nonfiring_analysis[depth] = nonfiring_analysis
        
        # Log to wandb
        cos_vals = [r['global_metrics']['cosine_similarity'] for r in results]
        cos_vals_wp = [r['global_metrics_wp']['cosine_similarity'] for r in results]
        cos_vals_both = [r['global_metrics_both']['cosine_similarity'] for r in results]
        run.log({
            f'depth_{depth}_cosine_mean': np.mean(cos_vals),
            f'depth_{depth}_cosine_mean_wp': np.mean(cos_vals_wp),
            f'depth_{depth}_cosine_mean_both': np.mean(cos_vals_both),
            f'depth_{depth}_spike_rate': spike_stats[-1]['overall_mean'] if spike_stats else 0,
        })
    
    run.finish()
    # Generate comparison plots
    print("\nGenerating comparison plots. . . . . . .. . ....  . .")
    plot_depth_comparison(all_results, output_dir)
    plot_spike_rates_vs_depth(all_spike_stats, output_dir)
    
    # Plot metrics vs h for the deepest network
    max_depth = args.depth_max
    plot_metrics_vs_h(all_results[max_depth], output_dir)
    
    # Save summary statistics
    summary = {
        'config': vars(args),
        'depth_analysis': {}
    }
    
    for depth in all_results.keys():
        results = all_results[depth]
        # Extract all three variants
        cos_vals = [r['global_metrics']['cosine_similarity'] for r in results]
        cos_vals_wp = [r['global_metrics_wp']['cosine_similarity'] for r in results]
        cos_vals_both = [r['global_metrics_both']['cosine_similarity'] for r in results]
        
        summary['depth_analysis'][f'depth_{depth}'] = {
            'cosine_mean': float(np.mean(cos_vals)),
            'cosine_std': float(np.std(cos_vals)),
            'cosine_mean_wp': float(np.mean(cos_vals_wp)),
            'cosine_std_wp': float(np.std(cos_vals_wp)),
            'cosine_mean_both': float(np.mean(cos_vals_both)),
            'cosine_std_both': float(np.std(cos_vals_both)),
            'spike_rate': float(all_spike_stats[depth][-1]['overall_mean']) if all_spike_stats[depth] else 0,
            'fraction_firing': float(all_spike_stats[depth][-1]['fraction_firing']) if all_spike_stats[depth] else 0,
            'num_results': len(results)
        }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    # Print summary with all three variants
    print("\nSummary (Cosine Similarity):")
    print(f"{'Depth':<8} {'All':<18} {'WP-only ⭐':<18} {'Both':<18} {'Spike Rate':<12} {'% Fire':<8}")
    print("-" * 90)
    for depth in sorted(all_results.keys()):
        info = summary['depth_analysis'][f'depth_{depth}']
        print(f"{depth:<8} "
              f"{info['cosine_mean']:.4f}±{info['cosine_std']:.3f}    "
              f"{info['cosine_mean_wp']:.4f}±{info['cosine_std_wp']:.3f}    "
              f"{info['cosine_mean_both']:.4f}±{info['cosine_std_both']:.3f}    "
              f"{info['spike_rate']:.4f}      "
              f"{info['fraction_firing']*100:.1f}%")

if __name__ == "__main__":
    main()




