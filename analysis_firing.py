import numpy as np
import torch

@torch.no_grad()
def measure_firing_stats(model, loader, device, num_batches=3, thr=1.0, near_delta=0.1, rare_k=2):
    """
    Measures firing regime on model.sgd
    Returns:
      per_layer: list of dicts with distributions + fractions
      summary: dict of scalars for shortlisting
    """
    model.sgd.to(device)
    model.sgd.eval()

    depth = model.sgd.depth

    # per layer accumulators
    rate_sums = [None] * depth              # sum of per neuron rates over batches
    count_sums = [None] * depth             # sum of per neuron spike counts over batches
    fired_any = [None] * depth              # per neuron bool across batches
    near_sum = [0.0] * depth
    near_cnt = [0.0] * depth

    for bi, batch in enumerate(loader):
        if bi >= num_batches:
            break
        xb, yb = batch[0], batch[1]
        xb = xb.to(device)

        logits, traces = model.forward_sgd(xb, record=True)

        for li in range(depth):
            spk = traces["spk"][li].float()     # (B,T,H)
            mem = traces["mem"][li].float()     # (B,T,H)
            H = spk.shape[-1]

            per_neuron_rate = spk.mean(dim=(0, 1)).detach().cpu()      # (H,)
            per_neuron_cnt = spk.sum(dim=(0, 1)).detach().cpu().long() # (H,)

            if rate_sums[li] is None:
                rate_sums[li] = per_neuron_rate.clone()
                count_sums[li] = per_neuron_cnt.clone()
                fired_any[li] = (per_neuron_cnt > 0)
            else:
                rate_sums[li] += per_neuron_rate
                count_sums[li] += per_neuron_cnt
                fired_any[li] |= (per_neuron_cnt > 0)

            near = ((mem - thr).abs() < near_delta).float()
            near_sum[li] += float(near.sum().item())
            near_cnt[li] += float(near.numel())

    per_layer = []
    for li in range(depth):
        if rate_sums[li] is None:
            continue

        avg_rate = (rate_sums[li] / max(1, num_batches)).numpy()     # (H,)
        cnts = count_sums[li].numpy()                                
        frac_firing = float(fired_any[li].float().mean().item())

        non_mask = (cnts == 0)
        rare_mask = (cnts > 0) & (cnts <= rare_k)
        fire_mask = (cnts > rare_k)

        per_layer.append({
            "layer": li,
            "overallMean": float(avg_rate.mean()),
            "overallStd": float(avg_rate.std()),
            "fracFiring": frac_firing,
            "fracNon": float(non_mask.mean()),
            "fracRare": float(rare_mask.mean()),
            "fracActive": float(fire_mask.mean()),
            "nearThrFrac": float(near_sum[li] / max(1.0, near_cnt[li])),
            "rateDist": avg_rate.astype(np.float32),   # perneuron rates 
        })

    # summary scalars for shortlisting
    means = np.array([p["overallMean"] for p in per_layer], dtype=float)
    fracs = np.array([p["fracFiring"] for p in per_layer], dtype=float)
    nears = np.array([p["nearThrFrac"] for p in per_layer], dtype=float)

    summary = {
        "meanRateAvg": float(means.mean()) if len(means) else 0.0,
        "meanRateMin": float(means.min()) if len(means) else 0.0,
        "meanRateStdAcrossLayers": float(means.std()) if len(means) else 0.0,
        "fracFiringAvg": float(fracs.mean()) if len(fracs) else 0.0,
        "fracFiringMin": float(fracs.min()) if len(fracs) else 0.0,
        "nearThrAvg": float(nears.mean()) if len(nears) else 0.0,
        "nearThrMin": float(nears.min()) if len(nears) else 0.0,
    }

    return per_layer, summary
