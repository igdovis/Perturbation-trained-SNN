import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

from generate_randman import generate_randman
import numpy as np

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dt = 2e-3
nb_time_steps = 100
nb_inputs = 20
nb_classes = 10

ds_train, ds_valid, ds_test = generate_randman(plot=False)  # returns tensors/arrays shaped (B,T,I) + labels
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(ds_valid, batch_size=512, shuffle=False)
test_loader  = torch.utils.data.DataLoader(ds_test,  batch_size=512, shuffle=False)

class SimpleSNN(nn.Module):
    "Simple SNN: Linear -> LIF -> Linear"
    def __init__(self, in_dim=20, hidden=128, nclass=10, beta=0.95):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden, bias=True)
        self.lif1 = snn.Leaky(beta=beta, threshold=0.5, reset_mechanism='subtract') 
        self.fc2  = nn.Linear(hidden, nclass, bias=True)
        with torch.no_grad():
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='linear')
            self.fc1.weight.mul_(0.8)       
            self.fc1.bias.zero_()
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')
            self.fc2.bias.zero_()
    @torch.no_grad()
    def forward_logits(self, x):
        B, T, D = x.shape
        x = x.to(device=device, dtype=torch.float32)
        # init state
        mem1 = self.lif1.reset_mem()
        logits_acc = torch.zeros(B, nb_classes, device=device, dtype=torch.float32)

        # unroll through time
        for t in range(T):
            cur = x[:, t, :]                       # (B, I)
            h    = self.fc1(cur)                   # (B, hidden)
            spk, mem1 = self.lif1(h, mem1)        # (B, hidden), (B, hidden)
            logits_t = self.fc2(spk)              # (B, C)
            logits_acc.add_(logits_t)

        # reduction over time
        logits = logits_acc / T                   # (B, C)
        return logits
    
model = SimpleSNN(in_dim=nb_inputs, hidden=128, nclass=nb_classes, beta=0.95).to(device)
for p in model.parameters():
    p.requires_grad_(False)
    
@torch.no_grad()
def batch_error_CE(m, xb, yb):
    xb = torch.as_tensor(xb, device=device, dtype=torch.float32)        # (B,T,I)
    yb = torch.as_tensor(yb, device=device)
    if yb.dtype != torch.long:
        yb = (yb.argmax(dim=-1) if yb.ndim > 1 else yb).long()
    logits = m.forward_logits(xb)                                       # (B,C)
    loss = F.cross_entropy(logits, yb)
    return float(loss.item())

@torch.no_grad()
def loader_accuracy(m, loader):
    corr = tot = 0
    for xb, yb in loader:
        xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
        yb = torch.as_tensor(yb, device=device)
        if yb.dtype != torch.long:
            yb = (yb.argmax(dim=-1) if yb.ndim > 1 else yb).long()
        pred = m.forward_logits(xb).argmax(dim=-1)
        corr += (pred == yb).sum().item()
        tot  += yb.numel()
    return corr / max(1, tot)

def sample_noise_like_params(m):
    """noise list matching model.parameters()."""
    return [torch.randn_like(p) for p in m.parameters()]

@torch.no_grad()
def normalize_to_unit(noise_list):
    norm2 = 0.0
    for n in noise_list:
        norm2 += (n*n).sum().item()
    norm = math.sqrt(norm2) + 1e-14
    return [n / norm for n in noise_list], norm

@torch.no_grad()
def add_scaled_params(m, direction_list, scale):
    """θ <- θ + scale * direction."""
    for p, d in zip(m.parameters(), direction_list):
        p.add_(scale * d)
        
#### training

h     = 0.5     # finite-difference step size
eta   = 0.05      # learning rate along u
epochs = 30
K_dirs = 1      # number of random directions to average each batch (variance reduction????)
weight_decay = 0 

with torch.no_grad():
    for epoch in range(1, epochs + 1):
        for bidx, (xb, yb) in enumerate(train_loader):
            # Average Du over K random directions
            Du_accum = 0.0
            u_accum  = [torch.zeros_like(p) for p in model.parameters()]

            for k in range(K_dirs):
                noise = sample_noise_like_params(model)
                u, _  = normalize_to_unit(noise)

                # central difference without deepcopy:
                # θ+ = θ + h u
                add_scaled_params(model, u, +h)
                E_plus  = batch_error_CE(model, xb, yb)
                # θ- = θ - h u  (from θ+ we subtract 2h u)
                add_scaled_params(model, u, -2*h)
                E_minus = batch_error_CE(model, xb, yb)
                # restore θ (add back h u)
                add_scaled_params(model, u, +h)

                Du_k = (E_plus - E_minus) / (2*h)
                Du_accum += Du_k
                for ua, ui in zip(u_accum, u):
                    ua.add_(ui)

            Du = Du_accum / K_dirs
            u  = [ua / K_dirs for ua in u_accum]

            # update θ ← θ − η Du u
            step = -eta * Du
            add_scaled_params(model, u, step)

            # optional weight decay tied to η (L2-style)
            if weight_decay > 0:
                decay = 1 - eta * weight_decay
                for p in model.parameters():
                    p.mul_(decay)

            # print a quick diagnostic once
            if epoch == 1 and bidx == 0:
                print(f"[dbg] E+={E_plus:.4f} E-={E_minus:.4f} Du={Du:.6f} step={step:.6f}")

        # epoch metrics
        train_acc = loader_accuracy(model, train_loader)
        val_acc   = loader_accuracy(model, valid_loader)
        print(f"epoch {epoch:03d} | acc(train) {train_acc:.3f} | acc(val) {val_acc:.3f}")

test_acc = loader_accuracy(model, test_loader)
print("test acc:", round(test_acc * 100, 2), "%")