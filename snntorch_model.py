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

ds_train, ds_valid, ds_test = generate_randman(dim_manifold=1, nb_classes=nb_classes, plot=False)  
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(ds_valid, batch_size=512, shuffle=False)
test_loader  = torch.utils.data.DataLoader(ds_test,  batch_size=512, shuffle=False)

class SimpleSNN(nn.Module):
    "Linear -> LIF -> Linear"
    def __init__(self, in_dim=20, hidden=128, nclass=10, beta=0.90):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden, bias=True)
        self.lif1 = snn.Leaky(beta=beta, threshold=0.15, reset_mechanism='subtract') 
        self.fc2  = nn.Linear(hidden, nclass, bias=True)
        with torch.no_grad():
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='linear')
            self.fc1.weight.mul_(0.8)       
            self.fc1.bias.zero_()
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')
            self.fc2.bias.zero_()
    @torch.no_grad()
    def forward_logits(self, x, return_diagnostics=False):
        B, T, D = x.shape
        x = x.to(device=device, dtype=torch.float32)
        x = 2.0 * x  # input scaling, doesnt seem to work anyways
        mem1 = self.lif1.reset_mem()
        logits_acc = torch.zeros(B, nb_classes, device=device, dtype=torch.float32)
        spike_counts = torch.zeros(B, self.fc1.out_features, device=device)
        spike_sum = 0.0
        vmin, vmax = 0.0, 0.0
        for t in range(T):
            cur = x[:, t, :]                       
            h = self.fc1(cur)                   
            spk, mem1 = self.lif1(h, mem1)        
            logits_t = self.fc2(spk)             
            logits_acc.add_(logits_t)
            spike_sum += spk.sum().item()
            vmin = min(vmin, mem1.min().item())
            vmax = max(vmax, mem1.max().item())
        # reduction over time
        logits = logits_acc / T 
        if not return_diagnostics:
            return logits
        avg_spk = spike_sum / (B * T * self.fc1.out_features + 1e-8)
        diagnostics = {'avg_spk': avg_spk, 'mem_min': vmin, 'mem_max': vmax}
        return logits, diagnostics
    
model = SimpleSNN(in_dim=nb_inputs, hidden=128, nclass=nb_classes, beta=0.95).to(device)
for p in model.parameters():
    p.requires_grad_(False)
    
@torch.no_grad()
def batch_error_CE(m, xb, yb, return_diagnostics=False):
    xb = torch.as_tensor(xb, device=device, dtype=torch.float32) # (B,T,I)
    yb = torch.as_tensor(yb, device=device)
    if yb.dtype != torch.long:
        yb = (yb.argmax(dim=-1) if yb.ndim > 1 else yb).long()
    out = m.forward_logits(xb, return_diagnostics)     
    if return_diagnostics:
        logits, diagnostics = out
    else:
        logits = out
    loss = F.cross_entropy(logits, yb)
    return (float(loss.item()), diagnostics) if return_diagnostics else float(loss.item())

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

h = 0.5
eta = 0.08
epochs = 30
weight_decay = 0 

with torch.no_grad():
    for epoch in range(1, epochs+1):
        step = 0
        Du_running, spk_running = 0.0, 0.0
        for xb, yb in train_loader: 
            E_base, diag = batch_error_CE(model, xb, yb, return_diagnostics=True)

            noise = sample_noise_like_params(model)
            v, _  = normalize_to_unit(noise) 

            add_scaled_params(model, v, +h)
            E_pert = batch_error_CE(model, xb, yb)
            add_scaled_params(model, v, -h)   # restore params
            
            Du = (E_pert - E_base) / h 
            for p, vi in zip(model.parameters(), v):
                p.add_(-eta * Du * vi)
            Du_running += Du
            spk_running += diag['avg_spk']
            step += 1
            if weight_decay > 0:
                decay = 1 - eta * weight_decay
                for p in model.parameters():
                    p.mul_(decay)
                    
            if step % 20 == 0:
                print(
                    f"step {step:04d} "
                    f"E_base {E_base:.3f} -> E_pert {E_pert:.3f} | "
                    f"Du {Du:+.2e} | spk {diag['avg_spk']:.2e} | "
                    f"V[{diag['mem_min']:.2f},{diag['mem_max']:.2f}]"
                )
        # epoch metrics
        train_acc = loader_accuracy(model, train_loader)
        val_acc   = loader_accuracy(model, valid_loader)
        print(f"epoch {epoch:03d} | acc(train) {train_acc:.3f} | acc(val) {val_acc:.3f} | "
            f"Du(avg) {Du_running/step:+.2e} | spk(avg) {spk_running/step:.2e}")
        

test_acc = loader_accuracy(model, test_loader)
print("test acc:", round(test_acc * 100, 2), "%")