import math
import sys

from generate_randman import generate_randman
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import stork
from stork.models import RecurrentSpikingModel
from stork.nodes import InputGroup, LIFGroup, ReadoutGroup
from stork.connections import Connection
from stork.generators import StandardGenerator
from stork.initializers import FluctuationDrivenCenteredNormalInitializer
from torch.nn import functional as F
dt = 2e-3
nb_time_steps = 100 
nb_inputs = 20    
nb_classes = 10
nb_spikes = 1

ds_train, ds_valid, ds_test = generate_randman(plot = False)

train_loader = torch.utils.data.DataLoader(
    ds_train, batch_size=32, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(ds_valid, batch_size=512, shuffle=False)
test_loader  = torch.utils.data.DataLoader(ds_test,  batch_size=512, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # with venv cuda is not available i think
print(f"Using device: {device}")

actfn_beta = 20
n_hidden = 128
tau_mem = 20e-3
tau_syn = 5e-3
tau_readout = nb_time_steps * dt
batch_size = 32


act_fn = stork.activations.SuperSpike
act_fn.beta = actfn_beta
model = RecurrentSpikingModel(batch_size=batch_size, nb_time_steps=nb_time_steps, nb_inputs=nb_inputs, 
                              device=device, dtype=torch.float32)
input_group = model.add_group(InputGroup(nb_inputs))
hidden_group = model.add_group(LIFGroup(n_hidden, tau_mem=tau_mem, tau_syn=tau_syn, activation=act_fn))
readout_group = model.add_group(
    ReadoutGroup(nb_classes, tau_mem=tau_readout, tau_syn=tau_syn, initial_state=-1e-2)
)
if not hasattr(model, "nb_steps"):
    setattr(model, "nb_steps", nb_time_steps)
con = model.add_connection(Connection(input_group, hidden_group))
con_ro = model.add_connection(Connection(hidden_group, readout_group))

sigma_u = 1.0
nu = nb_spikes / (nb_time_steps * dt)

initializer = FluctuationDrivenCenteredNormalInitializer(
    sigma_u=1.0, nu=nu, timestep=dt
)

con.init_parameters(initializer=initializer)
con_ro.init_parameters(initializer=initializer)
model.add_monitor(stork.monitors.SpikeCountMonitor(model.groups[1]))
model.add_monitor(stork.monitors.StateMonitor(model.groups[1], "out"))
loss_stack = stork.loss_stacks.MaxOverTimeCrossEntropy()

model.configure(
    input=input_group,
    output=readout_group,
    loss_stack=loss_stack,
    #optimizer=torch.optim.Adam,
    time_step=dt,
)

for p in model.parameters():
    p.requires_grad_(False)

@torch.no_grad()
def forward_logits(m, xb):
    out = m.forward_pass(xb, batch_size=xb.shape[0])
    # get the readout tensor
    logits = out["readout"] if (isinstance(out, dict) and "readout" in out) else out[-1]
    print(logits.shape, "logits shape before processing")
    if logits.dim() == 3:   # (B,T,C)
        logits = logits.mean(dim=1)  
    return logits

@torch.no_grad()
def batch_error_CE(m, xb, yb):
    """computes scalar cross entropy loss for one batch"""
    xb = torch.as_tensor(xb, dtype=torch.float32, device=device)
    yb = torch.as_tensor(yb, dtype=torch.long, device=device)
    logits = forward_logits(m, xb)
    print(f"logits shape: {logits.shape}, yb shape: {yb.shape}")
    print(f"logits sample: {logits[0]}, yb sample: {yb[0]}")
    print("xb shape:", xb.shape)
    loss = F.cross_entropy(logits, yb)
    m.out_loss = loss
    return float(m.out_loss.item())

def loader_accuracy(m, loader):
    """checks accuracy"""
    corr, tot = 0, 0
    for xb, yb in loader:
        xb = torch.as_tensor(xb, dtype=torch.float32, device=device)
        yb = torch.as_tensor(yb, dtype=torch.long, device=device)
        pred = forward_logits(m, xb).argmax(-1)
        corr += (pred == yb).sum().item()
        tot  += yb.numel()
    return corr / max(1, tot)

def sample_noise(model):
    """Samples Gaussian noise for all parameters of the model."""
    return [torch.randn_like(p) for p in model.parameters()]
            
@torch.no_grad()
def normalized_unit_direction(noise):
    """Returns unit direction and its norm."""
    norm2 = 0.0
    for i in noise:
        norm2 += torch.sum(i * i).item()
    norm = math.sqrt(norm2) + 1e-14
    return [i / norm for i in noise], norm

@torch.no_grad()
def perturb_weights(model, tensors, scale=1.0):
    """W <- W + scale * tensors (tensors is a list of same shape as model.parameters())."""
    for p, d in zip(model.parameters(), tensors):
        p.add_(scale * d)
        
@torch.no_grad()
def apply_weight_update_along_u(model, u, step):
    """W <- W + step * u  (u is unit direction list)."""
    for p, d in zip(model.parameters(), u):
        p.add_(step * d)
    
#### WP hyperparameters
h = 0.05 # or h
eta = 2 # learning rate
epochs = 50
weight_decay = 1e-4

#### training loop
with torch.no_grad():
    for epoch in range(1, epochs + 1):
        it = iter(train_loader)
        for _ in range(len(train_loader)):
            try:
                xb, yb = next(it)
            except StopIteration:
                break
            noise = sample_noise(model)
            u, _ = normalized_unit_direction(noise)
            params_backup = copy.deepcopy(model.state_dict())
            
            # E(theta)
            E_clean = batch_error_CE(model, xb, yb)
            
            # now add noise 
            perturb_weights(model, u, scale=h)
            # E(theta + noise)
            E_noise_plus = batch_error_CE(model, xb, yb)
            model.load_state_dict(params_backup) # reset weights
            perturb_weights(model, u, scale=-h)
            # E(theta - noise)
            E_noise_minus = batch_error_CE(model, xb, yb)
            model.load_state_dict(params_backup) # reset weights
            # directional derivative estimate
            Du = (E_noise_plus - E_noise_minus) / (2*h)
            
            # update params
            step = - eta * Du
            apply_weight_update_along_u(model, u, step)
            
            # weight decay 
            # if weight_decay > 0:
            #     decay = 1 - eta * weight_decay
            #     for p in model.parameters():
            #         p.mul_(decay)
            
            # print train and val accuracy
            if epoch == 1:   # first batch
                print(f"E0={E_clean:.4f}  E+={E_noise_plus:.4f}  E-={E_noise_minus:.4f}  Du={Du:.6f}  step={step:.6f}")
                # estimate update L2
                upd2 = sum(((step*ui)**2).sum().item() for ui in u)
                print(f"||Δθ||_2 ≈ {math.sqrt(upd2):.6f}")
        train_acc = loader_accuracy(model, train_loader)
        val_acc = loader_accuracy(model, valid_loader)
        print(f"epoch {epoch:03d} | acc(train) {train_acc:.3f} | acc(val) {val_acc:.3f}")

test_acc = loader_accuracy(model, test_loader)
print("test acc:", round(test_acc * 100, 2), "%")