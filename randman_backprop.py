### backprop with surrogate gradient approach to solve Randman task

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

from generate_randman import generate_randman
import numpy as np
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

dt = 2e-3
nb_time_steps = 100
nb_inputs = 20
nb_classes = 10
batch_size = 64
ds_train, ds_valid, ds_test = generate_randman(dim_manifold=1, nb_classes=nb_classes, plot=False)

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(ds_valid, batch_size=256, shuffle=False)
test_loader  = torch.utils.data.DataLoader(ds_test,  batch_size=256, shuffle=False)

class SurrogateSNN(nn.Module):
    def __init__(self, inDim=20, hidden=128, nClass=10, beta=0.95, thr=0.8):
        super().__init__()
        self.fc1  = nn.Linear(inDim, hidden, bias=True)
        self.lif1 = snn.Leaky(
            beta=beta, threshold=thr, reset_mechanism="subtract",
            spike_grad=surrogate.fast_sigmoid(slope=25.0) 
        )
        self.fc2  = nn.Linear(hidden, nClass, bias=True)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, returnDiag=False):
        B, T, D = x.shape
        x = x.to(dtype=torch.float32, device=device)
        x =  x  #

        # init mem to zeros on every forward (bptt through T)
        mem1 = torch.zeros(B, self.fc1.out_features, device=device)

        logitsAcc = torch.zeros(B, self.fc2.out_features, device=device)
        spikeCounts = torch.zeros_like(mem1)

        for t in range(T):
            cur = self.fc1(x[:, t, :])            # (B,H)
            spk1, mem1 = self.lif1(cur, mem1)     # (B,H), (B,H)
            spikeCounts = spikeCounts + spk1
            logitsAcc = logitsAcc + self.fc2(spk1)

        # time-avg logits + small rate skip to stabilize early training
        logits = logitsAcc / T
        logits = logits + 0.1 * self.fc2(spikeCounts / (T + 1e-6))
        return logits

model = SurrogateSNN(inDim=nb_inputs, hidden=128, nClass=nb_classes, beta=0.95, thr=0.15).to(device)

def loaderAccuracy(m, loader):
    m.eval()
    corr = tot = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = torch.as_tensor(yb, device=device)
            if yb.dtype != torch.long:
                yb = (yb.argmax(dim=-1) if yb.ndim > 1 else yb).long()
            pred = m(xb).argmax(dim=-1)
            corr += (pred == yb).sum().item()
            tot  += yb.numel()
    return corr / max(1, tot)
#### training setup
lr = 3e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
epochs = 25
clipGrad = 1.0

bestVal = float("inf")
for epoch in range(1, epochs + 1):
    model.train()
    runningLoss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = torch.as_tensor(yb, device=device)
        if yb.dtype != torch.long:
            yb = (yb.argmax(dim=-1) if yb.ndim > 1 else yb).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)       
        loss = F.cross_entropy(logits, yb)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clipGrad)
        optimizer.step()

        runningLoss += float(loss.item())

    # epoch metrics
    trainAcc = loaderAccuracy(model, train_loader)
    valAcc   = loaderAccuracy(model, valid_loader)

    # probe batch diags (use first batch of valid)

    print(
        f"epoch {epoch:03d} | "
        f"loss(train) {runningLoss/len(train_loader):.4f} | "
        f"acc(train) {trainAcc:.3f} | acc(val) {valAcc:.3f} | "
    )

###### test 
testAcc = loaderAccuracy(model, test_loader)
print(f"test acc: {testAcc*100:.2f}%")