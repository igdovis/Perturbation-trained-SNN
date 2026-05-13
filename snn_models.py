import math
import torch
import torch.nn as nn
import snntorch as snn

def init_eq31(fc, alpha=1.9, std=0.05, freeze_bias=False, eq31_alpha=1.0):
    n = fc.in_features
    mu = ((1.0 - 1.0 / math.sqrt(n)) / n) * eq31_alpha
    with torch.no_grad():
        fc.weight.fill_(mu)
        torch.nn.init.normal_(fc.weight, mu, std)
        if fc.bias is not None:
            fc.bias.zero_()
            fc.bias.requires_grad_(not freeze_bias)

class SimpleSNN(nn.Module):
    # Feedforward SNN with Leaky LIF neurons
    # x: number of hidden layers
    def __init__(self, inDim=20, hidden=128, nClass=10, beta=0.95, 
                 thr=1.0, spikeGrad=None, eq31=False, depth=1, eq31_alpha=1.0):
        super().__init__()
        self.device = torch.device("cuda" if     torch.cuda.is_available() else "cpu")
        self.inDim = inDim
        self.depth = depth
        self.fcs = nn.ModuleList()
        self.lifs = nn.ModuleList()
        self.hidden = hidden
        self.nClass = nClass
        self.threshold = thr

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
                    init_eq31(fc, freeze_bias=False, eq31_alpha=eq31_alpha)
                init_eq31(self.fc_out, freeze_bias=False, eq31_alpha=eq31_alpha)
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
                 thr_wp=1.0, thr_sgd=1.0, eq31=False, depth=1, surrogate_fn=None, eq31_alpha: float = 1.0):
        super().__init__()
        import copy
        self.eq31_alpha = eq31_alpha
        base = SimpleSNN(inDim, hidden, nClass, beta, thr_wp, 
                        spikeGrad=None, eq31=eq31, depth=depth, eq31_alpha=eq31_alpha)
        state0 = copy.deepcopy(base.state_dict())
        self.wp = base
        self.sgd = SimpleSNN(inDim, hidden, nClass, beta, thr_sgd,
                            spikeGrad=surrogate_fn, 
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
    