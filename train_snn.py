# train_snn_v2.py
# training runner for DualSNN with:
# - choose which layers/params to train (including output layer fc_out)
# - train modes:
#     * sync: train one branch (sgd or wp) and keep the other identical for meaningful cosine diagnostics
#     * both: train sgd and wp independently, but compute cosine diagnostics on a shared "probe" snapshot
# - wp: two-sided random-direction estimator, optionally orthogonalized directions
# - wp noise scope: directions can live in trainable subspace or full parameter space
# - periodic cosine similarity diagnostics (wp direction vs sgd direction at the same weights)
# - raster plots + spiking violin summaries
# - checkpointing + csv logs + simple curve plots
#

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from snn_models import DualSNN
from data_utils import get_dataset
from wp_sgd_metrics import compute_all_metrics, per_layer_cosines, analyze_zero_gradients
from experiment import plot_per_layer_cosine, plot_spiking_activity_violin, analyze_spiking_activity


######### utils #########

def toLongLabels(yb: torch.Tensor) -> torch.Tensor:
    yb = torch.as_tensor(yb)
    if yb.dtype != torch.long:
        yb = (yb.argmax(dim=-1) if yb.ndim > 1 else yb).long()
    return yb


def setupRun(args) -> Tuple[torch.device, Path]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.experimentName is None:
        args.experimentName = datetime.now().strftime("%Y%m%d_%H%M%S")

    outDir = Path(args.outputDir) / args.experimentName
    (outDir / "plots").mkdir(parents=True, exist_ok=True)
    (outDir / "data").mkdir(parents=True, exist_ok=True)
    (outDir / "checkpoints").mkdir(parents=True, exist_ok=True)

    with open(outDir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    return device, outDir


def syncParams(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    """Copy parameters (and buffers if any) from src to dst."""
    with torch.no_grad():
        dst.load_state_dict(src.state_dict(), strict=True)


def setModuleDeviceAttr(module: torch.nn.Module, device: torch.device) -> None:
    # set .device attribute on modules that have it 
    if hasattr(module, "device"):
        module.device = device


def betaFromTauMs(tauMs: float, dtMs: float) -> float:
    # beta = exp(-dt/tau)
    tau = max(1e-6, float(tauMs))
    dt = max(1e-6, float(dtMs))
    return float(math.exp(-dt / tau))

# beta scheduler
def scheduleValue(start: float, end: float, progress01: float, schedule: str) -> float:
    p = float(np.clip(progress01, 0.0, 1.0))
    a0, a1 = float(start), float(end)
    if schedule == "linear":
        return (1.0 - p) * a0 + p * a1
    if schedule == "exp":
        if a0 <= 0 or a1 <= 0:
            return (1.0 - p) * a0 + p * a1
        return float(a0 * ((a1 / a0) ** p))
    if schedule == "cosine":
        w = 0.5 * (1.0 - np.cos(np.pi * p))
        return float((1.0 - w) * a0 + w * a1)
    return a0


def updateNeuronHyperparams(model: torch.nn.Module, beta: Optional[float]) -> None:
    with torch.no_grad():
        for m in model.modules():
            if beta is not None and hasattr(m, "beta"):
                try:
                    m.beta = float(beta)
                except Exception:
                    print(f"warning: failed to set beta on module {m}")
                    pass


################# select params to train based on layer index, bias/weight, last N layers, etc.

_hiddenPat = re.compile(r"(?:^|.*\.)fcs\.(\d+)\.(weight|bias)$")
_outPat = re.compile(r"(?:^|.*\.)fc_out\.(weight|bias)$")


def paramLayerIndex(name: str, depthHidden: int) -> Optional[int]:
    """Map parameter name -> layer index (output layer is index=depthHidden)."""
    m = _hiddenPat.match(name)
    if m:
        return int(m.group(1))
    if _outPat.match(name):
        return int(depthHidden)
    return None


def buildIncludeFn(
    depthHidden: int,
    trainLayerIdxs: Optional[List[int]],
    trainLastN: Optional[int],
    trainBiasOnly: bool,
    trainBiasLastN: Optional[int],
    includeBias: bool,
    includeOutput: bool = True,
) -> Callable[[str], bool]:
    """
    include function over parameter names.

    layer indexing includes output layer as index depthHidden.
    so "last 2 layers" at depthHidden=5 means {fcs.4, fc_out}.
    """
    maxLayer = depthHidden + (1 if includeOutput else 0)

    def include(name: str) -> bool:
        li = paramLayerIndex(name, depthHidden)
        if li is None:
            return False
        if (not includeOutput) and li == depthHidden:
            return False

        kind = "bias" if name.endswith(".bias") else "weight"
        if (not includeBias) and kind == "bias":
            return False

        # bias only
        if trainBiasOnly and kind != "bias":
            return False

        # biases in last N
        if trainBiasLastN is not None and trainBiasLastN > 0:
            if kind != "bias":
                return False
            if li < (maxLayer - trainBiasLastN):
                return False

        # should also have just weights at some point


        # explicit layer indices
        if trainLayerIdxs is not None and len(trainLayerIdxs) > 0:
            return li in trainLayerIdxs

        # last N layers
        if trainLastN is not None and trainLastN > 0:
            return li >= (maxLayer - trainLastN)

        return True

    return include


def setTrainableParams(module: torch.nn.Module, includeFn: Callable[[str], bool]) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for name, p in module.named_parameters():
        ok = includeFn(name)
        p.requires_grad_(ok)
        if ok:
            params.append(p)
    return params


########### WP ###########


@dataclass
class FlatMeta:
    name: str
    shape: torch.Size
    start: int
    end: int


def buildFlatMetas(
    namedParams: List[Tuple[str, torch.nn.Parameter]],
    includeFn: Callable[[str], bool],
) -> Tuple[List[Optional[FlatMeta]], int, List[str]]:
    metas: List[Optional[FlatMeta]] = []
    keys: List[str] = []
    offset = 0
    for name, p in namedParams:
        if not includeFn(name):
            metas.append(None)
            continue
        n = p.numel()
        metas.append(FlatMeta(name=name, shape=p.shape, start=offset, end=offset + n))
        keys.append(name)
        offset += n
    return metas, offset, keys

def buildMaskForMetas(
    namedParams: List[Tuple[str, torch.nn.Parameter]],
    metasAll: List[Optional[FlatMeta]],
    includeMaskFn: Callable[[str], bool],
    totalLen: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask over the full flat parameter space."""
    mask = torch.zeros(totalLen, device=device, dtype=torch.bool)
    for (name, _), meta in zip(namedParams, metasAll):
        if meta is None:
            continue
        if not includeMaskFn(name):
            continue
        mask[meta.start : meta.end] = True
    return mask

def addFlatToParams(
    vFlat: torch.Tensor,
    namedParams: List[Tuple[str, torch.nn.Parameter]],
    metasAll: List[Optional[FlatMeta]],
    scale: float,
) -> None:
    with torch.no_grad():
        for (_, p), meta in zip(namedParams, metasAll):
            if meta is None:
                continue
            sl, sr = meta.start, meta.end
            p.add_(scale * vFlat[sl:sr].view_as(p))

def sampleNoiseActive(activeLen: int, device: torch.device, noise: str) -> torch.Tensor:
    if noise == "rademacher":
        return (torch.randint(0, 2, (activeLen,), device=device, dtype=torch.int8) * 2 - 1).float()
    if noise == "gaussian":
        return torch.randn(activeLen, device=device)
    raise ValueError(f"unknown noise: {noise}")

def gramSchmidtUnit(v: torch.Tensor, basis: List[torch.Tensor]) -> torch.Tensor:
    for u in basis:
        v = v - (v @ u) * u
    n = v.norm().clamp_min(1e-12)
    return v / n


@torch.no_grad()
def estimateWpUpdateDirection(
    modelWp: torch.nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    includeReturnFn: Callable[[str], bool],
    includeNoiseFn: Callable[[str], bool],
    h: float,
    kSamples: int,
    noise: str,
    sampling: str,  # two_sided | orthogonal
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float, List[str]]:
    """
    Two-sided WP direction estimator.

    - includeNoiseFn defines where v has non-zero entries ("noise only in train layers" vs "noise everywhere").
    - includeReturnFn defines which params are returned in the update dict and key order.
    """
    device = xb.device
    namedWp = list(modelWp.named_parameters())

    includeAll = lambda _n: True
    metasAll, totalLenAll, _ = buildFlatMetas(namedWp, includeAll)
    if totalLenAll == 0:
        return {}, torch.empty(0, device=device), float("nan"), []

    noiseMask = buildMaskForMetas(namedWp, metasAll, includeNoiseFn, totalLenAll, device=device)
    activeIdx = noiseMask.nonzero(as_tuple=False).flatten()
    activeLen = int(activeIdx.numel())
    if activeLen == 0:
        return {}, torch.empty(0, device=device), float("nan"), []

    metasRet, totalLenRet, returnKeys = buildFlatMetas(namedWp, includeReturnFn)
    if totalLenRet == 0:
        return {}, torch.empty(0, device=device), float("nan"), []

    logitsBase = modelWp.forward_logits(xb, record=False)
    lossBase = float(F.cross_entropy(logitsBase, yb))

    dWpFlatAllAccum = torch.zeros(totalLenAll, device=device)

    basisActive: List[torch.Tensor] = []

    for _ in range(int(kSamples)):
        vActive = sampleNoiseActive(activeLen, device=device, noise=noise)

        if sampling == "orthogonal":
            if len(basisActive) > 0:
                vActive = gramSchmidtUnit(vActive, basisActive)
            else:
                vActive = vActive / vActive.norm().clamp_min(1e-12)
            basisActive.append(vActive)

        vAll = torch.zeros(totalLenAll, device=device)
        vAll.index_copy_(0, activeIdx, vActive)

        addFlatToParams(vAll, namedWp, metasAll, +h)
        lossPlus = float(F.cross_entropy(modelWp.forward_logits(xb, record=False), yb))

        addFlatToParams(vAll, namedWp, metasAll, -2.0 * h)
        lossMinus = float(F.cross_entropy(modelWp.forward_logits(xb, record=False), yb))

        addFlatToParams(vAll, namedWp, metasAll, +h)

        delta = (lossPlus - lossMinus) / (2.0 * h)

        norm2 = float(vActive.pow(2).sum().item())
        num = float(vActive.numel())
        scale = max(norm2 / max(1.0, num), 1e-12)
        gHatAll = (delta / scale) * vAll
        dWpFlatAllAccum.add_(-gHatAll)

    dWpFlatAll = dWpFlatAllAccum / float(kSamples)

    # build return dict + compact flat vector
    dWpFlatRet = torch.zeros(totalLenRet, device=device)
    dWpDict: Dict[str, torch.Tensor] = {}

    # map name->metaAll once
    nameToMetaAll = {name: meta for (name, _), meta in zip(namedWp, metasAll)}

    for (name, p), metaRet in zip(namedWp, metasRet):
        if metaRet is None:
            continue
        metaAll = nameToMetaAll[name]
        slAll, srAll = metaAll.start, metaAll.end
        upd = dWpFlatAll[slAll:srAll].view_as(p).clone()
        dWpDict[name] = upd
        dWpFlatRet[metaRet.start : metaRet.end] = upd.reshape(-1)

    return dWpDict, dWpFlatRet, lossBase, returnKeys

def applyWpUpdateManual(
    modelWp: torch.nn.Module,
    dWpDict: Dict[str, torch.Tensor],
    lr: float,
    weightDecay: float,
    includeFn: Callable[[str], bool],
) -> None:
    """
    Apply a WP update directly 
    weight decay is applied as a separate step after the update
    Update rule: p <- p + lr * dWp[name]
    """
    with torch.no_grad():
        for name, p in modelWp.named_parameters():
            if not includeFn(name):
                continue
            upd = dWpDict.get(name, None)
            if upd is None:
                continue
            p.add_(lr * upd)

        if weightDecay and weightDecay > 0:
            decay = 1.0 - lr * float(weightDecay)
            for name, p in modelWp.named_parameters():
                if not includeFn(name):
                    continue
                p.mul_(decay)


# =========== sgd =======================================================================


def computeSgdUpdateDirection(
    dual: DualSNN,
    xb: torch.Tensor,
    yb: torch.Tensor,
    includeFn: Callable[[str], bool],
    record: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float, Optional[dict], List[str]]:
    """Compute SGD surrogate update direction (-grad) for the sgd branch."""
    dual.sgd.zero_grad(set_to_none=True)

    if record:
        logits, traces = dual.forward_sgd(xb, record=True)
    else:
        logits = dual.forward_sgd(xb, record=False)
        traces = None

    lossT = F.cross_entropy(logits, yb)
    lossT.backward()

    named = list(dual.sgd.named_parameters())
    metas, totalLen, keys = buildFlatMetas(named, includeFn)
    dSgdFlat = torch.zeros(totalLen, device=xb.device)

    dSgdDict: Dict[str, torch.Tensor] = {}
    for (name, p), meta in zip(named, metas):
        if meta is None:
            continue
        grad = p.grad if p.grad is not None else torch.zeros_like(p)
        upd = (-grad).detach()
        dSgdDict[name] = upd.clone()
        dSgdFlat[meta.start : meta.end] = upd.reshape(-1)

    dual.sgd.zero_grad(set_to_none=True)
    return dSgdDict, dSgdFlat, float(lossT.detach()), traces, keys


# diagnostics


def cosineDiagnosticsFromDirs(
    dWp: Dict[str, torch.Tensor],
    dSgd: Dict[str, torch.Tensor],
    keys: List[str],
    device: torch.device,
) -> dict:
    dWpVec = torch.cat([dWp[k].reshape(-1) for k in keys], dim=0) if keys else torch.empty(0, device=device)
    dSgdVec = torch.cat([dSgd[k].reshape(-1) for k in keys], dim=0) if keys else torch.empty(0, device=device)

    gm = compute_all_metrics(dWpVec, dSgdVec, exclude_zeros=False)
    gmWp = compute_all_metrics(dWpVec, dSgdVec, exclude_zeros="wp")
    gmBoth = compute_all_metrics(dWpVec, dSgdVec, exclude_zeros="both")

    wCos, bCos, combCos = per_layer_cosines(dWp, dSgd, exclude_zeros=False)
    wCosWp, bCosWp, combCosWp = per_layer_cosines(dWp, dSgd, exclude_zeros="wp")
    wCosBoth, bCosBoth, combCosBoth = per_layer_cosines(dWp, dSgd, exclude_zeros="both")

    zeroAnalysis = analyze_zero_gradients(dWp, dSgd)

    return {
        "global_metrics": gm,
        "global_metrics_wp": gmWp,
        "global_metrics_both": gmBoth,
        "w_cos": wCos,
        "b_cos": bCos,
        "comb_cos": combCos,
        "w_cos_wp": wCosWp,
        "b_cos_wp": bCosWp,
        "comb_cos_wp": combCosWp,
        "w_cos_both": wCosBoth,
        "b_cos_both": bCosBoth,
        "comb_cos_both": combCosBoth,
        "per_param_zero_analysis": zeroAnalysis,
        "dWP_norm": float(dWpVec.norm()),
        "dSGD_norm": float(dSgdVec.norm()),
    }


# ===================== plots =


def plotTrainingCurves(trainDf: pd.DataFrame, outDir: Path, mode: str) -> None:
    if trainDf is None or len(trainDf) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    if mode == "both":
        if "accSgd" in trainDf.columns:
            ax.plot(trainDf["step"], trainDf["accSgd"], label="sgd acc")
        if "accWp" in trainDf.columns:
            ax.plot(trainDf["step"], trainDf["accWp"], label="wp acc")
    else:
        if "acc" in trainDf.columns:
            ax.plot(trainDf["step"], trainDf["acc"], label="acc")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outDir / "plots" / "train_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    if mode == "both":
        if "lossSgd" in trainDf.columns:
            ax.plot(trainDf["step"], trainDf["lossSgd"], label="sgd loss")
        if "lossWp" in trainDf.columns:
            ax.plot(trainDf["step"], trainDf["lossWp"], label="wp loss")
    else:
        if "loss" in trainDf.columns:
            ax.plot(trainDf["step"], trainDf["loss"], label="loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outDir / "plots" / "train_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------- training ----------------------------

def parseArgs():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset", type=str, default="shd", choices=["randman", "shd"])
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--nbSamples", type=int, default=1000)
    p.add_argument("--batchSize", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=100)
    # model
    p.add_argument("--depth", type=int, default=5, help="number of hidden layers")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--threshold", type=float, default=1.0)

    # tau/beta
    p.add_argument("--dtMs", type=float, default=1.0, help="timestep in ms (for tau->beta mapping)")
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--tauMs", type=float, default=None, help="if set, compute beta=exp(-dt/tau)")
    p.add_argument("--betaSchedule", type=str, default="none", choices=["none", "linear", "exp", "cosine"])
    p.add_argument("--betaEnd", type=float, default=None)

    # eq31 init
    g = p.add_mutually_exclusive_group()
    g.add_argument("--eq31", dest="eq31", action="store_true", help="enable eq31 init")
    g.add_argument("--noEq31", dest="eq31", action="store_false", help="disable eq31 init")
    p.set_defaults(eq31=False)
    p.add_argument("--eq31Alpha", type=float, default=1.0)

    # surrogate
    p.add_argument("--surrogate", type=str, default="fast_sigmoid", choices=["fast_sigmoid", "sigmoid", "atan", "triangular"])
    p.add_argument("--slope", type=float, default=25.0)

    # training mode
    p.add_argument("--trainMode", type=str, default="sync", choices=["sync", "both"])
    p.add_argument("--trainMaster", type=str, default="sgd", choices=["sgd", "wp"], help="only used in trainMode=sync")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lrSgd", type=float, default=1e-3)
    p.add_argument("--lrWp", type=float, default=1e-3)
    p.add_argument("--weightDecay", type=float, default=0.0)

    # layer selection
    p.add_argument("--trainLayerIdxs", nargs="*", type=int, default=None, help="0-based, output layer index=depth")
    p.add_argument("--trainLastN", type=int, default=0, help="train last N layers including output")
    p.add_argument("--trainBiasOnly", action="store_true")
    p.add_argument("--trainBiasLastN", type=int, default=0, help="train biases in last N layers")
    p.add_argument("--includeBiasInDiagnostics", action="store_true")

    # wp estimator
    p.add_argument("--wpH", type=float, default=0.01)
    p.add_argument("--wpK", type=int, default=32)
    p.add_argument("--wpNoise", type=str, default="rademacher", choices=["rademacher", "gaussian"])
    p.add_argument("--wpSampling", type=str, default="two_sided", choices=["two_sided", "orthogonal"])
    p.add_argument("--wpNoiseScope", type=str, default="train", choices=["train", "all"], help="noise lives in trainable subspace vs full space")

    # logging/outputs
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outputDir", type=str, default="results_train")
    p.add_argument("--experimentName", type=str, default=None)

    p.add_argument("--logEvery", type=int, default=50)
    p.add_argument("--cosineEvery", type=int, default=200)
    p.add_argument("--rasterEvery", type=int, default=500)
    p.add_argument("--spikeViolinEvery", type=int, default=0)
    p.add_argument("--ckptEvery", type=int, default=500)

    p.add_argument("--cosineProbe", type=str, default="wp", choices=["sgd", "wp"], help="only used in trainMode=both")

    return p.parse_args()

def main():
    args = parseArgs()
    device, outDir = setupRun(args)

    # dataset
    datasetsList, inDim, outDim = get_dataset(args)
    dsTrain, dsValid, dsTest = datasetsList

    trainLoader = torch.utils.data.DataLoader(dsTrain, batch_size=args.batchSize, shuffle=True, drop_last=True)
    validLoader = torch.utils.data.DataLoader(dsValid, batch_size=args.batchSize, shuffle=False, drop_last=False)

    # surrogate
    from snntorch import surrogate as snnSurrogate

    def getSurrogate(name: str, slope: float):
        if name == "fast_sigmoid":
            return snnSurrogate.fast_sigmoid(slope=slope)
        if name == "sigmoid":
            return snnSurrogate.sigmoid(slope=slope)
        if name == "atan":
            return snnSurrogate.atan(alpha=slope)
        if name == "triangular":
            return snnSurrogate.triangular(threshold=slope)
        raise ValueError(name)

    surrogateFn = getSurrogate(args.surrogate, args.slope)

    beta0 = betaFromTauMs(args.tauMs, args.dtMs) if args.tauMs is not None else float(args.beta)

    dual = DualSNN(
        inDim=inDim,
        hidden=args.hidden,
        nClass=outDim,
        beta=beta0,
        thr_wp=args.threshold,
        thr_sgd=args.threshold,
        eq31=bool(args.eq31),
        depth=args.depth,
        surrogate_fn=surrogateFn,
        eq31_alpha=float(args.eq31Alpha),
    )

    dual.wp.to(device)
    dual.sgd.to(device)
    setModuleDeviceAttr(dual.wp, device)
    setModuleDeviceAttr(dual.sgd, device)

    # selection
    trainLayerIdxs = args.trainLayerIdxs if args.trainLayerIdxs is not None else None
    trainLastN = args.trainLastN if args.trainLastN and args.trainLastN > 0 else None
    trainBiasLastN = args.trainBiasLastN if args.trainBiasLastN and args.trainBiasLastN > 0 else None
    # TODO includeBias should be arg from the user, 
    includeTrain = buildIncludeFn(
        depthHidden=args.depth,
        trainLayerIdxs=trainLayerIdxs,
        trainLastN=trainLastN,
        trainBiasOnly=args.trainBiasOnly,
        trainBiasLastN=trainBiasLastN,
        includeBias=True,
        includeOutput=True,
    )

    includeDiag = buildIncludeFn(
        depthHidden=args.depth,
        trainLayerIdxs=None,
        trainLastN=None,
        trainBiasOnly=False,
        trainBiasLastN=None,
        includeBias=bool(args.includeBiasInDiagnostics),
        includeOutput=True,
    )

    includeNoiseTrain = includeTrain
    includeNoiseAll = lambda _n: True
    includeNoiseFn = includeNoiseTrain if args.wpNoiseScope == "train" else includeNoiseAll

    optSgd = None

    if args.trainMode == "sync":
        if args.trainMaster == "sgd":
            setTrainableParams(dual.sgd, includeTrain)
            paramsSgd = [p for (n, p) in dual.sgd.named_parameters() if includeTrain(n)]
            optSgd = torch.optim.Adam(paramsSgd, lr=args.lrSgd, weight_decay=args.weightDecay)
        else:
            # WP master: no optimizer needed
            optSgd = None
    else:
        # train both: SGD uses Adam and WP uses WP learning rule
        setTrainableParams(dual.sgd, includeTrain)
        paramsSgd = [p for (n, p) in dual.sgd.named_parameters() if includeTrain(n)]
        optSgd = torch.optim.Adam(paramsSgd, lr=args.lrSgd, weight_decay=args.weightDecay)

    
    trainRows: List[dict] = []
    cosRows: List[dict] = []

    globalStep = 0
    totalSteps = max(1, args.epochs * len(trainLoader))

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        dual.train()
        for xb, yb in trainLoader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = toLongLabels(yb).to(device)

            # beta schedule or just set a beta?
            if args.betaSchedule != "none" and args.betaEnd is not None:
                betaNow = scheduleValue(beta0, float(args.betaEnd), float(globalStep) / float(totalSteps), args.betaSchedule)
                updateNeuronHyperparams(dual.wp, beta=betaNow)
                updateNeuronHyperparams(dual.sgd, beta=betaNow)

            if args.trainMode == "sync":
                if args.trainMaster == "sgd":
                    optSgd.zero_grad(set_to_none=True)
                    logits = dual.forward_sgd(xb, record=False)
                    loss = F.cross_entropy(logits, yb)
                    loss.backward()
                    optSgd.step()
                    syncParams(dual.wp, dual.sgd)
                    with torch.no_grad():
                        pred = logits.argmax(dim=1)
                        acc = float((pred == yb).float().mean())

                    logRow = {"step": globalStep, "epoch": epoch, "loss": float(loss.detach()), "acc": acc, "trainMode": "sync", "trainMaster": "sgd"}

                else:
                    dWp, _, lossBase, _ = estimateWpUpdateDirection(
                        modelWp=dual.wp,
                        xb=xb,
                        yb=yb,
                        includeReturnFn=includeTrain,
                        includeNoiseFn=includeNoiseFn,
                        h=float(args.wpH),
                        kSamples=int(args.wpK),
                        noise=str(args.wpNoise),
                        sampling=str(args.wpSampling),
                    )

                    applyWpUpdateManual(dual.wp, dWp, lr=float(args.lrWp), weightDecay=float(args.weightDecay), includeFn=includeTrain)
                    syncParams(dual.sgd, dual.wp)
                    with torch.no_grad():
                        logitsWp = dual.wp.forward_logits(xb, record=False)
                        predWp = logitsWp.argmax(dim=1)
                        accWp = float((predWp == yb).float().mean())

                    logRow = {"step": globalStep, "epoch": epoch, "loss": float(lossBase), "acc": accWp, "trainMode": "sync", "trainMaster": "wp"}
                    
            else:
                # train both
                optSgd.zero_grad(set_to_none=True)
                logitsSgd = dual.forward_sgd(xb, record=False)
                lossSgd = F.cross_entropy(logitsSgd, yb)
                lossSgd.backward()
                optSgd.step()

                with torch.no_grad():
                    predSgd = logitsSgd.argmax(dim=1)
                    accSgd = float((predSgd == yb).float().mean())

                dWp, _, lossBase, _ = estimateWpUpdateDirection(
                    modelWp=dual.wp,
                    xb=xb,
                    yb=yb,
                    includeReturnFn=includeTrain,
                    includeNoiseFn=includeNoiseFn,
                    h=float(args.wpH),
                    kSamples=int(args.wpK),
                    noise=str(args.wpNoise),
                    sampling=str(args.wpSampling),
                )

                applyWpUpdateManual(dual.wp, dWp, lr=float(args.lrWp), weightDecay=float(args.weightDecay), includeFn=includeTrain)

                with torch.no_grad():
                    logitsWp = dual.wp.forward_logits(xb, record=False)
                    predWp = logitsWp.argmax(dim=1)
                    accWp = float((predWp == yb).float().mean())

                logRow = {
                    "step": globalStep,
                    "epoch": epoch,
                    "lossSgd": float(lossSgd.detach()),
                    "accSgd": accSgd,
                    "lossWp": float(lossBase),
                    "accWp": accWp,
                    "trainMode": "both",
                }

            if globalStep % args.logEvery == 0:
                trainRows.append(logRow)

            if args.cosineEvery > 0 and globalStep % args.cosineEvery == 0:
                wpState = None
                sgdState = None
                if args.trainMode == "both":
                    wpState = copy.deepcopy(dual.wp.state_dict())
                    sgdState = copy.deepcopy(dual.sgd.state_dict())
                    if args.cosineProbe == "sgd":
                        syncParams(dual.wp, dual.sgd)
                    else:
                        syncParams(dual.sgd, dual.wp)

                dWpDiag, _, wpLossBase, diagKeys = estimateWpUpdateDirection(
                    modelWp=dual.wp,
                    xb=xb,
                    yb=yb,
                    includeReturnFn=includeDiag,
                    includeNoiseFn=(includeNoiseFn if args.wpNoiseScope == "train" else (lambda _n: True)),
                    h=float(args.wpH),
                    kSamples=int(args.wpK),
                    noise=str(args.wpNoise),
                    sampling=str(args.wpSampling),
                )

                dSgdDiag, _, sgdLoss, traces, diagKeysSgd = computeSgdUpdateDirection(
                    dual=dual,
                    xb=xb,
                    yb=yb,
                    includeFn=includeDiag,
                    record=(args.rasterEvery > 0 and globalStep % args.rasterEvery == 0),
                )

                diag = cosineDiagnosticsFromDirs(dWpDiag, dSgdDiag, diagKeysSgd, device=device)
                diag.update({
                    "step": globalStep,
                    "epoch": epoch,
                    "lossSgd": sgdLoss,
                    "lossWpBase": wpLossBase,
                    "trainMode": args.trainMode,
                    "trainMaster": args.trainMaster if args.trainMode == "sync" else "both",
                    "cosineProbe": args.cosineProbe if args.trainMode == "both" else "synced",
                })

                cosRows.append({
                    "step": globalStep,
                    "epoch": epoch,
                    "trainMode": args.trainMode,
                    "trainMaster": args.trainMaster if args.trainMode == "sync" else "both",
                    "cos_all": diag["global_metrics"]["cosine_similarity"],
                    "cos_wp": diag["global_metrics_wp"]["cosine_similarity"],
                    "cos_both": diag["global_metrics_both"]["cosine_similarity"],
                    "activeFracWp": diag["global_metrics_wp"]["active_frac"],
                    "activeFracBoth": diag["global_metrics_both"]["active_frac"],
                    "probe": diag["cosineProbe"],
                })

                plot_per_layer_cosine([diag], outDir, suffix=f"_step{globalStep:07d}")

                # raster
                if traces is not None and args.rasterEvery > 0 and globalStep % args.rasterEvery == 0:
                    spkList = traces.get("spk")
                    if isinstance(spkList, (list, tuple)):
                        for li, spk in enumerate(spkList):
                            if not torch.is_tensor(spk):
                                continue
                            s = spk[0].detach().cpu()  # (T,H)
                            s = s[:, : min(64, s.shape[1])].T
                            ys, xs = torch.nonzero(s > 0.5, as_tuple=True)
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.scatter(xs.numpy(), ys.numpy(), s=2)
                            ax.set_xlabel("time step")
                            ax.set_ylabel("neuron")
                            ax.set_title(f"raster layer={li} step={globalStep}")
                            ax.grid(True, alpha=0.2)
                            fig.tight_layout()
                            fig.savefig(outDir / "plots" / f"raster_step{globalStep:07d}_layer{li}.png", dpi=150, bbox_inches="tight")
                            plt.close(fig)

                # violin
                if args.spikeViolinEvery and globalStep % args.spikeViolinEvery == 0:
                    spikeStats = analyze_spiking_activity(dual, validLoader, device, numBatches=4)
                    plot_spiking_activity_violin(spikeStats, outDir, depth=args.depth, step=globalStep)

                with open(outDir / "data" / f"cosine_step{globalStep:07d}.json", "w", encoding="utf-8") as f:
                    json.dump(diag, f, indent=2, default=str)

                if args.trainMode == "both":
                    dual.wp.load_state_dict(wpState, strict=True)
                    dual.sgd.load_state_dict(sgdState, strict=True)

            # checkpoint
            if args.ckptEvery > 0 and globalStep % args.ckptEvery == 0 and globalStep > 0:
                ckpt = {
                    "step": globalStep,
                    "epoch": epoch,
                    "args": vars(args),
                    "wpState": dual.wp.state_dict(),
                    "sgdState": dual.sgd.state_dict(),
                    "optSgd": optSgd.state_dict() if optSgd is not None else None,
                    "optWp": None,
                }
                torch.save(ckpt, outDir / "checkpoints" / f"ckpt_step{globalStep:07d}.pt")

            globalStep += 1
        if args.trainMode == "sync":
            if args.trainMaster == "sgd":
                syncParams(dual.wp, dual.sgd)
            else:  # wp master
                syncParams(dual.sgd, dual.wp)
        # epoch end validation acc for both
        dual.eval()
        with torch.no_grad():
            correctSgd = 0
            total = 0
            for xbV, ybV in validLoader:
                xbV = xbV.to(device=device, dtype=torch.float32)
                ybV = toLongLabels(ybV).to(device)
                logits = dual.forward_sgd(xbV, record=False)
                pred = logits.argmax(dim=1)
                correctSgd += int((pred == ybV).sum().item())
                total += int(ybV.numel())
            validAccSgd = correctSgd / max(1, total)

            correctWp = 0
            totalWp = 0
            for xbV, ybV in validLoader:
                xbV = xbV.to(device=device, dtype=torch.float32)
                ybV = toLongLabels(ybV).to(device)
                logits = dual.wp.forward_logits(xbV, record=False)
                pred = logits.argmax(dim=1)
                correctWp += int((pred == ybV).sum().item())
                totalWp += int(ybV.numel())
            validAccWp = correctWp / max(1, totalWp)

        trainRows.append({"step": globalStep, "epoch": epoch, "validAccSgd": float(validAccSgd), "validAccWp": float(validAccWp), "trainMode": args.trainMode})

        trainDf = pd.DataFrame(trainRows)
        cosDf = pd.DataFrame(cosRows)
        trainDf.to_csv(outDir / "data" / "train_log.csv", index=False)
        cosDf.to_csv(outDir / "data" / "cosine_log.csv", index=False)
        plotTrainingCurves(trainDf, outDir, mode=args.trainMode)

        print(f"epoch={epoch} done. validAccSgd={validAccSgd:.4f} validAccWp={validAccWp:.4f}")

    print(f"done. results in {outDir}")


if __name__ == "__main__":
    main()
