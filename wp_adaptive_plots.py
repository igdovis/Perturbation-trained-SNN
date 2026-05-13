from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_hidden_pat = re.compile(r"(?:^|.*\.)fcs\.(\d+)\.(weight|bias)$")
_out_pat = re.compile(r"(?:^|.*\.)fc_out\.(weight|bias)$")


def _as_int_key_dict(d: Dict[Any, Any]) -> Dict[int, float]:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = float(v)
        except Exception:
            continue
    return out


def _sorted_m_keys(*dicts: Dict[Any, Any]) -> List[int]:
    keys = set()
    for d in dicts:
        keys.update(_as_int_key_dict(d).keys())
    return sorted(keys)


def _ensure_plot_dir(out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _merge_hist_dicts(dicts: List[Dict[int, float]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for d in dicts:
        d = _as_int_key_dict(d)
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + float(v)
    return out


def _parse_param_name(name: str) -> Tuple[str, int, str]:
    """
    returns:
      layer_label, layer_index, kind
    """
    m = _hidden_pat.match(name)
    if m:
        li = int(m.group(1))
        kind = m.group(2)
        return f"layer{li}", li, kind

    m = _out_pat.match(name)
    if m:
        kind = m.group(1)
        return "fc_out", 10**9, kind

    kind = "bias" if name.endswith(".bias") else "weight"
    return name, 10**8, kind


def wp_stats_to_param_df(wp_stats: dict) -> pd.DataFrame:
    per_param = wp_stats.get("per_param", {})
    rows = []

    for name, st in per_param.items():
        if not isinstance(st, dict):
            continue

        layer_label, layer_index, kind = _parse_param_name(name)

        num_used = float(st.get("num_coords_used", 0))
        num_total = float(st.get("num_coords_total", 0))
        num_escalated = float(st.get("num_escalated", 0))
        num_failed = float(st.get("num_failed", 0))

        m_all = _as_int_key_dict(st.get("m_hist_all", {}))
        m_success = _as_int_key_dict(st.get("m_hist_success", {}))
        m_failed = _as_int_key_dict(st.get("m_hist_failed", {}))

        success_total = sum(m_success.values())
        failed_total = sum(m_failed.values())
        success_m1 = m_success.get(1, 0.0)
        success_escalated = success_total - success_m1

        rows.append({
            "param": name,
            "layer_label": layer_label,
            "layer_index": layer_index,
            "kind": kind,

            "num_coords_used": num_used,
            "num_coords_total": num_total,
            "used_frac": (num_used / num_total) if num_total > 0 else np.nan,

            "num_escalated": num_escalated,
            "num_failed": num_failed,
            "success_total": success_total,
            "failed_total": failed_total,
            "success_m1": success_m1,
            "success_escalated": success_escalated,

            "success_m1_frac_used": (success_m1 / num_used) if num_used > 0 else np.nan,
            "success_escalated_frac_used": (success_escalated / num_used) if num_used > 0 else np.nan,
            "failed_frac_used": (failed_total / num_used) if num_used > 0 else np.nan,
            "escalated_frac_used": (num_escalated / num_used) if num_used > 0 else np.nan,

            "m_hist_all": m_all,
            "m_hist_success": m_success,
            "m_hist_failed": m_failed,
        })

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["layer_index", "kind", "param"]).reset_index(drop=True)


def aggregate_param_df_to_layer_df(df: pd.DataFrame, separate_kind: bool = False) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if separate_kind:
        group_cols = ["layer_label", "layer_index", "kind"]
    else:
        group_cols = ["layer_label", "layer_index"]

    rows = []
    for group_key, sub in df.groupby(group_cols, dropna=False):
        if separate_kind:
            layer_label, layer_index, kind = group_key
            group_name = f"{layer_label}.{kind}"
        else:
            layer_label, layer_index = group_key
            kind = "both"
            group_name = layer_label

        num_used = float(sub["num_coords_used"].sum())
        num_total = float(sub["num_coords_total"].sum())
        num_escalated = float(sub["num_escalated"].sum())
        num_failed = float(sub["num_failed"].sum())
        success_total = float(sub["success_total"].sum())
        success_m1 = float(sub["success_m1"].sum())
        success_escalated = float(sub["success_escalated"].sum())

        rows.append({
            "group": group_name,
            "layer_label": layer_label,
            "layer_index": layer_index,
            "kind": kind,

            "num_coords_used": num_used,
            "num_coords_total": num_total,
            "used_frac": (num_used / num_total) if num_total > 0 else np.nan,

            "num_escalated": num_escalated,
            "num_failed": num_failed,
            "success_total": success_total,
            "success_m1": success_m1,
            "success_escalated": success_escalated,

            "success_m1_frac_used": (success_m1 / num_used) if num_used > 0 else np.nan,
            "success_escalated_frac_used": (success_escalated / num_used) if num_used > 0 else np.nan,
            "failed_frac_used": (num_failed / num_used) if num_used > 0 else np.nan,
            "escalated_frac_used": (num_escalated / num_used) if num_used > 0 else np.nan,

            "m_hist_all": _merge_hist_dicts(sub["m_hist_all"].tolist()),
            "m_hist_success": _merge_hist_dicts(sub["m_hist_success"].tolist()),
            "m_hist_failed": _merge_hist_dicts(sub["m_hist_failed"].tolist()),
        })

    return pd.DataFrame(rows).sort_values(["layer_index", "kind", "group"]).reset_index(drop=True)


def _filter_df_kind(df: pd.DataFrame, kind_filter: str | None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if kind_filter is None:
        return df
    return df[df["kind"] == kind_filter].copy()


def plot_wp_global_m_histograms(wp_stats: dict, out_dir, prefix: str = "") -> None:
    """
    x-axis:
      m=1,2,...,m_max plus one extra bin "failed"
    y-axis:
      fraction of used coordinates
    - success at m=1 means worked immediately
    - success at m>1 means needed escalation and first succeeded there
    - failed means never succeeded by m_max
    """
    out_dir = _ensure_plot_dir(out_dir)

    m_success = _as_int_key_dict(wp_stats.get("m_hist_success", {}))
    num_used = float(wp_stats.get("num_coords_used", 0))
    num_failed = float(wp_stats.get("num_failed", 0))

    ms = sorted(m_success.keys())
    if len(ms) == 0 and num_failed <= 0:
        return

    labels = [str(m) for m in ms] + ["failed"]
    vals = [m_success.get(m, 0.0) for m in ms] + [num_failed]

    if num_used > 0:
        vals = [v / num_used for v in vals]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(labels)), 5))
    ax.bar(x, vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("first successful multiplier m  (step = m*h), or failed")
    ax.set_ylabel("fraction of used coordinates")
    ax.set_title("adaptive WP method: global first success distribution")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}wp_global_m_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_fraction_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, out_path: Path) -> None:
    if df is None or len(df) == 0:
        return

    x = np.arange(len(df))
    y = df[y_col].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(df)), 4.8))
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].tolist(), rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_col)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_wp_per_param_outcomes(wp_stats: dict, out_dir, prefix: str = "", kind_filter: str | None = None) -> None:
    out_dir = _ensure_plot_dir(out_dir)
    df = _filter_df_kind(wp_stats_to_param_df(wp_stats), kind_filter)
    if df is None or len(df) == 0:
        return

    labels = df["param"].tolist()
    m1 = df["success_m1_frac_used"].fillna(0.0).to_numpy()
    escal = df["success_escalated_frac_used"].fillna(0.0).to_numpy()
    fail = df["failed_frac_used"].fillna(0.0).to_numpy()

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels)), 5))
    ax.bar(x, m1, label="success @ h")
    ax.bar(x, escal, bottom=m1, label="success @ >h")
    ax.bar(x, fail, bottom=m1 + escal, label="failed")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("fraction of used coordinates")
    ax.set_xlabel("parameter")
    suffix = "" if kind_filter is None else f" ({kind_filter})"
    ax.set_title(f"adaptive wp outcomes per parameter{suffix}")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    name_suffix = "" if kind_filter is None else f"_{kind_filter}"
    plt.savefig(out_dir / f"{prefix}wp_per_param_outcomes{name_suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_wp_per_param_used_fraction(wp_stats: dict, out_dir, prefix: str = "", kind_filter: str | None = None) -> None:
    out_dir = _ensure_plot_dir(out_dir)
    df = _filter_df_kind(wp_stats_to_param_df(wp_stats), kind_filter)
    if df is None or len(df) == 0:
        return

    _plot_fraction_bar(
        df=df,
        x_col="param",
        y_col="used_frac",
        title=f"adaptive wp sampled coverage per parameter{'' if kind_filter is None else f' ({kind_filter})'}",
        ylabel="used / total coordinates",
        out_path=out_dir / f"{prefix}wp_per_param_used_fraction{'' if kind_filter is None else f'_{kind_filter}'}.png",
    )

def plot_wp_failed_fraction_per_param(wp_stats: dict, out_dir, prefix: str = "", kind_filter: str | None = None) -> None:
    out_dir = _ensure_plot_dir(out_dir)
    df = _filter_df_kind(wp_stats_to_param_df(wp_stats), kind_filter)
    if df is None or len(df) == 0:
        return

    _plot_fraction_bar(
        df=df,
        x_col="param",
        y_col="failed_frac_used",
        title=f"adaptive wp failed fraction per parameter{'' if kind_filter is None else f' ({kind_filter})'}",
        ylabel="failed / used coordinates",
        out_path=out_dir / f"{prefix}wp_failed_fraction_per_param{'' if kind_filter is None else f'_{kind_filter}'}.png",
    )


def plot_wp_escalated_fraction_per_param(wp_stats: dict, out_dir, prefix: str = "", kind_filter: str | None = None) -> None:
    out_dir = _ensure_plot_dir(out_dir)
    df = _filter_df_kind(wp_stats_to_param_df(wp_stats), kind_filter)
    if df is None or len(df) == 0:
        return

    _plot_fraction_bar(
        df=df,
        x_col="param",
        y_col="escalated_frac_used",
        title=f"adaptive wp escalated fraction per parameter{'' if kind_filter is None else f' ({kind_filter})'}",
        ylabel="escalated-success / used coordinates",
        out_path=out_dir / f"{prefix}wp_escalated_fraction_per_param{'' if kind_filter is None else f'_{kind_filter}'}.png",
    )

def plot_wp_per_param_success_heatmap(wp_stats: dict, out_dir, prefix: str = "", kind_filter: str | None = None) -> None:
    out_dir = _ensure_plot_dir(out_dir)
    df = _filter_df_kind(wp_stats_to_param_df(wp_stats), kind_filter)
    if df is None or len(df) == 0:
        return

    all_ms = set()
    for _, row in df.iterrows():
        all_ms.update(row["m_hist_success"].keys())
    ms = sorted(all_ms)
    if len(ms) == 0:
        return

    mat = np.zeros((len(df), len(ms)), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        used = float(row["num_coords_used"])
        hist = row["m_hist_success"]
        for j, m in enumerate(ms):
            mat[i, j] = (hist.get(m, 0.0) / used) if used > 0 else np.nan

    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(ms)), max(4, 0.5 * len(df))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(ms)))
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["param"].tolist())

    ax.set_xlabel("multiplier m  (step = m*h)")
    ax.set_ylabel("parameter")
    ax.set_title(f"adaptive wp success fractions per parameter{'' if kind_filter is None else f' ({kind_filter})'}")
    plt.colorbar(im, ax=ax, label="fraction of used coordinates")

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}wp_per_param_success_heatmap{'' if kind_filter is None else f'_{kind_filter}'}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_wp_per_layer_outcomes(wp_stats: dict, out_dir, prefix: str = "", separate_kind: bool = False) -> None:
    out_dir = _ensure_plot_dir(out_dir)
    df = aggregate_param_df_to_layer_df(wp_stats_to_param_df(wp_stats), separate_kind=separate_kind)
    if df is None or len(df) == 0:
        return

    labels = df["group"].tolist()
    m1 = df["success_m1_frac_used"].fillna(0.0).to_numpy()
    escal = df["success_escalated_frac_used"].fillna(0.0).to_numpy()
    fail = df["failed_frac_used"].fillna(0.0).to_numpy()

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels)), 5))
    ax.bar(x, m1, label="success @ h")
    ax.bar(x, escal, bottom=m1, label="success @ >h")
    ax.bar(x, fail, bottom=m1 + escal, label="failed")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("fraction of used coordinates")
    ax.set_xlabel("layer")
    ax.set_title(f"adaptive wp outcomes per layer{' (separate weight/bias)' if separate_kind else ''}")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}wp_per_layer_outcomes{'_separate_kind' if separate_kind else ''}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_wp_failed_fraction_per_layer(wp_stats: dict, out_dir, prefix: str = "", separate_kind: bool = False) -> None:
    out_dir = _ensure_plot_dir(out_dir)
    df = aggregate_param_df_to_layer_df(wp_stats_to_param_df(wp_stats), separate_kind=separate_kind)
    if df is None or len(df) == 0:
        return

    _plot_fraction_bar(
        df=df.rename(columns={"group": "layer_group"}),
        x_col="layer_group",
        y_col="failed_frac_used",
        title=f"adaptive wp failed fraction per layer{' (separate weight/bias)' if separate_kind else ''}",
        ylabel="failed / used coordinates",
        out_path=out_dir / f"{prefix}wp_failed_fraction_per_layer{'_separate_kind' if separate_kind else ''}.png",
    )


def plot_wp_escalated_fraction_per_layer(wp_stats: dict, out_dir, prefix: str = "", separate_kind: bool = False) -> None:
    out_dir = _ensure_plot_dir(out_dir)
    df = aggregate_param_df_to_layer_df(wp_stats_to_param_df(wp_stats), separate_kind=separate_kind)
    if df is None or len(df) == 0:
        return

    _plot_fraction_bar(
        df=df.rename(columns={"group": "layer_group"}),
        x_col="layer_group",
        y_col="escalated_frac_used",
        title=f"adaptive wp escalated fraction per layer{' (separate weight/bias)' if separate_kind else ''}",
        ylabel="escalated-success / used coordinates",
        out_path=out_dir / f"{prefix}wp_escalated_fraction_per_layer{'_separate_kind' if separate_kind else ''}.png",
    )


def plot_wp_per_layer_success_heatmap(wp_stats: dict, out_dir, prefix: str = "", separate_kind: bool = False) -> None:
    out_dir = _ensure_plot_dir(out_dir)
    df = aggregate_param_df_to_layer_df(wp_stats_to_param_df(wp_stats), separate_kind=separate_kind)
    if df is None or len(df) == 0:
        return

    all_ms = set()
    for _, row in df.iterrows():
        all_ms.update(row["m_hist_success"].keys())
    ms = sorted(all_ms)
    if len(ms) == 0:
        return

    mat = np.zeros((len(df), len(ms)), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        used = float(row["num_coords_used"])
        hist = row["m_hist_success"]
        for j, m in enumerate(ms):
            mat[i, j] = (hist.get(m, 0.0) / used) if used > 0 else np.nan

    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(ms)), max(4, 0.5 * len(df))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(ms)))
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["group"].tolist())

    ax.set_xlabel("multiplier m  (step = m*h)")
    ax.set_ylabel("layer")
    ax.set_title(f"adaptive wp success fractions per layer{' (separate weight/bias)' if separate_kind else ''}")
    plt.colorbar(im, ax=ax, label="fraction of used coordinates")

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}wp_per_layer_success_heatmap{'_separate_kind' if separate_kind else ''}.png", dpi=150, bbox_inches="tight")
    plt.close()


def wp_stats_summary_row(label: str, wp_stats: dict) -> dict:
    used = float(wp_stats.get("num_coords_used", 0))
    total = float(wp_stats.get("num_coords_total", 0))
    escalated = float(wp_stats.get("num_escalated", 0))
    failed = float(wp_stats.get("num_failed", 0))
    m_success = _as_int_key_dict(wp_stats.get("m_hist_success", {}))
    success_total = float(sum(m_success.values()))
    success_m1 = float(m_success.get(1, 0.0))
    success_escalated = success_total - success_m1

    return {
        "label": label,
        "num_coords_used": used,
        "num_coords_total": total,
        "used_frac": (used / total) if total > 0 else np.nan,
        "success_m1_frac_used": (success_m1 / used) if used > 0 else np.nan,
        "success_escalated_frac_used": (success_escalated / used) if used > 0 else np.nan,
        "failed_frac_used": (failed / used) if used > 0 else np.nan,
        "escalated_frac_used": (escalated / used) if used > 0 else np.nan,
    }


def plot_wp_multi_run_global_outcomes(stats_items: List[Tuple[str, dict]], out_dir, prefix: str = "") -> None:
    """
    Compare multiple runs/regimes in one figure.
    stats_items: list of (label, wp_stats)
    """
    out_dir = _ensure_plot_dir(out_dir)
    rows = [wp_stats_summary_row(label, st) for label, st in stats_items if isinstance(st, dict)]
    if len(rows) == 0:
        return

    df = pd.DataFrame(rows)

    x = np.arange(len(df))
    m1 = df["success_m1_frac_used"].fillna(0.0).to_numpy()
    escal = df["success_escalated_frac_used"].fillna(0.0).to_numpy()
    fail = df["failed_frac_used"].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(df)), 5))
    ax.bar(x, m1, label="success @ h")
    ax.bar(x, escal, bottom=m1, label="success @ >h")
    ax.bar(x, fail, bottom=m1 + escal, label="failed")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("fraction of used coordinates")
    ax.set_xlabel("run / regime")
    ax.set_title("adaptive wp outcomes across runs/regimes")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}wp_multi_run_global_outcomes.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_wp_multi_run_layer_metric_heatmap(
    stats_items: List[Tuple[str, dict]],
    out_dir,
    metric: str = "failed_frac_used",
    separate_kind: bool = True,
    prefix: str = "",
) -> None:
    """
    a heatmap comparing multiple runs/regimes on a chosen layer metric.
    some metrics:
      - failed_frac_used
      - escalated_frac_used
      - success_m1_frac_used
      - success_escalated_frac_used
      - used_frac
    """
    out_dir = _ensure_plot_dir(out_dir)

    rows = []
    for label, st in stats_items:
        if not isinstance(st, dict):
            continue
        layer_df = aggregate_param_df_to_layer_df(wp_stats_to_param_df(st), separate_kind=separate_kind)
        if len(layer_df) == 0 or metric not in layer_df.columns:
            continue
        for _, row in layer_df.iterrows():
            rows.append({
                "label": label,
                "group": row["group"],
                "value": row[metric],
            })

    if len(rows) == 0:
        return

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="group", columns="label", values="value", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * piv.shape[1]), max(4, 0.5 * piv.shape[0])))
    im = ax.imshow(piv.values, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels(piv.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels(piv.index.tolist())

    ax.set_xlabel("run / regime")
    ax.set_ylabel("layer")
    ax.set_title(f"adaptive wp multi run heatmap: {metric}")
    plt.colorbar(im, ax=ax, label=metric)

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}wp_multi_run_layer_metric_heatmap_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_wp_multi_run_layer_success_m_heatmap(
    stats_items: List[Tuple[str, dict]],
    out_dir,
    m: int = 1,
    separate_kind: bool = True,
    prefix: str = "",
) -> None:
    """
    Compare, across runs/regimes, the fraction of used coordinates that succeeded at a chosen m
    """
    out_dir = _ensure_plot_dir(out_dir)

    rows = []
    for label, st in stats_items:
        if not isinstance(st, dict):
            continue
        layer_df = aggregate_param_df_to_layer_df(wp_stats_to_param_df(st), separate_kind=separate_kind)
        if len(layer_df) == 0:
            continue

        for _, row in layer_df.iterrows():
            hist = row["m_hist_success"]
            used = float(row["num_coords_used"])
            frac = (hist.get(int(m), 0.0) / used) if used > 0 else np.nan
            rows.append({
                "label": label,
                "group": row["group"],
                "value": frac,
            })

    if len(rows) == 0:
        return

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="group", columns="label", values="value", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * piv.shape[1]), max(4, 0.5 * piv.shape[0])))
    im = ax.imshow(piv.values, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels(piv.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels(piv.index.tolist())

    ax.set_xlabel("run / regime")
    ax.set_ylabel("layer")
    ax.set_title(f"adaptive wp multi-run heatmap: success fraction at m={m}")
    plt.colorbar(im, ax=ax, label="fraction of used coordinates")

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}wp_multi_run_layer_success_m_heatmap_m{m}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_wp_adaptive_suite(wp_stats: dict, out_dir, prefix: str = "") -> None:
    if not isinstance(wp_stats, dict):
        return
    if wp_stats.get("mode", None) == "random":
        return
    if "per_param" not in wp_stats:
        return

    plot_wp_global_m_histograms(wp_stats, out_dir, prefix=prefix)

    plot_wp_per_param_outcomes(wp_stats, out_dir, prefix=prefix)
    plot_wp_per_param_outcomes(wp_stats, out_dir, prefix=prefix, kind_filter="weight")
    plot_wp_per_param_outcomes(wp_stats, out_dir, prefix=prefix, kind_filter="bias")

    plot_wp_failed_fraction_per_param(wp_stats, out_dir, prefix=prefix)
    plot_wp_failed_fraction_per_param(wp_stats, out_dir, prefix=prefix, kind_filter="weight")
    plot_wp_failed_fraction_per_param(wp_stats, out_dir, prefix=prefix, kind_filter="bias")

    plot_wp_escalated_fraction_per_param(wp_stats, out_dir, prefix=prefix)
    plot_wp_escalated_fraction_per_param(wp_stats, out_dir, prefix=prefix, kind_filter="weight")
    plot_wp_escalated_fraction_per_param(wp_stats, out_dir, prefix=prefix, kind_filter="bias")

    plot_wp_per_param_used_fraction(wp_stats, out_dir, prefix=prefix)
    plot_wp_per_param_success_heatmap(wp_stats, out_dir, prefix=prefix)

    plot_wp_per_layer_outcomes(wp_stats, out_dir, prefix=prefix, separate_kind=False)
    plot_wp_per_layer_outcomes(wp_stats, out_dir, prefix=prefix, separate_kind=True)
    plot_wp_failed_fraction_per_layer(wp_stats, out_dir, prefix=prefix, separate_kind=False)
    plot_wp_failed_fraction_per_layer(wp_stats, out_dir, prefix=prefix, separate_kind=True)
    plot_wp_escalated_fraction_per_layer(wp_stats, out_dir, prefix=prefix, separate_kind=False)
    plot_wp_escalated_fraction_per_layer(wp_stats, out_dir, prefix=prefix, separate_kind=True)
    plot_wp_per_layer_success_heatmap(wp_stats, out_dir, prefix=prefix, separate_kind=False)
    plot_wp_per_layer_success_heatmap(wp_stats, out_dir, prefix=prefix, separate_kind=True)