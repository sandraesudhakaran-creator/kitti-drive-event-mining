"""Sliding-window features for pattern recognition.

Each window (default 2 s, 0.5 s stride) is summarised by statistics of the filtered
signals and, when the camera pipeline ran, by the minimum lead-vehicle TTC. Windows
get a label from the events they overlap, which lets the same table feed the
anomaly detector (unsupervised) and the classifier (supervised).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

WINDOW_SIGNALS = ["speed_f", "ax_f", "ay_f", "yaw_rate_f", "jerk"]
META_COLUMNS = ["sequence", "t_start", "t_end", "frame_start", "frame_end"]


def make_windows(
    sig: pd.DataFrame,
    sequence: str,
    window_s: float = 2.0,
    stride_s: float = 0.5,
    max_nan_fraction: float = 0.2,
    lead: pd.DataFrame | None = None,
    ttc_cap: float = 10.0,
) -> pd.DataFrame:
    """Cut one sequence into overlapping windows and compute features."""
    t = sig["t"].to_numpy()
    if len(t) < 2:
        return pd.DataFrame(columns=META_COLUMNS)
    dt = float(np.median(np.diff(t)))
    n = max(2, int(round(window_s / dt)))
    step = max(1, int(round(stride_s / dt)))
    frames = sig["frame"].to_numpy()
    data = {c: sig[c].to_numpy(dtype=float) for c in WINDOW_SIGNALS}

    ttc_series = None
    if lead is not None and not lead.empty:
        ttc_series = lead.set_index("frame")["ttc"].reindex(frames).to_numpy(dtype=float)

    rows = []
    for s in range(0, len(t) - n + 1, step):
        e = s + n
        if np.isnan(data["speed_f"][s:e]).mean() > max_nan_fraction:
            continue
        row = {"sequence": sequence, "t_start": float(t[s]), "t_end": float(t[e - 1]),
               "frame_start": int(frames[s]), "frame_end": int(frames[e - 1])}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for c in WINDOW_SIGNALS:
                x = data[c][s:e]
                row[f"{c}__mean"] = float(np.nanmean(x))
                row[f"{c}__std"] = float(np.nanstd(x))
                row[f"{c}__min"] = float(np.nanmin(x))
                row[f"{c}__max"] = float(np.nanmax(x))
            v = data["speed_f"][s:e]
            row["speed_f__delta"] = float(v[~np.isnan(v)][-1] - v[~np.isnan(v)][0])
            if ttc_series is not None:
                w = np.clip(ttc_series[s:e], None, ttc_cap)  # inf (not closing) -> cap
                has = ~np.isnan(w)
                row["ttc__min"] = float(np.min(w[has])) if has.any() else ttc_cap
                row["lead__fraction"] = float(has.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(windows: pd.DataFrame) -> list[str]:
    return [c for c in windows.columns if "__" in c]


def label_windows(
    windows: pd.DataFrame,
    events: pd.DataFrame,
    min_overlap: float = 0.5,
    type_col: str = "type",
    default: str = "normal",
) -> pd.Series:
    """Label each window with the event type it overlaps most.

    Overlap is measured relative to the shorter of window and event, so a short
    event fully inside a window still labels it. Windows without sufficient overlap
    get ``default``.
    """
    labels = np.full(len(windows), default, dtype=object)
    if events.empty or windows.empty:
        return pd.Series(labels, index=windows.index, name="label")
    ev_by_seq = {s: g for s, g in events.groupby(events["sequence"].astype(str))}
    for pos, (_, w) in enumerate(windows.iterrows()):
        ev = ev_by_seq.get(str(w["sequence"]))
        if ev is None:
            continue
        inter = (np.minimum(ev["t_end"], w["t_end"]) - np.maximum(ev["t_start"], w["t_start"]))
        inter = inter.clip(lower=0)
        shorter = np.minimum(ev["t_end"] - ev["t_start"], w["t_end"] - w["t_start"])
        shorter = shorter.clip(lower=1e-6)
        frac = (inter / shorter).to_numpy()
        k = int(np.argmax(frac))
        if frac[k] >= min_overlap:
            labels[pos] = ev[type_col].iloc[k]
    return pd.Series(labels, index=windows.index, name="label")
