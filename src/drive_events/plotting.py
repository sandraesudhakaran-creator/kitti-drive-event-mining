"""Shared plotting style and figures (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed colour per event type (categorical slots in a fixed order, never cycled),
# so a type keeps its colour in every figure.
EVENT_COLORS = {
    "braking": "#2a78d6",
    "acceleration": "#eb6834",
    "turn": "#1baf7a",
    "lateral": "#eda100",
    "jerk": "#e87ba4",
    "standstill": "#008300",
    "lead_approach": "#6250d6",
    "anomaly": "#e34948",
}
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"
LINE = "#2a2a28"

SIGNAL_LABELS = {
    "speed_f": "speed [m/s]",
    "ax_f": "long. accel [m/s²]",
    "ay_f": "lat. accel [m/s²]",
    "yaw_rate_f": "yaw rate [rad/s]",
    "jerk": "jerk [m/s³]",
    "ttc": "lead TTC [s]\n(10 = not closing)",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 1.4, "font.size": 9, "legend.frameon": False,
        "legend.fontsize": 8, "savefig.facecolor": SURFACE,
    })


def color_for(event_type: str) -> str:
    return EVENT_COLORS.get(event_type, "#6f6e69")


def plot_sequence_timeline(
    sig: pd.DataFrame, events: pd.DataFrame, out_path: str | Path, title: str = "",
    lead: pd.DataFrame | None = None,
) -> Path:
    """Signals of one drive above an event lane (one row per event type)."""
    apply_style()
    panels = ["speed_f", "ax_f", "yaw_rate_f"]
    if lead is not None and not lead.empty:
        panels.append("ttc")
    types = [k for k in EVENT_COLORS if len(events) and k in set(events["type"])]
    lane_h = 0.32 * max(len(types), 1) + 0.3
    fig, axes = plt.subplots(
        len(panels) + 1, 1, figsize=(12, 1.7 * len(panels) + lane_h + 0.8), sharex=True,
        gridspec_kw={"height_ratios": [1.0] * len(panels) + [lane_h / 1.7]},
    )
    t = sig["t"].to_numpy()
    for ax, name in zip(axes, panels):
        if name == "ttc":
            ttc = lead.set_index("frame")["ttc"].reindex(sig["frame"]).to_numpy(dtype=float)
            ax.plot(t, np.clip(ttc, 0, 10), color=LINE, linewidth=1.2)
            ax.set_ylim(0, 10.5)
        else:
            ax.plot(t, sig[name], color=LINE)
        ax.set_ylabel(SIGNAL_LABELS.get(name, name))

    lane = axes[-1]
    for row, typ in enumerate(types):
        g = events[events["type"] == typ]
        spans = [(r.t_start, max(r.t_end - r.t_start, 0.1)) for r in g.itertuples()]
        lane.broken_barh(spans, (row - 0.35, 0.7), color=color_for(typ), lw=0)
    lane.set_yticks(range(len(types)), types)
    lane.set_ylim(len(types) - 0.5, -0.5) if types else lane.set_yticks([])
    lane.grid(axis="y", visible=False)
    lane.set_xlabel("time [s]")
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_event_overview(events: pd.DataFrame, distance_km: float, out_path: str | Path) -> Path:
    """Event counts per type, split by severity, with the rate per km in the labels."""
    apply_style()
    order = [k for k in EVENT_COLORS if k in set(events["type"])]
    fig, ax = plt.subplots(figsize=(8, 0.45 * max(len(order), 1) + 1.4))
    for i, typ in enumerate(order):
        g = events[events["type"] == typ]
        n_notable = int((g["severity"] == "notable").sum())
        n_harsh = int((g["severity"] == "harsh").sum())
        c = color_for(typ)
        ax.barh(i, n_notable, color=c, alpha=0.55, height=0.6)
        ax.barh(i, n_harsh, left=n_notable, color=c, height=0.6, edgecolor=SURFACE, lw=2)
        rate = f"  {len(g)}  ({len(g) / distance_km:.1f}/km)" if distance_km > 0 else f"  {len(g)}"
        ax.text(n_notable + n_harsh, i, rate, va="center", color=TEXT_2, fontsize=8)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("events (light = notable, solid = harsh)")
    ax.set_title(f"Detected events over {distance_km:.2f} km", loc="left")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_ttc_comparison(per_frame: pd.DataFrame, out_path: str | Path, ttc_max: float = 10.0
                        ) -> Path | None:
    """Camera TTC vs. label-based TTC on frames where both are finite."""
    both = per_frame[np.isfinite(per_frame["ttc_pred"]) & np.isfinite(per_frame["ttc_gt"])]
    if both.empty:
        return None
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, ttc_max], [0, ttc_max], color=GRID, lw=1.2, zorder=1)
    ax.scatter(both["ttc_gt"], both["ttc_pred"], s=14, color=EVENT_COLORS["lead_approach"],
               edgecolor=SURFACE, linewidth=0.6, zorder=2)
    ax.set_xlim(0, ttc_max)
    ax.set_ylim(0, ttc_max)
    ax.set_xlabel("TTC from 3D labels [s]")
    ax.set_ylabel("TTC from camera box scale [s]")
    ax.set_title(f"Lead-vehicle TTC (n = {len(both)} frames)", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return Path(out_path)
