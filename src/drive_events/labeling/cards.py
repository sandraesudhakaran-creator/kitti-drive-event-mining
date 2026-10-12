"""Event cards: one image per candidate that shows a reviewer everything needed
to accept, reject or relabel it - camera frames at start / peak / end (with the
lead vehicle box) above the signals around the event."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from drive_events.plotting import LINE, SIGNAL_LABELS, TEXT_2, apply_style, color_for


def _nearest(frames: list[int], f: int) -> int | None:
    if not frames:
        return None
    arr = np.asarray(frames)
    return int(arr[np.argmin(np.abs(arr - f))])


def render_event_card(
    event: pd.Series | dict,
    sig: pd.DataFrame,
    out_path: str | Path,
    image_paths: dict[int, Path] | None = None,
    lead: pd.DataFrame | None = None,
    context_s: float = 3.0,
) -> Path:
    """Draw and save the review card for one event (or anomaly window)."""
    ev = pd.Series(event)
    apply_style()
    color = color_for(str(ev["type"]))
    image_paths = image_paths or {}
    show_images = bool(image_paths)

    panels = ["speed_f", "ax_f", "yaw_rate_f"]
    if lead is not None and not lead.empty:
        panels.append("ttc")
    # KITTI frames are ~3.3:1, so one image row of 1.3 in fills a third of 12 in.
    n_rows = len(panels) + (1 if show_images else 0)
    fig = plt.figure(figsize=(12, 1.3 * len(panels) + (1.5 if show_images else 0) + 0.7))
    gs = fig.add_gridspec(n_rows, 3, height_ratios=([1.15] if show_images else [])
                          + [1] * len(panels))

    if show_images:
        frames = sorted(image_paths)
        f_peak = int(ev.get("frame_peak", (ev["frame_start"] + ev["frame_end"]) // 2))
        picks = [("start", int(ev["frame_start"])), ("peak", f_peak), ("end", int(ev["frame_end"]))]
        for col, (name, f) in enumerate(picks):
            ax = fig.add_subplot(gs[0, col])
            fn = _nearest(frames, f)
            if fn is not None:
                ax.imshow(Image.open(image_paths[fn]))
                if lead is not None and fn in set(lead["frame"]):
                    b = lead[lead["frame"] == fn].iloc[0]
                    ax.add_patch(plt.Rectangle((b.x1, b.y1), b.x2 - b.x1, b.y2 - b.y1,
                                               fill=False, edgecolor=color, lw=2))
            ax.set_title(f"{name} · frame {fn}", loc="left", fontsize=9, color=TEXT_2)
            ax.axis("off")

    lo, hi = ev["t_start"] - context_s, ev["t_end"] + context_s
    win = sig[(sig["t"] >= lo) & (sig["t"] <= hi)]
    first = None
    for i, name in enumerate(panels):
        row = i + (1 if show_images else 0)
        ax = fig.add_subplot(gs[row, :], sharex=first)
        first = first or ax
        if name == "ttc":
            ttc = lead.set_index("frame")["ttc"].reindex(win["frame"]).to_numpy(dtype=float)
            ax.plot(win["t"], np.clip(ttc, 0, 10), color=LINE)
            ax.set_ylim(0, 10.5)
        else:
            ax.plot(win["t"], win[name], color=LINE)
        if str(ev.get("signal", "")) in (name, name.replace("_f", "_abs")) and \
                pd.notna(ev.get("threshold")):
            thr = float(ev["threshold"])
            levels = [thr, -thr] if str(ev["signal"]).endswith("_abs") else [thr]
            for lv in levels:
                ax.axhline(lv, color=color, lw=1, ls="--")
        ax.axvspan(ev["t_start"], ev["t_end"], color=color, alpha=0.18, lw=0)
        ax.set_ylabel(SIGNAL_LABELS.get(name, name), fontsize=8)
        if i < len(panels) - 1:
            ax.tick_params(labelbottom=False)
    first.set_xlim(lo, hi)
    ax.set_xlabel("time [s]")

    peak = ev.get("peak")
    peak_txt = f" · peak {float(peak):.2f}" if peak is not None and pd.notna(peak) else ""
    sev = f" ({ev['severity']})" if "severity" in ev and pd.notna(ev.get("severity")) else ""
    fig.suptitle(
        f"{ev.get('event_id', '')}   {ev['type']}{sev}{peak_txt} · "
        f"{ev['t_end'] - ev['t_start']:.1f} s",
        x=0.01, ha="left", fontsize=11, fontweight="bold", color=color,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def write_clip(
    image_paths: dict[int, Path], frame_start: int, frame_end: int, out_path: str | Path,
    fps: float = 10.0, pad_frames: int = 10,
) -> Path:
    """Write the camera frames around an event as an MP4 (needs the ``clips`` extra)."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Clips need the 'clips' extra: pip install -e '.[clips]'") from exc
    frames = [f for f in sorted(image_paths)
              if frame_start - pad_frames <= f <= frame_end + pad_frames]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
        for f in frames:
            writer.append_data(np.asarray(Image.open(image_paths[f]).convert("RGB")))
    return out_path
