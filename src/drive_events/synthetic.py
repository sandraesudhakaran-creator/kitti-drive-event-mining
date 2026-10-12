"""Synthetic drives in exact KITTI tracking format, with known ground truth.

Used by the tests and the CI demo, so the whole pipeline can run without
downloading KITTI. Each drive contains scripted manoeuvres - a harsh brake, a
stop, a pull-away, a turn and a closing lead vehicle - recorded with simulated
sensor noise, IMU bias and a few faulty samples.

Ground truth is defined as "where the *noise-free* signal crosses the configured
thresholds" (and, for the lead vehicle, where the analytic TTC drops below the
critical value). Running the pipeline on the noisy recording and comparing with
this truth therefore measures how robust the processing chain is to sensor noise,
outliers and filtering delay.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from drive_events.config import DEFAULT_CONFIG
from drive_events.events import detect_events, events_to_frame
from drive_events.video.lead import lead_approach_events

FOCAL, CX, CY = 721.5377, 609.5593, 172.854
IMG_W, IMG_H = 1242, 375
CAM_HEIGHT = 1.65
CAR_H, CAR_W, CAR_L = 1.5, 1.7, 4.2


def _cosine_segments(t: np.ndarray, segments: list[tuple[float, float, float, float]],
                     v_init: float) -> tuple[np.ndarray, np.ndarray]:
    """Speed and acceleration from smooth (raised-cosine) speed transitions."""
    v = np.full_like(t, v_init)
    a = np.zeros_like(t)
    for t0, t1, v0, v1 in segments:
        inside = (t >= t0) & (t < t1)
        tau = (t[inside] - t0) / (t1 - t0)
        v[inside] = v0 + (v1 - v0) * (1 - np.cos(np.pi * tau)) / 2
        a[inside] = (v1 - v0) * np.pi / (2 * (t1 - t0)) * np.sin(np.pi * tau)
        v[t >= t1] = v1
    return v, a


def _project(x: float, z: float) -> tuple[float, float, float, float] | None:
    """Approximate image box of a car centred at lateral ``x`` and depth ``z``."""
    if z <= 1.0:
        return None
    x1 = FOCAL * (x - CAR_W / 2) / z + CX
    x2 = FOCAL * (x + CAR_W / 2) / z + CX
    y1 = FOCAL * (CAM_HEIGHT - CAR_H) / z + CY
    y2 = FOCAL * CAM_HEIGHT / z + CY
    if x2 <= 0 or x1 >= IMG_W:
        return None
    return (max(x1, 0.0), max(y1, 0.0), min(x2, IMG_W - 1.0), min(y2, IMG_H - 1.0))


def generate_sequence(seq: str, seed: int, fps: float = 10.0, duration_s: float = 45.0,
                      cfg: dict | None = None) -> dict:
    """Simulate one drive. Returns oxts array, labels, truth events and clean signals."""
    cfg = cfg or DEFAULT_CONFIG
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    t = np.arange(n) / fps
    j = lambda: rng.uniform(-1.0, 1.0)  # noqa: E731 - timing jitter

    tb = 5.0 + j()
    v_mid = rng.uniform(3.0, 5.0)
    v_cruise = rng.uniform(10.0, 12.0)
    segs = [
        (tb, tb + 3.0, 12.0, v_mid),                       # harsh brake
        (tb + 6.0, tb + 9.0, v_mid, 0.0),                  # gentle stop
        (tb + 13.0, tb + 19.0, 0.0, v_cruise),             # pull away
    ]
    v, a = _cosine_segments(t, segs, 12.0)

    t_turn, turn_peak = 27.0 + j(), rng.uniform(0.25, 0.32)
    w = np.where((t >= t_turn) & (t < t_turn + 4.0),
                 turn_peak * np.sin(np.pi * (t - t_turn) / 4.0), 0.0)
    ay = v * w

    clean = pd.DataFrame({"t": t, "frame": np.arange(n), "speed_f": v, "ax_f": a,
                          "ay_f": ay, "yaw_rate_f": w})
    clean["jerk"] = np.gradient(a, t)
    clean["yaw_rate_abs"] = np.abs(w)
    clean["ay_abs"] = np.abs(ay)
    clean["jerk_abs"] = np.abs(clean["jerk"])

    # Measured signals: noise, a small IMU bias and two faulty spikes.
    v_m = v + rng.normal(0, 0.05, n)
    a_m = a + 0.05 + rng.normal(0, 0.15, n)
    ay_m = ay + rng.normal(0, 0.15, n)
    w_m = w + rng.normal(0, 0.008, n)
    a_m[rng.choice(n, size=2, replace=False)] = 40.0

    yaw = np.cumsum(w) / fps
    px = np.cumsum(v * np.cos(yaw)) / fps
    py = np.cumsum(v * np.sin(yaw)) / fps
    lat = 49.0 + py / 111_320.0
    lon = 8.4 + px / (111_320.0 * np.cos(np.radians(49.0)))
    z0 = np.zeros(n)
    oxts = np.column_stack([
        lat, lon, z0 + 115.0, z0, z0, yaw,
        v * np.sin(yaw), v * np.cos(yaw), v_m, z0, z0,
        a_m, ay_m, z0 + 9.81, a_m, ay_m, z0 + 9.81,
        z0, z0, w_m, z0, z0, w_m,
        z0 + 0.5, z0 + 0.05, z0 + 4, z0 + 10, z0 + 5, z0 + 5, z0 + 6,
    ])

    # Lead vehicle: constant gap, then closes at 6 m/s from 30 m to 6 m.
    t_close = 34.0 + j()
    z_lead = np.clip(30.0 - 6.0 * np.clip(t - t_close, 0, None), 6.0, None)
    closing = (t >= t_close) & (z_lead > 6.0)
    ttc_true = np.where(closing, z_lead / 6.0, np.inf)

    # Parked cars on both sides; they approach with the ego distance travelled.
    travelled = np.cumsum(v) / fps
    parked = [(-4.5 if k % 2 else 5.0, z) for k, z in enumerate(
        [12, 30, 48, 70, 95, 120, 150, 190, 230, 280, 330, 380])]

    rows = []
    for f in range(n):
        box = _project(0.2, z_lead[f])
        if box:
            rows.append((f, 0, "Car", 0.2, z_lead[f], box))
        for k, (x, zp) in enumerate(parked):
            z = zp - travelled[f]
            if 2.0 < z < 60.0:
                b = _project(x, z)
                if b:
                    rows.append((f, k + 1, "Car", x, z, b))
    labels = pd.DataFrame(
        [{"frame": f, "track_id": tid, "type": typ, "truncated": 0, "occluded": 0,
          "alpha": 0.0, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
          "h": CAR_H, "w": CAR_W, "l": CAR_L, "x": x, "y": CAM_HEIGHT, "z": z, "ry": 0.0}
         for f, tid, typ, x, z, b in rows]
    )

    events = detect_events(clean, cfg, seq)
    lead_truth = pd.DataFrame({"frame": np.arange(n), "ttc": ttc_true})
    ecfg = cfg["events"]
    events += lead_approach_events(lead_truth, clean, cfg["video"], ecfg["hysteresis_ratio"],
                                   ecfg["merge_gap_s"], seq)
    truth = events_to_frame(events)
    return {"oxts": oxts, "labels": labels, "truth": truth, "clean": clean, "fps": fps}


def _calib_text() -> str:
    p = f"{FOCAL:.6e} 0 {CX:.6e} 0 0 {FOCAL:.6e} {CY:.6e} 0 0 0 1 0"
    eye = "1 0 0 0 1 0 0 0 1"
    rt = "0 -1 0 0 0 0 -1 -0.08 1 0 0 -0.27"
    return "\n".join([f"P0: {p}", f"P1: {p}", f"P2: {p}", f"P3: {p}",
                      f"R_rect {eye}", f"Tr_velo_cam {rt}", f"Tr_imu_velo {rt}", ""])


def _draw_frame(labels_f: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (IMG_W, IMG_H), (196, 206, 214))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(CY), IMG_W, IMG_H], fill=(118, 118, 112))
    d.polygon([(CX - 6, CY), (CX + 6, CY), (CX + 260, IMG_H), (CX + 230, IMG_H)],
              fill=(150, 150, 142))
    for r in labels_f.sort_values("z", ascending=False).itertuples():
        fill = (52, 64, 92) if r.track_id == 0 else (70, 70, 66)
        d.rectangle([r.x1, r.y1, r.x2, r.y2], fill=fill)
        band = r.y1 + 0.25 * (r.y2 - r.y1)
        d.rectangle([r.x1 + 0.15 * (r.x2 - r.x1), r.y1 + 0.08 * (r.y2 - r.y1),
                     r.x2 - 0.15 * (r.x2 - r.x1), band], fill=(140, 160, 176))
    img.save(path, optimize=False, compress_level=1)


def write_dataset(out_root: str | Path, n_sequences: int = 3, seed: int = 0,
                  images: bool = False, fps: float = 10.0) -> pd.DataFrame:
    """Write ``n_sequences`` synthetic drives in KITTI tracking layout.

    Also writes ``<out_root>/truth_events.csv`` and returns it.
    """
    root = Path(out_root)
    base = root / "training"
    for sub in ("oxts", "label_02", "calib"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    truths = []
    for i in range(n_sequences):
        seq = f"{i:04d}"
        sim = generate_sequence(seq, seed=seed * 1000 + i, fps=fps)
        np.savetxt(base / "oxts" / f"{seq}.txt", sim["oxts"], fmt="%.10g")
        lab = sim["labels"]
        with open(base / "label_02" / f"{seq}.txt", "w") as fh:
            fh.write("0 -1 DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10\n")
            for r in lab.itertuples(index=False):
                fh.write(
                    f"{r.frame} {r.track_id} {r.type} {r.truncated} {r.occluded} {r.alpha:.2f} "
                    f"{r.x1:.2f} {r.y1:.2f} {r.x2:.2f} {r.y2:.2f} {r.h:.2f} {r.w:.2f} {r.l:.2f} "
                    f"{r.x:.3f} {r.y:.3f} {r.z:.3f} {r.ry:.2f}\n"
                )
        (base / "calib" / f"{seq}.txt").write_text(_calib_text())
        if images:
            img_dir = base / "image_02" / seq
            img_dir.mkdir(parents=True, exist_ok=True)
            by_frame = dict(tuple(lab.groupby("frame")))
            empty = lab.iloc[0:0]
            for f in range(len(sim["oxts"])):
                _draw_frame(by_frame.get(f, empty), img_dir / f"{f:06d}.png")
        truths.append(sim["truth"])
    truth = pd.concat(truths, ignore_index=True)
    truth.to_csv(root / "truth_events.csv", index=False)
    return truth
