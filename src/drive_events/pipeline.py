"""End-to-end pipeline: KITTI sequences in, events + review material + report out.

Steps per run::

    load OXTS -> quality check -> clean / resample / filter
              -> rule events (absolute or percentile thresholds)
    images    -> vehicle detection -> tracking -> lead vehicle -> TTC
              -> lead_approach events, TTC evaluation against 3D labels
    windows   -> features -> anomaly scores -> review candidates
    output    -> events.csv, windows.csv, quality.csv, event cards, figures, summary.md

Detections are cached per sequence, so re-running with other thresholds does not
repeat the (slow) YOLO inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from drive_events.events import detect_events, events_to_frame, percentile_thresholds
from drive_events.features import label_windows, make_windows
from drive_events.io.kitti import TrackingSequence, list_sequences
from drive_events.labeling.cards import render_event_card
from drive_events.models import anomaly_scores, review_candidates
from drive_events.plotting import (
    plot_event_overview,
    plot_sequence_timeline,
    plot_ttc_comparison,
)
from drive_events.signals import preprocess
from drive_events.video.detect import YoloDetector, detections_from_labels
from drive_events.video.lead import (
    CameraModel,
    evaluate_ttc,
    lead_approach_events,
    select_lead,
    summarize_ttc,
    ttc_from_scale,
)
from drive_events.video.tracking import track_detections


@dataclass
class RunResult:
    out_dir: Path
    events: pd.DataFrame
    windows: pd.DataFrame
    quality: pd.DataFrame
    candidates: pd.DataFrame
    thresholds: dict
    ttc_metrics: dict = field(default_factory=dict)
    distance_km: float = 0.0


def _detect(seq: TrackingSequence, images: dict, vcfg: dict, detector_name: str,
            cache: Path, detector_obj, log: Callable) -> pd.DataFrame | None:
    """Vehicle detections for one sequence (cached as CSV)."""
    if detector_name == "ground_truth":
        if not seq.has_labels():
            return None
        return detections_from_labels(seq.load_labels(), vcfg["gt_classes"])
    if cache.exists():
        log(f"  detections: using cache {cache.name}")
        return pd.read_csv(cache)
    if not images:
        return None
    log(f"  detections: running YOLO on {len(images)} frames")
    det = detector_obj.detect(images, batch=vcfg["yolo_batch"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    det.to_csv(cache, index=False)
    return det


def run_pipeline(
    data_root: str | Path,
    out_dir: str | Path,
    cfg: dict,
    sequences: list[str] | None = None,
    detector: str | None = None,
    make_cards: bool = True,
    max_cards: int | None = None,
    log: Callable[[str], None] = print,
) -> RunResult:
    data_root, out_dir = Path(data_root), Path(out_dir)
    for sub in ("signals", "lead", "figures", "cards"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    dcfg, vcfg, ecfg = cfg["data"], cfg["video"], cfg["events"]
    detector = detector or vcfg["detector"]
    fps = float(dcfg["fps"])
    seq_ids = sequences or list_sequences(data_root, dcfg["split"])
    with open(out_dir / "run_config.yaml", "w") as fh:
        yaml.safe_dump({**cfg, "_run": {"data_root": str(data_root), "sequences": seq_ids,
                                        "detector": detector}}, fh, sort_keys=False)

    # 1) signals --------------------------------------------------------------
    seqs, sigs, quality = {}, {}, []
    for sid in seq_ids:
        seq = TrackingSequence(data_root, sid, dcfg["split"], dcfg["camera"])
        sig, rep = preprocess(seq.load_oxts(fps), cfg, sid)
        sig.to_csv(out_dir / "signals" / f"{sid}.csv", index=False)
        seqs[sid], sigs[sid] = seq, sig
        quality.append(rep.to_row())
    quality_df = pd.DataFrame(quality)
    quality_df.to_csv(out_dir / "quality.csv", index=False)
    distance_km = float(sum(s["distance_m"].iloc[-1] for s in sigs.values()) / 1000.0)
    log(f"Loaded {len(seq_ids)} sequences, {distance_km:.2f} km of driving")

    # 2) rule events ----------------------------------------------------------
    overrides = None
    if ecfg["threshold_mode"] == "percentile":
        overrides = percentile_thresholds(list(sigs.values()), cfg)
    thresholds = {name: {**d, **(overrides or {}).get(name, {})}
                  for name, d in ecfg["definitions"].items()}
    events = []
    for sid, sig in sigs.items():
        events += detect_events(sig, cfg, sid, overrides)

    # 3) camera: detection -> tracking -> lead vehicle -> TTC --------------------
    leads, ttc_frames, images_by_seq = {}, [], {}
    yolo = None
    if detector != "none":
        for sid, seq in seqs.items():
            images = seq.image_paths()
            images_by_seq[sid] = images
            if detector == "yolo" and images and yolo is None and \
                    not (out_dir / "detections" / f"{sid}.csv").exists():
                yolo = YoloDetector(vcfg["yolo_weights"], vcfg["yolo_conf"],
                                    vcfg["yolo_classes"], vcfg["yolo_imgsz"])
            log(f"[{sid}] camera pipeline ({detector})")
            det = _detect(seq, images, vcfg, detector, out_dir / "detections" / f"{sid}.csv",
                          yolo, log)
            if det is None or det.empty:
                continue
            det = track_detections(det, vcfg["tracker_iou"], vcfg["tracker_max_age"])
            det = ttc_from_scale(det, fps, vcfg["ttc_window"], vcfg["ttc_min_frames"],
                                 vcfg["ttc_max_s"])
            camera = CameraModel.from_projection(seq.load_camera_matrix(),
                                                 vcfg["camera_height_m"])
            lead = select_lead(det, camera, vcfg["lane_half_width_m"],
                               vcfg["min_box_height_px"])
            lead.to_csv(out_dir / "lead" / f"{sid}.csv", index=False)
            leads[sid] = lead
            events += lead_approach_events(lead, sigs[sid], vcfg, ecfg["hysteresis_ratio"],
                                           ecfg["merge_gap_s"], sid)
            if seq.has_labels():
                per_frame, _ = evaluate_ttc(lead, seq.load_labels(), vcfg, camera, fps)
                ttc_frames.append(per_frame.assign(sequence=sid))

    events_df = events_to_frame(events)
    events_df.to_csv(out_dir / "events.csv", index=False)
    log(f"Detected {len(events_df)} events: "
        + ", ".join(f"{k} {v}" for k, v in events_df["type"].value_counts().items()))

    ttc_metrics: dict = {}
    if ttc_frames:
        ttc_all = pd.concat(ttc_frames, ignore_index=True)
        ttc_all.to_csv(out_dir / "ttc_per_frame.csv", index=False)
        ttc_metrics = summarize_ttc(ttc_all, vcfg["critical_ttc_s"])
        plot_ttc_comparison(ttc_all, out_dir / "figures" / "ttc_scatter.png", vcfg["ttc_max_s"])

    # 4) windows, anomaly detection, review candidates ---------------------------
    wcfg, mcfg = cfg["windows"], cfg["models"]
    windows = pd.concat(
        [make_windows(sig, sid, wcfg["window_s"], wcfg["stride_s"], wcfg["max_nan_fraction"],
                      leads.get(sid), vcfg["ttc_max_s"]) for sid, sig in sigs.items()],
        ignore_index=True,
    )
    windows["label"] = label_windows(windows, events_df, wcfg["min_overlap"])
    candidates = pd.DataFrame()
    if len(windows) >= 10:
        windows["anomaly_score"] = anomaly_scores(windows, mcfg["contamination"], mcfg["seed"])
        candidates = review_candidates(windows, windows["anomaly_score"].to_numpy(),
                                       mcfg["review_top_k"])
        candidates = candidates.assign(
            type="anomaly",
            event_id=[f"{s}_anomaly_{int(f):06d}" for s, f in
                      zip(candidates["sequence"], candidates["frame_start"])],
        )
        candidates.to_csv(out_dir / "review_candidates.csv", index=False)
    windows.to_csv(out_dir / "windows.csv", index=False)

    # 5) review material and figures ---------------------------------------------
    if make_cards:
        limit = cfg["cards"]["max_cards"] if max_cards is None else max_cards
        for _, ev in _card_order(events_df).head(limit).iterrows():
            sid = str(ev["sequence"])
            render_event_card(ev, sigs[sid], out_dir / "cards" / f"{ev.event_id}.png",
                              images_by_seq.get(sid), leads.get(sid), cfg["cards"]["context_s"])
        for _, c in candidates.head(limit).iterrows():
            sid = str(c["sequence"])
            render_event_card(c, sigs[sid], out_dir / "cards" / "candidates" / f"{c.event_id}.png",
                              images_by_seq.get(sid), leads.get(sid), cfg["cards"]["context_s"])

    for sid, sig in sigs.items():
        plot_sequence_timeline(sig, events_df[events_df["sequence"].astype(str) == sid],
                               out_dir / "figures" / f"timeline_{sid}.png",
                               f"Sequence {sid}", leads.get(sid))
    if len(events_df):
        plot_event_overview(events_df, distance_km, out_dir / "figures" / "overview.png")

    with open(out_dir / "thresholds.json", "w") as fh:
        json.dump(thresholds, fh, indent=2)
    result = RunResult(out_dir, events_df, windows, quality_df, candidates, thresholds,
                       ttc_metrics, distance_km)
    write_summary(result, cfg, detector)
    log(f"Done. Summary: {out_dir / 'summary.md'}")
    return result


def _card_order(events: pd.DataFrame) -> pd.DataFrame:
    """Order events for card rendering: round-robin over types, harsh first, so a
    card limit still covers every event type."""
    if events.empty:
        return events
    ranked = events.sort_values(["severity", "duration_s"], ascending=[True, False])
    ranked = ranked.assign(_rank=ranked.groupby("type").cumcount())
    return ranked.sort_values(["_rank", "severity"]).drop(columns="_rank")


def _fmt(x) -> str:
    if isinstance(x, (float, np.floating)):
        return "–" if np.isnan(x) else f"{x:.3f}"
    return str(x)


def write_summary(res: RunResult, cfg: dict, detector: str) -> Path:
    """Markdown run summary with the key numbers and data-quality findings."""
    ev, q = res.events, res.quality
    lines = [
        "# Run summary", "",
        f"- Sequences: {len(q)}  ·  driving time: {q['duration_s'].sum() / 60:.1f} min  ·  "
        f"distance: {res.distance_km:.2f} km",
        f"- Threshold mode: `{cfg['events']['threshold_mode']}`  ·  detector: `{detector}`",
        f"- Events: {len(ev)}  ·  anomaly candidates for review: {len(res.candidates)}", "",
        "## Events by type", "",
        "| type | count | harsh | per km | median duration [s] |",
        "|---|---:|---:|---:|---:|",
    ]
    for typ, g in ev.groupby("type"):
        per_km = len(g) / res.distance_km if res.distance_km > 0 else float("nan")
        lines.append(f"| {typ} | {len(g)} | {(g['severity'] == 'harsh').sum()} | "
                     f"{_fmt(per_km)} | {_fmt(g['duration_s'].median())} |")
    lines += ["", "## Data quality", "",
              "| sequence | samples | gaps | out-of-range (ax) | accel consistency RMSE "
              "[m/s²] | IMU bias [m/s²] | valid |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in q.itertuples():
        lines.append(f"| {r.sequence} | {r.n_samples} | {r.n_time_gaps} | "
                     f"{r.n_out_of_range__ax} | {_fmt(r.accel_consistency_rmse)} | "
                     f"{_fmt(r.accel_bias)} | {r.valid_fraction:.1%} |")
    if res.ttc_metrics:
        lines += ["", "## Camera TTC vs. 3D labels", "", "| metric | value |", "|---|---:|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in res.ttc_metrics.items()]
    lines += ["", "## Thresholds used", "", "```json", json.dumps(res.thresholds, indent=2),
              "```", ""]
    path = res.out_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
