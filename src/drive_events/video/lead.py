"""Lead-vehicle selection and monocular time-to-collision (TTC) from the camera.

Why box scale gives TTC
-----------------------
For a pinhole camera the image size ``s`` of an object at depth ``Z`` is ``s ~ 1/Z``.
Differentiating ``ln s`` gives

    d ln(s) / dt = -(dZ/dt) / Z = 1 / TTC

so the TTC follows from how fast the box grows - no depth estimate, calibration or
object size is needed. The slope is estimated by a least-squares fit of ``ln s`` over
the last few frames of the same track, which is far less noisy than a two-frame ratio.
The estimate assumes a constant closing speed and a rigid object seen from roughly the
same angle, which is why it is evaluated against KITTI 3D labels in
:func:`evaluate_ttc`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from drive_events.events import Event, detect_intervals, exit_level
from drive_events.video.detect import detections_from_labels
from drive_events.video.tracking import box_iou


@dataclass
class CameraModel:
    """Pinhole camera mounted ``height_m`` above a flat road (KITTI defaults)."""

    fx: float = 721.5377
    fy: float = 721.5377
    cx: float = 609.5593
    cy: float = 172.854
    height_m: float = 1.65

    @classmethod
    def from_projection(cls, P: np.ndarray | None, height_m: float = 1.65) -> CameraModel:
        if P is None:
            return cls(height_m=height_m)
        return cls(fx=float(P[0, 0]), fy=float(P[1, 1]), cx=float(P[0, 2]),
                   cy=float(P[1, 2]), height_m=height_m)

    def ground_position(self, det: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Depth ``z`` and lateral offset ``x`` [m] of each box's bottom-centre point,
        assuming it touches a flat road. Boxes whose bottom is at or above the
        horizon get ``inf`` depth."""
        dy = det["y2"].to_numpy(dtype=float) - self.cy
        with np.errstate(divide="ignore"):
            z = np.where(dy > 1.0, self.fy * self.height_m / np.maximum(dy, 1e-9), np.inf)
        u = (det["x1"].to_numpy(dtype=float) + det["x2"].to_numpy(dtype=float)) / 2.0
        x = np.where(np.isfinite(z), (u - self.cx) * z / self.fx, np.inf)
        return z, x


def _window_slope(t: np.ndarray, y: np.ndarray, span: float, min_points: int) -> np.ndarray:
    """Least-squares slope of ``y`` over samples in the trailing time span
    ``(t_i - span, t_i]``; NaN where fewer than ``min_points`` samples fall inside.
    ``t`` must be sorted."""
    out = np.full(len(t), np.nan)
    los = np.searchsorted(t, t - span + 1e-9, side="left")
    for i in range(len(t)):
        tt, yy = t[los[i] : i + 1], y[los[i] : i + 1]
        if len(tt) >= min_points and np.ptp(tt) > 0:
            tc = tt - tt.mean()
            out[i] = float(np.sum(tc * (yy - yy.mean())) / np.sum(tc**2))
    return out


def ttc_from_scale(
    det: pd.DataFrame, fps: float, window: int = 5, min_frames: int = 3, ttc_max: float = 10.0
) -> pd.DataFrame:
    """Add ``scale_rate`` [1/s] and ``ttc`` [s] to every tracked detection.

    ``ttc`` is ``inf`` when the object is not approaching (or slower than ``ttc_max``)
    and NaN when the track is too short to estimate.
    """
    out = det.copy()
    out["scale_rate"] = np.nan
    out["ttc"] = np.nan
    for _, g in out.groupby("track_id"):
        g = g.sort_values("frame")
        # The fit uses only frames inside the time window, so a track that was lost
        # and re-found is never compared against a box from long ago.
        t = g["frame"].to_numpy(dtype=float) / fps
        size = np.sqrt((g["x2"] - g["x1"]).to_numpy() * (g["y2"] - g["y1"]).to_numpy())
        rate = _window_slope(t, np.log(np.clip(size, 1e-6, None)), window / fps, min_frames)
        ttc = np.where(rate > 1.0 / ttc_max, 1.0 / np.where(rate > 0, rate, 1), np.inf)
        ttc = np.where(np.isnan(rate), np.nan, ttc)
        out.loc[g.index, "scale_rate"] = rate
        out.loc[g.index, "ttc"] = ttc
    return out


def select_lead(
    det: pd.DataFrame,
    camera: CameraModel,
    lane_half_width_m: float = 1.8,
    min_height_px: float = 12.0,
) -> pd.DataFrame:
    """Pick the lead vehicle per frame: the nearest vehicle inside the ego lane.

    A fixed pixel corridor does not work: a car parked at the kerb 30 m ahead
    appears close to the image centre. Instead each box is placed on the road with
    the flat-ground assumption (:meth:`CameraModel.ground_position`) and kept only if
    its lateral offset is within ``lane_half_width_m``. The nearest of those (smallest
    estimated depth) is the lead vehicle. Adds ``z_est`` and ``x_est`` columns.
    """
    if det.empty:
        return det.assign(z_est=pd.Series(dtype=float), x_est=pd.Series(dtype=float))
    z_est, x_est = camera.ground_position(det)
    height = (det["y2"] - det["y1"]).to_numpy()
    keep = np.isfinite(z_est) & (np.abs(x_est) <= lane_half_width_m) & (height >= min_height_px)
    cand = det.assign(z_est=z_est, x_est=x_est)[keep]
    if cand.empty:
        return cand.copy()
    idx = cand.groupby("frame")["z_est"].idxmin()
    return cand.loc[idx].sort_values("frame").reset_index(drop=True)


def ttc_ground_truth(
    labels: pd.DataFrame, fps: float, window: int = 5, min_frames: int = 3,
    ttc_max: float = 10.0,
) -> pd.DataFrame:
    """Reference TTC from KITTI 3D labels: depth ``z`` over closing speed ``-dz/dt``.

    KITTI label positions are in the camera frame, so ``z`` is already the
    longitudinal distance relative to the ego vehicle.
    """
    cols = ["frame", "track_id", "z", "closing_speed", "ttc_gt"]
    if labels.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for tid, g in labels.groupby("track_id"):
        g = g.sort_values("frame")
        t = g["frame"].to_numpy(dtype=float) / fps
        z = g["z"].to_numpy(dtype=float)
        closing = -_window_slope(t, z, window / fps, min_frames)
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(closing > 0, z / closing, np.inf)
        ttc = np.where(ttc > ttc_max, np.inf, ttc)
        ttc = np.where(np.isnan(closing), np.nan, ttc)
        rows.append(pd.DataFrame({"frame": g["frame"].to_numpy(), "track_id": tid, "z": z,
                                  "closing_speed": closing, "ttc_gt": ttc}))
    return pd.concat(rows, ignore_index=True)[cols]


def evaluate_ttc(
    pred_lead: pd.DataFrame,
    labels: pd.DataFrame,
    vcfg: dict,
    camera: CameraModel,
    fps: float,
) -> tuple[pd.DataFrame, dict]:
    """Compare predicted lead vehicle and TTC with the KITTI 3D labels.

    Returns a per-frame table and summary metrics:

    * ``lead_recall`` / ``lead_precision`` - does the camera pipeline pick the same
      lead vehicle as the labels would (box IoU >= ``match_iou``)?
    * TTC error on frames where both TTCs are finite (object approaching).
    * Frame-level precision / recall of the ``TTC < critical_ttc_s`` warning.
    """
    gt_det = detections_from_labels(labels, vcfg["gt_classes"])
    gt_ttc = ttc_ground_truth(labels, fps, vcfg["ttc_window"], vcfg["ttc_min_frames"],
                              vcfg["ttc_max_s"])
    gt_det = gt_det.merge(gt_ttc[["frame", "track_id", "z", "ttc_gt"]],
                          on=["frame", "track_id"], how="left")
    gt_lead = select_lead(gt_det, camera, vcfg["lane_half_width_m"], vcfg["min_box_height_px"])

    match_iou = vcfg["match_iou"]
    crit = vcfg["critical_ttc_s"]
    frames = sorted(set(pred_lead["frame"]) | set(gt_lead["frame"]))
    p_by_f = pred_lead.set_index("frame")
    g_by_f = gt_lead.set_index("frame")
    rows = []
    for f in frames:
        row = {"frame": f, "has_pred": f in p_by_f.index, "has_gt_lead": f in g_by_f.index,
               "lead_match": False, "ttc_pred": np.nan, "ttc_gt": np.nan, "z_gt": np.nan}
        if row["has_pred"]:
            pb = p_by_f.loc[f, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            row["ttc_pred"] = float(p_by_f.loc[f, "ttc"])
            if row["has_gt_lead"]:
                gb = g_by_f.loc[f, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
                row["lead_match"] = bool(box_iou(pb, gb)[0, 0] >= match_iou)
            # TTC reference: the labelled vehicle the predicted box actually covers.
            cand = gt_det[gt_det["frame"] == f]
            if not cand.empty:
                ious = box_iou(pb, cand[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float))[0]
                k = int(np.argmax(ious))
                if ious[k] >= match_iou:
                    row["ttc_gt"] = float(cand.iloc[k]["ttc_gt"])
                    row["z_gt"] = float(cand.iloc[k]["z"])
        rows.append(row)
    per_frame = pd.DataFrame(rows, columns=["frame", "has_pred", "has_gt_lead", "lead_match",
                                            "ttc_pred", "ttc_gt", "z_gt"])
    return per_frame, summarize_ttc(per_frame, crit)


def summarize_ttc(per_frame: pd.DataFrame, critical_ttc_s: float) -> dict:
    """Summary metrics from the per-frame table of :func:`evaluate_ttc`
    (can be applied to several sequences concatenated)."""
    crit = critical_ttc_s
    m: dict = {"frames_with_gt_lead": int(per_frame["has_gt_lead"].sum()),
               "frames_with_pred_lead": int(per_frame["has_pred"].sum())}
    if per_frame.empty:
        return m
    gl, pl = per_frame["has_gt_lead"], per_frame["has_pred"]
    m["lead_recall"] = float(per_frame.loc[gl, "lead_match"].mean()) if gl.any() else np.nan
    m["lead_precision"] = float(per_frame.loc[pl, "lead_match"].mean()) if pl.any() else np.nan

    both = per_frame[np.isfinite(per_frame["ttc_pred"]) & np.isfinite(per_frame["ttc_gt"])]
    m["ttc_pairs"] = int(len(both))
    if len(both):
        err = both["ttc_pred"] - both["ttc_gt"]
        m["ttc_mae_s"] = float(np.abs(err).mean())
        m["ttc_median_abs_err_s"] = float(np.abs(err).median())
        m["ttc_median_rel_err"] = float((np.abs(err) / both["ttc_gt"]).median())

    ref = per_frame[per_frame["ttc_gt"].notna() & per_frame["ttc_pred"].notna()]
    pred_c, gt_c = ref["ttc_pred"] < crit, ref["ttc_gt"] < crit
    tp = int((pred_c & gt_c).sum())
    m["critical_frames_gt"] = int(gt_c.sum())
    m["critical_precision"] = tp / int(pred_c.sum()) if pred_c.sum() else np.nan
    m["critical_recall"] = tp / int(gt_c.sum()) if gt_c.sum() else np.nan
    return m


def lead_approach_events(
    lead: pd.DataFrame,
    sig: pd.DataFrame,
    vcfg: dict,
    hysteresis_ratio: float,
    merge_gap_s: float,
    sequence: str,
) -> list[Event]:
    """Turn the per-frame lead TTC into ``lead_approach`` events (TTC < critical)."""
    if lead.empty:
        return []
    by_frame = lead.set_index("frame")["ttc"]
    frames = sig["frame"].to_numpy()
    ttc = by_frame.reindex(frames).to_numpy(dtype=float)
    ttc = np.where(np.isinf(ttc), vcfg["ttc_max_s"] + 1.0, ttc)
    t = sig["t"].to_numpy()
    crit = float(vcfg["critical_ttc_s"])
    harsh = float(vcfg.get("harsh_ttc_s", crit / 2))
    mask = sig["speed_f"].to_numpy() >= vcfg["critical_min_ego_speed_mps"]
    events = []
    for s, e in detect_intervals(t, ttc, crit, exit_level(crit, "below", hysteresis_ratio),
                                 "below", vcfg["critical_min_duration_s"], merge_gap_s, mask):
        k = s + int(np.nanargmin(ttc[s : e + 1]))
        events.append(Event(
            sequence=sequence, type="lead_approach", t_start=float(t[s]), t_end=float(t[e]),
            t_peak=float(t[k]), frame_start=int(frames[s]), frame_end=int(frames[e]),
            frame_peak=int(frames[k]), peak=float(ttc[k]),
            severity="harsh" if ttc[k] < harsh else "notable",
            signal="ttc", threshold=crit, source="video",
        ))
    return events
