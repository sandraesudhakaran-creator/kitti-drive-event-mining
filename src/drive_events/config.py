"""Pipeline configuration.

The defaults live here so the package works without a config file;
``configs/default.yaml`` mirrors them for documentation and is kept in sync by a test.
A user config is deep-merged on top of the defaults.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "split": "training",
        "camera": "image_02",
        "fps": 10.0,
    },
    "signals": {
        "resample_hz": 10.0,
        "lowpass_cutoff_hz": 1.5,
        "filter_order": 2,
        "max_gap_s": 0.5,
        # Physically plausible ranges; samples outside are treated as sensor faults.
        "plausibility": {
            "speed": [-1.0, 70.0],
            "ax": [-12.0, 12.0],
            "ay": [-12.0, 12.0],
            "yaw_rate": [-1.5, 1.5],
        },
    },
    "events": {
        # "absolute": thresholds below are used as-is.
        # "percentile": per-signal thresholds are set from the data distribution
        # (useful on calm datasets where fixed thresholds rarely fire).
        "threshold_mode": "absolute",
        "percentile": 97.0,
        "hysteresis_ratio": 0.7,
        "merge_gap_s": 0.5,
        "definitions": {
            "braking": {
                "signal": "ax_f", "direction": "below",
                "threshold": -1.5, "harsh": -3.0, "min_duration_s": 0.5,
            },
            "acceleration": {
                "signal": "ax_f", "direction": "above",
                "threshold": 1.5, "harsh": 2.5, "min_duration_s": 0.5,
            },
            "turn": {
                "signal": "yaw_rate_abs", "direction": "above",
                "threshold": 0.15, "harsh": 0.35, "min_duration_s": 1.0,
                "min_speed_mps": 2.0,
            },
            "lateral": {
                "signal": "ay_abs", "direction": "above",
                "threshold": 2.0, "harsh": 3.5, "min_duration_s": 0.5,
            },
            "jerk": {
                "signal": "jerk_abs", "direction": "above",
                "threshold": 3.0, "harsh": 6.0, "min_duration_s": 0.2,
            },
            "standstill": {
                "signal": "speed_f", "direction": "below",
                "threshold": 0.3, "min_duration_s": 1.0,
                "percentile_exempt": True,
            },
        },
    },
    "video": {
        "detector": "yolo",  # yolo | ground_truth | none
        "yolo_weights": "yolov8n.pt",
        "yolo_conf": 0.35,
        "yolo_imgsz": 1280,
        "yolo_batch": 16,
        "yolo_classes": ["car", "truck", "bus"],
        "gt_classes": ["Car", "Van", "Truck"],
        "tracker_iou": 0.3,
        "tracker_max_age": 2,
        "camera_height_m": 1.65,  # KITTI camera height above the road
        "lane_half_width_m": 1.8,  # lateral offset that still counts as "in the ego lane"
        "min_box_height_px": 12,
        "ttc_window": 5,
        "ttc_min_frames": 3,
        "ttc_max_s": 10.0,
        "critical_ttc_s": 3.0,
        "harsh_ttc_s": 1.5,
        "critical_min_duration_s": 0.3,
        "critical_min_ego_speed_mps": 1.0,
        "match_iou": 0.5,
    },
    "windows": {
        "window_s": 2.0,
        "stride_s": 0.5,
        "min_overlap": 0.5,
        "max_nan_fraction": 0.2,
    },
    "models": {
        "contamination": 0.03,
        "review_top_k": 20,
        "cv_splits": 5,
        "seed": 42,
    },
    "cards": {
        "context_s": 3.0,
        "max_cards": 60,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of ``base`` with ``override`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML config on top of the defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        cfg = deep_merge(cfg, user)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
