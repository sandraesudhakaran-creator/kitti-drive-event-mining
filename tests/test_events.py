import numpy as np
import pandas as pd
import pytest

from drive_events.evaluation import interval_iou, match_events, summarize_matches
from drive_events.events import (
    detect_events,
    detect_intervals,
    events_to_frame,
    exit_level,
    percentile_thresholds,
)

T = np.arange(0, 10, 0.1)


def test_exit_level_is_softer_than_threshold():
    assert exit_level(-2.0, "below", 0.7) == pytest.approx(-1.4)
    assert exit_level(2.0, "above", 0.7) == pytest.approx(1.4)
    assert exit_level(0.3, "below", 0.7) == pytest.approx(0.3 / 0.7)


def test_hysteresis_keeps_one_event_through_noise():
    x = np.zeros_like(T)
    x[20:50] = -2.5
    x[35] = -1.6  # dips back above the threshold but not past the exit level
    iv = detect_intervals(T, x, -2.0, -1.4, "below", 0.5, 0.0)
    assert iv == [(20, 49)]


def test_min_duration_and_merge_gap():
    x = np.zeros_like(T)
    x[10:12] = 3.0           # 0.2 s blip -> dropped
    x[30:40] = 3.0
    x[42:50] = 3.0           # 0.3 s gap -> merged with the previous run
    iv = detect_intervals(T, x, 2.0, 1.4, "above", 0.5, 0.5)
    assert iv == [(30, 49)]


def test_nan_and_mask_end_intervals():
    x = np.full_like(T, 3.0)
    x[50] = np.nan
    mask = np.ones_like(T, dtype=bool)
    mask[80:] = False
    iv = detect_intervals(T, x, 2.0, 1.4, "above", 0.0, 0.0, mask)
    assert iv == [(0, 49), (51, 79)]


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        detect_intervals(T, T, 1, 1, "sideways", 0, 0)


def _sig(**cols):
    n = len(T)
    base = {"t": T, "frame": np.arange(n), "speed_f": np.full(n, 10.0), "ax_f": np.zeros(n),
            "ay_abs": np.zeros(n), "yaw_rate_abs": np.zeros(n), "jerk_abs": np.zeros(n)}
    base.update(cols)
    return pd.DataFrame(base)


def test_detect_events_types_and_severity(cfg):
    ax = np.zeros_like(T)
    ax[20:40] = -3.5          # harsh brake
    ax[60:75] = 1.8           # notable acceleration
    ev = events_to_frame(detect_events(_sig(ax_f=ax), cfg, "0007"))
    assert list(ev["type"]) == ["braking", "acceleration"]
    assert list(ev["severity"]) == ["harsh", "notable"]
    assert ev.loc[0, "peak"] == pytest.approx(-3.5)
    assert ev.loc[0, "event_id"] == "0007_braking_000020"


def test_turn_requires_minimum_speed(cfg):
    yaw = np.full_like(T, 0.3)
    speed = np.where(T < 5, 1.0, 10.0)
    ev = events_to_frame(detect_events(_sig(yaw_rate_abs=yaw, speed_f=speed), cfg, "s"))
    turns = ev[ev["type"] == "turn"]
    assert len(turns) == 1 and turns.iloc[0]["t_start"] >= 5.0


def test_percentile_thresholds_never_below_exit_floor(cfg):
    calm = _sig(ax_f=np.random.default_rng(0).normal(0, 0.1, len(T)))
    thr = percentile_thresholds([calm], cfg)
    assert thr["braking"]["threshold"] <= exit_level(-1.5, "below", 0.7)
    assert thr["acceleration"]["threshold"] >= exit_level(1.5, "above", 0.7)
    assert "standstill" not in thr  # exempt


def test_interval_iou():
    assert interval_iou(0, 2, 1, 3) == pytest.approx(1 / 3)
    assert interval_iou(0, 1, 2, 3) == 0.0


def test_match_events_counts():
    ref = pd.DataFrame({"sequence": ["a", "a"], "type": ["braking", "turn"],
                        "t_start": [1.0, 5.0], "t_end": [2.0, 7.0]})
    pred = pd.DataFrame({"sequence": ["a", "a", "a"], "type": ["braking", "braking", "turn"],
                         "t_start": [1.1, 8.0, 0.0], "t_end": [2.1, 9.0, 0.5]})
    s = summarize_matches(match_events(pred, ref, 0.3)).set_index("type")
    assert s.loc["braking", ["tp", "fp", "fn"]].tolist() == [1, 1, 0]
    assert s.loc["turn", ["tp", "fp", "fn"]].tolist() == [0, 1, 1]
    assert s.loc["ALL", "precision"] == pytest.approx(1 / 3)
