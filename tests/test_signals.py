import numpy as np
import pandas as pd

from drive_events.io.kitti import load_oxts
from drive_events.signals import (
    check_quality,
    clean,
    lowpass,
    preprocess,
    resample_uniform,
)

PLAUS = {"speed": [-1, 70], "ax": [-12, 12], "ay": [-12, 12], "yaw_rate": [-1.5, 1.5]}


def _sig(t, **cols):
    n = len(t)
    base = {"t": t, "frame": np.arange(n), "speed": np.ones(n), "ax": np.zeros(n),
            "ay": np.zeros(n), "yaw_rate": np.zeros(n)}
    base.update(cols)
    return pd.DataFrame(base)


def test_quality_flags_gaps_order_and_range():
    t = np.array([0.0, 0.1, 0.2, 0.9, 0.85, 1.0])
    sig = _sig(t, ax=np.array([0, 0, 50.0, 0, 0, np.nan]))
    q = check_quality(sig, PLAUS, 0.1)
    assert q.n_time_gaps == 1 and abs(q.max_gap_s - 0.7) < 1e-9
    assert q.n_non_monotonic == 1
    assert q.n_out_of_range["ax"] == 1 and q.n_nan["ax"] == 1


def test_clean_sorts_dedups_and_blanks_outliers():
    sig = _sig(np.array([0.2, 0.0, 0.1, 0.1]), ax=np.array([1.0, 2.0, 99.0, 3.0]))
    out = clean(sig, PLAUS)
    assert list(out["t"]) == [0.0, 0.1, 0.2]
    assert np.isnan(out.loc[1, "ax"])


def test_resample_keeps_long_gaps_empty():
    t = np.concatenate([np.arange(0, 1.0, 0.1), np.arange(2.0, 3.0, 0.1)])
    sig = _sig(t, speed=t.copy())
    out = resample_uniform(sig, 10.0, max_gap_s=0.5)
    inside_gap = (out["t"] > 0.95) & (out["t"] < 1.95)
    assert out.loc[inside_gap, "speed"].isna().all()
    assert np.allclose(out.loc[~inside_gap, "speed"], out.loc[~inside_gap, "t"], atol=1e-9)


def test_resample_bridges_short_gaps():
    t = np.array([0.0, 0.1, 0.4, 0.5])
    out = resample_uniform(_sig(t, speed=t * 2), 10.0, max_gap_s=0.5)
    assert np.allclose(out["speed"], out["t"] * 2)


def test_lowpass_keeps_dc_and_removes_high_frequency():
    fs = 10.0
    t = np.arange(0, 30, 1 / fs)
    x = 2.0 + np.sin(2 * np.pi * 4.0 * t)  # 4 Hz noise on a constant
    y = lowpass(x, fs, cutoff_hz=1.0, order=2)
    core = y[50:-50]
    assert abs(core.mean() - 2.0) < 0.02 and core.std() < 0.05


def test_lowpass_respects_nan_segments():
    x = np.concatenate([np.ones(40), [np.nan] * 5, np.full(40, 3.0)])
    y = lowpass(x, 10.0, 1.5)
    assert np.isnan(y[40:45]).all()
    assert np.allclose(y[:40], 1.0) and np.allclose(y[45:], 3.0)


def test_preprocess_end_to_end(synthetic_root, cfg):
    root, _ = synthetic_root
    sig, q = preprocess(load_oxts(root / "training" / "oxts" / "0000.txt"), cfg, "0000")
    for col in ("speed_f", "ax_f", "jerk", "yaw_rate_abs", "distance_m"):
        assert col in sig
    # the two injected 40 m/s^2 spikes are caught, cleaned and interpolated over
    assert q.n_out_of_range["ax"] == 2
    assert sig["ax_f"].abs().max() < 6.0
    # IMU and speed derivative agree up to the simulated bias and noise
    assert q.accel_consistency_rmse < 0.3
    assert 0.0 < q.accel_bias < 0.1
    assert q.valid_fraction == 1.0
