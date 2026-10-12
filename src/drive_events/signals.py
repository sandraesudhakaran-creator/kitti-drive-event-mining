"""Vehicle-dynamics signal preparation: plausibility checks, resampling and filtering.

The OXTS unit provides the ego motion in the vehicle frame:

* ``vf``  forward velocity            -> ``speed``     [m/s]
* ``af``  forward acceleration        -> ``ax``        [m/s^2]
* ``al``  leftward acceleration       -> ``ay``        [m/s^2]
* ``wu``  angular rate about up axis  -> ``yaw_rate``  [rad/s]

Processing steps (each one is a function so it can be tested on its own):

1. :func:`check_quality`      report timing gaps, non-monotonic time, NaNs, out-of-range values
2. :func:`clean`              sort by time, drop duplicates, set implausible samples to NaN
3. :func:`resample_uniform`   interpolate onto a uniform time grid, leave real gaps as NaN
4. :func:`lowpass`            zero-phase Butterworth filter, applied per contiguous segment
5. derived signals            jerk, absolute values, speed-derived acceleration, distance
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

BASE_SIGNALS = ["speed", "ax", "ay", "yaw_rate"]


def oxts_to_signals(oxts: pd.DataFrame) -> pd.DataFrame:
    """Select and rename the OXTS fields used for event analysis."""
    return pd.DataFrame(
        {
            "t": oxts["t"].to_numpy(dtype=float),
            "frame": oxts["frame"].to_numpy(dtype=int),
            "speed": oxts["vf"].to_numpy(dtype=float),
            "ax": oxts["af"].to_numpy(dtype=float),
            "ay": oxts["al"].to_numpy(dtype=float),
            "yaw_rate": oxts["wu"].to_numpy(dtype=float),
            "lat": oxts["lat"].to_numpy(dtype=float),
            "lon": oxts["lon"].to_numpy(dtype=float),
        }
    )


@dataclass
class QualityReport:
    """Data-quality summary of one recording."""

    sequence: str
    n_samples: int
    duration_s: float
    median_dt_s: float
    n_time_gaps: int
    max_gap_s: float
    n_non_monotonic: int
    n_nan: dict = field(default_factory=dict)
    n_out_of_range: dict = field(default_factory=dict)
    accel_consistency_rmse: float = float("nan")
    accel_bias: float = float("nan")
    valid_fraction: float = float("nan")

    def to_row(self) -> dict:
        row = asdict(self)
        for key in ("n_nan", "n_out_of_range"):
            for sig, val in row.pop(key).items():
                row[f"{key}__{sig}"] = val
        return row


def check_quality(
    sig: pd.DataFrame,
    plausibility: dict,
    expected_dt: float,
    sequence: str = "",
) -> QualityReport:
    """Summarise timing and value problems in raw (not yet cleaned) signals."""
    t = sig["t"].to_numpy(dtype=float)
    dt = np.diff(t)
    gaps = dt[dt > 1.5 * expected_dt]
    n_nan, n_oor = {}, {}
    for name in BASE_SIGNALS:
        x = sig[name].to_numpy(dtype=float)
        n_nan[name] = int(np.isnan(x).sum())
        lo, hi = plausibility.get(name, (-np.inf, np.inf))
        with np.errstate(invalid="ignore"):
            n_oor[name] = int(((x < lo) | (x > hi)).sum())
    return QualityReport(
        sequence=sequence,
        n_samples=len(sig),
        duration_s=float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        median_dt_s=float(np.median(dt)) if len(dt) else float("nan"),
        n_time_gaps=int(len(gaps)),
        max_gap_s=float(gaps.max()) if len(gaps) else 0.0,
        n_non_monotonic=int((dt <= 0).sum()),
        n_nan=n_nan,
        n_out_of_range=n_oor,
    )


def clean(sig: pd.DataFrame, plausibility: dict) -> pd.DataFrame:
    """Sort by time, drop duplicate timestamps and blank out implausible values."""
    out = sig.sort_values("t", kind="stable").drop_duplicates("t", keep="first").copy()
    for name in BASE_SIGNALS:
        lo, hi = plausibility.get(name, (-np.inf, np.inf))
        x = out[name].to_numpy(dtype=float, copy=True)
        with np.errstate(invalid="ignore"):
            bad = (x < lo) | (x > hi)
        x[bad] = np.nan
        out[name] = x
    return out.reset_index(drop=True)


def resample_uniform(
    sig: pd.DataFrame,
    hz: float,
    max_gap_s: float,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Linearly interpolate onto a uniform grid of ``hz``.

    Grid points that fall inside a gap longer than ``max_gap_s`` between valid samples
    stay NaN, so the analysis never invents data across dropouts. ``frame`` is the
    nearest original frame (used to look up the matching camera image).
    """
    columns = columns or BASE_SIGNALS
    t = sig["t"].to_numpy(dtype=float)
    step = 1.0 / hz
    grid = np.round(np.arange(t[0], t[-1] + step / 2, step), 9)
    out = pd.DataFrame({"t": grid})

    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    nearest = np.where(np.abs(t[idx - 1] - grid) <= np.abs(t[idx] - grid), idx - 1, idx)
    out["frame"] = sig["frame"].to_numpy()[nearest]

    for col in columns:
        x = sig[col].to_numpy(dtype=float)
        valid = ~np.isnan(x)
        if valid.sum() < 2:
            out[col] = np.nan
            continue
        tv, xv = t[valid], x[valid]
        y = np.interp(grid, tv, xv)
        j = np.searchsorted(tv, grid)
        prev_t = np.where(j > 0, tv[np.clip(j - 1, 0, len(tv) - 1)], -np.inf)
        next_t = np.where(j < len(tv), tv[np.clip(j, 0, len(tv) - 1)], np.inf)
        exact = np.isclose(next_t, grid)
        in_gap = (next_t - prev_t > max_gap_s + 1e-9) & ~exact
        y[in_gap] = np.nan
        out[col] = y
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, stop) index pairs of consecutive True values."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    change = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(change[0::2], change[1::2]))


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter that respects NaN gaps.

    Each contiguous valid segment is filtered on its own (``filtfilt`` cannot
    cross NaNs); segments too short to filter are passed through unchanged.
    """
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan)
    b, a = butter(order, cutoff_hz / (fs / 2.0))
    padlen = 3 * max(len(a), len(b))
    for s, e in _runs(~np.isnan(x)):
        seg = x[s:e]
        if len(seg) > padlen:
            y[s:e] = filtfilt(b, a, seg, padlen=padlen)
        else:
            y[s:e] = seg
    return y


def _gradient(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Time derivative that stays NaN next to NaN samples."""
    if len(x) < 2:
        return np.full_like(x, np.nan)
    return np.gradient(x, t)


def preprocess(
    oxts: pd.DataFrame, cfg: dict, sequence: str = ""
) -> tuple[pd.DataFrame, QualityReport]:
    """Full signal preparation for one sequence.

    Returns the processed signal table (uniform time grid, raw + filtered + derived
    columns) and its :class:`QualityReport`.
    """
    scfg = cfg["signals"]
    fs = float(scfg["resample_hz"])
    raw = oxts_to_signals(oxts)
    report = check_quality(raw, scfg["plausibility"], 1.0 / cfg["data"]["fps"], sequence)

    sig = resample_uniform(clean(raw, scfg["plausibility"]), fs, scfg["max_gap_s"])
    for name in BASE_SIGNALS:
        sig[f"{name}_f"] = lowpass(
            sig[name].to_numpy(), fs, scfg["lowpass_cutoff_hz"], scfg["filter_order"]
        )

    t = sig["t"].to_numpy()
    sig["jerk"] = _gradient(sig["ax_f"].to_numpy(), t)
    sig["ax_from_speed"] = _gradient(sig["speed_f"].to_numpy(), t)
    sig["yaw_rate_abs"] = sig["yaw_rate_f"].abs()
    sig["ay_abs"] = sig["ay_f"].abs()
    sig["jerk_abs"] = sig["jerk"].abs()
    step = np.diff(t, prepend=t[0])
    sig["distance_m"] = np.cumsum(np.nan_to_num(sig["speed_f"].clip(lower=0).to_numpy()) * step)

    # Cross-check two independent measurements: the IMU acceleration and the
    # derivative of the measured speed. A large residual points to sensor bias,
    # a timing problem or a wrong sign convention.
    resid = (sig["ax_f"] - sig["ax_from_speed"]).to_numpy()
    resid = resid[~np.isnan(resid)]
    if len(resid):
        report.accel_consistency_rmse = float(np.sqrt(np.mean(resid**2)))
        report.accel_bias = float(np.mean(resid))
    report.valid_fraction = float(sig[[f"{n}_f" for n in BASE_SIGNALS]].notna().all(axis=1).mean())
    return sig, report
