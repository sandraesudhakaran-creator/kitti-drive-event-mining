"""Rule-based detection of driving events in processed vehicle signals.

Each event type is a threshold rule on one signal with

* **hysteresis** - an event starts when the signal crosses ``threshold`` and ends
  only when it falls back past a softer exit level, so noise near the threshold
  does not split one manoeuvre into many short events;
* **merging** - events of the same type separated by less than ``merge_gap_s``
  are joined;
* **minimum duration** - shorter detections are discarded as noise;
* **severity** - ``harsh`` if the peak passes the second threshold, else ``notable``.

The rules produce *candidate labels*; they are reviewed by a person in the labeling
step, and that review is what measures how good the rules are.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "event_id", "sequence", "type", "severity", "source",
    "t_start", "t_end", "t_peak", "duration_s",
    "frame_start", "frame_end", "frame_peak", "peak", "signal", "threshold",
]


@dataclass
class Event:
    sequence: str
    type: str
    t_start: float
    t_end: float
    t_peak: float
    frame_start: int
    frame_end: int
    frame_peak: int
    peak: float
    severity: str
    signal: str
    threshold: float
    source: str = "rule"

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    @property
    def event_id(self) -> str:
        return f"{self.sequence}_{self.type}_{self.frame_start:06d}"

    def to_row(self) -> dict:
        row = asdict(self)
        row["event_id"] = self.event_id
        row["duration_s"] = self.duration_s
        return row


def _beyond(x: np.ndarray, level: float, direction: str) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return x > level if direction == "above" else x < level


def exit_level(threshold: float, direction: str, ratio: float) -> float:
    """Softer level at which an active event ends (hysteresis)."""
    if direction == "above":
        return threshold * ratio if threshold > 0 else threshold / ratio
    return threshold * ratio if threshold < 0 else threshold / ratio


def detect_intervals(
    t: np.ndarray,
    x: np.ndarray,
    enter: float,
    exit: float,
    direction: str,
    min_duration_s: float,
    merge_gap_s: float,
    mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Find index intervals ``[start, end]`` (inclusive) where a rule is active.

    NaN samples and samples where ``mask`` is False always end an interval.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
    ok = ~np.isnan(x)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    start_ok = ok & _beyond(x, enter, direction)
    stay_ok = ok & _beyond(x, exit, direction)

    raw: list[tuple[int, int]] = []
    active, start = False, 0
    for i in range(len(x)):
        if not active and start_ok[i]:
            active, start = True, i
        elif active and not stay_ok[i]:
            raw.append((start, i - 1))
            active = False
    if active:
        raw.append((start, len(x) - 1))

    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and t[s] - t[merged[-1][1]] <= merge_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    return [(s, e) for s, e in merged if (t[e] - t[s] + dt) >= min_duration_s]


def percentile_thresholds(signals: list[pd.DataFrame], cfg: dict) -> dict[str, dict]:
    """Data-driven thresholds: the configured percentile of each rule's signal.

    The result never goes below the absolute threshold's *exit* level, so a very calm
    recording cannot turn ordinary driving into events.
    """
    ecfg = cfg["events"]
    q = float(ecfg["percentile"])
    out: dict[str, dict] = {}
    for name, d in ecfg["definitions"].items():
        if d.get("percentile_exempt"):
            continue
        x = np.concatenate([s[d["signal"]].to_numpy(dtype=float) for s in signals])
        x = x[~np.isnan(x)]
        if not len(x):
            continue
        floor = exit_level(d["threshold"], d["direction"], ecfg["hysteresis_ratio"])
        if d["direction"] == "above":
            thr = max(float(np.percentile(x, q)), floor)
        else:
            thr = min(float(np.percentile(x, 100 - q)), floor)
        out[name] = {"threshold": thr}
    return out


def detect_events(
    sig: pd.DataFrame,
    cfg: dict,
    sequence: str,
    overrides: dict[str, dict] | None = None,
) -> list[Event]:
    """Apply every configured rule to one sequence."""
    ecfg = cfg["events"]
    t = sig["t"].to_numpy()
    frames = sig["frame"].to_numpy()
    events: list[Event] = []
    for name, d in ecfg["definitions"].items():
        d = {**d, **(overrides or {}).get(name, {})}
        direction = d["direction"]
        thr = float(d["threshold"])
        x = sig[d["signal"]].to_numpy(dtype=float)
        mask = None
        if "min_speed_mps" in d:
            mask = sig["speed_f"].to_numpy() >= d["min_speed_mps"]
        ex = d.get("exit", exit_level(thr, direction, ecfg["hysteresis_ratio"]))
        for s, e in detect_intervals(
            t, x, thr, ex, direction, d["min_duration_s"], ecfg["merge_gap_s"], mask
        ):
            seg = x[s : e + 1]
            k = s + int(np.nanargmax(seg) if direction == "above" else np.nanargmin(seg))
            harsh = d.get("harsh")
            severity = "notable"
            if harsh is not None and _beyond(np.array([x[k]]), harsh, direction)[0]:
                severity = "harsh"
            events.append(
                Event(
                    sequence=sequence, type=name,
                    t_start=float(t[s]), t_end=float(t[e]), t_peak=float(t[k]),
                    frame_start=int(frames[s]), frame_end=int(frames[e]),
                    frame_peak=int(frames[k]), peak=float(x[k]), severity=severity,
                    signal=d["signal"], threshold=thr,
                )
            )
    return events


def events_to_frame(events: list[Event]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.DataFrame([e.to_row() for e in events])[EVENT_COLUMNS]
    return df.sort_values(["sequence", "t_start", "type"]).reset_index(drop=True)
