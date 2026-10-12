import numpy as np
import pandas as pd
import pytest

from drive_events.features import feature_columns, label_windows, make_windows
from drive_events.labeling.store import LabelStore, apply_reviews, rule_agreement
from drive_events.models import anomaly_scores, cross_validate_classifier, review_candidates


def _sig(n=200, fs=10.0):
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    return pd.DataFrame({"t": t, "frame": np.arange(n), "speed_f": 10 + rng.normal(0, .1, n),
                         "ax_f": rng.normal(0, .1, n), "ay_f": rng.normal(0, .1, n),
                         "yaw_rate_f": rng.normal(0, .01, n), "jerk": rng.normal(0, .3, n)})


def test_make_windows_shape_and_ttc_feature():
    sig = _sig()
    lead = pd.DataFrame({"frame": np.arange(100, 130), "ttc": np.linspace(5, 2, 30)})
    w = make_windows(sig, "s", 2.0, 0.5, lead=lead, ttc_cap=10.0)
    assert len(w) == (200 - 20) // 5 + 1
    assert "ttc__min" in w and w["ttc__min"].min() == pytest.approx(2.0, abs=0.11)
    assert w.loc[0, "ttc__min"] == 10.0 and w.loc[0, "lead__fraction"] == 0.0
    assert all("__" in c for c in feature_columns(w))


def test_label_windows_uses_overlap_with_shorter_interval():
    w = pd.DataFrame({"sequence": ["s"] * 3, "t_start": [0.0, 2.0, 4.0],
                      "t_end": [1.9, 3.9, 5.9]})
    ev = pd.DataFrame({"sequence": ["s"], "type": ["braking"], "t_start": [2.5],
                       "t_end": [3.0]})
    assert label_windows(w, ev).tolist() == ["normal", "braking", "normal"]


def test_anomaly_candidates_exclude_labelled_and_overlapping_windows():
    sig = _sig(400)
    sig.loc[300:315, "ax_f"] = -4.0  # unusual moment the "rules" did not label
    w = make_windows(sig, "s", 2.0, 0.5)
    w["label"] = "normal"
    w.loc[0, "label"] = "turn"
    scores = anomaly_scores(w, contamination=0.05, seed=0)
    cand = review_candidates(w, scores, top_k=3)
    assert "turn" not in set(cand["label"])
    top = cand.iloc[0]
    assert top["t_start"] <= 31.5 and top["t_end"] >= 30.0
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = cand.iloc[i], cand.iloc[j]
            assert not (a.t_start < b.t_end and b.t_start < a.t_end)


def test_grouped_cross_validation_runs_and_needs_two_groups():
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({"sequence": np.repeat(["a", "b", "c"], n // 3),
                       "f__x": rng.normal(0, 1, n)})
    df["label"] = np.where(df["f__x"] > 0.5, "event", "normal")
    res = cross_validate_classifier(df, n_splits=3)
    assert res["n_splits"] == 3 and res["report"]["accuracy"] > 0.9
    with pytest.raises(ValueError):
        cross_validate_classifier(df[df.sequence == "a"])


def test_label_store_roundtrip_history_and_validation(tmp_path):
    path = tmp_path / "labels.csv"
    store = LabelStore(path)
    store.record("e1", "0001", "braking", "accept", reviewer="me")
    store.record("e2", "0001", "jerk", "reject")
    store.record("e2", "0001", "jerk", "relabel", final_type="braking", comment="changed mind")
    with pytest.raises(ValueError):
        store.record("e3", "0001", "turn", "relabel")
    with pytest.raises(ValueError):
        store.record("e3", "0001", "turn", "maybe")

    reloaded = LabelStore(path)
    assert len(reloaded) == 2
    assert reloaded.df.set_index("event_id").loc["e2", "final_type"] == "braking"
    assert len(pd.read_csv(tmp_path / "labels_history.csv")) == 3

    events = pd.DataFrame({"event_id": ["e1", "e2", "e3"], "sequence": ["0001"] * 3,
                           "type": ["braking", "jerk", "turn"]})
    assert reloaded.pending(events)["event_id"].tolist() == ["e3"]
    reviewed = apply_reviews(events, reloaded.df)
    assert reviewed["type"].tolist() == ["braking", "braking"]
    agree = rule_agreement(events, reloaded.df).set_index("type")
    assert agree.loc["ALL", "n_reviewed"] == 2
    assert agree.loc["ALL", "precision_estimate"] == 0.5
