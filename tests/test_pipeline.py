from pathlib import Path

import pandas as pd
import yaml

from drive_events.cli import main
from drive_events.config import DEFAULT_CONFIG
from drive_events.evaluation import match_events, summarize_matches
from drive_events.labeling.store import LabelStore
from drive_events.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[1]


def test_yaml_config_matches_defaults():
    with open(REPO / "configs" / "default.yaml") as fh:
        assert yaml.safe_load(fh) == DEFAULT_CONFIG


def test_pipeline_recovers_synthetic_events(synthetic_root, cfg, tmp_path):
    root, truth = synthetic_root
    res = run_pipeline(root, tmp_path / "out", cfg, detector="ground_truth",
                       max_cards=2, log=lambda *_: None)
    summary = summarize_matches(match_events(res.events, truth, 0.3)).set_index("type")
    assert summary.loc["ALL", "recall"] >= 0.9
    assert summary.loc["ALL", "precision"] >= 0.9
    assert summary.loc["lead_approach", "tp"] == 2
    out = tmp_path / "out"
    for f in ("events.csv", "windows.csv", "quality.csv", "summary.md", "thresholds.json",
              "figures/overview.png", "figures/timeline_0000.png", "run_config.yaml"):
        assert (out / f).exists(), f
    assert len(list((out / "cards").glob("*.png"))) == 2
    assert res.ttc_metrics["lead_recall"] == 1.0


def test_pipeline_percentile_mode_and_no_video(synthetic_root, cfg, tmp_path):
    root, _ = synthetic_root
    cfg = {**cfg, "events": {**cfg["events"], "threshold_mode": "percentile"}}
    res = run_pipeline(root, tmp_path / "o", cfg, detector="none", make_cards=False,
                       log=lambda *_: None)
    assert "lead_approach" not in set(res.events["type"])
    assert res.thresholds["braking"]["threshold"] <= -1.05


def test_cards_with_images(synthetic_root_images, cfg, tmp_path):
    root, _ = synthetic_root_images
    run_pipeline(root, tmp_path / "o", cfg, detector="ground_truth", max_cards=1,
                 log=lambda *_: None)
    cards = list((tmp_path / "o" / "cards").glob("*.png"))
    assert len(cards) == 1 and cards[0].stat().st_size > 20_000


def test_cli_end_to_end_with_review(tmp_path, capsys):
    data, out = tmp_path / "syn", tmp_path / "out"
    assert main(["synth", "--out", str(data), "--n", "3", "--seed", "2"]) == 0
    assert main(["run", "--data-root", str(data), "--out", str(out),
                 "--detector", "ground_truth", "--no-cards"]) == 0
    assert main(["evaluate", "--events", str(out / "events.csv"),
                 "--truth", str(data / "truth_events.csv")]) == 0

    # simulate a reviewer: accept everything except jerk events
    events = pd.read_csv(out / "events.csv", dtype={"sequence": str})
    store = LabelStore(tmp_path / "labels.csv")
    for ev in events.itertuples():
        store.record(ev.event_id, ev.sequence, ev.type,
                     "reject" if ev.type == "jerk" else "accept", reviewer="test")
    assert main(["evaluate-labels", "--run", str(out),
                 "--labels", str(tmp_path / "labels.csv")]) == 0
    printed = capsys.readouterr().out
    assert "Rule precision estimated from human review" in printed
    agree = pd.read_csv(out / "rule_agreement.csv").set_index("type")
    assert agree.loc["jerk", "precision_estimate"] == 0.0
    assert (out / "classifier_report.csv").exists()
