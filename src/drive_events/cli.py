"""Command-line interface.

    drive-events synth           --out data/synthetic --n 3 [--images]
    drive-events run             --data-root data/kitti_tracking --out outputs/kitti
    drive-events evaluate        --events outputs/x/events.csv \
                                 --truth data/synthetic/truth_events.csv
    drive-events evaluate-labels --run outputs/kitti --labels labels.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from drive_events.config import load_config


def _cmd_synth(args: argparse.Namespace) -> int:
    from drive_events.synthetic import write_dataset

    truth = write_dataset(args.out, args.n, args.seed, images=args.images)
    print(f"Wrote {args.n} synthetic sequences to {args.out} "
          f"({len(truth)} ground-truth events)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from drive_events.pipeline import run_pipeline

    overrides = {}
    if args.threshold_mode:
        overrides = {"events": {"threshold_mode": args.threshold_mode}}
    cfg = load_config(args.config, overrides)
    run_pipeline(args.data_root, args.out, cfg, args.sequences, args.detector,
                 make_cards=not args.no_cards, max_cards=args.max_cards)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from drive_events.evaluation import match_events, summarize_matches

    pred = pd.read_csv(args.events, dtype={"sequence": str})
    ref = pd.read_csv(args.truth, dtype={"sequence": str})
    summary = summarize_matches(match_events(pred, ref, args.iou))
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.out:
        summary.to_csv(args.out, index=False)
    return 0


def _cmd_evaluate_labels(args: argparse.Namespace) -> int:
    from drive_events.features import label_windows
    from drive_events.labeling.store import apply_reviews, rule_agreement
    from drive_events.models import cross_validate_classifier

    run = Path(args.run)
    cfg = load_config(run / "run_config.yaml")
    events = pd.read_csv(run / "events.csv", dtype={"sequence": str})
    labels = pd.read_csv(args.labels, dtype={"sequence": str}, keep_default_na=False)

    agreement = rule_agreement(events, labels)
    print("Rule precision estimated from human review:\n")
    print(agreement.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    agreement.to_csv(run / "rule_agreement.csv", index=False)

    # Train only on reviewed sequences: windows of unreviewed events would
    # otherwise be counted as 'normal'.
    reviewed = apply_reviews(events, labels)
    windows = pd.read_csv(run / "windows.csv", dtype={"sequence": str})
    windows = windows[windows["sequence"].isin(labels["sequence"].astype(str).unique())]
    windows = windows.drop(columns=["label", "anomaly_score"], errors="ignore")
    windows["label"] = label_windows(windows, reviewed, cfg["windows"]["min_overlap"])
    try:
        res = cross_validate_classifier(windows, n_splits=cfg["models"]["cv_splits"],
                                        seed=cfg["models"]["seed"])
    except ValueError as exc:
        print(f"\nClassifier skipped: {exc}")
        return 0
    rep = pd.DataFrame(res["report"]).T
    print(f"\nWindow classifier on reviewed labels ({res['n_windows']} windows, "
          f"{res['n_groups']} sequences, GroupKFold k={res['n_splits']}):\n")
    print(rep.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nMajority-class baseline accuracy: {res['baseline_majority_accuracy']:.3f}")
    print("\nTop features:\n" + res["feature_importance"].head(8).to_string())
    rep.to_csv(run / "classifier_report.csv")
    res["confusion"].to_csv(run / "classifier_confusion.csv")
    with open(run / "classifier_summary.json", "w") as fh:
        json.dump({k: res[k] for k in ("n_windows", "n_groups", "n_splits", "classes",
                                       "baseline_majority_accuracy")}, fh, indent=2)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="drive-events", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write synthetic KITTI-format drives with ground truth")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--images", action="store_true", help="also render camera frames")
    s.set_defaults(func=_cmd_synth)

    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument("--data-root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--config", default=None)
    r.add_argument("--sequences", nargs="*", default=None)
    r.add_argument("--detector", choices=["yolo", "ground_truth", "none"], default=None)
    r.add_argument("--threshold-mode", choices=["absolute", "percentile"], default=None)
    r.add_argument("--max-cards", type=int, default=None)
    r.add_argument("--no-cards", action="store_true")
    r.set_defaults(func=_cmd_run)

    e = sub.add_parser("evaluate", help="compare detected events with reference events")
    e.add_argument("--events", required=True)
    e.add_argument("--truth", required=True)
    e.add_argument("--iou", type=float, default=0.3)
    e.add_argument("--out", default=None)
    e.set_defaults(func=_cmd_evaluate)

    v = sub.add_parser("evaluate-labels",
                       help="rule precision from human review + classifier on reviewed labels")
    v.add_argument("--run", required=True, help="output folder of a pipeline run")
    v.add_argument("--labels", required=True, help="CSV written by the review tool")
    v.set_defaults(func=_cmd_evaluate_labels)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
