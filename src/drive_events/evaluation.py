"""Event-level evaluation: match detected intervals to reference intervals.

A detection counts as a true positive when it has the same sequence and type as a
reference event and their time intervals overlap with IoU >= ``iou_threshold``.
Matching is one-to-one and greedy by IoU. The reference can be synthetic ground
truth or the human-reviewed labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else float(a0 == b0 and a1 == b1)


def match_events(
    pred: pd.DataFrame, ref: pd.DataFrame, iou_threshold: float = 0.3
) -> pd.DataFrame:
    """Return one row per pred/ref pair or unmatched event, with ``outcome`` tp/fp/fn."""
    rows = []
    keys = set(map(tuple, pred[["sequence", "type"]].astype(str).to_numpy())) | set(
        map(tuple, ref[["sequence", "type"]].astype(str).to_numpy())
    )
    for seq, typ in sorted(keys):
        p = pred[(pred["sequence"].astype(str) == seq) & (pred["type"] == typ)]
        r = ref[(ref["sequence"].astype(str) == seq) & (ref["type"] == typ)]
        pairs = []
        for i, pe in p.iterrows():
            for j, re_ in r.iterrows():
                iou = interval_iou(pe.t_start, pe.t_end, re_.t_start, re_.t_end)
                if iou >= iou_threshold:
                    pairs.append((iou, i, j))
        used_p, used_r = set(), set()
        for iou, i, j in sorted(pairs, reverse=True):
            if i in used_p or j in used_r:
                continue
            used_p.add(i)
            used_r.add(j)
            rows.append({"sequence": seq, "type": typ, "outcome": "tp", "iou": iou,
                         "pred_index": i, "ref_index": j})
        for i in p.index.difference(list(used_p)):
            rows.append({"sequence": seq, "type": typ, "outcome": "fp", "iou": np.nan,
                         "pred_index": i, "ref_index": None})
        for j in r.index.difference(list(used_r)):
            rows.append({"sequence": seq, "type": typ, "outcome": "fn", "iou": np.nan,
                         "pred_index": None, "ref_index": j})
    return pd.DataFrame(rows, columns=["sequence", "type", "outcome", "iou",
                                       "pred_index", "ref_index"])


def summarize_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Precision / recall / F1 per event type plus an ``ALL`` row."""
    def _row(df: pd.DataFrame, name: str) -> dict:
        tp = int((df["outcome"] == "tp").sum())
        fp = int((df["outcome"] == "fp").sum())
        fn = int((df["outcome"] == "fn").sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        return {"type": name, "tp": tp, "fp": fp, "fn": fn, "precision": prec,
                "recall": rec, "f1": f1, "mean_iou": float(df["iou"].mean())}

    rows = [_row(g, t) for t, g in matches.groupby("type")]
    rows.append(_row(matches, "ALL"))
    return pd.DataFrame(rows)
