"""Human review of rule-generated event candidates.

The :class:`LabelStore` keeps one current decision per event in a CSV file and
appends every decision to a ``*_history.csv`` audit log, so changes of mind are
traceable (who, when, what). Writes are atomic, which matters when the file sits
on a mounted Google Drive and a Colab session may disconnect.

Decisions:

* ``accept``  - the rule was right; ``final_type`` = rule type
* ``reject``  - false alarm; the event is dropped from the reviewed set
* ``relabel`` - a real event of a different type; ``final_type`` must be given
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LABEL_COLUMNS = ["event_id", "sequence", "rule_type", "decision", "final_type",
                 "reviewer", "reviewed_at", "comment"]
DECISIONS = ("accept", "reject", "relabel")


class LabelStore:
    def __init__(self, path: str | Path, autosave: bool = True) -> None:
        self.path = Path(path)
        self.history_path = self.path.with_name(self.path.stem + "_history.csv")
        self.autosave = autosave
        if self.path.exists():
            self._df = pd.read_csv(self.path, dtype={"sequence": str}, keep_default_na=False)
        else:
            self._df = pd.DataFrame(columns=LABEL_COLUMNS)

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)

    def record(
        self,
        event_id: str,
        sequence: str,
        rule_type: str,
        decision: str,
        final_type: str | None = None,
        reviewer: str = "",
        comment: str = "",
    ) -> dict:
        """Store (or overwrite) the decision for one event."""
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
        if decision == "accept":
            final_type = rule_type
        elif decision == "reject":
            final_type = ""
        elif not final_type:
            raise ValueError("relabel needs a final_type")
        row = {
            "event_id": event_id, "sequence": str(sequence), "rule_type": rule_type,
            "decision": decision, "final_type": final_type, "reviewer": reviewer,
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "comment": comment,
        }
        self._df = pd.concat(
            [self._df[self._df["event_id"] != event_id], pd.DataFrame([row])],
            ignore_index=True,
        )[LABEL_COLUMNS]
        if self.autosave:
            self.save()
            header = not self.history_path.exists()
            pd.DataFrame([row]).to_csv(self.history_path, mode="a", header=header, index=False)
        return row

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self._df.to_csv(tmp, index=False)
        os.replace(tmp, self.path)

    def pending(self, events: pd.DataFrame) -> pd.DataFrame:
        """Events that have no decision yet."""
        return events[~events["event_id"].isin(self._df["event_id"])]


def apply_reviews(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Reviewed event set: rejected events removed, relabelled ones get the new type.

    Only reviewed events are returned; ``type`` holds the final (human) type and
    ``rule_type`` the original rule output.
    """
    merged = events.merge(labels[["event_id", "decision", "final_type"]], on="event_id",
                          how="inner")
    merged = merged[merged["decision"] != "reject"].copy()
    merged["rule_type"] = merged["type"]
    merged["type"] = merged["final_type"]
    return merged.drop(columns=["final_type"]).reset_index(drop=True)


def rule_agreement(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """How often the reviewer confirmed each rule - an estimate of rule precision."""
    merged = events.merge(labels[["event_id", "decision"]], on="event_id", how="left")
    rows = []
    for typ, g in list(merged.groupby("type")) + [("ALL", merged)]:
        reviewed = g["decision"].isin(DECISIONS)
        acc = int((g["decision"] == "accept").sum())
        n = int(reviewed.sum())
        rows.append({
            "type": typ, "n_events": len(g), "n_reviewed": n, "accepted": acc,
            "relabelled": int((g["decision"] == "relabel").sum()),
            "rejected": int((g["decision"] == "reject").sum()),
            "precision_estimate": acc / n if n else float("nan"),
        })
    return pd.DataFrame(rows)
