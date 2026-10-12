"""Machine-learning layer on top of the window features.

* :func:`anomaly_scores` - Isolation Forest over all windows. Rules only find what
  they were written for; the anomaly detector surfaces unusual driving the rules
  missed, so those windows can be sent to review (:func:`review_candidates`).
* :func:`cross_validate_classifier` - Random Forest that learns event types from
  labelled windows. Folds are split **by sequence** (GroupKFold): overlapping windows
  of one drive are nearly identical, and a random split would leak them between
  train and test and inflate the scores.

Note on labels: trained on rule labels, the classifier can at best reproduce the
rules - that run is only a check that the features carry the information. The
meaningful run is on human-reviewed labels (see ``drive-events evaluate-labels``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drive_events.features import feature_columns


def anomaly_scores(
    windows: pd.DataFrame, contamination: float = 0.03, seed: int = 42
) -> np.ndarray:
    """Isolation Forest anomaly score per window (higher = more unusual)."""
    cols = feature_columns(windows)
    X = windows[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        IsolationForest(n_estimators=300, contamination=contamination, random_state=seed),
    )
    model.fit(X)
    return -model.decision_function(X)


def review_candidates(
    windows: pd.DataFrame, scores: np.ndarray, top_k: int = 20, label_col: str = "label"
) -> pd.DataFrame:
    """Most anomalous windows that no rule event covers - candidates for manual review.

    Overlapping windows are de-duplicated so each candidate is a distinct moment.
    """
    df = windows.assign(anomaly_score=scores)
    df = df[df[label_col] == "normal"].sort_values("anomaly_score", ascending=False)
    picked: list[int] = []
    for idx, row in df.iterrows():
        clash = any(
            str(df.at[p, "sequence"]) == str(row["sequence"])
            and df.at[p, "t_start"] < row["t_end"] and row["t_start"] < df.at[p, "t_end"]
            for p in picked
        )
        if not clash:
            picked.append(idx)
        if len(picked) >= top_k:
            break
    return df.loc[picked].reset_index(drop=True)


def cross_validate_classifier(
    windows: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "sequence",
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Grouped cross-validation of a Random Forest window classifier."""
    groups = windows[group_col].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("Grouped cross-validation needs at least two sequences")
    cols = feature_columns(windows)
    X = windows[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y = windows[label_col].astype(str).to_numpy()
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               min_samples_leaf=2, random_state=seed, n_jobs=-1),
    )
    cv = GroupKFold(n_splits=min(n_splits, n_groups))
    pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
    classes = sorted(np.unique(np.concatenate([y, pred])))
    majority = pd.Series(y).value_counts().idxmax()

    model.fit(X, y)
    importances = pd.Series(model[-1].feature_importances_, index=cols).sort_values(
        ascending=False)
    return {
        "n_windows": int(len(y)),
        "n_groups": int(n_groups),
        "n_splits": int(cv.get_n_splits()),
        "classes": classes,
        "report": classification_report(y, pred, labels=classes, output_dict=True,
                                        zero_division=0),
        "confusion": pd.DataFrame(confusion_matrix(y, pred, labels=classes),
                                  index=classes, columns=classes),
        "baseline_majority_accuracy": float(np.mean(y == majority)),
        "feature_importance": importances,
        "predictions": pred,
    }
