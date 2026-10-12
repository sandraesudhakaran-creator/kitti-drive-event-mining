"""Frame-to-frame association of detections (IoU + Hungarian matching).

Kept deliberately simple: at 10 Hz and for the vehicle directly ahead, box overlap
between consecutive frames is large, so IoU association is sufficient and fully
deterministic. Tracks survive ``max_age`` frames without a match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between box sets ``a`` (N,4) and ``b`` (M,4) in x1,y1,x2,y2."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def track_detections(
    det: pd.DataFrame, iou_threshold: float = 0.3, max_age: int = 2
) -> pd.DataFrame:
    """Assign a ``track_id`` to every detection (overwrites existing ids)."""
    out = det.sort_values("frame", kind="stable").reset_index(drop=True).copy()
    ids = np.full(len(out), -1, dtype=int)
    tracks: dict[int, tuple[np.ndarray, int]] = {}  # id -> (last box, last frame)
    next_id = 0
    for frame, idx in out.groupby("frame").groups.items():
        idx = np.asarray(idx)
        boxes = out.loc[idx, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        live = [k for k, (_, f) in tracks.items() if frame - f <= max_age]
        assigned = np.full(len(idx), -1)
        if live and len(idx):
            iou = box_iou(np.stack([tracks[k][0] for k in live]), boxes)
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= iou_threshold:
                    assigned[c] = live[r]
        for j in range(len(idx)):
            if assigned[j] < 0:
                assigned[j] = next_id
                next_id += 1
            tracks[int(assigned[j])] = (boxes[j], int(frame))
        ids[idx] = assigned
    out["track_id"] = ids
    return out
