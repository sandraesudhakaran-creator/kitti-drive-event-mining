"""Vehicle detection on camera frames.

Two interchangeable sources produce the same table
(``frame, x1, y1, x2, y2, conf, cls, track_id``):

* :class:`YoloDetector` - a pretrained Ultralytics YOLOv8 model (COCO classes),
  the realistic setting: no labels are needed at inference time.
* :func:`detections_from_labels` - the KITTI ground-truth boxes, used as an upper
  bound ("oracle detector") and in tests and CI, where no GPU or model is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DET_COLUMNS = ["frame", "x1", "y1", "x2", "y2", "conf", "cls", "track_id"]


def empty_detections() -> pd.DataFrame:
    return pd.DataFrame(columns=DET_COLUMNS).astype(
        {"frame": int, "x1": float, "y1": float, "x2": float, "y2": float,
         "conf": float, "cls": str, "track_id": int}
    )


def detections_from_labels(
    labels: pd.DataFrame, classes: tuple[str, ...] | list[str] = ("Car", "Van", "Truck")
) -> pd.DataFrame:
    """Turn KITTI tracking labels into detections (confidence 1, GT track ids)."""
    df = labels[labels["type"].isin(list(classes))]
    if df.empty:
        return empty_detections()
    out = df[["frame", "x1", "y1", "x2", "y2"]].copy()
    out["conf"] = 1.0
    out["cls"] = df["type"].str.lower().to_numpy()
    out["track_id"] = df["track_id"].to_numpy()
    return out.reset_index(drop=True)[DET_COLUMNS]


class YoloDetector:
    """Thin wrapper around an Ultralytics YOLO model.

    Requires ``pip install ultralytics`` (the ``video`` extra). Weights such as
    ``yolov8n.pt`` are downloaded by Ultralytics on first use. Runs on GPU when one
    is available (e.g. Colab T4), otherwise on CPU.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        classes: tuple[str, ...] | list[str] = ("car", "truck", "bus"),
        imgsz: int = 1280,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "YOLO detection needs the 'video' extra: pip install -e '.[video]'"
            ) from exc
        self.model = YOLO(weights)
        names = self.model.names
        self.class_ids = [i for i, n in names.items() if n in set(classes)]
        if not self.class_ids:
            raise ValueError(f"None of {classes} are classes of {weights}")
        self.names = names
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, image_paths: dict[int, Path], batch: int = 16) -> pd.DataFrame:
        """Run detection on ``{frame: image_path}``; returns a detection table."""
        frames = sorted(image_paths)
        rows = []
        for i in range(0, len(frames), batch):
            chunk = frames[i : i + batch]
            results = self.model.predict(
                [str(image_paths[f]) for f in chunk],
                conf=self.conf, classes=self.class_ids, imgsz=self.imgsz,
                device=self.device, verbose=False,
            )
            for frame, res in zip(chunk, results):
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                    rows.append((frame, x1, y1, x2, y2, float(c), self.names[k], -1))
        if not rows:
            return empty_detections()
        df = pd.DataFrame(rows, columns=DET_COLUMNS)
        df[["x1", "y1", "x2", "y2"]] = df[["x1", "y1", "x2", "y2"]].astype(np.float64)
        return df
