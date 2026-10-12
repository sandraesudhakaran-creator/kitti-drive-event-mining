"""Readers for the KITTI tracking benchmark and KITTI raw recordings.

Tracking benchmark layout (what this project uses by default)::

    <root>/training/oxts/0000.txt        one OXTS row per camera frame (10 Hz)
    <root>/training/label_02/0000.txt    2D/3D object labels with track ids
    <root>/training/calib/0000.txt       camera projection matrices
    <root>/training/image_02/0000/*.png  left colour camera

KITTI raw recordings store OXTS as ``oxts/data/0000000000.txt`` plus
``oxts/timestamps.txt``; :func:`load_oxts` accepts that directory as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Field order of an OXTS row, from the KITTI devkit (dataformat.txt).
OXTS_FIELDS = [
    "lat", "lon", "alt", "roll", "pitch", "yaw",
    "vn", "ve", "vf", "vl", "vu",
    "ax", "ay", "az", "af", "al", "au",
    "wx", "wy", "wz", "wf", "wl", "wu",
    "posacc", "velacc", "navstat", "numsats", "posmode", "velmode", "orimode",
]

LABEL_FIELDS = [
    "frame", "track_id", "type", "truncated", "occluded", "alpha",
    "x1", "y1", "x2", "y2", "h", "w", "l", "x", "y", "z", "ry",
]

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def load_oxts(path: str | Path, fps: float = 10.0) -> pd.DataFrame:
    """Load OXTS GPS/IMU data.

    ``path`` is either a tracking-benchmark file (one row per frame) or a raw-recording
    ``oxts`` directory. Returns one row per frame with ``frame``, ``t`` (seconds from
    the first sample) and the 30 OXTS fields. Tracking files carry no timestamps, so
    ``t = frame / fps`` there; raw recordings use their real timestamps.
    """
    path = Path(path)
    ts_file = None
    if path.is_dir():
        data_dir = path / "data" if (path / "data").is_dir() else path
        files = sorted(f for f in data_dir.glob("*.txt") if f.name != "timestamps.txt")
        if not files:
            raise FileNotFoundError(f"No OXTS .txt files in {data_dir}")
        arr = np.vstack([np.loadtxt(f, ndmin=1) for f in files])
        ts_file = path / "timestamps.txt"
    else:
        arr = np.loadtxt(path, ndmin=2)
    if arr.shape[1] != len(OXTS_FIELDS):
        raise ValueError(f"{path}: expected {len(OXTS_FIELDS)} OXTS columns, got {arr.shape[1]}")

    df = pd.DataFrame(arr, columns=OXTS_FIELDS)
    df.insert(0, "frame", np.arange(len(df), dtype=int))

    t = None
    if ts_file is not None and ts_file.exists():
        lines = [ln.strip() for ln in ts_file.read_text().splitlines() if ln.strip()]
        if len(lines) == len(df):
            ts = pd.to_datetime(pd.Series(lines))
            t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    if t is None:
        t = df["frame"].to_numpy() / float(fps)
    df.insert(1, "t", t)
    return df


def load_tracking_labels(path: str | Path) -> pd.DataFrame:
    """Load a KITTI tracking label file (``DontCare`` rows are dropped).

    Result files with an extra 18th ``score`` column are accepted too.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=LABEL_FIELDS)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] == len(LABEL_FIELDS):
        df.columns = LABEL_FIELDS
    elif df.shape[1] == len(LABEL_FIELDS) + 1:
        df.columns = LABEL_FIELDS + ["score"]
    else:
        raise ValueError(f"{path}: unexpected number of label columns ({df.shape[1]})")
    df = df[df["type"] != "DontCare"].reset_index(drop=True)
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df


def load_calib(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a KITTI calibration file into ``{key: values}``.

    Handles both ``P2: ...`` (tracking/object) and ``P_rect_02: ...`` (raw) styles,
    and keys written without a colon (``R_rect ...`` in tracking calib files).
    """
    calib: dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.strip().replace(":", " ", 1).split()
        if len(parts) < 2:
            continue
        try:
            calib[parts[0]] = np.array([float(v) for v in parts[1:]])
        except ValueError:
            continue  # e.g. calib_time lines in raw calibration files
    return calib


def camera_matrix(calib: dict[str, np.ndarray], camera: str = "image_02") -> np.ndarray:
    """Return the 3x4 rectified projection matrix of the given camera."""
    idx = camera[-1]
    for key in (f"P{idx}", f"P_rect_0{idx}"):
        if key in calib and calib[key].size == 12:
            return calib[key].reshape(3, 4)
    raise KeyError(f"No projection matrix for {camera} in calibration")


@dataclass
class TrackingSequence:
    """One sequence of the KITTI tracking benchmark."""

    root: Path
    seq: str
    split: str = "training"
    camera: str = "image_02"

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def base(self) -> Path:
        return self.root / self.split

    @property
    def oxts_path(self) -> Path:
        return self.base / "oxts" / f"{self.seq}.txt"

    @property
    def label_path(self) -> Path:
        return self.base / f"label_{self.camera[-2:]}" / f"{self.seq}.txt"

    @property
    def calib_path(self) -> Path:
        return self.base / "calib" / f"{self.seq}.txt"

    @property
    def image_dir(self) -> Path:
        return self.base / self.camera / self.seq

    def image_paths(self) -> dict[int, Path]:
        """Map frame index to image file (empty if images were not downloaded)."""
        if not self.image_dir.is_dir():
            return {}
        out = {}
        for p in self.image_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTS and p.stem.isdigit():
                out[int(p.stem)] = p
        return dict(sorted(out.items()))

    def has_labels(self) -> bool:
        return self.label_path.exists()

    def load_oxts(self, fps: float = 10.0) -> pd.DataFrame:
        return load_oxts(self.oxts_path, fps=fps)

    def load_labels(self) -> pd.DataFrame:
        return load_tracking_labels(self.label_path)

    def load_camera_matrix(self) -> np.ndarray | None:
        if not self.calib_path.exists():
            return None
        return camera_matrix(load_calib(self.calib_path), self.camera)


def list_sequences(root: str | Path, split: str = "training") -> list[str]:
    """List sequence ids that have OXTS data."""
    oxts_dir = Path(root) / split / "oxts"
    if not oxts_dir.is_dir():
        raise FileNotFoundError(f"Missing OXTS folder: {oxts_dir}")
    return sorted(p.stem for p in oxts_dir.glob("*.txt"))
