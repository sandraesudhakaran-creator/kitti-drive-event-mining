import numpy as np
import pandas as pd
import pytest

from drive_events.io.kitti import TrackingSequence
from drive_events.video import (
    CameraModel,
    box_iou,
    detections_from_labels,
    evaluate_ttc,
    select_lead,
    track_detections,
    ttc_from_scale,
    ttc_ground_truth,
)

CAM = CameraModel()


def _box_at(x, z, h=1.5, w=1.7):
    """Image box of a car at lateral x / depth z (flat road, KITTI camera)."""
    return (CAM.fx * (x - w / 2) / z + CAM.cx, CAM.fy * (CAM.height_m - h) / z + CAM.cy,
            CAM.fx * (x + w / 2) / z + CAM.cx, CAM.fy * CAM.height_m / z + CAM.cy)


def test_box_iou_basics():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(box_iou(a, b), [[1.0, 50 / 150, 0.0]])


def test_tracker_keeps_identity_and_survives_short_miss():
    rows = []
    for f in range(10):
        if f == 4:
            continue  # missed detection
        rows.append((f, 100 + f, 100, 150 + f, 140, 0.9, "car", -1))
        rows.append((f, 600, 100, 650, 140, 0.9, "car", -1))
    det = pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "cls", "track_id"])
    out = track_detections(det, 0.3, max_age=2)
    assert out.groupby("x1")["track_id"].nunique().max() == 1
    assert out["track_id"].nunique() == 2


def test_ground_position_recovers_depth_and_offset():
    x1, y1, x2, y2 = _box_at(1.0, 20.0)
    z, x = CAM.ground_position(pd.DataFrame({"x1": [x1], "y1": [y1], "x2": [x2], "y2": [y2]}))
    assert z[0] == pytest.approx(20.0) and x[0] == pytest.approx(1.0)


def test_lead_ignores_parked_car_that_looks_central():
    rows = []
    b_lead = _box_at(0.0, 40.0)
    b_kerb = _box_at(5.0, 30.0)  # near the image centre in pixels, but at the kerb
    for tid, b in ((0, b_lead), (1, b_kerb)):
        rows.append((0, *b, 1.0, "car", tid))
    det = pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "cls", "track_id"])
    lead = select_lead(det, CAM, lane_half_width_m=1.8)
    assert lead["track_id"].tolist() == [0]


def test_ttc_from_scale_matches_constant_closing():
    fps, v = 10.0, 5.0
    frames = np.arange(30)
    z = 60.0 - v * frames / fps        # TTC falls from 12 s to 9 s
    rows = [(f, *_box_at(0.0, zz), 1.0, "car", 0) for f, zz in zip(frames, z)]
    det = pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "cls", "track_id"])
    out = ttc_from_scale(det, fps, window=5, min_frames=3, ttc_max=10.0)
    true = z / v
    ok = np.isfinite(out["ttc"])
    rel = np.abs(out.loc[ok, "ttc"] - true[ok]) / true[ok]
    assert np.isnan(out.loc[0, "ttc"])                 # not enough history yet
    assert rel.max() < 0.12                              # window lag only
    assert np.isinf(out.loc[3, "ttc"])     # ~11.7 s > ttc_max -> reported as not closing
    assert ok.iloc[-1]                     # below ttc_max at the end


def test_ttc_ground_truth_and_receding_object():
    frames = np.arange(20)
    lab = pd.concat([
        pd.DataFrame({"frame": frames, "track_id": 0, "z": 20.0 - 0.5 * frames}),  # 5 m/s closing
        pd.DataFrame({"frame": frames, "track_id": 1, "z": 20.0 + 0.5 * frames}),  # receding
    ])
    gt = ttc_ground_truth(lab, fps=10.0, window=5, min_frames=3, ttc_max=10.0)
    closing = gt[(gt.track_id == 0) & (gt.frame == 10)].iloc[0]
    assert closing["closing_speed"] == pytest.approx(5.0)
    assert closing["ttc_gt"] == pytest.approx(15.0 / 5.0)
    assert np.isinf(gt[(gt.track_id == 1) & (gt.frame == 10)]["ttc_gt"]).all()


def test_evaluate_ttc_with_oracle_detector(synthetic_root, cfg):
    root, _ = synthetic_root
    seq = TrackingSequence(root, "0000")
    labels = seq.load_labels()
    vcfg = cfg["video"]
    det = ttc_from_scale(detections_from_labels(labels), 10.0, vcfg["ttc_window"],
                         vcfg["ttc_min_frames"], vcfg["ttc_max_s"])
    cam = CameraModel.from_projection(seq.load_camera_matrix())
    lead = select_lead(det, cam, vcfg["lane_half_width_m"], vcfg["min_box_height_px"])
    _, m = evaluate_ttc(lead, labels, vcfg, cam, 10.0)
    assert m["lead_recall"] == 1.0 and m["lead_precision"] == 1.0
    assert m["ttc_pairs"] > 10 and m["ttc_median_rel_err"] < 0.15
    assert m["critical_precision"] == 1.0 and m["critical_recall"] > 0.8


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Arr(xyxy), _Arr(conf), _Arr(cls)

    def __len__(self):
        return len(self.xyxy.a)


class _FakeYOLO:
    """Stands in for ultralytics.YOLO (no torch in CI): same result structure."""

    names = {0: "person", 2: "car", 7: "truck"}
    calls = []

    def __init__(self, weights):
        self.weights = weights

    def predict(self, paths, conf, classes, imgsz, device, verbose):
        _FakeYOLO.calls.append({"n": len(paths), "classes": classes, "imgsz": imgsz})
        out = []
        for i, _ in enumerate(paths):
            boxes = _Boxes([[10, 20, 50, 60]], [0.8], [2]) if i % 2 == 0 else _Boxes(
                np.zeros((0, 4)), [], [])
            out.append(type("R", (), {"boxes": boxes})())
        return out


def test_yolo_detector_wrapper_parses_results(monkeypatch, tmp_path):
    import sys
    import types

    from drive_events.video.detect import YoloDetector

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO))
    det = YoloDetector("yolov8n.pt", conf=0.3, classes=["car", "truck", "bus"], imgsz=1280)
    assert det.class_ids == [2, 7]
    paths = {f: tmp_path / f"{f:06d}.png" for f in range(5)}
    out = det.detect(paths, batch=2)
    assert out["frame"].tolist() == [0, 2, 4]
    assert out["cls"].unique().tolist() == ["car"]
    assert [c["n"] for c in _FakeYOLO.calls[-3:]] == [2, 2, 1]
    with pytest.raises(ValueError):
        YoloDetector("x.pt", classes=["giraffe"])
