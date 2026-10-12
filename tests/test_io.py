import numpy as np
import pandas as pd
import pytest

from drive_events.io.kitti import (
    OXTS_FIELDS,
    TrackingSequence,
    camera_matrix,
    list_sequences,
    load_calib,
    load_oxts,
    load_tracking_labels,
)


def test_tracking_oxts_uses_frame_rate(synthetic_root):
    root, _ = synthetic_root
    df = load_oxts(root / "training" / "oxts" / "0000.txt", fps=10.0)
    assert list(df.columns[:2]) == ["frame", "t"]
    assert list(df.columns[2:]) == OXTS_FIELDS
    assert np.allclose(np.diff(df["t"]), 0.1)


def test_raw_oxts_directory_uses_timestamps(tmp_path):
    data = tmp_path / "oxts" / "data"
    data.mkdir(parents=True)
    for i in range(3):
        np.savetxt(data / f"{i:010d}.txt", np.arange(30, dtype=float)[None, :] + i)
    (tmp_path / "oxts" / "timestamps.txt").write_text(
        "2011-09-26 13:02:25.000000000\n2011-09-26 13:02:25.103000000\n"
        "2011-09-26 13:02:25.300000000\n"
    )
    df = load_oxts(tmp_path / "oxts")
    assert len(df) == 3
    assert np.allclose(df["t"], [0.0, 0.103, 0.3])
    assert df.loc[2, "vf"] == 8 + 2


def test_oxts_wrong_width_raises(tmp_path):
    f = tmp_path / "bad.txt"
    np.savetxt(f, np.zeros((2, 5)))
    with pytest.raises(ValueError):
        load_oxts(f)


def test_labels_drop_dontcare_and_parse_types(synthetic_root):
    root, _ = synthetic_root
    lab = load_tracking_labels(root / "training" / "label_02" / "0000.txt")
    assert "DontCare" not in set(lab["type"])
    assert lab["frame"].dtype.kind == "i" and lab["track_id"].min() >= 0


def test_missing_labels_give_empty_frame(tmp_path):
    assert load_tracking_labels(tmp_path / "nope.txt").empty


def test_calib_parses_both_key_styles(tmp_path):
    f = tmp_path / "calib.txt"
    f.write_text("P2: 1 0 3 0 0 2 4 0 0 0 1 0\nR_rect 1 0 0 0 1 0 0 0 1\n"
                 "calib_time: 09-Jan-2012 14:00:00\n")
    calib = load_calib(f)
    assert calib["R_rect"].size == 9
    P = camera_matrix(calib, "image_02")
    assert P.shape == (3, 4) and P[0, 2] == 3
    raw = {"P_rect_02": np.arange(12.0)}
    assert camera_matrix(raw, "image_02")[2, 3] == 11


def test_sequence_paths_and_listing(synthetic_root):
    root, _ = synthetic_root
    assert list_sequences(root) == ["0000", "0001"]
    seq = TrackingSequence(root, "0001")
    assert seq.has_labels()
    assert seq.image_paths() == {}
    assert seq.load_camera_matrix().shape == (3, 4)
    assert isinstance(seq.load_oxts(), pd.DataFrame)
