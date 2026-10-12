from __future__ import annotations

import pytest

from drive_events.config import load_config
from drive_events.synthetic import write_dataset


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """Two synthetic drives in KITTI tracking layout (no images, fast)."""
    root = tmp_path_factory.mktemp("synthetic")
    truth = write_dataset(root, n_sequences=2, seed=7)
    return root, truth


@pytest.fixture(scope="session")
def synthetic_root_images(tmp_path_factory):
    """One short synthetic drive with rendered camera frames."""
    root = tmp_path_factory.mktemp("synthetic_img")
    truth = write_dataset(root, n_sequences=1, seed=11, images=True)
    return root, truth
