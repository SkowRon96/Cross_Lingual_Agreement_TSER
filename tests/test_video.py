import glob
import numpy as np
import pytest
from pathlib import Path
from app.data.video import sample_frame_indices, detect_and_crop_face, extract_face_frames


def test_sample_frame_indices_uniform():
    assert sample_frame_indices(100, 5) == [0, 25, 50, 74, 99]  # endpoints included, spread uniformly


def test_sample_frame_indices_fewer_than_n():
    assert sample_frame_indices(3, 8) == [0, 1, 2]


def test_sample_frame_indices_empty():
    assert sample_frame_indices(0, 8) == []


def test_detect_face_returns_none_on_blank_frame():
    blank = np.zeros((240, 320, 3), dtype=np.uint8)  # no face
    assert detect_and_crop_face(blank) is None


@pytest.mark.skipif(not glob.glob("datasets/RAVDESS/Actor_01/01-*.mp4"),
                    reason="RAVDESS dataset not present")
def test_extract_face_frames_on_real_ravdess_video():
    path = Path(sorted(glob.glob("datasets/RAVDESS/Actor_01/01-*.mp4"))[0])
    crops = extract_face_frames(path, num_frames=6)
    assert len(crops) >= 1                         # at least one face found
    for c in crops:
        assert c.ndim == 3 and c.shape[2] == 3     # RGB
        assert c.dtype == np.uint8
        assert c.shape[0] > 0 and c.shape[1] > 0
