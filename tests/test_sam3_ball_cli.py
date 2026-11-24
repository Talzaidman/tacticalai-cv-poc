from unittest.mock import patch

import numpy as np
import pytest

from tacticalai.sam3_ball_cli import read_frame_chunks


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


def test_frame_reader_bounds_memory_by_yielding_fixed_chunks():
    capture = FakeCapture([np.zeros((2, 3, 3), np.uint8) for _ in range(5)])
    with patch('tacticalai.sam3_ball_cli.cv2.VideoCapture', return_value=capture):
        chunks = list(read_frame_chunks('video.mp4', 2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert capture.released


def test_frame_reader_rejects_nonpositive_chunk_size():
    with pytest.raises(ValueError):
        list(read_frame_chunks('video.mp4', 0))
