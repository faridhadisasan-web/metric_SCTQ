from sctq.utils.video_utils import sample_frame_indices


def test_uniform_sampling_returns_requested_number():
    idx = sample_frame_indices(200, max_frames=50, sample_mode="uniform")
    assert len(idx) == 50
    assert idx[0] == 0
    assert idx[-1] == 199


def test_head_sampling_returns_first_frames():
    idx = sample_frame_indices(20, max_frames=5, sample_mode="head")
    assert idx == [0, 1, 2, 3, 4]
