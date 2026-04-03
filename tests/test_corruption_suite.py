import json

from sctq.cli.run_corruption_suite import normalize_corruption_config, try_remove_bad_cache, validate_corruption_outputs


def test_normalize_corruption_config_supports_nested_and_flat_shapes():
    nested = {
        "corruptions": {
            "image_level": [{"name": "gaussian_noise"}],
            "detection_level": [{"name": "random_drop"}],
            "runner": {"runs_per_severity": 2},
        }
    }
    flat = {
        "image_level": [{"name": "gaussian_noise"}],
        "detection_level": [{"name": "random_drop"}],
        "runner": {"runs_per_severity": 2},
    }
    assert normalize_corruption_config(nested) == normalize_corruption_config(flat)


def test_try_remove_bad_cache_deletes_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not-valid-json", encoding="utf-8")
    assert try_remove_bad_cache(path) is True
    assert not path.exists()


def test_validate_corruption_outputs_fails_when_no_noisy_runs():
    cfg = {"image_level": [{"name": "gaussian_noise", "enabled": True, "severities": [1, 3]}], "detection_level": [], "runner": {}}
    try:
        validate_corruption_outputs([], cfg)
    except RuntimeError as exc:
        assert "No corrupted runs" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for clean-only output")


def test_validate_corruption_outputs_accepts_valid_noisy_rows():
    cfg = {"image_level": [{"name": "gaussian_noise", "enabled": True, "severities": [1, 3]}], "detection_level": [], "runner": {}}
    rows = [
        {"tracker_name": "sort", "corruption": "gaussian_noise", "corruption_type": "image", "severity": 1, "run_index": 0},
        {"tracker_name": "sort", "corruption": "gaussian_noise", "corruption_type": "image", "severity": 3, "run_index": 0},
    ]
    validate_corruption_outputs(rows, cfg)
