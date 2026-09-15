from __future__ import annotations

import numpy as np

from cuhkx_har.automl import _hyperparameters, _sequence_summary, _visual_summary


def test_sequence_summary_is_finite_and_tracks_temporal_change() -> None:
    values = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    result = _sequence_summary(values, "sensor")
    assert result["sensor_change_0"] == 4.0
    assert result["sensor_abs_delta_1"] == 2.0
    assert all(np.isfinite(list(result.values())))


def test_visual_summary_handles_absent_stream(tmp_path) -> None:
    result = _visual_summary(tmp_path / "missing", "thermal", frames=4)
    assert result["thermal_available"] == 0.0
    assert not any(result[key] for key in result if key != "thermal_available")


def test_gpu_assignment_keeps_tree_models_that_oom_on_cpu() -> None:
    parameters = _hyperparameters(use_gpu=True)
    assert parameters["GBM"]["ag_args_fit"]["num_gpus"] == 0
    assert parameters["CAT"]["ag_args_fit"]["num_gpus"] == 0
    assert parameters["XGB"]["ag_args_fit"]["num_gpus"] == 1
    assert parameters["NN_TORCH"]["ag_args_fit"]["num_gpus"] == 1
    assert "task_type" not in parameters["CAT"]
