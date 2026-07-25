import numpy as np

from cuhkx_har.ensemble_oof import crossfit_weight_selection, scan_weights


def test_scan_weights_finds_complementary_blend() -> None:
    labels = np.array([0, 1, 0, 1])
    baseline = np.zeros((4, 40))
    candidate = np.zeros((4, 40))
    baseline[:, :2] = [[0.9, 0.1], [0.6, 0.4], [0.6, 0.4], [0.1, 0.9]]
    candidate[:, :2] = [[0.6, 0.4], [0.1, 0.9], [0.9, 0.1], [0.4, 0.6]]

    results = scan_weights(labels, baseline, candidate, np.array([0, 0, 1, 1]), steps=4)

    assert results[0]["oof_accuracy"] == 1.0
    assert results[0]["candidate_weight"] == 0.25
    assert results[0]["fold_accuracy"] == {"0": 1.0, "1": 1.0}


def test_crossfit_weight_selection_never_uses_held_out_fold_labels() -> None:
    labels = np.array([0, 1, 0, 1])
    baseline = np.zeros((4, 40))
    candidate = np.zeros((4, 40))
    baseline[:, :2] = [[0.9, 0.1], [0.6, 0.4], [0.6, 0.4], [0.1, 0.9]]
    candidate[:, :2] = [[0.6, 0.4], [0.1, 0.9], [0.9, 0.1], [0.4, 0.6]]

    result = crossfit_weight_selection(labels, baseline, candidate, np.array([0, 0, 1, 1]), 4)

    assert result["examples"] == 4
    assert {row["held_out_fold"] for row in result["folds"]} == {0, 1}
