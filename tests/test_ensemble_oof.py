import numpy as np

from cuhkx_har.ensemble_oof import scan_weights


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
