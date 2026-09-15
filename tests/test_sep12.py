from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cuhkx_sep12.common import cache_path, digest, submit, write_json
from cuhkx_sep12.data import validate_cache
from cuhkx_sep12.lightgbm import fit
from cuhkx_sep12.model import VisualHAR
from cuhkx_sep12.prepare import aligned_indices, audit_detector
from cuhkx_sep12.train import fit_fold, predict_fold


def config(method=8):
    return dict(
        method=method,
        dim=16,
        dropout=0.0,
        frames=2,
        image_size=32,
        batch_size=2,
        workers=0,
        seed=0,
        epochs=1,
        patience=1,
        learning_rate=0.001,
        weight_decay=0.01,
    )


@pytest.mark.parametrize("method", range(2, 9))
def test_models_backward_missing_modalities_and_size(method):
    torch.set_num_threads(2)
    model = VisualHAR(config(method))
    images = torch.rand(2, 3, 2, 3, 32, 32)
    mask = torch.ones(2, 3, 2, dtype=torch.bool)
    mask[0, 2] = False
    mask[1] = False
    output = model(images, mask)
    p = output["probabilities"]
    assert p.shape == (2, 40)
    assert torch.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(2))
    loss = model.loss(output, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    size = sum(x.numel() * x.element_size() for x in model.state_dict().values())
    assert size + 6_237_994 < 100_000_000


def test_probability_sum_ignores_missing_branch():
    model = VisualHAR(config(4)).eval()
    images = torch.rand(2, 3, 2, 3, 32, 32)
    mask = torch.ones(2, 3, 2, dtype=torch.bool)
    mask[:, 2] = False
    with torch.no_grad():
        output = model(images, mask)
    expected = output["branches"][:, :2].softmax(-1).mean(1)
    torch.testing.assert_close(output["probabilities"], expected)


def test_hard_vote_beats_single_confident_branch():
    model = VisualHAR(config(2)).eval()
    with torch.no_grad():
        for head in model.branches:
            head.weight.zero_()
            head.bias.zero_()
            head.bias[2] = 1.0
        model.branches[2].bias.zero_()
        model.branches[2].bias[3] = 100.0
        model.head[-1].weight.zero_()
        model.head[-1].bias.zero_()
        model.head[-1].bias[3] = 100.0
        # Two tied hard votes, probability tie-break favors class 3.
        output = model(torch.zeros(2, 3, 2, 3, 32, 32), torch.ones(2, 3, 2).bool())
        assert output["probabilities"].argmax(1).tolist() == [3, 3]
        # Remove the dissenting branch: two class-2 votes beat one confident class-3 vote.
        mask = torch.ones(2, 3, 2).bool()
        mask[:, 2] = False
        output = model(torch.zeros(2, 3, 2, 3, 32, 32), mask)
        assert output["probabilities"].argmax(1).tolist() == [2, 2]


def test_alignment_uses_clock_overlap_and_shared_thermal_time():
    ir = [Path(f"IR_2025-01-01_12-00-{s:02d}.000_000.png") for s in (0, 1, 2, 3, 4)]
    depth = [Path(f"Depth_2025-01-01_12-00-{s:02d}.000_000_Color.png") for s in (1, 2, 3)]
    thermal = [Path(f"frame_{s:06d}.jpg") for s in range(9)]
    indices, mode = aligned_indices([ir, depth, thermal], 3)
    assert indices[0].tolist() == [1, 2, 3]
    assert indices[1].tolist() == [0, 1, 2]
    assert indices[2].tolist() == [0, 4, 8]
    assert mode == "ir_depth_timestamp_thermal_relative"
    assert aligned_indices([[], [], thermal], 3)[0][2].tolist() == [0, 4, 8]


def test_detector_rejects_cross_fold_leakage(tmp_path):
    weights = tmp_path / "detector.pt"
    weights.write_bytes(b"detector")
    write_json(
        tmp_path / "detector_summary.json",
        {"training_users": ["u2"], "weight_sha256": digest(weights)},
    )
    manifest = pd.DataFrame({"user": ["u2", "u4"], "fold": [2, 4]})
    audit_detector(weights, manifest, "4")
    with pytest.raises(ValueError, match="leaks fold"):
        audit_detector(weights, manifest, "2")


def test_submission_identity_order(tmp_path):
    rows = pd.DataFrame({"clip_id": ["b", "a"], "submission_path": ["test/b/", "test/a/"]})
    official = tmp_path / "official.csv"
    pd.DataFrame({"path": ["test/a/", "test/b/"]}).to_csv(official, index=False)
    p = np.zeros((2, 40))
    p[0, 3], p[1, 7] = 1, 1
    submit(rows, p, official, tmp_path / "submission.csv")
    assert pd.read_csv(tmp_path / "submission.csv").prediction.tolist() == [7, 3]
    with pytest.raises(ValueError, match="Invalid class probabilities"):
        submit(rows, p * np.nan, official, tmp_path / "bad.csv")


def test_lightgbm_40_class_serialization(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(160, 4)), columns=list("abcd"))
    model = fit(x, np.tile(np.arange(40), 4), rounds=2)
    model.save_model(str(tmp_path / "model.txt"))
    restored = lgb.Booster(model_file=str(tmp_path / "model.txt"))
    p = restored.predict(x)
    assert p.shape == (160, 40)
    np.testing.assert_allclose(p.sum(1), 1)
    np.testing.assert_allclose(p, model.predict(x))


@pytest.mark.parametrize("method", range(2, 9))
def test_fold_training_validation_reload_and_submission(tmp_path, method):
    torch.set_num_threads(2)
    cfg = config(method)
    rows = pd.DataFrame(
        {
            "clip_id": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "user": ["u0", "u0", "u2", "u2"],
            "fold": [0, 0, 2, 2],
        }
    )
    test = pd.DataFrame(
        {"clip_id": ["e", "f"], "label": [-1, -1], "submission_path": ["test/e/", "test/f/"]}
    )
    manifest, test_manifest = tmp_path / "train.csv", tmp_path / "test.csv"
    rows.to_csv(manifest, index=False)
    test.to_csv(test_manifest, index=False)
    weights = tmp_path / "detector.pt"
    weights.write_bytes(b"synthetic-detector")
    write_json(tmp_path / "detector_summary.json", {"training_users": ["u0"]})
    cache = tmp_path / "cache/fold_2"
    for split, records, source in [("train", rows, manifest), ("test", test, test_manifest)]:
        write_json(
            cache / f"{split}_metadata.json",
            {
                "split": split,
                "fold": "2",
                "manifest_sha256": digest(source),
                "detector_sha256": digest(weights),
                "frames": 2,
                "image_size": 32,
            },
        )
        for clip_id in records.clip_id:
            path = cache_path(cache, split, clip_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                images=np.full((3, 2, 32, 32, 3), 120, np.uint8),
                mask=np.ones((3, 2), bool),
                clip_id=clip_id,
            )
    official = tmp_path / "official.csv"
    pd.DataFrame({"path": test.submission_path[::-1]}).to_csv(official, index=False)
    args = SimpleNamespace(
        cache_root=tmp_path / "cache",
        detector2=weights,
        output=tmp_path / "run",
        manifest=manifest,
        test_manifest=test_manifest,
        test_csv=official,
        reuse_completed=False,
    )
    summary = fit_fold(args, cfg, rows, 2, torch.device("cpu"))
    assert summary["train_rows"] == summary["validation_rows"] == 2
    probabilities = predict_fold(args, cfg, test, 2, torch.device("cpu"))
    assert probabilities.shape == (2, 40)
    assert (tmp_path / "run/fold_2/submission.csv").exists()
    validation = pd.read_csv(tmp_path / "run/fold_2/validation_predictions.csv")
    assert validation.clip_id.tolist() == ["c", "d"]
    args.reuse_completed = True
    assert (
        fit_fold(args, cfg, rows, 2, torch.device("cpu"))["checkpoint_sha256"]
        == summary["checkpoint_sha256"]
    )
    with pytest.raises(ValueError, match="Cache|cache"):
        validate_cache(cache, "test", test_manifest, 4, weights, cfg)


@pytest.mark.parametrize("method", range(2, 9))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_training_step(method):
    from cuhkx_sep12.train import epoch

    model = VisualHAR(config(method)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda")
    batches = [(torch.rand(2, 3, 2, 3, 32, 32), torch.ones(2, 3, 2).bool(), torch.tensor([0, 1]))]
    batches[0][1][0, 2] = False
    batches[0][1][1] = False
    metrics, probabilities = epoch(model, batches, torch.device("cuda"), optimizer, scaler)
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-3)


def test_lightgbm_complete_cv_full_and_submission(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    import sys

    from cuhkx_har.features import cache_key
    from cuhkx_sep12 import lightgbm as workflow

    rows = pd.DataFrame(
        [
            {"clip_id": f"f{fold}_c{label}", "label": label, "user": f"u{fold}", "fold": fold}
            for fold in (0, 2, 4)
            for label in range(40)
        ]
    )
    test = pd.DataFrame(
        {
            "clip_id": ["test_a", "test_b"],
            "label": [-1, -1],
            "submission_path": ["official/a/", "official/b/"],
        }
    )
    manifest, test_manifest, official = (
        tmp_path / f"{s}.csv" for s in ("train", "test", "official")
    )
    rows.to_csv(manifest, index=False)
    test.to_csv(test_manifest, index=False)
    pd.DataFrame({"path": test.submission_path[::-1]}).to_csv(official, index=False)
    sensor_cache = tmp_path / "sensors"
    sensor_cache.mkdir()
    for split, records in [("train", rows), ("test", test)]:
        for row in records.itertuples():
            sequence = np.full((2, 2), max(0, row.label), np.float32)
            np.savez_compressed(
                sensor_cache / cache_key(split, row.clip_id),
                skeleton=sequence,
                imu=sequence,
                radar=sequence,
                sensor_mask=np.ones(3, bool),
            )
    detectors = {}
    for fold in ("2", "4", "full"):
        detector_dir = tmp_path / f"detector_{fold}"
        detector_dir.mkdir()
        weights = detector_dir / "weights.pt"
        weights.write_bytes(f"detector_{fold}".encode())
        detectors[fold] = weights
        users = sorted(
            set(rows.user if fold == "full" else rows.loc[rows.fold != int(fold), "user"])
        )
        write_json(detector_dir / "detector_summary.json", {"training_users": users})
        cache = tmp_path / "cache" / ("full" if fold == "full" else f"fold_{fold}")
        for split, records, source in [("train", rows, manifest), ("test", test, test_manifest)]:
            meta = dict(
                split=split,
                fold=fold,
                manifest_sha256=digest(source),
                detector_sha256=digest(weights),
                frames=2,
                image_size=4,
                confidence=0.25,
                padding=0.1,
                missing_policy="zero",
                alignment="relative",
            )
            write_json(cache / f"{split}_metadata.json", meta)
            for row in records.itertuples():
                path = cache_path(cache, split, row.clip_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path,
                    images=np.full((3, 2, 4, 4, 3), 100, np.uint8),
                    mask=np.ones((3, 2), bool),
                    clip_id=row.clip_id,
                )
    monkeypatch.setattr(workflow, "read_frozen_manifest", lambda path: rows)
    # Reduce boosting only in this integration test; production remains fixed at 1000.
    real_fit = workflow.fit
    monkeypatch.setattr(
        workflow, "fit", lambda features, labels: real_fit(features, labels, rounds=2)
    )
    output = tmp_path / "run"
    argv = [
        "lightgbm",
        "--manifest",
        str(manifest),
        "--test-manifest",
        str(test_manifest),
        "--test-csv",
        str(official),
        "--sensor-cache",
        str(sensor_cache),
        "--cache-root",
        str(tmp_path / "cache"),
        "--detector2",
        str(detectors["2"]),
        "--detector4",
        str(detectors["4"]),
        "--detector-full",
        str(detectors["full"]),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    workflow.main()
    summary = json.loads((output / "full/summary.json").read_text())
    assert summary["training_rows"] == 120
    assert summary["training_users"] == ["u0", "u2", "u4"]
    for fold in (2, 4):
        predictions = pd.read_csv(output / f"fold_{fold}/validation_predictions.csv")
        assert set(predictions.clip_id) == set(rows.loc[rows.fold == fold, "clip_id"])
    assert (
        pd.read_csv(output / "submission.csv").path.tolist() == test.submission_path[::-1].tolist()
    )
    monkeypatch.setattr(sys, "argv", argv + ["--stage", "predict"])
    workflow.main()


def test_detector_training_filters_held_out_annotations(tmp_path, monkeypatch):
    import sys

    from cuhkx_sep12 import prepare as preprocessing

    manifest = pd.DataFrame({"clip_id": ["a", "b"], "user": ["u2", "u4"], "fold": [2, 4]})
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    records = []
    for row in manifest.itertuples():
        image = annotations / f"{row.clip_id}.png"
        label = annotations / f"{row.clip_id}.txt"
        image.write_bytes(row.clip_id.encode())
        label.write_text("0 .5 .5 .5 .5\n")
        records.append(
            {
                "sample_id": row.clip_id,
                "clip_id": row.clip_id,
                "user": row.user,
                "image_path": image.name,
                "label_path": label.name,
                "image_sha256": digest(image),
            }
        )
    pd.DataFrame(records).to_csv(annotations / "annotation_manifest.csv", index=False)
    base = tmp_path / "base.pt"
    base.write_bytes(b"base")
    output = tmp_path / "detector"

    class Detector:
        def __init__(self, weights):
            assert weights == str(base)

        def train(self, **kwargs):
            assert kwargs["val"] is False and kwargs["patience"] == 0
            assert sorted(p.name for p in (output / "dataset/images/train").iterdir()) == ["a.png"]
            run = output / "runs/train"
            (run / "weights").mkdir(parents=True)
            (run / "weights/last.pt").write_bytes(b"last")
            self.trainer = SimpleNamespace(save_dir=run)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))
    monkeypatch.setattr(preprocessing, "read_frozen_manifest", lambda path: manifest)
    args = SimpleNamespace(
        manifest="unused.csv",
        annotations=annotations,
        fold="4",
        output=output,
        base_weights=str(base),
        epochs=1,
        batch_size=1,
        device="cpu",
    )
    preprocessing.detector(args)
    summary = json.loads((output / "detector_summary.json").read_text())
    assert summary["training_users"] == ["u2"]
    assert summary["held_out_users"] == ["u4"]
    audit_detector(output / "yolov8n_4.pt", manifest, "4")


def test_crop_misses_use_same_stream_box_and_never_raw_frame(tmp_path, monkeypatch):
    import sys

    from PIL import Image

    from cuhkx_sep12 import prepare as preprocessing

    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    write_json(tmp_path / "detector_summary.json", {"training_users": ["u0"]})
    rows = pd.DataFrame(
        [
            {
                "clip_id": "a",
                "user": "u0",
                "fold": 0,
                "IR": "IR",
                "Depth_Color": "Depth_Color",
                "Thermal": "Thermal",
            }
        ]
    )
    manifest = tmp_path / "manifest.csv"
    rows.to_csv(manifest, index=False)
    for modality in ("IR", "Depth_Color", "Thermal"):
        folder = tmp_path / modality
        folder.mkdir()
        for index in range(2):
            Image.new("RGB", (32, 32), (255, 100, 20)).save(folder / f"frame_{index}.png")

    class Boxes:
        def __init__(self, present):
            self.conf = torch.tensor([0.8]) if present else torch.empty(0)
            self.xyxyn = torch.tensor([[0.2, 0.2, 0.8, 0.8]])

        def __len__(self):
            return len(self.conf)

    class Detector:
        def __init__(self, weights):
            pass

        def predict(self, source, **kwargs):
            return [
                SimpleNamespace(boxes=Boxes("Thermal" not in str(p) and i == 0))
                for i, p in enumerate(source)
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))
    monkeypatch.setattr(preprocessing, "read_frozen_manifest", lambda path: rows)
    output = tmp_path / "cache"
    args = SimpleNamespace(
        manifest=manifest,
        weights=weights,
        fold="2",
        split="train",
        output=output,
        frames=2,
        image_size=16,
        confidence=0.25,
        data_root=tmp_path,
        batch_size=2,
        device="cpu",
    )
    preprocessing.prepare(args)
    with np.load(cache_path(output, "train", "a")) as saved:
        assert saved["mask"].tolist() == [[True, True], [True, True], [False, False]]
        assert saved["detected"].tolist() == [[True, False], [True, False], [False, False]]
        assert saved["images"][2].sum() == 0
        np.testing.assert_array_equal(saved["boxes"][0, 0], saved["boxes"][0, 1])
    report = json.loads((output / "train_coverage.json").read_text())
    assert report["zero_crop_clips"]["Thermal"] == 1
    preprocessing.prepare(args)  # Resume must preserve coverage information.
    assert json.loads((output / "train_coverage.json").read_text()) == report
