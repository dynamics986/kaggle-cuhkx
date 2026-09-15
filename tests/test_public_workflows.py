from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from cuhkx_public import train, yolo
from cuhkx_public.data import ThermalDataset
from cuhkx_public.runner import combine

ROOT = Path(__file__).resolve().parents[1]


def fixtures(tmp_path):
    files = []
    for index in range(3):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (32, 32), (80 + index * 30, 20, 30)).save(path)
        files.append(path)
    table = pd.DataFrame(
        [
            {
                "clip_id": f"c{fold}_{label}",
                "label": label,
                "fold": fold,
                "user": f"u{fold}",
                "frames": files,
            }
            for fold in (0, 2)
            for label in (0, 1)
        ]
    )
    test = pd.DataFrame(
        {
            "clip_id": ["a", "b"],
            "label": [-1, -1],
            "frames": [files, files],
            "submission_path": ["test/a/", "test/b/"],
        }
    )
    official = tmp_path / "official.csv"
    pd.DataFrame({"path": test.submission_path[::-1]}).to_csv(official, index=False)
    return table, test, official


@pytest.mark.parametrize(
    "baseline, improved",
    [
        ("r1_thermal_baseline", "i1_temporal_baseline"),
        ("r3_thermal_specialist", "i3_se_specialist"),
    ],
)
def test_thermal_train_finetune_validation_submission(tmp_path, monkeypatch, baseline, improved):
    torch.set_num_threads(2)
    table, test, official = fixtures(tmp_path)
    monkeypatch.setattr(train, "index_thermal", lambda *a, **k: test if k.get("test") else table)
    parent = None
    for recipe in (baseline, improved):
        cfg = json.loads((ROOT / f"configs/public/{recipe}.json").read_text())
        cfg.update(epochs=1, frames=2, image_size=32, effective_batch=2, micro_batch=2, workers=0)
        config = tmp_path / f"{recipe}.json"
        config.write_text(json.dumps(cfg))
        args = SimpleNamespace(
            config=config,
            device="cpu",
            manifest="train.csv",
            test_manifest="test.csv",
            train_root="unused",
            test_root="unused",
            fold=2,
            split=None,
            smoke=False,
            init=parent,
            output=tmp_path / recipe,
            test_csv=official,
        )
        summary = train.fit(args)
        assert summary["train_rows"] == summary["validation_rows"] == 2
        assert (args.output / "submission.csv").is_file()
        assert pd.read_csv(args.output / "submission.csv").path.tolist() == ["test/b/", "test/a/"]
        history = json.loads((args.output / "history.json").read_text())
        assert history[0]["train"]["optimizer_updates"] == 1
        if parent:
            assert summary["initialization"]["path"] == str(parent.resolve())
        parent = args.output / "model.pt"


def test_thermal_bad_image_replacement(tmp_path):
    table, _, _ = fixtures(tmp_path)
    bad = tmp_path / "bad.png"
    bad.write_bytes(bytes(100))
    table.at[0, "frames"] = [bad, table.iloc[0].frames[0]]
    cfg = json.loads((ROOT / "configs/public/r1_thermal_baseline.json").read_text())
    cfg.update(frames=2, image_size=32)
    images, _, bad_count = ThermalDataset(table, cfg)[0]
    assert bad_count == 1
    torch.testing.assert_close(images[0], images[1])


def test_combine_averages_logits_and_preserves_paths(tmp_path):
    outputs = {}
    official = tmp_path / "official.csv"
    pd.DataFrame({"path": ["test/a/"]}).to_csv(official, index=False)
    expected = []
    for fold in (2, 4):
        out = tmp_path / f"fold_{fold}"
        out.mkdir()
        (out / "model.pt").write_bytes(b"model")
        (out / "summary.json").write_text(json.dumps({"validation_accuracy": 0.5}))
        p = np.full(40, 0.1 / 38)
        p[0], p[1] = (0.85, 0.05) if fold == 2 else (0.2, 0.7)
        pd.DataFrame([{"clip_id": "a", **{f"prob_{c}": p[c] for c in range(40)}}]).to_csv(
            out / "test_predictions.csv", index=False
        )
        outputs[fold] = out
        expected.append(np.log(p))
    result = combine("combined", outputs, tmp_path, official)
    assert (
        pd.read_csv(result["submission"]).prediction.iloc[0] == np.mean(expected, axis=0).argmax()
    )


@pytest.mark.parametrize("improve", [False, True])
def test_yolo_complete_pipeline_with_synthetic_assets(tmp_path, monkeypatch, improve):
    import sys

    torch.set_num_threads(2)
    assets = tmp_path / "assets"
    assets.mkdir()
    source = assets / "ig65m_models.py"
    source.write_text("""from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv3d(3,4,1))
        self.pool=nn.AdaptiveAvgPool3d(1)
        self.fc=nn.Linear(4,400)
    def forward(self,x):
        return self.fc(self.pool(self.stem(x)).flatten(1))
def r2plus1d_34_32_kinetics(**kwargs):
    return Net()
""")
    detector = assets / "yolov8n_fold_2.pt"
    detector.write_bytes(b"fake-detector")
    model = yolo.R2Plus1D34(source)
    state = model.state_dict()
    torch.save(
        {
            "schema_version": "kuno-yolo-r2p1d-packed-ensemble/v1",
            "bits": [5, 6],
            "folds": [0, 1],
            "weights": [0.5, 0.5],
            "models_packed": [state, state],
        },
        assets / "ensemble_packed.pt",
    )
    folder = tmp_path / "images"
    folder.mkdir()
    Image.new("RGB", (32, 32), (80, 30, 20)).save(folder / "frame_0.png")
    (folder / "frame_1.png").write_bytes(b"bad-png")
    records = [
        {
            "clip_id": f"f{f}_{c}",
            "label": c,
            "fold": f,
            "user": f"u{f}",
            "Depth_Color": "images",
            "IR": "images",
        }
        for f in (0, 2, 4)
        for c in (0, 1)
    ]
    training = pd.DataFrame(records)
    test = pd.DataFrame(
        [
            {
                "clip_id": "a",
                "label": -1,
                "Depth_Color": "images",
                "IR": "images",
                "submission_path": "test/a/",
            }
        ]
    )
    official = tmp_path / "official.csv"
    pd.DataFrame({"path": ["test/a/"]}).to_csv(official, index=False)
    real_read = pd.read_csv
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda path, *a, **k: (
            test.copy() if str(path) == "manifests/cv5/test.csv" else real_read(path, *a, **k)
        ),
    )
    monkeypatch.setattr(yolo, "read_frozen_manifest", lambda path: training.copy())

    class Detector:
        def __init__(self, path):
            pass

        def predict(self, probes, **kwargs):
            return [
                SimpleNamespace(
                    orig_shape=(32, 32),
                    boxes=SimpleNamespace(xyxy=torch.tensor([[4.0, 4.0, 28.0, 28.0]])),
                )
                for _ in probes
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))
    args = SimpleNamespace(
        assets=assets,
        detector=detector,
        model_definition=source,
        output=tmp_path / "output",
        device="cpu",
        improve=improve,
        test_root=tmp_path,
        train_root=tmp_path,
        cache=tmp_path / "cache",
        micro_batch=1,
        epochs=1,
        test_csv=official,
    )
    yolo.run(args)
    assert (args.output / "submission.csv").is_file()
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["validation_status"].startswith("diagnostic_only")
    if improve:
        assert (args.output / "heads.pt").is_file()
        assert (args.output / "diagnostic_fold_2.csv").is_file()
        assert (args.output / "diagnostic_fold_4.csv").is_file()
