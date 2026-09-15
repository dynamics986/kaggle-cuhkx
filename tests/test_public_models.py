from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cuhkx_public.data import sample_indices
from cuhkx_public.models import FrameNet, SEThermalGRUNet, TemporalFrameNet, ThermalGRUNet
from cuhkx_public.yolo import dequantize_state, require_assets, unpack_signed, window_from_boxes

ROOT = Path(__file__).resolve().parents[1]


def source_classes(filename, cell):
    notebook = json.loads((ROOT / "references/kaggle" / filename).read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][cell]["source"]))
    selected = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]
    )
    scope = {"torch": torch, "nn": nn, "F": F, "NUM_CLASSES": 40}
    exec(compile(selected, filename, "exec"), scope)
    return scope


@pytest.mark.parametrize("specialist", [False, True])
def test_reproduction_exact_source_architecture(specialist):
    torch.set_num_threads(2)
    if specialist:
        original = source_classes("cuhk-x-thermal-specialist-v2.ipynb", 5)["ThermalGRUNet"]().eval()
        port = ThermalGRUNet().eval()
    else:
        original = source_classes("cuhk-x-14th-place-0-8-thermal-baseline(1).ipynb", 5)[
            "FrameNet"
        ]().eval()
        port = FrameNet().eval()
    port.load_state_dict(original.state_dict(), strict=True)
    x = torch.rand(2, 2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(port(x), original(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    "base, improved", [(FrameNet, TemporalFrameNet), (ThermalGRUNet, SEThermalGRUNet)]
)
def test_finetuning_identity_initialization(base, improved):
    original, candidate = base().eval(), improved().eval()
    candidate.load_state_dict(original.state_dict(), strict=False)
    x = torch.rand(2, 2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(candidate(x), original(x), atol=1e-6, rtol=1e-5)
    candidate.train()
    nn.functional.cross_entropy(candidate(x), torch.tensor([0, 1])).backward()
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)


def test_sampling_short_sequences_and_source_midpoints():
    assert sample_indices(1, 8, False) == [0] * 8
    assert sample_indices(8, 4, False) == [0, 2, 4, 6]
    assert sample_indices(0, 8, False) == []
    assert sample_indices(8, 4, False, jitter=-1) == [0, 1, 3, 5]


@pytest.mark.parametrize("bits", [5, 6])
def test_quantized_signed_unpack(bits):
    values = np.array([-(1 << (bits - 1)), -1, 0, 1, (1 << (bits - 1)) - 1], dtype=np.int8)
    packed = np.zeros((len(values) * bits + 7) // 8, dtype=np.uint8)
    for index, value in enumerate(values):
        code = int(value) % (1 << bits)
        for bit in range(bits):
            position = index * bits + bit
            packed[position // 8] |= ((code >> bit) & 1) << (position % 8)
    result = unpack_signed(torch.from_numpy(packed), (len(values),), bits)
    assert result.tolist() == values.tolist()
    state = dequantize_state(
        {
            "w": {
                "packed": torch.from_numpy(packed),
                "shape": [len(values)],
                "bits": bits,
                "scale": torch.tensor(0.25),
            }
        }
    )
    np.testing.assert_array_equal(state["w"].numpy(), values.astype(float) * 0.25)


def test_median_window_matches_notebook():
    notebook = json.loads(
        (ROOT / "references/kaggle/yolo-for-cuhk-x.ipynb").read_text(encoding="utf-8")
    )
    tree = ast.parse("".join(notebook["cells"][6]["source"]))
    selected = ast.Module(
        body=[
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "window_from_boxes"
        ],
        type_ignores=[],
    )
    scope = {"np": np, "CROP_MARGIN": 1.4, "MIN_SIDE_FRACTION": 0.35}
    exec(compile(selected, "notebook_window", "exec"), scope)
    boxes = [[0.1, 0.1, 0.3, 0.9], [0.2, 0.1, 0.4, 0.8], [0.7, 0.5, 0.8, 0.6]]
    assert window_from_boxes(boxes) == scope["window_from_boxes"](boxes)


def test_missing_assets_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="ensemble_packed.pt"):
        require_assets(tmp_path)
