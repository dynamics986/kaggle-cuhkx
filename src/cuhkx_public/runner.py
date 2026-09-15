"""Two independently resumable serial queues for reproduction and fine-tuning."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from cuhkx_sep12.common import check_size, digest, submit, write_json
from cuhkx_sep12.serial import ROOT, Runner, exclusive_lock, watch

THERMAL = {
    "reproduce": ["r1_thermal_baseline", "r3_thermal_specialist"],
    "improve": ["i1_temporal_baseline", "i3_se_specialist"],
}


def parent_checkpoint(reproduction, name, fold):
    path = Path(reproduction) / "status.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    job = state["jobs"].get(f"{name}_fold_{fold}", {})
    output = Path(job.get("output", "__missing__"))
    checkpoint = output / "model.pt"
    return checkpoint if checkpoint.is_file() and (output / "summary.json").is_file() else None


def combine(recipe, outputs, directory, test_csv):
    frames, paths = [], []
    for output in outputs.values():
        table = pd.read_csv(output / "test_predictions.csv")
        if frames and table.clip_id.tolist() != frames[0].clip_id.tolist():
            raise ValueError("Fold test predictions are misaligned")
        frames.append(table)
        paths.append(output / "model.pt")
    size = check_size(paths)
    # Mean logits, as in the source specialist (mean log-probabilities is equivalent).
    log_probabilities = np.mean(
        [
            np.log(np.maximum(f[[f"prob_{c}" for c in range(40)]].to_numpy(), 1e-300))
            for f in frames
        ],
        axis=0,
    )
    probabilities = np.exp(log_probabilities - log_probabilities.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    official = pd.read_csv(test_csv)
    rows = pd.DataFrame(
        {
            "clip_id": frames[0].clip_id,
            "submission_path": ["small_model_track_test/" + c + "/" for c in frames[0].clip_id],
        }
    )
    # Preserve exact official paths, independent of test manifest row order.
    path_by_clip = {
        str(p).rstrip("/\\").replace("\\", "/").split("/")[-1]: p for p in official.path
    }
    rows["submission_path"] = rows.clip_id.map(path_by_clip)
    destination = directory / "submissions" / f"{recipe}.csv"
    submit(rows, probabilities, test_csv, destination)
    return {
        "submission": str(destination),
        "inference_bytes": size,
        "folds": {
            str(f): json.loads((p / "summary.json").read_text(encoding="utf-8"))[
                "validation_accuracy"
            ]
            for f, p in outputs.items()
        },
    }


def run(args, mode):
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = {
        "mode": mode,
        "folds": args.folds,
        "split": args.split,
        "micro_batch": args.micro_batch,
        "smoke": args.smoke,
        "device": args.device,
        "configs": {p.name: digest(p) for p in (ROOT / "configs/public").glob("*.json")},
        "assets": str(Path(args.assets).resolve()),
        "detector": str(Path(args.detector).resolve()),
        "model_definition": str(Path(args.model_definition).resolve()),
        "reproduction_dir": str(Path(args.reproduction_dir).resolve()),
    }
    spec_path = root / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise ValueError("Run settings changed. Use a new --run-dir rather than reuse old results.")
    write_json(spec_path, spec)
    runner = Runner(root, timeout_hours=args.timeout_hours)
    missing_assets = [
        name for name in ("ensemble_packed.pt",) if not (Path(args.assets) / name).is_file()
    ]
    if missing_assets:
        print(
            f"YOLO classifier assets missing in {args.assets}: {missing_assets}. "
            "Thermal tasks will still run; YOLO cannot reproduce without its original assets.",
            flush=True,
        )
    for recipe in THERMAL[mode]:
        outputs = {}
        for fold in args.folds:
            parent = None
            if mode == "improve":
                source = (
                    "r1_thermal_baseline" if recipe.startswith("i1") else "r3_thermal_specialist"
                )
                parent = parent_checkpoint(args.reproduction_dir, source, fold)
            arguments = [
                "--config",
                f"configs/public/{recipe}.json",
                "--fold",
                fold,
                "--micro-batch",
                args.micro_batch,
                "--split",
                args.split,
                "--device",
                args.device,
            ]
            if parent:
                arguments += ["--init", parent]
            if args.smoke:
                arguments += ["--smoke"]
            output = runner.job(
                f"{recipe}_fold_{fold}",
                "cuhkx_public.train",
                lambda out, extra=arguments: ["--output", out, *extra],
                required=("model.pt", "summary.json", "submission.csv"),
                dependencies=mode != "improve" or parent is not None,
            )
            if output:
                outputs[fold] = output
        record = {
            "status": "success"
            if len(outputs) == len(args.folds)
            else "partial"
            if outputs
            else "failed"
        }
        if len(outputs) == len(args.folds):
            try:
                if args.smoke:
                    record["outputs"] = {str(f): str(p) for f, p in outputs.items()}
                else:
                    record.update(
                        combine(
                            recipe, outputs, root, "../Small-Model-Track/Testing/test_file/test.csv"
                        )
                    )
            except Exception as error:
                record.update(status="failed", error=str(error))
        runner.state["methods"][recipe] = record
        runner.flush()
    # The inference-only notebook is a separate task and cannot block either Thermal recipe.
    recipe = "r2_yolo_v9" if mode == "reproduce" else "i2_yolo_head_finetune"
    output = runner.job(
        recipe,
        "cuhkx_public.yolo",
        lambda out: (
            [
                "--output",
                out,
                "--config",
                f"configs/public/{recipe}.json",
                "--assets",
                args.assets,
                "--detector",
                args.detector,
                "--model-definition",
                args.model_definition,
                "--micro-batch",
                "1",
                "--device",
                args.device,
            ]
            + (["--improve"] if mode == "improve" else [])
        ),
        required=("summary.json", "submission.csv"),
        dependencies=not args.smoke,
    )
    runner.state["methods"][recipe] = {"status": "success" if output else "blocked_or_failed"}
    if output:
        target = root / "submissions" / f"{recipe}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "submission.csv", target)
        runner.state["methods"][recipe]["submission"] = str(target)
    runner.state.update(status="finished", current_job=None)
    runner.flush()
    print(json.dumps(runner.state["methods"], indent=2), flush=True)
    return 0 if all(v["status"] == "success" for v in runner.state["methods"].values()) else 1


def main(mode):
    parser = argparse.ArgumentParser(description=f"Serial {mode} of the three supplied notebooks")
    parser.add_argument("--run-dir", default=f"artifacts/public/{mode}")
    parser.add_argument("--reproduction-dir", default="artifacts/public/reproduce")
    parser.add_argument("--assets", default="artifacts/public/assets/yolo_v9")
    parser.add_argument(
        "--detector",
        default="artifacts/yolo/detectors/fold_2/yolov8n_fold_2.pt",
        help="Local person detector used for the YOLO notebook pipeline",
    )
    parser.add_argument(
        "--model-definition",
        default="src/cuhkx_public/ig65m_models.py",
        help="Vendored architecture from moabitcoin/ig65m-pytorch",
    )
    parser.add_argument("--folds", type=int, choices=range(5), nargs="+", default=[2, 4])
    parser.add_argument("--split", choices=["notebook", "frozen"], default="notebook")
    parser.add_argument("--micro-batch", type=int, choices=[1, 2, 4, 8, 16, 32], default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--timeout-hours", type=float, default=12)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if len(args.folds) != len(set(args.folds)) or args.timeout_hours <= 0:
        parser.error("Invalid folds/timeout")
    # All documented commands work from the repository root; normalize relative paths explicitly.
    import os

    os.chdir(ROOT)
    if args.watch:
        watch(Path(args.run_dir), 5)
        return
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    with exclusive_lock(Path(args.run_dir) / "runner.lock"):
        raise SystemExit(run(args, mode))
