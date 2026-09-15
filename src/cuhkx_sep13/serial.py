"""Failure-isolated serial runner for the Sep13 robustness experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from cuhkx_sep12.serial import ROOT, Runner, exclusive_lock, now, watch

NAMES = (
    "m01_robust_lightgbm",
    "m02_attention_concat_vote",
    "m03_attention_concat_ir_depth",
    "m04_attention_probability_sum",
    "m05_dual_resnet18",
    "m06_independent_concat",
    "m07_temporal_transformer",
    "m08_se_attention",
)


def run(directory, timeout_hours=12, cache_root=None, sensor_cache=None, detector2=None, detector4=None, detector_full=None):
    runner = Runner(directory, timeout_hours)
    cache_root = Path(cache_root or "artifacts/sep12/serial_depth_align/cache")
    sensor_cache = Path(sensor_cache or "artifacts/sep12/serial_depth_align/sensors")
    detector2 = Path(detector2 or "artifacts/yolo/detectors/fold_2/yolov8n_fold_2.pt")
    detector4 = Path(detector4 or "artifacts/sep12/serial_depth_align/jobs/detector_4/attempt_1/yolov8n_4.pt")
    detector_full = Path(detector_full or "artifacts/sep12/serial_depth_align/jobs/detector_full/attempt_1/yolov8n_full.pt")
    prerequisites = [cache_root / tag for tag in ("fold_2", "fold_4", "full")] + [sensor_cache, detector2, detector4, detector_full]
    ready = all(path.exists() for path in prerequisites)
    if not ready:
        missing = [str(path) for path in prerequisites if not path.exists()]
        print("[BLOCKED] Sep13 needs the completed Sep12 depth-align cache/detectors: " + "; ".join(missing), flush=True)

    result = runner.job(
        NAMES[0], "cuhkx_sep13.lightgbm",
        lambda out: ["--output", out, "--cache-root", cache_root, "--sensor-cache", sensor_cache,
                     "--detector2", detector2, "--detector4", detector4, "--detector-full", detector_full],
        required=("comparison.json", "submission.csv"), dependencies=ready,
    )
    runner.state["methods"][NAMES[0]] = {"status": "success" if result else "failed_or_blocked", "submission": str(result / "submission.csv") if result else None}
    if result:
        destination = runner.directory / "submissions" / f"{NAMES[0]}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result / "submission.csv", destination)
    runner.flush()

    for name in NAMES[1:]:
        results = {}
        for fold in (2, 4):
            result = runner.job(
                f"{name}_fold_{fold}", "cuhkx_sep12.train",
                lambda out, n=name, f=fold: ["--config", f"configs/sep13/{n}.json", "--output", out,
                                               "--folds", f, "--cache-root", cache_root, "--detector2", detector2,
                                               "--detector4", detector4, "--device", "cuda"],
                required=(f"fold_{fold}/summary.json", "submission.csv"),
                dependencies=ready,
            )
            if result:
                summary = json.loads((result / f"fold_{fold}" / "summary.json").read_text(encoding="utf-8"))
                results[str(fold)] = {"accuracy": summary["validation_accuracy"], "submission": str(result / "submission.csv")}
        record = {"status": "success" if len(results) == 2 else "partial" if results else "failed", "folds": results}
        if len(results) == 2:
            selected = max(results, key=lambda f: (results[f]["accuracy"], -int(f)))
            destination = runner.directory / "submissions" / f"{name}.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(results[selected]["submission"], destination)
            record.update(selected_fold=selected, submission=str(destination))
        runner.state["methods"][name] = record
        runner.flush()
    runner.state.update(status="finished", current_job=None, finished=now())
    runner.flush()
    print(json.dumps(runner.state["methods"], indent=2), flush=True)
    return 0 if all(item["status"] == "success" for item in runner.state["methods"].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="artifacts/sep13/serial")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=12)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--cache-root")
    parser.add_argument("--sensor-cache")
    parser.add_argument("--detector2")
    parser.add_argument("--detector4")
    parser.add_argument("--detector-full")
    args = parser.parse_args()
    if args.timeout_hours <= 0:
        parser.error("--timeout-hours must be positive")
    directory = (ROOT / args.run_dir).resolve()
    if args.watch:
        watch(directory, max(1, args.interval))
        return
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / "runner.lock"):
        raise SystemExit(run(directory, args.timeout_hours, args.cache_root, args.sensor_cache, args.detector2, args.detector4, args.detector_full))


if __name__ == "__main__":
    main()
