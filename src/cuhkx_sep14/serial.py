"""Failure-isolated serial execution for six Sep14 experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from cuhkx_sep12.serial import ROOT, Runner, exclusive_lock, now, watch

VISUAL = (
    "m05_depth_resnet18",
    "m05_dual_resnet18_ir_dropout",
    "m05_dual_resnet18_gate",
)


def run(directory, timeout_hours=12):
    runner = Runner(directory, timeout_hours)
    cache = Path("artifacts/sep12/serial_depth_align/cache")
    sensors = Path("artifacts/sep12/serial_depth_align/sensors")
    detector2 = Path("artifacts/yolo/detectors/fold_2/yolov8n_fold_2.pt")
    detector4 = Path("artifacts/sep12/serial_depth_align/jobs/detector_4/attempt_1/yolov8n_4.pt")
    detector_full = Path("artifacts/sep12/serial_depth_align/jobs/detector_full/attempt_1/yolov8n_full.pt")
    visual_ready = all(path.exists() for path in [cache / "fold_2", cache / "fold_4", sensors, detector2, detector4, detector_full])
    results = {}
    for name in VISUAL:
        folds = {}
        for fold in (2, 4):
            result = runner.job(
                f"{name}_fold_{fold}", "cuhkx_sep12.train",
                lambda out, n=name, f=fold: ["--config", f"configs/sep14/{n}.json", "--output", out, "--folds", f, "--cache-root", cache, "--detector2", detector2, "--detector4", detector4, "--device", "cuda"],
                required=(f"fold_{fold}/summary.json", "submission.csv"), dependencies=visual_ready,
            )
            if result:
                summary = json.loads((result / f"fold_{fold}" / "summary.json").read_text())
                folds[str(fold)] = {"accuracy": summary["validation_accuracy"], "submission": str(result / "submission.csv")}
        record = {"status": "success" if len(folds) == 2 else "partial" if folds else "failed", "folds": folds}
        if len(folds) == 2:
            selected = max(folds, key=lambda f: (folds[f]["accuracy"], -int(f)))
            target = runner.directory / "submissions" / f"{name}.csv"; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folds[selected]["submission"], target)
            record.update(selected_fold=selected, submission=str(target))
        runner.state["methods"][name] = record; runner.flush(); results[name] = record
    result = runner.job(
        "m01_yolo_box_grid", "cuhkx_sep14.lightgbm",
        lambda out: ["--output", out, "--cache-root", cache, "--sensor-cache", sensors, "--detector2", detector2, "--detector4", detector4, "--detector-full", detector_full],
        required=("comparison.json", "submission.csv"), dependencies=visual_ready,
    )
    runner.state["methods"]["m01_yolo_box_grid"] = {"status": "success" if result else "failed_or_blocked", "submission": str(result / "submission.csv") if result else None}; runner.flush()
    if result:
        target = runner.directory / "submissions" / "m01_yolo_box_grid.csv"; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(result / "submission.csv", target)
    for name, config in (("r3_full", "r3_thermal_specialist_full.json"), ("r3_full_aug", "r3_thermal_specialist_full_aug.json")):
        result = runner.job(name, "cuhkx_public.full", lambda out, c=config: ["--config", f"configs/public/{c}", "--output", out, "--device", "cuda"], required=("model.pt", "summary.json", "submission.csv"))
        runner.state["methods"][name] = {"status": "success" if result else "failed", "submission": str(result / "submission.csv") if result else None}; runner.flush()
        if result:
            target = runner.directory / "submissions" / f"{name}.csv"; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(result / "submission.csv", target)
    runner.state.update(status="finished", current_job=None, finished=now()); runner.flush()
    print(json.dumps(runner.state["methods"], indent=2), flush=True)
    return 0 if all(item["status"] == "success" for item in runner.state["methods"].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="artifacts/sep14/serial")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=12)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    directory = (ROOT / args.run_dir).resolve()
    if args.watch:
        watch(directory, max(1, args.interval)); return
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / "runner.lock"):
        raise SystemExit(run(directory, args.timeout_hours))


if __name__ == "__main__":
    main()
