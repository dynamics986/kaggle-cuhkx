"""Serial, failure-isolated preparation and execution of all eight Sep12 methods."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

NAMES = (
    "m01_lightgbm",
    "m02_attention_concat_vote",
    "m03_attention_concat_ir_depth",
    "m04_attention_probability_sum",
    "m05_dual_resnet18",
    "m06_independent_concat",
    "m07_temporal_transformer",
    "m08_se_attention",
)
ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now().isoformat(timespec="seconds")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


@contextlib.contextmanager
def exclusive_lock(path):
    handle = path.open("a+b")
    handle.seek(0)
    if not handle.read(1):
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("Another serial runner is using this run directory") from None
    try:
        yield
    finally:
        handle.close()


class Runner:
    def __init__(self, directory, timeout_hours=12):
        self.timeout_seconds = timeout_hours * 3600
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "status.json"
        self.state = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {"created": now(), "jobs": {}, "methods": {}}
        )
        self.state.update(status="running", pid=os.getpid(), updated=now())
        self.flush()

    def flush(self):
        self.state["updated"] = now()
        save(self.path, self.state)

    def job(self, name, module, arguments, required=(), dependencies=True):
        previous = self.state["jobs"].get(name, {})
        if previous.get("status") == "success":
            output = Path(previous["output"])
            if (
                output.is_dir()
                and any(output.rglob("*"))
                and all((output / item).is_file() for item in required)
            ):
                print(f"[REUSE] {name}", flush=True)
                return output
        if not dependencies:
            self.state["jobs"][name] = {
                **previous,
                "status": "blocked",
                "reason": "dependency failed",
            }
            self.flush()
            print(f"[BLOCKED] {name}: prerequisite failed; continuing", flush=True)
            return None
        attempt = int(previous.get("attempt", 0)) + 1
        output = self.directory / "jobs" / name / f"attempt_{attempt}"
        # Training tools create their own output directories; never overwrite failed attempts.
        log = self.directory / "logs" / f"{name}_attempt_{attempt}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", "-m", module, *map(str, arguments(output))]
        record = {
            "status": "running",
            "attempt": attempt,
            "started": now(),
            "output": str(output),
            "log": str(log),
            "command": command,
        }
        self.state["jobs"][name] = record
        self.state["current_job"] = name
        self.flush()
        banner = f"[{now()}] START {name} (attempt {attempt})"
        print(banner, flush=True)
        process = None
        watchdog = None
        timed_out = threading.Event()

        def stop_process():
            if process is not None and process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    process.kill()

        def timeout():
            timed_out.set()
            stop_process()

        try:
            with (
                log.open("w", encoding="utf-8") as stream,
                (self.directory / "console.log").open("a", encoding="utf-8") as combined,
            ):
                stream.write(banner + "\n")
                combined.write(banner + "\n")
                combined.flush()
                env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                watchdog = threading.Timer(self.timeout_seconds, timeout)
                watchdog.daemon = True
                watchdog.start()
                record["child_pid"] = process.pid
                self.flush()
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                    combined.write(f"[{name}] {line}")
                    combined.flush()
                code = process.wait()
            if watchdog is not None:
                watchdog.cancel()
            record["exit_code"] = code
            if timed_out.is_set():
                record["error"] = "Task exceeded configured timeout"
            missing = [item for item in required if not (output / item).is_file()]
            record["status"] = (
                "success" if code == 0 and not missing and not timed_out.is_set() else "failed"
            )
            if missing:
                record["missing_outputs"] = missing
        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                stop_process()
                process.wait()
            record["status"] = "interrupted"
            self.state["status"] = "interrupted"
            raise
        except Exception as error:
            if process is not None and process.poll() is None:
                stop_process()
                process.wait()
            record.update(status="failed", error=str(error))
        finally:
            if watchdog is not None:
                watchdog.cancel()
            record["finished"] = now()
            self.flush()
        print(f"[{record['status'].upper()}] {name}; log={log}", flush=True)
        return output if record["status"] == "success" else None


def run(directory, timeout_hours=12):
    runner = Runner(directory, timeout_hours)
    train_root = "../Small-Model-Track/Training/extracted/HAR/data"
    test_root = "../Small-Model-Track/Testing/data/small_model_track_test"
    prep = "cuhkx_sep12.prepare"
    detectors = {"2": ROOT / "artifacts/yolo/detectors/fold_2/yolov8n_fold_2.pt"}
    if not detectors["2"].is_file():
        detectors["2"] = None
    for fold in ("4", "full"):
        result = runner.job(
            f"detector_{fold}",
            prep,
            lambda out, f=fold: ["detector", "--fold", f, "--output", out, "--device", "0"],
            required=(f"yolov8n_{fold}.pt", "detector_summary.json"),
        )
        detectors[fold] = result / f"yolov8n_{fold}.pt" if result else None
    # Each successful split lives independently; assemble a common cache root via paths below.
    cache_root = runner.directory / "cache"
    ready = {}
    for fold in ("2", "4", "full"):
        tag = "full" if fold == "full" else f"fold_{fold}"
        ready[fold] = True
        for split, data_root in (("train", train_root), ("test", test_root)):
            result = runner.job(
                f"crops_{fold}_{split}",
                prep,
                lambda out, f=fold, s=split, root=data_root: [
                    "crops",
                    "--fold",
                    f,
                    "--weights",
                    detectors[f],
                    "--split",
                    s,
                    "--data-root",
                    root,
                    "--output",
                    out,
                    "--device",
                    "0",
                ],
                required=(f"{split}_coverage.json",),
                dependencies=detectors[fold] is not None,
            )
            ready[fold] &= result is not None
            if result:
                destination = cache_root / tag
                destination.mkdir(parents=True, exist_ok=True)
                # Hard-link caches on the same volume: no extra large image-cache copy.
                for source in result.rglob("*"):
                    if source.is_file():
                        target = destination / source.relative_to(result)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            if os.path.samefile(source, target):
                                continue
                            target.unlink()
                        os.link(source, target)
    sensors = runner.directory / "sensors"
    sensors_ready = True
    for split, root in (("train", train_root), ("test", test_root)):
        result = runner.job(
            f"sensors_{split}",
            "cuhkx_har.features",
            lambda out, s=split, r=root: [
                "--manifest",
                f"manifests/cv5/{s}.csv",
                "--data-root",
                r,
                "--output-dir",
                out,
                "--split",
                s,
                "--steps",
                "64",
            ],
        )
        sensors_ready &= result is not None
        if result:
            sensors.mkdir(parents=True, exist_ok=True)
            for source in result.glob("*.npz"):
                target = sensors / source.name
                if target.exists():
                    if os.path.samefile(source, target):
                        continue
                    target.unlink()
                os.link(source, target)
    detector_args = [
        arg
        for fold, flag in [("2", "--detector2"), ("4", "--detector4"), ("full", "--detector-full")]
        for arg in (flag, str(detectors[fold]))
    ]
    result = runner.job(
        NAMES[0],
        "cuhkx_sep12.lightgbm",
        lambda out: [
            "--output",
            out,
            "--cache-root",
            cache_root,
            "--sensor-cache",
            sensors,
            *detector_args,
        ],
        required=("submission.csv",),
        dependencies=all(ready.values()) and sensors_ready,
    )
    if result:
        target = runner.directory / "submissions" / f"{NAMES[0]}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result / "submission.csv", target)
    runner.state["methods"][NAMES[0]] = {
        "status": "success" if result else "failed_or_blocked",
        "submission": str(result / "submission.csv") if result else None,
    }
    runner.flush()
    for name in NAMES[1:]:
        results = {}
        for fold in ("2", "4"):
            result = runner.job(
                f"{name}_fold_{fold}",
                "cuhkx_sep12.train",
                lambda out, f=fold, n=name: [
                    "--config",
                    f"configs/sep12/{n}.json",
                    "--output",
                    out,
                    "--folds",
                    f,
                    "--cache-root",
                    cache_root,
                    "--device",
                    "cuda",
                    "--detector2",
                    str(detectors["2"]),
                    "--detector4",
                    str(detectors["4"]),
                ],
                required=(f"fold_{fold}/summary.json", "submission.csv"),
                dependencies=ready[fold],
            )
            if result:
                summary = json.loads((result / f"fold_{fold}/summary.json").read_text())
                results[fold] = {
                    "accuracy": summary["validation_accuracy"],
                    "submission": str(result / "submission.csv"),
                }
        record = {
            "status": "success" if len(results) == 2 else "partial" if results else "failed",
            "folds": results,
        }
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
    return 0 if all(m["status"] == "success" for m in runner.state["methods"].values()) else 1


def watch(directory, interval):
    path = directory / "status.json"
    previous = None
    while True:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state != previous:
                print(
                    f"\n[{now()}] {state['status']}; current={state.get('current_job')}", flush=True
                )
                for name, record in state.get("methods", {}).items():
                    print(f"  {name}: {record}", flush=True)
                current = state.get("jobs", {}).get(state.get("current_job"), {})
                if current.get("log"):
                    print(f"  log: {current['log']}", flush=True)
                previous = state
            if state.get("status") in ("finished", "interrupted"):
                return
        except (FileNotFoundError, json.JSONDecodeError):
            print("Waiting for status.json...", flush=True)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="artifacts/sep12/serial")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=12,
        help="Maximum hours per task before terminating it and continuing",
    )
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.timeout_hours <= 0:
        parser.error("--timeout-hours must be positive")
    directory = (ROOT / args.run_dir).resolve()
    if args.watch:
        watch(directory, max(1, args.interval))
    else:
        directory.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(directory / "runner.lock"):
            raise SystemExit(run(directory, args.timeout_hours))


if __name__ == "__main__":
    main()
