from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return default


def sparkline(values: list[float]) -> str:
    if not values:
        return "waiting"
    blocks = "._-=+*#@"
    low, high = min(values), max(values)
    if high == low:
        return blocks[3] * len(values)
    return "".join(blocks[round((value - low) / (high - low) * 7)] for value in values)


def progress_bar(step: int, total: int, width: int = 30) -> str:
    fraction = min(max(step / max(total, 1), 0.0), 1.0)
    filled = round(fraction * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {fraction:6.1%}"


def render_dashboard(
    run_dir: Path,
    progress: dict[str, Any],
    history: list[dict[str, Any]],
    summary: dict[str, Any],
    patience: int | None,
) -> str:
    lines = [f"CUHK-X live training | {run_dir}", "=" * 88]
    if progress:
        epoch = int(progress.get("epoch", 0))
        total_epochs = int(progress.get("total_epochs", 0))
        step = int(progress.get("step", 0))
        total_steps = int(progress.get("total_steps", 0))
        phase = str(progress.get("phase", "waiting")).upper()
        lines.append(
            f"Epoch {epoch}/{total_epochs} | {phase:<5} batch {step}/{total_steps} "
            f"{progress_bar(step, total_steps)}"
        )
        lines.append(
            f"Running loss {float(progress.get('loss', 0)):.4f} | "
            f"accuracy {float(progress.get('accuracy', 0)):.4f} | "
            f"phase time {float(progress.get('elapsed_seconds', 0)) / 60:.1f} min"
        )
    else:
        lines.append("Waiting for the first batch update...")

    if history:
        best = max(history, key=lambda row: float(row["valid_accuracy"]))
        since_best = len(history) - 1 - history.index(best)
        patience_text = f"/{patience}" if patience is not None else ""
        lines.extend(
            [
                "",
                f"Best validation accuracy {float(best['valid_accuracy']):.4f} "
                f"at epoch {int(best['epoch'])} | early-stop wait {since_best}{patience_text}",
                "",
                " epoch | train loss | valid loss | train acc | valid acc | learning rate",
                "-------+------------+------------+-----------+-----------+--------------",
            ]
        )
        for row in history[-8:]:
            lines.append(
                f"{int(row['epoch']):6d} | {float(row['train_loss']):10.4f} | "
                f"{float(row['valid_loss']):10.4f} | {float(row['train_accuracy']):9.4f} | "
                f"{float(row['valid_accuracy']):9.4f} | {float(row['learning_rate']):.3e}"
            )
        train_losses = [float(row["train_loss"]) for row in history]
        valid_losses = [float(row["valid_loss"]) for row in history]
        lines.extend(
            [
                "",
                f"Train loss: {sparkline(train_losses)}  {train_losses[-1]:.4f}",
                f"Valid loss: {sparkline(valid_losses)}  {valid_losses[-1]:.4f}",
            ]
        )

    if summary:
        lines.extend(
            [
                "",
                "TRAINING COMPLETE",
                f"Best validation accuracy: {float(summary['best_valid_accuracy']):.4f}",
                f"Elapsed: {float(summary['elapsed_minutes']):.1f} min",
                f"Checkpoint: {summary['checkpoint']}",
            ]
        )
    lines.append("\nPress Ctrl+C to stop monitoring; training will continue.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live terminal dashboard for a CUHK-X run")
    parser.add_argument("--run-dir", required=True, help="Fold directory containing history.json")
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh seconds")
    parser.add_argument("--patience", type=int, help="Show the configured early-stop patience")
    parser.add_argument("--no-clear", action="store_true", help="Print snapshots without clearing")
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    run_dir = Path(args.run_dir).resolve()
    try:
        while True:
            progress = read_json(run_dir / "progress.json", {})
            history = read_json(run_dir / "history.json", [])
            summary = read_json(run_dir / "summary.json", {})
            dashboard = render_dashboard(run_dir, progress, history, summary, args.patience)
            if not args.no_clear:
                print("\033[2J\033[H", end="")
            print(dashboard, flush=True)
            if args.once or summary:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped. The training process was not interrupted.")


if __name__ == "__main__":
    main()
