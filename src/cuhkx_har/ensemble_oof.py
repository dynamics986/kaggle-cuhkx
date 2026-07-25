from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_prediction_pair(
    baseline_path: Path, candidate_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = pd.read_csv(baseline_path)
    candidate = pd.read_csv(candidate_path)
    probability_columns = [f"prob_{index}" for index in range(40)]
    required = ["clip_id", "label", *probability_columns]

    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} predictions are missing columns: {missing}")
    if not baseline["clip_id"].equals(candidate["clip_id"]):
        raise ValueError(f"Prediction rows are not aligned: {baseline_path} vs {candidate_path}")
    if not baseline["label"].equals(candidate["label"]):
        raise ValueError(f"Labels do not match: {baseline_path} vs {candidate_path}")

    labels = baseline["label"].to_numpy(dtype=np.int64)
    baseline_probabilities = baseline[probability_columns].to_numpy(dtype=np.float64)
    candidate_probabilities = candidate[probability_columns].to_numpy(dtype=np.float64)
    return labels, baseline_probabilities, candidate_probabilities


def scan_weights(
    labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    fold_ids: np.ndarray,
    steps: int,
) -> list[dict[str, object]]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    results: list[dict[str, object]] = []
    unique_folds = np.unique(fold_ids)
    for index in range(steps + 1):
        candidate_weight = index / steps
        probabilities = (
            (1.0 - candidate_weight) * baseline_probabilities
            + candidate_weight * candidate_probabilities
        )
        predictions = probabilities.argmax(axis=1)
        correct = predictions == labels
        fold_accuracy = {
            str(int(fold)): float(correct[fold_ids == fold].mean()) for fold in unique_folds
        }
        results.append(
            {
                "baseline_weight": 1.0 - candidate_weight,
                "candidate_weight": candidate_weight,
                "oof_accuracy": float(correct.mean()),
                "correct": int(correct.sum()),
                "examples": int(len(labels)),
                "fold_accuracy": fold_accuracy,
            }
        )
    return sorted(results, key=lambda row: (-float(row["oof_accuracy"]), row["candidate_weight"]))


def crossfit_weight_selection(
    labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    fold_ids: np.ndarray,
    steps: int,
) -> dict[str, object]:
    """Evaluate blending without choosing a fold's weight on that fold.

    The ordinary pooled OOF scan is useful to choose a final deployment weight,
    but its best score is optimistic because it selects after seeing every OOF
    label.  This leave-one-fold-out estimate is the unbiased guardrail used
    while deciding whether an ensemble idea is worth keeping.
    """
    unique_folds = np.unique(fold_ids)
    rows: list[dict[str, object]] = []
    all_predictions = np.empty(len(labels), dtype=np.int64)
    for held_out in unique_folds:
        selection_mask = fold_ids != held_out
        held_out_mask = ~selection_mask
        selection = scan_weights(
            labels[selection_mask],
            baseline_probabilities[selection_mask],
            candidate_probabilities[selection_mask],
            fold_ids[selection_mask],
            steps,
        )[0]
        weight = float(selection["candidate_weight"])
        probabilities = (
            (1.0 - weight) * baseline_probabilities[held_out_mask]
            + weight * candidate_probabilities[held_out_mask]
        )
        predictions = probabilities.argmax(axis=1)
        all_predictions[held_out_mask] = predictions
        rows.append(
            {
                "held_out_fold": int(held_out),
                "selected_on_other_folds_candidate_weight": weight,
                "selection_accuracy": float(selection["oof_accuracy"]),
                "held_out_accuracy": float((predictions == labels[held_out_mask]).mean()),
            }
        )
    correct = all_predictions == labels
    return {
        "crossfit_oof_accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "examples": int(len(labels)),
        "folds": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a common weight for two OOF model families")
    parser.add_argument("--baseline", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if len(args.baseline) != len(args.candidate):
        raise ValueError("Baseline and candidate must contain the same number of folds")

    label_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    for fold, (baseline_path, candidate_path) in enumerate(
        zip(args.baseline, args.candidate, strict=True)
    ):
        labels, baseline_probabilities, candidate_probabilities = load_prediction_pair(
            baseline_path, candidate_path
        )
        label_parts.append(labels)
        baseline_parts.append(baseline_probabilities)
        candidate_parts.append(candidate_probabilities)
        fold_parts.append(np.full(len(labels), fold, dtype=np.int64))

    results = scan_weights(
        np.concatenate(label_parts),
        np.concatenate(baseline_parts),
        np.concatenate(candidate_parts),
        np.concatenate(fold_parts),
        args.steps,
    )
    crossfit = crossfit_weight_selection(
        np.concatenate(label_parts),
        np.concatenate(baseline_parts),
        np.concatenate(candidate_parts),
        np.concatenate(fold_parts),
        args.steps,
    )
    payload = {
        "folds": len(args.baseline),
        "steps": args.steps,
        "results": results,
        "crossfit_weight_selection": crossfit,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for rank, row in enumerate(results[: args.top], start=1):
        folds = ", ".join(
            f"fold_{fold}={accuracy:.5f}"
            for fold, accuracy in dict(row["fold_accuracy"]).items()
        )
        print(
            f"{rank:2d}. baseline={row['baseline_weight']:.3f} "
            f"candidate={row['candidate_weight']:.3f} "
            f"OOF={row['oof_accuracy']:.5f} ({row['correct']}/{row['examples']}) "
            f"{folds}"
        )
    print(
        "Cross-fitted blend accuracy="
        f"{crossfit['crossfit_oof_accuracy']:.5f} ({crossfit['correct']}/{crossfit['examples']})"
    )


if __name__ == "__main__":
    main()
