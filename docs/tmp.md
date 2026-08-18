

> The following command will run successfully, provided that `cache-64` already contains the sensor cache for the test set.

```powershell
uv run cuhkx-predict `
  --checkpoints artifacts\cv5_pose_motion_residual\fold_4\best.pt `
  --manifest manifests\cv5\test.csv `
  --data-root ..\Small-Model-Track\Testing\data\small_model_track_test `
  --cache-dir cache-64 `
  --views 3 `
  --output artifacts\submission1.csv
```

Then validate the format:

```powershell
uv run cuhkx-check-submission `
  --submission artifacts\submission1.csv `
  --test-csv ..\Small-Model-Track\Testing\test_file\test.csv
```

Note the distinctions:

- A single `fold_4\best.pt`: can stand alone for demonstration and produce independent results, but has only seen training users from the other 4 folds; its stability is typically lower than an ensemble.
- Five folds from the same model family: when all are listed after `--checkpoints`, the program will sequentially run inference and average the probabilities.
- Current final approach: all ten checkpoints are included, but they are not run in parallel; their predictions are fused with weights of `0.05 × 5` baseline plus `0.15 × 5` pose-motion. This is the competition output with the strongest current CV evidence.

Therefore, when explaining to judges, you can say: "Each `.pt` is a complete, independently reproducible model; the final submission uses a weighted probability ensemble across folds and architectures to improve cross-user generalization stability."


### Regarding `tests/`:

- They are not meant to be invoked during training, but rather to be invoked by `pytest`.
- When running `uv run pytest`, pytest automatically discovers `test_*.py` files and `test_*` functions inside them, executes them, and checks `assert` statements.
- Therefore a `main()` is not needed.
- For example, `test_model.py` verifies that the model can forward-propagate, parameter size is within limits, and configurations are mutually exclusive; `test_submission.py` validates CSV format; `test_splits.py` verifies that no subject leakage occurs.
- `test_gpu.py` currently only has top-level print statements and no real `test_*` assertions; it is more of a manual GPU inspection script than a valid unit test.


### The `seed` in `configs/*.json`

The `seed` in `configs/*.json` is the random number seed, controlling initialization, shuffle, augmentation sampling, and DataLoader randomness. `20260719` is simply the primary seed selected when the project was established on 2026-07-19 and has no special meaning. It can be changed, but the rules are:

- Do not change the configuration corresponding to a checkpoint being restored;
- Do not arbitrarily change it for a certain fold/LB score;
- A new seed should be treated as a new experiment directory with a complete CV run;
- Multiple seeds can serve as a source of ensemble diversity in the future, but each seed must have OOF evidence.