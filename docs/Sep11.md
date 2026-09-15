# AutoGluon Architecture Search

`cuhkx_har.automl` uses the frozen five-fold, divide-by-user split. It summarizes cached Skeleton, IMU, and Radar sequences using temporal and frequency-domain features, plus small RGB and motion summaries for Depth Color, IR, and Thermal.

After building `manifests/cv5` and `cache-64`, run these commands from `har-solution`. The feature files can be reused between runs and always use identical feature options for training and test.

```powershell
uv run python -m cuhkx_har.automl features `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --manifest manifests/cv5/train.csv `
  --cache-dir cache-64 `
  --split train --output artifacts/automl_features_train.csv

uv run python -m cuhkx_har.automl features `
  --data-root ..\Small-Model-Track\Testing\data\small_model_track_test `
  --manifest manifests/cv5/test.csv `
  --cache-dir cache-64 `
  --split test --output artifacts/automl_features_test.csv

# Before this command, close memory-heavy applications and check that at least 6 GB system RAM is free. 
# The time limit is per fold, in seconds.
uv run python -m cuhkx_har.automl cv `
  --features artifacts/automl_features_train.csv `
  --output-dir artifacts/automl_v3 --device cuda --time-limit 7200

uv run python -m cuhkx_har.automl submit `
  --model-dir artifacts/automl_v3 `
  --test-features artifacts/automl_features_test.csv `
  --test-csv ..\Small-Model-Track\Testing\test_file\test.csv `
  --output artifacts/automl_v3/submission.csv
```

The CV command writes fold predictions, model leaderboards, `oof_predictions.csv`,
and `cv_summary.json`. Each `fold_N/best_model.json` reports the winning AutoGluon
model, its held-out-fold accuracy, resolved hyperparameters, single-model disk
size, and total predictor size. The model chosen for deployment is the five-fold
ensemble written by `submit`, not the winner of one unusually easy fold. Compare
the OOF accuracy with the deep model; only blend models after cross-fitted
ensemble evaluation. The submission command averages all five fold predictors
and validates the exact official row order.

AutoGluon is imported only at training/inference time, so feature extraction
and unit tests do not require it. The existing environment must supply
`autogluon.tabular` and its desired optional model backends.

This run intentionally uses LightGBM and CatBoost on CPU, while XGBoost and
PyTorch use CUDA. CatBoost GPU training exceeds 8 GB VRAM for this 40-class
dataset. The run disables AutoGluon's automatic weighted ensemble and
`refit_full` so it does not retain duplicate models. Every class present in a
fold is retained, including classes with fewer than ten training clips.
