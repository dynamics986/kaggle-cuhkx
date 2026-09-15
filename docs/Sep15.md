# Sep15: budgeted AutoML and temporal sensor search

Each architecture family receives four hours total, split into two hours for each representative subject-held-out validation fold (Fold 2 and Fold 4). The serial runner isolates failures, records `status.json`, and writes one official-format submission for each successful family.

`ag_tree_search` uses AutoGluon random HPO over LightGBM, CatBoost and XGBoost on the existing leakage-safe clip-summary table. `ag_nn_search` searches the AutoGluon tabular MLP. These are tabular models: AutoGluon is not asked to interpret a 64-frame stream as a Transformer or RNN.

`sensor_gru_search` and `sensor_transformer_search` read the cached Skeleton, IMU and Radar sequences directly. Each modality has its own projection and temporal encoder, attention pooling, sensor-presence mask and final classifier. They randomly evaluate compact hyperparameter candidates until their fold budget expires, retain the best validation checkpoint, average Fold 2/4 test probabilities, and write `submission.csv`.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -u -m cuhkx_sep15.serial --timeout-hours 4.25
```

View current progress in another PowerShell window:

```powershell
.\.venv\Scripts\python.exe -u -m cuhkx_sep15.serial --run-dir artifacts/sep15/serial --watch
Get-Content artifacts\sep15\serial\console.log -Wait
```
