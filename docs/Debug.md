# Debug Tools

## To monitor the training process:
```powershell
uv run cuhkx-monitor `
  --run-dir artifacts\cv5_synced_flip_imu_dropout\fold_3 `
  --patience 8
```

## To see whether GPU is available:
```python
import torch
print(torch.cuda.is_available())  # should return True
print(torch.cuda.device_count())  # should >= 1
print(torch.cuda.get_device_name(0))  # should return RTX 5060
```

## GPU usage can be viewed continuously instead of just once:
```powershell
nvidia-smi `
  --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu `
  --format=csv `
  -l 1
```

## Model Training
```powershell
cd path\to\CUHK-X\har-solution

$yours = TBC
$fold = 10086
uv run cuhkx-train `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --manifest manifests\cv5\train.csv `
  --config configs\$yours .json `
  --cache-dir cache-64 `
  --output-dir artifacts\$yours `
  --fold $fold `
  --resume artifacts\$yours\$fold\last.pt
```

Notice that we can not modify the config file and resume a unfinished training process.

## Submission

The following command will run successfully, provided that `cache-64` already contains the sensor cache for the test set.

```powershell
uv run cuhkx-predict `
  --data-root ..\Small-Model-Track\Testing\data\small_model_track_test `
  --manifest manifests\cv5\test.csv `
  --cache-dir cache-64 `
  --checkpoints artifacts\path\to\your\best.pt `
  --views 3 `
  --output artifacts\submission_yours.csv
```

Then validate the format:

```powershell
uv run cuhkx-check-submission `
  --submission artifacts\submission_yours.csv `
  --test-csv ..\Small-Model-Track\Testing\test_file\test.csv
```

## explanation

Regarding the 5 fold training, we used ensemble training to weighted average the 5 fold model results. Actually each and every checkppoint can be an independent model to get prediction.


Regarding `tests/`:
- When running `uv run pytest`, pytest automatically discovers `test_*.py` files and `test_*` functions inside them, executes them, and checks `assert` statements.
- For example, `test_model.py` verifies that the model can forward-propagate, parameter size is within limits, and configurations are mutually exclusive; `test_submission.py` validates CSV format; `test_splits.py` verifies that no subject leakage occurs.