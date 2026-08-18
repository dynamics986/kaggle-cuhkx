
# July 19 training commands

Run commands from:

```powershell
cd C:\Users\dynam\Documents\CUHK-X\har-solution
```

The synchronized-flip experiment uses `configs\synced_flip.json`. Fold 0 completed at `0.50823`
validation accuracy (best epoch 24, stopped epoch 32). Its contribution to the existing ensemble
was only one additional validation clip, so folds 1 and 2 are currently **on hold** until a
fixed-cache no-flip Base provides a controlled comparison.

Each fold trains for at most
40 epochs and stops after 8 consecutive epochs without a validation-accuracy improvement. Always
use `best.pt` for inference. Do not launch folds 1 and 2 until fold 0 has been reviewed.

## Fold 1

Training window:

```powershell
uv run cuhkx-train `
  --config configs\synced_flip.json `
  --manifest manifests\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 `
  --output-dir artifacts\synced_flip `
  --fold 1
```

Second PowerShell monitoring window:

```powershell
uv run cuhkx-monitor `
  --run-dir artifacts\synced_flip\fold_1 `
  --patience 8
```

If fold 1 is interrupted, resume it with the original training command plus:

```powershell
--resume artifacts\synced_flip\fold_1\last.pt
```

## Fold 2

Training window:

```powershell
uv run cuhkx-train `
  --config configs\synced_flip.json `
  --manifest manifests\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 `
  --output-dir artifacts\synced_flip `
  --fold 2
```

Second PowerShell monitoring window:

```powershell
uv run cuhkx-monitor `
  --run-dir artifacts\synced_flip\fold_2 `
  --patience 8
```

If fold 2 is interrupted, resume it with the original training command plus:

```powershell
--resume artifacts\synced_flip\fold_2\last.pt
```

Each fold writes to its own directory, so fold 1 and fold 2 do not overwrite fold 0. For this
laptop, train one fold at a time rather than running folds concurrently on the same GPU.
