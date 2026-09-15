# Sep13: depth-led robustness experiments

These experiments reuse the completed, leak-checked `sep12/serial_depth_align` crop and sensor caches. They do not train a new YOLO detector or overwrite any Sep12 result. Each job writes into `artifacts/sep13/serial/jobs/<job>/attempt_N`; a failed job is recorded and the runner continues to the next method.

`m01` evaluates a fixed baseline plus eight regularized LightGBM candidates on folds 2 and 4. A candidate is selected only if neither fold declines and its validation-size-weighted score is strictly greater than the baseline. Only then is it refit on all training data and used to create `submission.csv`.

The visual experiments share an epoch-level log format.  Their changes are:

| Method | Sep13 change |
|---|---|
| m02 | Depth-weighted branch vote plus a learned availability/content gate |
| m03 | Learned gate for the IR/Depth fusion |
| m04 | Depth-weighted probability sum instead of equal probability sum |
| m05 | Clip-consistent crop, scale, brightness/contrast, erasing, and temporal-offset augmentation |
| m06 | Temporal attention and learned modality gate replace simple masked means |
| m07 | Attention pools time first; the Transformer then fuses only the modality tokens |
| m08 | Spatial attention after SE plus a learned modality gate |

All geometric and photometric augmentation draws are shared by every frame and every visual modality in a clip. Thus augmented pixels cannot create frame-to-frame motion.

Run from `CUHK-X\har-solution`:

```powershell
.\.venv\Scripts\python.exe -u -m cuhkx_sep13.serial --run-dir artifacts/sep13/serial --timeout-hours 12
```

Watch without starting a second runner:

```powershell
.\.venv\Scripts\python.exe -u -m cuhkx_sep13.serial --run-dir artifacts/sep13/serial --watch --interval 5
```

The runner expects the existing Sep12 depth-align cache and detector files. If a required file is absent, every dependent job is marked blocked and no training begins.
