# Debug Tools

To monitor the training process:
```powershell
uv run cuhkx-monitor `
  --run-dir artifacts\cv5_synced_flip_imu_dropout\fold_3 `
  --patience 8
```

To see whether GPU is available:
```python
import torch
print(torch.cuda.is_available())  # should return True
print(torch.cuda.device_count())  # should >= 1
print(torch.cuda.get_device_name(0))  # should return RTX 5060
```

GPU usage can be viewed continuously instead of just once:
```powershell
nvidia-smi `
  --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu `
  --format=csv `
  -l 1
```

Notice that we can not modify the config file and resume a unfinished training process.