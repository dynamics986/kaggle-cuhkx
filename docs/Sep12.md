# Sep 12 Training and Results

## 1. Repository Status and Rules for This Run

```text
CUHK-X/
  Small-Model-Track/
    Training/extracted/HAR/data/           # Raw training data, six modalities
    Testing/data/small_model_track_test/   # 405 test clips
    Testing/test_file/test.csv             # Official output order
  har-solution/
    src/cuhkx_har/                         # Existing multimodal, AutoML, feature, and submission-checking code
    src/cuhkx_modality/                    # Existing single-modality five-fold code
    src/cuhkx_sep12/                       # New code for this run; it shares infrastructure but uses separate experiments
      prepare.py                           # Fold-safe YOLO fine-tuning, alignment, and cropping
      data.py                              # Reads YOLO-cropped caches
      model.py                             # Methods 2–8
      train.py                             # Neural-network training, validation, and inference
      lightgbm.py                          # Method 1 CV, full retraining, and inference
      common.py                            # Probabilities, submission format, and size checks
    configs/sep12/m01_*.json ... m08_*.json
    manifests/cv5/{train,test}.csv         # Reuses the frozen split; no new folds
    cache-64/                              # Existing Skeleton/IMU/Radar sequence cache
    cache-sep12/{fold_2,fold_4,full}/      # New YOLO-cropped cache, separate from the old cache
    artifacts/yolo/                        # Existing labels, detectors, and old crop results
    artifacts/sep12/                       # Detectors and outputs for these eight experiments
    docs/Sep12.md
```

- The frozen training set has **3,036** clips, 40 classes, and 18 training users.
- `fold 2` and `fold 4` here use the zero-based names `fold_2` and `fold_4` from the original code.
- Fold 2 has 2,496 training clips and 540 validation clips; validation users are `user20,user22,user9`.
- Fold 4 has 2,396 training clips and 640 validation clips; validation users are `user1,user21,user7,user8`.
- Each fold trains on the other four folds. It does not train only on fold-2/fold-4 data and does not create a new two-fold split.
- The full frozen clip-to-fold fingerprint is checked, and training and validation users must not overlap.
- Only the separate scores of the two folds and their validation-size-weighted accuracy are reported. This process does not create or claim a full five-fold OOF result.
- Existing code, results, and old documents remain usable. The two-fold process on this page is the rule for these experiments.

The existing `artifacts/automl_v3/fold_3/best_model.json` confirms that the model is LightGBM and its hyperparameters match the requirement. Its recorded validation accuracy is **0.6773504273504274**. This is historical model information, not a new result from this run. This run fixes `learning_rate=0.035, num_boost_round=1000, num_leaves=48, seed=0`. Visual inputs change from old full-image statistics to YOLO-crop statistics, so only the parameters are reused; it does not claim to exactly reproduce the old features or score.

## 2. Eight Methods

|No.|Config/output directory prefix|Model and fusion method|
|---|---|---|
|1|`m01_lightgbm`|Existing Skeleton/IMU/Radar summary features plus YOLO-crop visual summaries from three modalities; after two-fold validation, one LightGBM is fit on all training samples.|
|2|`m02_attention_concat_vote`|Three independent 2D CNNs → temporal Conv1D plus attention pooling; concatenated features are classified, and the three separate heads and concatenated head use hard voting.|
|3|`m03_attention_concat_ir_depth`|IR and Depth_Color only; independent 2D CNNs → temporal attention → concatenated feature classification.|
|4|`m04_attention_probability_sum`|IR, Depth_Color, and Thermal each use a CNN, temporal attention, and 40-class softmax; probabilities from all available modalities are added, normalized, and passed to argmax.|
|5|`m05_dual_resnet18`|One independent ResNet-18 each for IR and Depth_Color, with the original 1,000-class head replaced, followed by temporal attention and concatenated feature classification.|
|6|`m06_independent_concat`|Three independent CNNs, masked temporal mean pooling, then concatenated feature classification; this is the control without temporal attention.|
|7|`m07_temporal_transformer`|Three independent lightweight CNNs → time-position and modality embeddings → a two-layer, four-head Transformer with CLS classification and an explicit missing-token mask.|
|8|`m08_se_attention`|SE channel attention in every convolution stage of three CNNs, followed by temporal attention and concatenated feature classification.|

Method 2 gives one vote to each available modality head and to the concatenated head. Ties use the mean softmax probability; the probability term can add at most 0.25 vote, so it cannot overturn a full-vote lead. Output `prob_*` values are normalized vote scores, not calibrated probabilities. Method 4 uses softmax instead of 40 separate sigmoids because classes are mutually exclusive; summing or averaging available-modality probabilities gives the same argmax. Methods 2 and 4 also supervise the independent classification heads. Other methods supervise the final classification head directly.

Method 7 implements the user-described “temporal transformation” as a lightweight multimodal temporal **Transformer**. It uses the visual modalities from these experiments, IR, Depth_Color, and Thermal, and does not add the three sensors. Except for reuse or fine-tuning of the specified YOLO model, all HAR CNNs, ResNets, and Transformers train from scratch; ResNet uses `weights=None`. The default uses 16 frames, 128×128 images, and embedding size 128; Method 5 uses batch size 4 and the others use batch size 8. Training runs for at most 60 epochs and stops after validation accuracy does not improve for 15 consecutive epochs. Copy a config to a new file to change it, and use a new result directory.

## 3. Environment

```powershell
uv sync --extra sep12 --dev
uv run --extra sep12 python -c "import torch, ultralytics, lightgbm; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CUDA must be available. Neural-network commands specify `--device cuda`; YOLO commands specify `--device 0`. Neural networks do not silently fall back to CPU. For small CPU debugging, explicitly use `--device cpu`.

The new console entry points are `cuhkx-sep12-prepare`, `cuhkx-sep12-train`, and `cuhkx-sep12-lightgbm`. The commands below use `python -m` to avoid entry points from an old editable install that were not refreshed.

## 4. YOLO Training Scope and Cropping

The existing detector is `artifacts/yolo/detectors/fold_2/yolov8n_fold_2.pt`, 6,237,994 bytes. `detector_summary.json` records its training users. It excludes Fold 2 validation users but **includes Fold 4 validation users**. Therefore it is reused only for Fold 2; this fine-tuned weight cannot be further fine-tuned for Fold 4 while claiming no leakage.

The existing 800 person annotations are in `artifacts/yolo/annotations_800/` and come from Depth_Color. Initialize Fold 4 and full from the original `yolov8n.pt`:

```powershell
uv run --extra sep12 python -m cuhkx_sep12.prepare detector `
  --fold 4 --output artifacts/sep12/yolo/fold_4 --device 0

uv run --extra sep12 python -m cuhkx_sep12.prepare detector `
  --fold full --output artifacts/sep12/yolo/full --device 0
```

Detector training filters annotations by the user range in the frozen manifest and checks the clip-to-user mapping and image hash. It uses `last.pt` after a fixed 60 epochs and does not use HAR validation users to select a detector. In the YOLO data config, `val` points to training images and `val=False`, so its internal metrics cannot be treated as independent validation scores. The full detector uses all existing annotations only for final full-data LightGBM and does not take part in two-fold validation.

To rebuild the Fold 2 detector, use the same command with `--fold 2 --output artifacts/sep12/yolo/fold_2`, then set `--detector2 artifacts/sep12/yolo/fold_2/yolov8n_2.pt` and rebuild its cache.

### Alignment and Missing-data Rules

1. Each clip uses the same 16 time positions. When IR and Depth have readable timestamps, frames are paired by nearest time in their shared time range.
2. Thermal has only `frame_*.jpg` and no reliable absolute clock. It maps to normalized time in the same clip. This is approximate alignment that assumes corresponding clip starts and ends; it must not be interpreted as precise hardware synchronization.
3. When no readable clock exists, normalized time positions are shared. An error is raised if IR and Depth time ranges do not overlap at all, rather than silently pairing the wrong frames.
4. **All three modalities run the fine-tuned YOLO on their own images**. A Depth box is not moved directly to the Thermal camera.
5. Each frame uses the highest-confidence person box, expands it by 10% on every side, crops it, and letterboxes it to 128×128.
6. If some frames of one modality are not detected, they borrow the box from the nearest detected sampled frame of that modality. If the whole stream is not detected, it is zeroed and masked. It does not silently fall back to the original full image. A missing original modality is also zeroed and masked, while the clip remains to fully cover validation and test rows.
7. Data augmentation is enabled only during training: all modalities and time positions share the horizontal flip and brightness factor.

Existing detector annotations come from Depth_Color, so detection rates on IR and Thermal must be checked in practice. The cache saves `boxes`, `detected`, `mask`, and `alignment`, and writes `train_coverage.csv/json` and `test_coverage.csv/json`. They count direct detections, cropped frames, and clips with a fully undetected stream for each modality; a masked image must not be described as a successful detection. For the 16 sampled positions of `SM_test_0001`, IR directly detects 1 frame, Depth detects 16, and Thermal detects 0. After borrowing boxes from the same modality, valid crops are 16/16/0. Thermal for this clip is explicitly masked and does not use the full image instead. This shows that a detector fine-tuned with Depth annotations has a domain gap on other modalities. Formal experiments must inspect coverage reports, and the current implementation does not claim to locate a person successfully in every original Thermal stream.

### Generate Once, Share Across Seven Visual Experiments

```powershell
$detectors = @{
  "2" = "artifacts/yolo/detectors/fold_2/yolov8n_fold_2.pt"
  "4" = "artifacts/sep12/yolo/fold_4/yolov8n_4.pt"
  "full" = "artifacts/sep12/yolo/full/yolov8n_full.pt"
}
foreach ($fold in @("2", "4", "full")) {
  $tag = if ($fold -eq "full") { "full" } else { "fold_$fold" }
  uv run --extra sep12 python -m cuhkx_sep12.prepare crops `
    --fold $fold --weights $detectors[$fold] --split train `
    --data-root ..\Small-Model-Track\Training\extracted\HAR\data --output "cache-sep12/$tag" --device 0
  if ($LASTEXITCODE -ne 0) { throw "Train crops failed: $tag" }
  uv run --extra sep12 python -m cuhkx_sep12.prepare crops `
    --fold $fold --weights $detectors[$fold] --split test `
    --data-root ..\Small-Model-Track\Testing\data\small_model_track_test --output "cache-sep12/$tag" --device 0
  if ($LASTEXITCODE -ne 0) { throw "Test crops failed: $tag" }
}
```

The full test cache is for Method 1; fold_2/fold_4 test caches are for neural networks in each fold. Each directory saves `train_metadata.json`, `test_metadata.json`, `train/*.npz`, and `test/*.npz`. Metadata includes the manifest and detector SHA256, sample count, image size, thresholds, and policy, preventing caches from different folds or detectors from being mixed. The same crop command can continue after interruption and reuses completed clips; changed parameters require a new directory. The old `artifacts/yolo/crops/fold_2/bboxes.json` cannot directly replace the new three-modality cache.

## 5. Method 1: LightGBM Two-fold Validation, Full Training, and CSV

Reuse the sensor cache when it already exists; otherwise generate the training and test caches separately:

```powershell
uv run --extra sep12 python -m cuhkx_har.features `
  --manifest manifests/cv5/train.csv --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --output-dir cache-64 --split train --steps 64
uv run --extra sep12 python -m cuhkx_har.features `
  --manifest manifests/cv5/test.csv --data-root ..\Small-Model-Track\Testing\data\small_model_track_test `
  --output-dir cache-64 --split test --steps 64

uv run --extra sep12 python -m cuhkx_sep12.lightgbm `
  --output artifacts/sep12/m01_lightgbm
```

By default it runs Fold 2 validation, Fold 4 validation, full training, and full-model test inference in order. It can also run stages separately with `--stage cv`, `--stage full`, and `--stage predict`.

This entry point uses native LightGBM directly for a fixed 1,000 rounds, with no early stopping and no hidden AutoGluon holdout. Each fold training run reads only labels from its training fold. The full model really uses all 3,036 samples and holds out no validation set. Full retraining does not report a “full validation accuracy”; its only generalization reference is the earlier two-fold result. Method 1 JSON files describe fixed parameters; `PARAMETERS` in the entry point is fixed to the same set and does not accept arbitrary tuning.

```text
artifacts/sep12/m01_lightgbm/
  fold_2/{model.txt,summary.json,validation_predictions.csv}
  fold_4/{model.txt,summary.json,validation_predictions.csv}
  comparison.json
  full/{model.txt,summary.json,test_predictions.csv}
  submission.csv                        # Output of one full model
```

## 6. Methods 2–8: Each Trains Independently to a Submission

```powershell
$configs = @(
  "m02_attention_concat_vote",
  "m03_attention_concat_ir_depth",
  "m04_attention_probability_sum",
  "m05_dual_resnet18",
  "m06_independent_concat",
  "m07_temporal_transformer",
  "m08_se_attention"
)
foreach ($name in $configs) {
  uv run --extra sep12 python -m cuhkx_sep12.train `
    --config "configs/sep12/$name.json" `
    --output "artifacts/sep12/$name" --folds 2 4 --device cuda
  if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $name" }
}
```

To run one method alone, use its corresponding command. By default, each method completes training and testing for both folds:

```text
artifacts/sep12/m0N_name/
  fold_2/
    best.pt                             # Inference-only state_dict and config, without optimizer
    history.json                        # Training/validation loss and accuracy per epoch
    summary.json                        # Samples, users, best epoch, GPU, size, and source
    validation_predictions.csv          # clip_id,label,prediction,prob_0..39
    test_predictions.csv
    submission.csv
  fold_4/                               # Same structure
  comparison.json                       # Two-fold report and deployment-fold selection
  submission.csv                        # Predictions from the deployment fold, not a two-fold ensemble
```

Use `--stage train` for training only, then use the same command with `--stage predict` to reload the best checkpoint and create CSV files. `--folds 2` and `--folds 4` can be scheduled separately. After completion, run `--stage predict --folds 2 4` to collect both folds and perform inference. Completed training is never overwritten. To skip completed folds and continue with remaining folds, add `--reuse-completed`; the code checks the config and source. **Epoch-level checkpoint resume is not supported**; an incomplete training directory must be rerun in a new output directory and is not treated as complete.

Each fold chooses `best.pt` by validation accuracy. Final deployment selects the fold with higher validation accuracy by default; a tie selects the lower fold number. This is only a compute-saving deployment rule and does not mean that fold is certainly better for all users; both fold accuracies remain separate. Neural networks have no additional full retraining and do not deploy both dual-ResNet weight sets together. The final package must include the selected fold `best.pt`, its YOLO weights, and preprocessing code; see `summary` for paths and hashes.

## 7. Model Size, Validation, and Reproduction

Before writing every submission, the size of one inference file set, **including the detector**, is checked to be strictly below decimal 100,000,000 bytes. Training optimizer state is excluded from deployment because `best.pt` does not save it. Method 5 has two ResNet-18 models plus one YOLO model; actual saved file size is the criterion, not only an estimate such as “45MB×2.” The default saved dual-ResNet-18 and existing YOLO are **96,498,441 bytes** (checkpoint metadata can vary slightly); other visual methods are about 7.9–9.8 MB. LightGBM also performs the size check; it raises an error after real training if the limit is exceeded rather than claiming compliance. This checks only the weight-file budget, not runtime VRAM or the full Python environment size.

The seed is fixed at 0. Configurations, torch/GPU information, and manifest/YOLO/checkpoint hashes are saved; CUDA/cuDNN use deterministic settings and a fixed DataLoader random seed. Different software and hardware versions can still cause floating-point differences. Inference uses the same crop size, sampling rule, and detector as training, and restores row order by the official `path` mapping. The submission CSV has only `path,prediction`, uses integer classes 0–39, and covers all 405 test clips.

```powershell
uv run --extra sep12 python -m cuhkx_har.submission `
  --submission artifacts/sep12/m01_lightgbm/submission.csv `
  --test-csv ../Small-Model-Track/Testing/test_file/test.csv

uv run --extra sep12 pytest tests/test_sep12.py -q -p no:cacheprovider --basetemp=.sep12-test-tmp
```

Before running, copy a neural-network config, set `epochs=1, patience=1, workers=0`, keep the default frames/image_size, and use `--folds 2 --output artifacts/sep12/smoke_m08` for a real-data smoke test; it still iterates over every sample in that fold. Test code uses small synthetic data to check structure, computation, and data contracts; it is not a full model performance test.

## 8. Delivery Validation Record

- The repository structure, frozen split, existing LightGBM parameters, and YOLO training-user record were read.
- The Windows environment was tested with PyTorch `2.11.0+cu128`; CUDA is available and the GPU is NVIDIA GeForce RTX 5060 Laptop GPU.
- Final related regression tests had **41 passed**, and Ruff checks passed for the new code. The run included `tests/test_sep12.py`, `tests/test_automl.py`, `tests/test_splits.py`, `tests/test_submission.py`, and `tests/test_modality.py`.
- New tests cover forward/backward passes for every method, complete missing modalities under GPU AMP, voting/probability sums, timestamp alignment, cross-fold detector leakage, official path restoration, and the training → validation → checkpoint reload → submission flow.
- YOLO cropping was run on real `SM_test_0001`; the record is at `artifacts/sep12/verification/real_smoke.json`.
- The formal eight training jobs, Fold 4/full YOLO fine-tuning, full cropping, and competition submission have not yet run. This document does not fill in unexecuted accuracies or use old experiment results as new results.
