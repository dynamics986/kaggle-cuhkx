# Kaggle Competition Strategy & Principles Checklist

> **Mission Objective:** Win the competition by building a highly generalizable solution, mastering local cross-validation, and producing top-tier technical documentation for final verification and presentation.

---

## I. Core Philosophy: Trust Your CV, Not the LB

The Public Leaderboard (LB) evaluates only a small fraction of the test dataset and can lead to severe over-fitting ("Shake-up"). Our iterations must be guided by a robust, local validation scheme.

* **100% CV-Driven Iteration:** Every code change, feature addition, model architecture swap, or hyperparameter tweak **must** be validated against our local Cross-Validation (CV) metric before considering a submission.
* **Zero Leakage Policy:** Strictly separate training and validation splits. Apply scaling, target encoding, and data augmentations *inside* each CV fold to prevent data leakage.
* **Distribution Alignment:** Align the local split scheme (e.g., `StratifiedKFold`, `GroupKFold`, or `StratifiedGroupKFold`) strictly with the dataset's characteristics and Kaggle’s evaluation protocol to mirror the Private Test Set distribution.
* **Ensemble Strategy (OOF):** Save model weights and Out-Of-Fold (OOF) predictions for all $K$ folds. Use fold-averaging for test set predictions to maximize generalization and stability.

---

## II. Winning Roadmap: Documenting for the Final Audit

Winning requires passing the post-competition code audit, reproducing results cleanly, and presenting a winning solution. We design our codebase to answer the three core questions expected in the final **Solution Overview** and **Live Presentation**:

```
           ┌───────────────────────────────────────────────────────────┐
           │                   SOLUTION ARCHITECTURE                   │
           └──────────────┬─────────────────────────────┬──────────────┘
                          │                             │
    ┌─────────────────────▼─────┐         ┌─────────────▼─────────────┐
    │  1. Model Architectures   │         │ 2. Augmentations & FE     │
    │  • Backbone selection     │         │ • Domain-specific transforms│
    │  • Temporal modeling      │         │ • Feature extraction      │
    └─────────────────────┬─────┘         └─────────────┬─────────────┘
                          │                             │
                          └──────────────┬──────────────┘
                                         │
                           ┌─────────────▼─────────────┐
                           │   3. Generalization Proof │
                           │   • CV vs. LB correlation │
                           │   • Ablation studies      │
                           └───────────────────────────┘

```

### 1. Model Architecture Design

* **Multi-Model Diversity:** Combine structurally different backbones (e.g., 3D CNNs like *SlowFast* / *X3D* paired with Transformers like *Video Swin Transformer* / *Timesformer*) to capture distinct spatial-temporal patterns.
* **Modular Pipeline:** Keep model definitions clean and configurable so changing backbones requires minimal code refactoring.

### 2. Augmentations & Feature Engineering

* **Domain-Tailored Augmentations:** Implement augmentations designed for the modality (e.g., Mixup/CutMix, spatial cropping, frame sampling strategies, color jittering) to force the model to learn robust features rather than memorizing background details.
* **Feature Extraction Quality:** Document every feature engineering choice with rationale and benchmark its impact on the local CV score.

### 3. Generalization Proof & Ablation Studies

* **Ablation Logging:** Track every experiment meticulously (using tools like W&B or MLflow). We must be able to prove *why* our pipeline works through systematic ablation studies (e.g., "Feature X improved 5-fold CV by +0.008").
* **Reproducibility Guarantee:** Maintain deterministic seeds, clean requirements files, and end-to-end inference scripts so our solution can be re-run seamlessly during official verification.

---

## III. Submission Rules of Engagement

1. **Rule of 2 Submissions:**
* **Submission A (Safe Pick):** The solution with the **highest local 5-fold CV score** and most conservative pipeline.
* **Submission B (Aggressive Pick):** The solution with high CV *and* the top Public LB score, provided its pipeline remains logical and non-overfitted.


2. **Never Overfit the Public LB:** Never tune hyperparameters solely to push the Public LB score up by a tiny fraction if it hurts or stagnates our local CV score.