# AI-LesionNet — Multi-Sequence MRI Lesion Segmentation

**Research implementation based on the methodology described in the supplied manuscript**, *AI-LesionNet: Context-Uncertainty Coupled Multi-Sequence MRI Segmentation for Precision Multiple-Sclerosis Lesion Delineation*.

This repository is a fresh implementation based on the manuscript's descriptions, **not the original training source or pretrained model**. It contains source code for three-modality 3D segmentation, train/validation splitting, NIfTI preprocessing, transformer contextual fusion, Monte Carlo dropout uncertainty, decision-active postprocessing, quantitative evaluation, and smoke tests. Medical research software only; not validated or approved for diagnosis or clinical use.

## Features

- Inputs: **T1-weighted, T2-weighted, FLAIR** co-registered and skull-stripped NIfTI volumes; per-case lesion masks for training.
- Header-aware isotropic resampling, optional histogram equalization, nonzero-region z-score intensity normalization, 3D center crop/padding.
- Hierarchical 3D CNN encoder–decoder, modality-specific stems, 3D patch tokens, **6-layer transformer (8 heads, 256-dimensional embeddings)**, and cross-modality attention.
- BCE + Dice loss, Adam, cosine LR decay with warm-up, early stopping, AMP for NVIDIA GPUs, gradient clipping, randomized augmentation.
- **20 MC dropout passes**, voxel-level predictive variance, `alpha=0.7` uncertainty weighting, adaptive acceptance threshold, connected components, morphological cleanup.
- Native-space segmentation NIfTI, probability NIfTI, uncertainty NIfTI; Dice, precision, recall, voxel false-positive rate, lesion-level F1, HD95 metrics.
- Subject-disjoint 70/15/15 train/validation/test split.

## 1. Installation

Python 3.10+ is recommended. Use a compatible GPU build of PyTorch for your CUDA environment if training full-size 3D MRI scans.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run tests after installing development dependencies:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## 2. Prepare a dataset manifest

Obtain legally accessible, subject-level labelled brain MRI data (e.g., via the official ISBI 2015 or MSSEG-2 distribution and terms). **No patient data or dataset labels are included in this repository.**

Create `data/raw_cases.csv` with `subject_id,t1,t2,flair,mask`, as shown in [`examples/raw_cases.csv`](examples/raw_cases.csv). Paths can be absolute or relative to the manifest file. Repeated longitudinal scans must have the **same subject_id** to prevent train/test leakage.

```bash
python prepare_splits.py --input data/raw_cases.csv --output data/cases_split.csv --seed 42
```

This creates subject-disjoint 70/15/15 splits (rounded by subject count). With fewer than seven subjects, a split can be empty; collect enough unique subjects or assign appropriate splits manually. When subjects have unequal scan counts, the *scan percentages* may not be exactly 70/15/15.

## 3. NIfTI preprocessing

```bash
python preprocess.py --manifest data/cases_split.csv \
  --output data/processed --shape 32 256 256 --spacing 1.0
```

The output contains compressed `.npz` arrays (`image` shape `[3,D,H,W]` and `mask` shape `[D,H,W]`), per-case mapping metadata, and `data/processed/manifest.csv` for training. The channel order is T1, T2, FLAIR. Lesion masks are resampled with nearest-neighbor interpolation.

**Important preprocessing boundary:** All sequences and manual masks must be co-registered and skull-stripped *before* running this pipeline. Aligning NIfTI header grids is not equivalent to anatomical affine registration, and this repository does not silently claim to do skull stripping, bias correction, or full registration. Those upstream tasks require separate validated neuroimaging preprocessing (e.g., FSL/ANTs/SimpleITK). Use harmonization consistently across sites.

## 4. Train

The manuscript Table 5 settings are reflected in the defaults: 200 epochs, batch size 8, learning rate 1e-4, 10 warm-up epochs, 20-epoch patience. The training narrative also contains contradictory values (150 epochs, batch size 2). This implementation selects **Table 5** and documents the discrepancy instead of claiming both were used.

```bash
python train.py --manifest data/processed/manifest.csv \
  --output runs/ai-lesionnet --epochs 200 --batch-size 8
```

Full-resolution volumes consume considerable GPU memory. To experiment on modest hardware, start with a reduced preprocessing shape (divisible by eight), batch size 1, and a smaller model:

```bash
python train.py --manifest data/processed/manifest.csv \
  --output runs/debug --batch-size 1 --epochs 2 \
  --base-channels 8 --embed-dim 64 --heads 4 --layers 2
```

The best checkpoint is saved as `runs/ai-lesionnet/best.pt`; validation metrics are appended to `history.jsonl`. Training is needed to generate meaningful MRI predictions; using randomly initialized weights will not recreate manuscript results.

## 5. Predict on a case

```bash
python infer.py \
  --t1 /path/to/case_T1.nii.gz \
  --t2 /path/to/case_T2.nii.gz \
  --flair /path/to/case_FLAIR.nii.gz \
  --checkpoint runs/ai-lesionnet/best.pt \
  --output outputs/case_001 --iterations 20
```

Generated files:

- `outputs/case_001_mask.nii.gz` — binary lesion segmentation, restored to the native T1 grid.
- `outputs/case_001_probability.nii.gz` — mean probability before uncertainty-based refinement.
- `outputs/case_001_uncertainty.nii.gz` — voxel-level predictive variance.
- `outputs/case_001_run.json` — elapsed time and inference metadata.

Use **the same `--shape` and `--spacing` used at training time**. This implementation performs MC dropout passes serially, so do not assume the paper's reported 0.87 seconds per case; timing depends on GPU and preprocessing costs.

## 6. Evaluate independently on the held-out test set

Create a CSV with `subject_id,mask,prediction` referring to true masks and predicted native-grid masks (see `examples/evaluation_manifest.csv`).

```bash
python evaluate.py --manifest data/test_predictions.csv \
  --output evaluation/test_metrics.csv
```

The script writes case-level measurements and an aggregate `summary.json`. It **does not hardcode or generate** the manuscript's reported performance values. Run inference and evaluation on held-out images to obtain evidence-backed numbers. The lesion F1 here uses one-to-one connected-component overlap matching, and HD95 uses a 95th-percentile symmetric surface distance in mm; other manuscripts may use different matching conventions.

## Paper-to-code mapping

| Paper section | Implementation |
|---|---|
| 3.1 Hierarchical 3D encoder–decoder | `ailesionnet/model.py` (`ConvBlock`, `AILesionNet`) |
| 3.2 Transformer / cross-modality attention | `ailesionnet/model.py` (`ContextualFusion`) |
| 3.3 MC epistemic uncertainty / adaptive decisions | `predict_mc`, `ailesionnet/refinement.py` |
| 3.4 Training config / augmentation | `train.py`, `ailesionnet/data.py` |
| 4.1–4.2 Quantitative evaluation and ablations | `evaluate.py`, `ABLATIONS.md` and optional model/inference switches |
| 4.4 CCDataset MRI reconstruction experiment | **Not implemented**: a separate benchmark of other reconstruction models; it is not part of the lesion-segmentation model training pipeline. |

### Implementation differences and limitations

1. This reference implementation reconstructs the described mechanism, but the manuscript does not provide exact source code, original weights, patient-level split definitions, pretrained checkpoints, all geometric registration parameters, or a full specification for morphology/adaptive-threshold coefficients. Some parameters are therefore exposed as **documented, tunable choices** (`beta=0.15`, min component size 3 voxels, closing radius 1). They must be tuned on validation data, not the test set.
2. We implement attention on heavily downsampled 3D patch tokens (a memory-efficient global token transformer). The manuscript mentions windowed attention without sufficient details to uniquely reconstruct its particular window operation. This implementation should not be described as bit-for-bit identical to the authors' model.
3. No training has been performed on the manuscript's 4,567-scan cohort. Results such as Dice `0.913`, lesion-wise F1 `0.887`, FPR `0.051`, and inference time `0.87 s` are **claims of the supplied manuscript, not independently verified outputs of this code**.
4. The NIfTI preprocessing requires appropriately co-registered modality images; this repository does not distribute private/controlled clinical images.
5. `prepare_splits.py` and `preprocess.py` are reproducibility tools. To match an existing manuscript's results, the exact approved case list, acquisition-specific preprocessing, and a documented patient-level split must be recovered.

The code's default model currently has about **34.6 million** trainable parameters; the manuscript reports roughly 42 million, but does not specify every block in enough detail to match the exact number. These values should not be conflated.

## Reproducibility / citation

If you use this repo, cite the accompanying AI-LesionNet manuscript using its final publication metadata once available. Do not imply external validation or clinical readiness from passing the source-code tests.

**License:** MIT for the source implementation. MRI dataset use remains governed by the original dataset permissions.

For five component ablation configurations, see [`ABLATIONS.md`](ABLATIONS.md).
