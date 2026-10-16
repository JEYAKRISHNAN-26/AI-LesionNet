# Reproducing the five section-4.2 ablation configurations

Run each configuration using the **same patient-level train/validation/test manifest**, random seed, and preprocessing volume shape. Keep all other hyperparameters identical. Each row below describes actual code switches; it does not substitute for retraining and measuring the results.

| ID | Contextual transformer | MC uncertainty refinement | Intensity equalization | Training | Inference |
|---|---|---|---|---|---|
| (a) 3D CNN baseline | off | off | off | `train.py --disable-fusion` | `infer.py --disable-refinement --no-equalize` |
| (b) + transformer | on | off | off | default | `infer.py --disable-refinement --no-equalize` |
| (c) + uncertainty | off | on | off | `train.py --disable-fusion` | `infer.py --no-equalize` |
| (d) + transformer + uncertainty | on | on | off | default | `infer.py --no-equalize` |
| (e) full AI-LesionNet | on | on | on | default | default |

To disable intensity equalization for (a)–(d), preprocess the SAME raw scans once into a separate output directory with `python preprocess.py --manifest data/cases_split.csv --output data/processed_no_eq --no-equalize`; use that cached manifest for training. The full model uses `data/processed/manifest.csv` generated with default equalization. Importantly, the paper's 'harmonization' also mentions prior co-registration, skull stripping and bias correction. Those externally performed steps should be consistent for all variants rather than unintentionally changing the dataset. The switch here isolates the **in-code histogram equalization** component only.

Example commands (replace source files/paths):

```bash
# CNN-only baseline, without in-code equalization
python train.py --manifest data/processed_no_eq/manifest.csv --output runs/ablation_a --disable-fusion --batch-size 1
python infer.py --t1 YOUR_T1.nii.gz --t2 YOUR_T2.nii.gz --flair YOUR_FLAIR.nii.gz \
  --checkpoint runs/ablation_a/best.pt --output outputs/ablation_a --disable-refinement --no-equalize

# Full model, with in-code equalization and MC-based uncertainty
python train.py --manifest data/processed/manifest.csv --output runs/ablation_e --batch-size 1
python infer.py --t1 YOUR_T1.nii.gz --t2 YOUR_T2.nii.gz --flair YOUR_FLAIR.nii.gz \
  --checkpoint runs/ablation_e/best.pt --output outputs/ablation_e --iterations 20
```

Use `evaluate.py` on predictions of **all** held-out subjects, and calculate summary metrics without leaking subject or lesion-mask information into model-selection steps. Do not populate paper tables with example/test figures.
