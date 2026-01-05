# ARCNet-MODEL-CODE
# ARC-Net (Reference Skeleton)

This is a minimal reference implementation of ARC-Net with:
- Idealized Response Estimator (IRE) placeholder (`SimpleRetinexFormer`)
- Artifact Rectification Module (ARM) placeholder (`SimpleNAFNet`)
- Self-Correction Loop (residual extraction + proxy contamination)
- Stochastic Information Occlusion (binary masking)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout
```
data/
  paired/
    low/*.png   # synthetic low-light
    gt/*.png    # corresponding ground truth
  unpaired/
    real/*.png  # real low-light (unpaired)
```

## Train
```bash
python -m src.engine.train_arcnet \
  --paired_root data/paired \
  --unpaired_root data/unpaired \
  --size 256 256 \
  --device cuda
```

Checkpoints are stored in `checkpoints/`.

## Notes
- Replace `SimpleRetinexFormer` with your pretrained RetinexFormer weights.
- Replace `SimpleNAFNet` with your full NAFNet blocks for best performance.
- Adjust occlusion ratios, cycles (NUM_CYCLES), and learning rate schedule in `src/config.py`.
