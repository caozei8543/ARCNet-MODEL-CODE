import pathlib

# Root paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Training
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 60
BASE_LR = 1e-4
MIN_LR = 1e-7
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)

# Self-Correction Loop
NUM_CYCLES = 4          # K
MASK_START = 0.8        # 80% occlusion
MASK_END = 0.7          # 70% occlusion
GAUSS_NOISE_STD = 0.01  # small e for synthetic branch

# Logging / Checkpoints
LOG_INTERVAL = 50
CKPT_DIR = PROJECT_ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True, parents=True)