"""
config.py — Central settings for the AI Takeoff project
========================================================
All paths, hyperparameters, and class definitions live here.
Change things here rather than hunting through multiple files.
"""

import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"          # unannotated drawings (model input)
ANNOTATED_DIR   = DATA_DIR / "annotated"    # color-highlighted drawings (ground truth masks)
OUTPUT_DIR      = ROOT_DIR / "outputs"
CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
LOG_DIR         = OUTPUT_DIR / "logs"

for d in [RAW_DIR, ANNOTATED_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Material Classes ──────────────────────────────────────────────────────────
# Sampled directly from DAS10's color legend (2026-06-06).
# Index 0 = background (uncolored areas).

CLASSES = [
    "background",                       # 0
    "aluminum_cladding",                # 1  — yellow        #FBF734
    "beams_ornaments_mouldings",        # 2  — purple        #8550FF
    "vinyl_trim",                       # 3  — pink-purple   #DF63F5
    "downpipes",                        # 4  — blue          #254AFB
    "hardie_pvc_trim_orange",           # 5  — orange        #EB8032
    "red",                              # 6  — red           #EA3F2D
    "mac_coil",                         # 7  — light purple  #817CFA
    "hardie_pvc_trim_beige",            # 8  — beige         #F5E1C9
    "hardie_pvc_trim_brown",            # 9  — brown-orange  #CA6E2D
    "light_pink",                       # 10 — light pink    #F1BEFF
    "mac_siding",                       # 11 — teal          #61BEC5
    "black_element",                    # 12 — black         #000000
    "green",                            # 13 — green         #82F118
    # Note: White (#FFFEFF) is treated as background
]

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

# ── Color Map ─────────────────────────────────────────────────────────────────
# Sampled from DAS10 legend screenshot (2026-06-06).
# Format: (R, G, B) : class_index
# These are center-pixel samples — dataset.py uses HSV range matching
# to handle anti-aliasing and JPEG compression variation around these values.

COLOR_TO_CLASS = {
    (251, 247,  52): 1,   # aluminum cladding     — yellow
    (133,  80, 255): 2,   # beams & ornaments     — purple
    (223,  99, 245): 3,   # vinyl trim            — pink-purple
    ( 37,  74, 251): 4,   # downpipes             — blue
    (235, 128,  50): 5,   # hardie/pvc trim       — orange
    (234,  63,  45): 6,   # red
    (129, 124, 250): 7,   # mac coil              — light purple
    (245, 225, 201): 8,   # hardie/pvc trim       — beige
    (202, 110,  45): 9,   # hardie/pvc trim       — brown-orange
    (241, 190, 255): 10,  # light pink
    ( 97, 190, 197): 11,  # mac siding            — teal
    (  0,   0,   0): 12,  # black element
    (130, 241,  24): 13,  # green
}

# RGB colors for inference output and visualizations (index = class index).
# Background is white (uncolored drawing areas); material classes use DAS10 legend.
_idx_to_rgb = {idx: list(rgb) for rgb, idx in COLOR_TO_CLASS.items()}
CLASS_VIS_COLORS = [[255, 255, 255]]  # 0 — background
for class_idx in range(1, NUM_CLASSES):
    CLASS_VIS_COLORS.append(_idx_to_rgb.get(class_idx, [128, 128, 128]))

# HSV ranges used by dataset.py to build training masks from annotated drawings.
# Tune these if dataset.py reports missing classes — not used at inference time.
COLOR_HSV_RANGES = {
    1:  {"h": 60,  "h_tol": 8,  "s_min": 150, "v_min": 180},  # yellow
    2:  {"h": 270, "h_tol": 15, "s_min": 100, "v_min": 80 },  # purple
    3:  {"h": 290, "h_tol": 15, "s_min": 80,  "v_min": 150},  # pink-purple
    4:  {"h": 230, "h_tol": 15, "s_min": 150, "v_min": 80 },  # blue
    5:  {"h": 28,  "h_tol": 12, "s_min": 120, "v_min": 150},  # orange
    6:  {"h": 5,   "h_tol": 10, "s_min": 150, "v_min": 150},  # red
    7:  {"h": 245, "h_tol": 15, "s_min": 60,  "v_min": 150},  # light purple
    8:  {"h": 30,  "h_tol": 15, "s_min": 60,  "v_min": 200},  # beige
    9:  {"h": 25,  "h_tol": 10, "s_min": 100, "v_min": 100},  # brown-orange
    10: {"h": 300, "h_tol": 15, "s_min": 30,  "v_min": 200},  # light pink
    11: {"h": 185, "h_tol": 15, "s_min": 80,  "v_min": 120},  # teal
    12: {"h": 0,   "h_tol": 0,  "s_min": 0,   "v_min": 0  },  # black (special case)
    13: {"h": 90,  "h_tol": 15, "s_min": 150, "v_min": 100},  # green
}


# ── Class Weights ─────────────────────────────────────────────────────────────
# Computed from real pixel counts across the 16-image dataset (2026-06-17).
# Background dominates at 96.4% of pixels, so it's down-weighted heavily.
# Material classes are up-weighted so the loss actually penalizes missing them.
# Re-run scripts/compute_class_weights.py whenever the dataset grows meaningfully
# (e.g. after adding the new 400-image batch) — weights will shift.
CLASS_WEIGHTS = [
    0.300,   # background
    0.887,   # aluminum_cladding
    1.149,   # beams_ornaments_mouldings
    1.149,   # vinyl_trim
    1.149,   # downpipes
    1.149,   # hardie_pvc_trim_orange
    1.149,   # red
    1.149,   # mac_coil
    1.149,   # hardie_pvc_trim_beige
    1.149,   # hardie_pvc_trim_brown
    1.091,   # light_pink
    1.149,   # mac_siding
    0.528,   # black_element
    1.149,   # green
]

# Weight given to Dice loss vs CrossEntropy in the combined loss (model.py).
# Raised from 0.5 to 0.8 — Dice loss handles severe class imbalance much
# better than CE alone, which is critical given background is 96%+ of pixels.
DICE_LOSS_WEIGHT = 0.8
CE_LOSS_WEIGHT   = 1.0 - DICE_LOSS_WEIGHT


# ── Image Settings ────────────────────────────────────────────────────────────

IMAGE_HEIGHT = 1024
IMAGE_WIDTH  = 1024
# Raised from 512 → 1024. At 512, fine trim/soffit lines on these 5100x3300
# architectural sheets were getting crushed to a few pixels — visible as
# blocky/pixelated output and contributing to class collapse (thin material
# regions had almost no pixels left for the model to learn from).
# If you hit CUDA out-of-memory errors, drop BATCH_SIZE before lowering this.

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Training Hyperparameters ──────────────────────────────────────────────────

BATCH_SIZE     = 1
# Lowered from 2 → 1 to compensate for IMAGE_HEIGHT/WIDTH increasing to 1024.
# Memory usage scales with image_size^2 * batch_size — doubling resolution
# roughly quadruples memory use, so batch size must come down to compensate.
# If training on GPU with room to spare, try BATCH_SIZE=2 at 1024px first.
NUM_EPOCHS     = 50
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
TRAIN_SPLIT    = 0.8
SAVE_EVERY     = 5
EARLY_STOPPING_PATIENCE = 10


# ── Model Settings ────────────────────────────────────────────────────────────

ENCODER_NAME    = "resnet34"
ENCODER_WEIGHTS = "imagenet"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"


# ── Device ────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
