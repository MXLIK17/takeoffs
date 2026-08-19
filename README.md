# AI Takeoff — Neural Network Starter Kit

U-Net segmentation model for DAS10 architectural elevation drawings.

## Setup

```bash
conda env create -f environment.yml
conda activate takeoff-env
```

Or install manually:

```bash
conda create -n takeoff-env python=3.11
conda activate takeoff-env
conda install pytorch torchvision -c pytorch
conda install opencv matplotlib pillow numpy -c conda-forge
pip install albumentations segmentation-models-pytorch
```

For scale bar detection (`detect_scale_bar.py`), also install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).

## Project Structure

```
takeoff-nn/
  config.py       ← all settings (paths, classes, hyperparameters)
  dataset.py      ← loads image/mask pairs into PyTorch
  model.py        ← U-Net architecture + loss + optimizer
  train.py        ← training loop, run this to train
  evaluate.py     ← test accuracy, visualize predictions, colored inference output
  align_pairs.py  ← align annotated images to raw when crops differ
  detect_scale_bar.py  ← scale bar detection (Stage 2, optional)
  data/
    raw/          ← put unnannotated drawings here
    annotated/    ← put color-annotated drawings here (same filenames)
  outputs (after running) /
    checkpoints/  ← saved model weights
    predictions/  ← visualizations from evaluate.py
```

## Usage

**Step 1 — Add your data**
- Place raw drawings in `data/raw/`
- Place matching annotated drawings in `data/annotated/`
- Filenames must match exactly (e.g. `drawing_01.jpg` in both folders)
- See `data/README.md` for DAS10 data requirements (20–30 pairs recommended)

**Step 2 — Verify color map**
Run `python dataset.py` and check which classes are detected. If materials are
missing, tune `COLOR_HSV_RANGES` in `config.py` (this is what builds training
masks). `COLOR_TO_CLASS` and `CLASS_VIS_COLORS` are the DAS10 legend reference
colors used for inference output.

**Step 2b — Align pairs (if needed)**
If raw and annotated images have different crops or resolutions:
```bash
python align_pairs.py
```

**Step 3 — Test data loading**
```bash
python dataset.py
```

**Step 4 — Test model builds**
```bash
python model.py
```

**Step 5 — Train**
```bash
python train.py
```

**Step 6 — Evaluate**
```bash
python evaluate.py

# Or color in a single unannotated drawing:
python evaluate.py --image path/to/drawing.jpg
# Saves: {name}_colored_mask.png, {name}_colored_overlay.png, {name}_comparison.png
```

## Key Concepts

| Term | What it means |
|------|---------------|
| Epoch | One full pass through all training images |
| Loss | How wrong the model is — lower is better |
| IoU | Accuracy metric — 0 = wrong, 1 = perfect |
| Batch size | How many images processed at once |
| Encoder | Pre-trained feature extractor (ResNet34) |
| Decoder | Reconstructs spatial layout from features |
| Mask | Image where each pixel = a material class index |

## Adjusting for Your Hardware

**No GPU (CPU only):** Training will be slow (~10-30min/epoch).
Reduce `IMAGE_HEIGHT` and `IMAGE_WIDTH` to 256 in `config.py` to speed it up.

**With GPU:** Set `BATCH_SIZE = 4` or higher in `config.py` for faster training.

## Common Issues

**"No images found"** → Check filenames match between raw/ and annotated/ folders.

**"CUDA out of memory"** → Reduce `BATCH_SIZE` to 1 in `config.py`.

**IoU stuck near 0** → Check `COLOR_HSV_RANGES` in config.py matches your annotation colors; run `align_pairs.py` if crops differ.

**Training loss not decreasing** → Reduce `LEARNING_RATE` to `1e-5` in `config.py`.
