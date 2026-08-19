# Training Data — DAS10 Elevation Drawings

Place matched **raw** and **annotated** image pairs here to train the segmentation model.

## Folder layout

```
data/
  raw/         ← unannotated elevation drawings (model input at inference)
  annotated/   ← DAS10 color-shaded versions (ground truth for training)
```

## Requirements (ask DAS10)

Request **20–30 matched pairs** for a credible POC (5–10 minimum for a rough demo).

Each pair must have:

1. **Identical filename** in both folders (including extension)
   - Good: `raw/P05.jpg` + `annotated/P05.jpg`
   - Bad:  `raw/P05.png` + `annotated/P05.jpg`
2. **Same resolution and crop** (same page region)
3. **Elevation views only** (not floor plans)
4. **Legend colors** matching `COLOR_HSV_RANGES` in `config.py`

## Validate before training

```bash
conda activate takeoff-env
python dataset.py
```

Check the printed "classes found" and pixel counts. If classes are missing, tune `COLOR_HSV_RANGES` in `config.py`.

## If raw and annotated crops differ

Run alignment before training:

```bash
python align_pairs.py
```

This warps annotated images to match raw dimensions using feature matching and saves aligned copies (originals are not overwritten).
