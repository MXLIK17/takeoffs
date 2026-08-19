"""
dataset.py — PyTorch Dataset for architectural elevation drawings
=================================================================
Uses HSV color range matching to handle JPEG compression artifacts
and anti-aliasing in DAS10's annotated drawings.
"""

import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import config


# ── Color → Mask Conversion (HSV-based) ───────────────────────────────────────

def rgb_mask_to_class_mask(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert a color-annotated image to a class index mask using HSV ranges.

    Why HSV instead of exact RGB?
      - JPEG compression shifts RGB values slightly (e.g. pure yellow 255,255,0
        becomes 251,247,52 after compression)
      - Annotations are semi-transparent overlays — colors blend with drawing lines
      - HSV separates hue (the "color") from brightness, making matching much
        more robust to these variations

    Args:
        rgb_image: (H, W, 3) numpy array, the annotated drawing

    Returns:
        class_mask: (H, W) numpy array, dtype int64, values 0..NUM_CLASSES-1
    """
    h, w = rgb_image.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)  # 0 = background

    # Convert to HSV — OpenCV uses H:0-180, S:0-255, V:0-255
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    for class_idx, ranges in config.COLOR_HSV_RANGES.items():
        h_center = ranges["h"] / 2   # convert 0-360 → 0-180 (OpenCV scale)
        h_tol    = ranges["h_tol"] / 2
        s_min    = ranges["s_min"]
        v_min    = ranges["v_min"]

        # Special case: black (very low value regardless of hue)
        if class_idx == 12:
            mask = (val < 30)
        else:
            h_lo = max(0,   h_center - h_tol)
            h_hi = min(180, h_center + h_tol)

            # Handle red hue wrap-around (hue near 0/180)
            if h_lo < 0:
                mask = ((hue >= 180 + h_lo) | (hue <= h_hi)) & (sat >= s_min) & (val >= v_min)
            else:
                mask = (hue >= h_lo) & (hue <= h_hi) & (sat >= s_min) & (val >= v_min)

        class_mask[mask] = class_idx

    return class_mask


def verify_mask(mask: np.ndarray, filename: str):
    """Print which classes were found in a mask — useful for debugging."""
    unique = np.unique(mask)
    found = [config.CLASSES[i] for i in unique if i < len(config.CLASSES)]
    print(f"  [{filename}] classes found: {found}")


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-5, 5), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class ElevationDataset(Dataset):
    """
    Loads raw/annotated elevation drawing pairs.

    Folder structure:
        data/raw/        ← unannotated JPEGs (e.g. drawing_01.jpg)
        data/annotated/  ← color-annotated JPEGs (same filename)
    """

    def __init__(self, raw_dir: Path, annotated_dir: Path, transform=None):
        self.raw_dir       = Path(raw_dir)
        self.annotated_dir = Path(annotated_dir)
        self.transform     = transform

        valid_ext = {'.jpg', '.jpeg', '.png'}
        self.filenames = sorted([
            f.name for f in self.raw_dir.iterdir()
            if f.suffix.lower() in valid_ext
        ])

        if not self.filenames:
            print(f"⚠️  No images found in {raw_dir}")
            return

        print(f"[Dataset] {len(self.filenames)} images found")

        # Warn about missing annotation pairs
        for fname in self.filenames:
            if not (self.annotated_dir / fname).exists():
                print(f"  ⚠️  No annotation for: {fname}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load raw image
        image = np.array(Image.open(self.raw_dir / fname).convert("RGB"))

        # Load annotated image and convert to class mask
        annotation = np.array(Image.open(self.annotated_dir / fname).convert("RGB"))

        # Resize annotation to match raw if sizes differ
        if image.shape[:2] != annotation.shape[:2]:
            h, w = image.shape[:2]
            annotation = cv2.resize(annotation, (w, h), interpolation=cv2.INTER_NEAREST)

        mask = rgb_mask_to_class_mask(annotation)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        return image, mask.long()

    def load_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load raw image and class mask without applying transforms."""
        fname = self.filenames[idx]

        image = np.array(Image.open(self.raw_dir / fname).convert("RGB"))
        annotation = np.array(Image.open(self.annotated_dir / fname).convert("RGB"))

        if image.shape[:2] != annotation.shape[:2]:
            h, w = image.shape[:2]
            annotation = cv2.resize(annotation, (w, h), interpolation=cv2.INTER_NEAREST)

        mask = rgb_mask_to_class_mask(annotation)
        return image, mask


class TransformSubset(Dataset):
    """
    Wraps a subset of ElevationDataset with its own transform.

    random_split/Subset share one underlying dataset — assigning transforms
    on .dataset would overwrite the other split. This wrapper keeps train
    augmentation and val transforms independent.
    """

    def __init__(self, base_dataset: ElevationDataset, indices: list, transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, mask = self.base_dataset.load_pair(self.indices[idx])

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders():
    full_dataset = ElevationDataset(
        raw_dir       = config.RAW_DIR,
        annotated_dir = config.ANNOTATED_DIR,
    )

    if not full_dataset.filenames:
        print(f"\n📂 Add raw drawings to:       {config.RAW_DIR}")
        print(f"   Add annotated drawings to: {config.ANNOTATED_DIR}")
        return None, None

    n_total = len(full_dataset)
    n_train = max(1, int(n_total * config.TRAIN_SPLIT))
    n_val   = max(1, n_total - n_train)

    # If only 1 image, use it for both train and val (POC mode)
    if n_total == 1:
        print("⚠️  Only 1 image — using it for both train and val (POC mode)")
        train_indices = [0]
        val_indices = [0]
    else:
        train_split, val_split = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        train_indices = train_split.indices
        val_indices = val_split.indices

    train_subset = TransformSubset(full_dataset, train_indices, get_train_transforms())
    val_subset = TransformSubset(full_dataset, val_indices, get_val_transforms())

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"[DataLoader] Train: {n_train} | Val: {n_val}")
    return train_loader, val_loader


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test color extraction directly on your uploaded images.
    Run: python dataset.py
    """
    import sys

    # Test color mask on the annotated sample if provided
    test_annotated = "data/annotated/P05.jpg"  # update path as needed
    if Path(test_annotated).exists():
        print(f"Testing color extraction on {test_annotated}...")
        img = np.array(Image.open(test_annotated).convert("RGB"))
        mask = rgb_mask_to_class_mask(img)
        verify_mask(mask, test_annotated)

        unique, counts = np.unique(mask, return_counts=True)
        print("\nPixel counts per class:")
        for cls_idx, count in zip(unique, counts):
            pct = count / mask.size * 100
            name = config.CLASSES[cls_idx] if cls_idx < len(config.CLASSES) else "unknown"
            print(f"  [{cls_idx}] {name:<35} {count:>10,} px  ({pct:.1f}%)")
    else:
        print("No test image found — add your annotated drawing to data/annotated/")

    # Test dataloader
    train_loader, val_loader = get_dataloaders()
    if train_loader:
        images, masks = next(iter(train_loader))
        print(f"\n✅ Batch loaded: images {images.shape} | masks {masks.shape}")
        print(f"   Classes in batch: {[config.CLASSES[i] for i in masks.unique().tolist()]}")
