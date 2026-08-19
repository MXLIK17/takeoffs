"""
evaluate.py — Test and visualize trained model predictions
===========================================================
Run this after training to see how well the model performs.

What it does:
  1. Loads the best saved model checkpoint
  2. Runs it on validation images
  3. Saves side-by-side comparisons: raw | predicted mask | true mask
  4. Prints per-class IoU so you know which materials need more work

Run with:
  python evaluate.py
  python evaluate.py --image path/to/drawing.jpg   (run on a single image)
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from model import build_model, build_optimizer, build_loss, compute_iou
from dataset import get_dataloaders, rgb_mask_to_class_mask, get_val_transforms


def class_mask_to_rgb(class_mask: np.ndarray) -> np.ndarray:
    """Convert a (H, W) class index mask to an (H, W, 3) RGB image using DAS10 legend colors."""
    h, w = class_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(config.CLASS_VIS_COLORS):
        if cls_idx >= config.NUM_CLASSES:
            break
        rgb[class_mask == cls_idx] = color
    return rgb


def overlay_mask_on_image(image: np.ndarray, class_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend colored material mask over the raw drawing for a DAS10-style annotated look."""
    mask_rgb = class_mask_to_rgb(class_mask).astype(np.float32) / 255.0
    base = image.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0

    material = class_mask > 0
    blended = base.copy()
    blended[material] = (
        (1 - alpha) * base[material] + alpha * mask_rgb[material]
    )
    return (blended.clip(0, 1) * 255).astype(np.uint8)


def build_legend():
    """Build a matplotlib legend showing class → color mapping."""
    patches = []
    for i, cls in enumerate(config.CLASSES):
        if i >= len(config.CLASS_VIS_COLORS):
            break
        color = [c / 255.0 for c in config.CLASS_VIS_COLORS[i]]
        patches.append(mpatches.Patch(color=color, label=cls))
    return patches


# ── Per-class IoU ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_per_class_iou(model, loader, device) -> dict:
    """
    Compute IoU separately for each class across the entire dataset.
    This tells you which materials your model handles well vs poorly.

    Returns dict: { class_name: iou_score }
    """
    model.eval()

    intersection_per_class = torch.zeros(config.NUM_CLASSES)
    union_per_class        = torch.zeros(config.NUM_CLASSES)

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        predictions  = model(images)
        pred_classes = predictions.argmax(dim=1)

        for cls in range(config.NUM_CLASSES):
            pred_mask = (pred_classes == cls).cpu()
            true_mask = (masks == cls).cpu()
            intersection_per_class[cls] += (pred_mask & true_mask).sum()
            union_per_class[cls]        += (pred_mask | true_mask).sum()

    results = {}
    for cls in range(config.NUM_CLASSES):
        if union_per_class[cls] > 0:
            iou = (intersection_per_class[cls] / union_per_class[cls]).item()
            results[config.CLASSES[cls]] = round(iou, 4)
        else:
            results[config.CLASSES[cls]] = None  # class not present in dataset

    return results


# ── Visualize Predictions ─────────────────────────────────────────────────────

@torch.no_grad()
def visualize_predictions(model, loader, device, n_samples: int = 50):
    """
    Save side-by-side comparison images:
      [ Raw drawing | Predicted mask | Ground truth mask ]

    Saved to outputs/predictions/
    """
    model.eval()
    pred_dir = config.OUTPUT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True)

    saved = 0
    for images, masks in loader:
        if saved >= n_samples:
            break

        images_dev = images.to(device)
        predictions = model(images_dev)
        pred_classes = predictions.argmax(dim=1).cpu()

        for i in range(images.shape[0]):
            if saved >= n_samples:
                break

            # De-normalize the image for display
            img_np = images[i].permute(1, 2, 0).numpy()
            mean = np.array(config.MEAN)
            std  = np.array(config.STD)
            img_np = (img_np * std + mean).clip(0, 1)

            pred_rgb = class_mask_to_rgb(pred_classes[i].numpy())
            true_rgb = class_mask_to_rgb(masks[i].numpy())

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            axes[0].imshow(img_np, interpolation="nearest");       axes[0].set_title("Raw Drawing");         axes[0].axis("off")
            axes[1].imshow(pred_rgb, interpolation="nearest");     axes[1].set_title("Predicted Mask");      axes[1].axis("off")
            axes[2].imshow(true_rgb, interpolation="nearest");     axes[2].set_title("Ground Truth Mask");   axes[2].axis("off")

            # Add legend
            legend = build_legend()
            fig.legend(handles=legend, loc="lower center", ncol=5,
                      bbox_to_anchor=(0.5, -0.05), fontsize=8)

            out_path = pred_dir / f"prediction_{saved+1:03d}.png"
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close()
            print(f"  Saved → {out_path}")
            saved += 1


# ── Single Image Inference ────────────────────────────────────────────────────

@torch.no_grad()
def predict_single_image(model, image_path: str, device):
    """
    Run the model on a single raw drawing and save the predicted mask.
    Useful for testing on new drawings outside the dataset.
    """
    transform = get_val_transforms()

    img = np.array(Image.open(image_path).convert("RGB"))
    augmented = transform(image=img)
    tensor = augmented["image"].unsqueeze(0).to(device)  # add batch dim

    model.eval()
    output = model(tensor)
    pred_class = output.argmax(dim=1).squeeze(0).cpu().numpy()
    pred_rgb = class_mask_to_rgb(pred_class)

    # De-normalize for display
    img_display = augmented["image"].permute(1, 2, 0).numpy()
    img_display = (img_display * np.array(config.STD) + np.array(config.MEAN)).clip(0, 1)
    overlay_rgb = overlay_mask_on_image(img_display, pred_class)

    pred_dir = config.OUTPUT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True)
    stem = Path(image_path).stem

    # Standalone colored outputs (DAS10 legend colors)
    mask_only_path = pred_dir / f"{stem}_colored_mask.png"
    overlay_path = pred_dir / f"{stem}_colored_overlay.png"
    Image.fromarray(pred_rgb).save(mask_only_path)
    Image.fromarray(overlay_rgb).save(overlay_path)
    print(f"✅ Colored mask saved → {mask_only_path}")
    print(f"✅ Colored overlay saved → {overlay_path}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(img_display);   axes[0].set_title("Input Drawing");      axes[0].axis("off")
    axes[1].imshow(pred_rgb, interpolation="nearest");      axes[1].set_title("Predicted Mask");      axes[1].axis("off")
    axes[2].imshow(overlay_rgb);   axes[2].set_title("Colored Overlay");     axes[2].axis("off")
    fig.legend(handles=build_legend(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05), fontsize=8)

    comparison_path = pred_dir / f"{stem}_comparison.png"
    plt.tight_layout()
    plt.savefig(comparison_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✅ Comparison saved → {comparison_path}")

    # Print detected classes
    unique_classes = np.unique(pred_class)
    print("\nDetected materials:")
    for cls_idx in unique_classes:
        if cls_idx < len(config.CLASSES):
            pixel_count = (pred_class == cls_idx).sum()
            print(f"  {config.CLASSES[cls_idx]}: {pixel_count:,} pixels")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained U-Net model")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image for inference")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {config.BEST_MODEL_PATH}...")
    if not config.BEST_MODEL_PATH.exists():
        print("⛔ No trained model found. Run train.py first.")
        return

    model     = build_model()
    optimizer = build_optimizer(model)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    print(f"✅ Model loaded (trained to epoch {checkpoint['epoch']+1}, "
          f"val loss: {checkpoint['val_loss']:.4f})")

    # Single image mode
    if args.image:
        predict_single_image(model, args.image, config.DEVICE)
        return

    # Full evaluation mode
    _, val_loader = get_dataloaders()
    if val_loader is None:
        return

    print("\n── Per-class IoU ─────────────────────────────────")
    iou_scores = compute_per_class_iou(model, val_loader, config.DEVICE)
    valid_scores = {k: v for k, v in iou_scores.items() if v is not None}
    mean_iou = sum(valid_scores.values()) / len(valid_scores) if valid_scores else 0

    for cls_name, score in iou_scores.items():
        if score is None:
            print(f"  {cls_name:<25} — (not present in dataset)")
        else:
            bar = "█" * int(score * 20)
            print(f"  {cls_name:<25} {score:.4f}  {bar}")

    print(f"\n  Mean IoU: {mean_iou:.4f}")
    if mean_iou < 0.3:
        print("  → Model needs more training data or epochs.")
    elif mean_iou < 0.6:
        print("  → Decent POC result. Tune augmentations and train longer.")
    else:
        print("  → Good result for a POC!")

    print("\n── Saving prediction visualizations ──────────────")
    visualize_predictions(model, val_loader, config.DEVICE, n_samples=50)
    print(f"\nDone. Check {config.OUTPUT_DIR / 'predictions'} for output images.")


if __name__ == "__main__":
    main()