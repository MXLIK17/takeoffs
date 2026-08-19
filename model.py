"""
model.py — U-Net segmentation model for elevation drawings
==========================================================
We use the segmentation_models_pytorch library which gives us
a production-quality U-Net with a pre-trained encoder in ~5 lines.

Architecture overview:
  Input image (3, H, W)
       ↓
  Encoder (ResNet34) — extracts visual features at multiple scales
       ↓
  Decoder — reconstructs spatial layout, uses encoder features via skip connections
       ↓
  Segmentation head — outputs one score per class per pixel
       ↓
  Output mask (NUM_CLASSES, H, W)

The encoder starts with ImageNet weights — it already knows what edges,
textures, and shapes look like. We only need to teach it what material
regions in architectural drawings look like.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import config


def build_model() -> nn.Module:
    """
    Build and return the U-Net model.

    segmentation_models_pytorch handles all the architecture complexity.
    You just specify what encoder to use and how many output classes.
    """
    model = smp.Unet(
        encoder_name    = config.ENCODER_NAME,      # resnet34
        encoder_weights = config.ENCODER_WEIGHTS,   # 'imagenet'
        in_channels     = 3,                        # RGB input
        classes         = config.NUM_CLASSES,       # one output channel per material class
        activation      = None,                     # raw logits — loss function handles this
    )

    model = model.to(config.DEVICE)

    # Print a summary
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] U-Net with {config.ENCODER_NAME} encoder")
    print(f"[Model] Trainable parameters: {n_params:,}")
    print(f"[Model] Output classes: {config.NUM_CLASSES} ({', '.join(config.CLASSES)})")
    print(f"[Model] Running on: {config.DEVICE.upper()}")

    return model


def build_loss() -> nn.Module:
    class_weights = torch.tensor(
        config.CLASS_WEIGHTS, dtype=torch.float32
    ).to(config.DEVICE)
    ce_loss   = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    class CombinedLoss(nn.Module):
        def forward(self, predictions, targets):
            return config.CE_LOSS_WEIGHT * ce_loss(predictions, targets) + \
                   config.DICE_LOSS_WEIGHT * dice_loss(predictions, targets)

    return CombinedLoss()


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    Adam optimizer — the standard choice for segmentation tasks.
    Weight decay adds regularization to prevent overfitting on small datasets.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )


def build_scheduler(optimizer):
    """
    Learning rate scheduler — reduces LR when validation loss plateaus.
    Helps squeeze out extra accuracy in later training epochs.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",   # we want loss to go DOWN
        factor   = 0.5,     # multiply LR by 0.5 when plateauing
        patience = 5,       # wait 5 epochs before reducing
        verbose  = True,
    )


def compute_iou(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute mean Intersection over Union (mIoU) — the standard
    accuracy metric for segmentation.

    IoU per class = (predicted ∩ true) / (predicted ∪ true)
    mIoU = average IoU across all classes

    A score of 1.0 = perfect; 0.0 = completely wrong.
    For a POC, aim for >0.5 on your validation set.

    Args:
        predictions: (B, NUM_CLASSES, H, W) raw logits from the model
        targets:     (B, H, W) ground truth class indices

    Returns:
        mean IoU as a float
    """
    # Convert logits to class predictions
    pred_classes = predictions.argmax(dim=1)  # (B, H, W)

    iou_per_class = []
    for cls in range(config.NUM_CLASSES):
        pred_mask = (pred_classes == cls)
        true_mask = (targets == cls)

        intersection = (pred_mask & true_mask).sum().float()
        union        = (pred_mask | true_mask).sum().float()

        if union == 0:
            # Class not present in this batch — skip it
            continue

        iou_per_class.append((intersection / union).item())

    return sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0.0


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Run this file directly to verify the model builds and runs a forward pass."""
    model = build_model()

    # Simulate one batch of images, using actual config values
    test_batch_size = config.BATCH_SIZE
    dummy_images = torch.randn(
        test_batch_size, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH
    ).to(config.DEVICE)

    with torch.no_grad():
        output = model(dummy_images)

    print(f"\n✅ Forward pass successful!")
    print(f"   Input shape:  {dummy_images.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Each pixel gets {config.NUM_CLASSES} scores, one per material class.")

    # Also sanity-check the loss function builds and runs on dummy data
    loss_fn = build_loss()
    dummy_targets = torch.randint(
        0, config.NUM_CLASSES, (test_batch_size, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    ).to(config.DEVICE)
    loss_value = loss_fn(output, dummy_targets)
    print(f"\n✅ Loss function check passed! Combined loss value: {loss_value.item():.4f}")
    print(f"   (CE weight: {config.CE_LOSS_WEIGHT} | Dice weight: {config.DICE_LOSS_WEIGHT})")
