"""
train.py — Training loop for the U-Net segmentation model
==========================================================
This is the core script you run to train your neural network.

What happens each epoch:
  1. Train loop  — model sees training images, makes predictions,
                   computes loss, updates weights via backpropagation
  2. Val loop    — model sees validation images (no weight updates),
                   we measure how well it generalizes
  3. Logging     — print loss and IoU for both splits
  4. Checkpointing — save the model whenever validation improves
  5. Early stopping — stop if no improvement for N epochs

Run with:
  python train.py
"""

import time
import torch
import numpy as np
from pathlib import Path

import config
from dataset import get_dataloaders
from model import build_model, build_loss, build_optimizer, build_scheduler, compute_iou


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, device) -> tuple:
    """
    Run one full pass through the training data.

    Returns:
        avg_loss: average loss across all batches
        avg_iou:  average mIoU across all batches
    """
    model.train()  # puts model in training mode (enables dropout, batch norm)

    total_loss = 0.0
    total_iou  = 0.0
    n_batches  = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        # ── Forward pass ──────────────────────────────────────────────────────
        predictions = model(images)              # (B, NUM_CLASSES, H, W)
        loss = loss_fn(predictions, masks)

        # ── Backward pass (the learning step) ─────────────────────────────────
        optimizer.zero_grad()   # clear gradients from previous step
        loss.backward()         # compute gradients
        optimizer.step()        # update model weights

        # ── Track metrics ──────────────────────────────────────────────────────
        with torch.no_grad():
            iou = compute_iou(predictions, masks)

        total_loss += loss.item()
        total_iou  += iou
        n_batches  += 1

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"  Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | IoU: {iou:.4f}")

    return total_loss / n_batches, total_iou / n_batches


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> tuple:
    """
    Run one full pass through the validation data.
    No gradient computation — we're only measuring, not learning.

    Returns:
        avg_loss, avg_iou
    """
    model.eval()  # puts model in evaluation mode (disables dropout)

    total_loss = 0.0
    total_iou  = 0.0
    n_batches  = 0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, masks)
        iou  = compute_iou(predictions, masks)

        total_loss += loss.item()
        total_iou  += iou
        n_batches  += 1

    return total_loss / n_batches, total_iou / n_batches


# ── Checkpoint Helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_loss, path: Path):
    """Save model weights and training state to disk."""
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
    }, path)
    print(f"  💾 Checkpoint saved → {path.name}")


def load_checkpoint(model, optimizer, path: Path) -> tuple:
    """
    Load a saved checkpoint to resume training.
    Returns (model, optimizer, start_epoch, best_val_loss).
    """
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"  ✅ Resumed from epoch {checkpoint['epoch']} "
          f"(val loss: {checkpoint['val_loss']:.4f})")
    return model, optimizer, checkpoint["epoch"] + 1, checkpoint["val_loss"]


# ── Main Training Function ────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  AI Takeoff — U-Net Segmentation Training")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────────────────────
    device    = config.DEVICE
    model     = build_model()
    loss_fn   = build_loss()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    train_loader, val_loader = get_dataloaders()

    if train_loader is None:
        print("\n⛔ Cannot start training — no data found.")
        print("   See instructions above to add your drawings.")
        return

    # ── Resume from checkpoint if one exists ───────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float("inf")
    no_improve     = 0  # counter for early stopping

    if config.BEST_MODEL_PATH.exists():
        print(f"\nFound existing checkpoint: {config.BEST_MODEL_PATH}")
        resume = input("Resume training from checkpoint? (y/n): ").strip().lower()
        if resume == "y":
            model, optimizer, start_epoch, best_val_loss = load_checkpoint(
                model, optimizer, config.BEST_MODEL_PATH
            )

    # ── History log (for plotting later) ──────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_iou":  [], "val_iou":  [],
    }

    # ── Epoch Loop ─────────────────────────────────────────────────────────────
    print(f"\nStarting training on {device.upper()} for up to {config.NUM_EPOCHS} epochs\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 40)

        # Train
        train_loss, train_iou = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        # Validate
        val_loss, val_iou = validate(model, val_loader, loss_fn, device)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        elapsed = time.time() - epoch_start
        print(f"\n  Train → Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f}   | IoU: {val_iou:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # ── Save best model ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            save_checkpoint(model, optimizer, epoch, val_loss, config.BEST_MODEL_PATH)
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{config.EARLY_STOPPING_PATIENCE} epochs")

        # ── Periodic checkpoint ────────────────────────────────────────────────
        if (epoch + 1) % config.SAVE_EVERY == 0:
            periodic_path = config.CHECKPOINT_DIR / f"epoch_{epoch+1:03d}.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, periodic_path)

        # ── Early stopping ─────────────────────────────────────────────────────
        if no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹  Early stopping triggered after {epoch+1} epochs.")
            break

        print()

    # ── Training Complete ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model saved at:  {config.BEST_MODEL_PATH}")
    print("=" * 60)

    # Save history for analysis
    history_path = config.LOG_DIR / "training_history.npy"
    np.save(history_path, history)
    print(f"\n  Training history saved → {history_path}")
    print("  Run evaluate.py to test the model on your drawings.")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
