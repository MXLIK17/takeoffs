"""
align_pairs.py — Align annotated drawings to raw drawing crops
================================================================
When DAS10 exports raw and annotated images at different resolutions
or crops, resizing alone misaligns the ground-truth mask.

This script uses ORB feature matching + homography to warp each
annotated image onto its raw pair, then saves aligned copies to
data/annotated_aligned/ (originals are preserved).

Run:
  python align_pairs.py
  python align_pairs.py --dry-run   # report pairs without writing files
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config


def align_annotation_to_raw(raw_rgb: np.ndarray, annotated_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Warp annotated image to align with raw using ORB features + homography.

    Returns:
        aligned annotated image (same H×W as raw), success flag
    """
    raw_gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    ann_gray = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(raw_gray, None)
    kp2, des2 = orb.detectAndCompute(ann_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return cv2.resize(annotated_rgb, (raw_rgb.shape[1], raw_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST), False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 8:
        return cv2.resize(annotated_rgb, (raw_rgb.shape[1], raw_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST), False

    matches = sorted(matches, key=lambda m: m.distance)[:200]
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if matrix is None or inliers is None or inliers.sum() < 6:
        return cv2.resize(annotated_rgb, (raw_rgb.shape[1], raw_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST), False

    h, w = raw_rgb.shape[:2]
    aligned = cv2.warpPerspective(
        annotated_rgb, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return aligned, True


def main():
    parser = argparse.ArgumentParser(description="Align annotated images to raw drawing pairs")
    parser.add_argument("--dry-run", action="store_true", help="Report status without writing files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DATA_DIR / "annotated_aligned",
        help="Where to save aligned annotated images",
    )
    args = parser.parse_args()

    valid_ext = {".jpg", ".jpeg", ".png"}
    raw_files = sorted(
        f for f in config.RAW_DIR.iterdir()
        if f.suffix.lower() in valid_ext
    )

    if not raw_files:
        print(f"⚠️  No images in {config.RAW_DIR}")
        return

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    aligned_count = 0
    fallback_count = 0

    for raw_path in raw_files:
        ann_path = config.ANNOTATED_DIR / raw_path.name
        if not ann_path.exists():
            print(f"  ⚠️  Missing annotation for {raw_path.name}")
            continue

        raw_rgb = np.array(Image.open(raw_path).convert("RGB"))
        ann_rgb = np.array(Image.open(ann_path).convert("RGB"))

        if raw_rgb.shape[:2] == ann_rgb.shape[:2]:
            print(f"  ✅ {raw_path.name}: sizes already match ({raw_rgb.shape[1]}×{raw_rgb.shape[0]})")
            if not args.dry_run:
                Image.fromarray(ann_rgb).save(args.output_dir / raw_path.name)
            aligned_count += 1
            continue

        aligned, ok = align_annotation_to_raw(raw_rgb, ann_rgb)
        status = "aligned" if ok else "resize fallback"
        print(f"  {'✅' if ok else '⚠️ '} {raw_path.name}: {ann_rgb.shape[1]}×{ann_rgb.shape[0]} "
              f"→ {raw_rgb.shape[1]}×{raw_rgb.shape[0]} ({status})")

        if not args.dry_run:
            Image.fromarray(aligned).save(args.output_dir / raw_path.name)

        if ok:
            aligned_count += 1
        else:
            fallback_count += 1

    print(f"\n[Align] Done: {aligned_count} ok, {fallback_count} fallback resize")
    if not args.dry_run and aligned_count + fallback_count > 0:
        print(f"  Saved aligned annotations to: {args.output_dir}")
        print("  To use them for training, copy into data/annotated/ or update ANNOTATED_DIR in config.py")


if __name__ == "__main__":
    main()
