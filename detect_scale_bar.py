"""
Scale Bar Detector for Architectural Elevation Drawings
========================================================
POC script for DAS10 takeoff automation project.

What this script does:
  1. Loads a JPEG elevation drawing
  2. Finds the scale bar using edge/line detection
  3. Reads the scale label using OCR
  4. Computes pixels_per_foot — the ratio needed for all measurements

Dependencies:
  conda install -c conda-forge opencv pytesseract pillow numpy
  (also install Tesseract OCR engine: https://github.com/UB-Mannheim/tesseract/wiki)

Usage:
  python detect_scale_bar.py                        # uses default image path
  python detect_scale_bar.py my_drawing.jpg         # pass your own image
"""

import sys
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image

# ── If Tesseract isn't on your PATH, set it here ──────────────────────────────
# Windows example:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Mac/Linux: usually works without this if installed via brew/apt


# =============================================================================
# STEP 1 — Load and pre-process the image
# =============================================================================

def load_image(path: str):
    """Load image and return both colour and grayscale versions."""
    img_color = cv2.imread(path)
    if img_color is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    print(f"[1] Loaded image: {path}  ({img_color.shape[1]}×{img_color.shape[0]} px)")
    return img_color, img_gray


def preprocess(img_gray: np.ndarray) -> np.ndarray:
    """
    Clean up the image for detection.
    Architectural drawings are usually light background + dark lines,
    so we threshold to get a clean black-and-white version.
    """
    # Gaussian blur removes minor noise without blurring structural lines
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Otsu's threshold automatically picks the best cut-off value
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    print("[1] Pre-processing done (blur + Otsu threshold)")
    return binary


# =============================================================================
# STEP 2 — Detect horizontal line segments (scale bar candidates)
# =============================================================================

def find_horizontal_lines(binary: np.ndarray, img_shape: tuple) -> list:
    """
    Use morphological operations to isolate horizontal lines.
    A scale bar is typically a solid horizontal line with tick marks at each end.

    Returns a list of (x, y, w, h) bounding boxes for candidate lines.
    """
    h, w = img_shape[:2]

    # Create a kernel that's wide (captures horizontal lines) but only 1px tall
    # Minimum line length = 3% of image width — filters out tiny marks
    min_line_len = max(30, int(w * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len, 1))

    # Erode then dilate: only features wider than kernel survive
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find connected components (each surviving line segment)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / max(ch, 1)

        # Filter: must be wide relative to height (truly horizontal)
        # and not span the full image width (would be a drawing border)
        if aspect_ratio > 5 and cw < w * 0.6:
            candidates.append((x, y, cw, ch))

    # Sort by y-position — scale bars are usually near the bottom of drawings
    candidates.sort(key=lambda c: c[1], reverse=True)

    print(f"[2] Found {len(candidates)} horizontal line candidates")
    return candidates


# =============================================================================
# STEP 3 — OCR: read the scale label near candidate lines
# =============================================================================

def read_scale_label(img_gray: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """
    Crop a region around the candidate line and run OCR on it.
    We look for text like '10 ft', '1" = 10'-0"', '1:50', etc.

    Architectural drawings put the scale label either:
      - directly below the line
      - to the right of the line
    So we search a generous area around it.
    """
    img_h, img_w = img_gray.shape
    padding = 60  # pixels to search around the line

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding * 2)  # extra below

    region = img_gray[y1:y2, x1:x2]

    # Upscale small regions — Tesseract works better on larger text
    scale_factor = max(1, int(200 / region.shape[0]))
    if scale_factor > 1:
        region = cv2.resize(region, None, fx=scale_factor, fy=scale_factor,
                            interpolation=cv2.INTER_CUBIC)

    # Tesseract config: treat as a single block of text, allow digits + symbols
    tess_config = "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz=:'\". -/"
text = pytesseract.image_to_string(region, config=tess_config).strip()

    return text


def parse_scale_from_text(text: str) -> float | None:
    """
    Try to extract a feet-per-unit value from OCR'd text.

    Handles common formats:
      '1" = 10 ft'       → 10.0
      "1/4\" = 1'-0\""   → 48.0  (quarter inch = 1 foot → 1 inch = 4 feet)
      '1:100'            → ~8.33 feet (100 units at 1:1 scale in inches)
      '10 ft'            → assumes line = 10 ft
      '10'               → assumes line = 10 ft (fallback)
    """
    text = text.replace('\n', ' ').strip()

    # Pattern: 1" = 10' or 1 = 10 ft
    m = re.search(r'1\s*["\']?\s*=\s*(\d+\.?\d*)\s*[\'ft]', text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Pattern: ratio like 1:100
    m = re.search(r'1\s*:\s*(\d+)', text)
    if m:
        ratio = float(m.group(1))
        # In architectural drawings, 1:100 means 1mm = 100mm real
        # Convert to feet: 100mm / 25.4mm per inch / 12 inches per foot
        return round(ratio / 25.4 / 12, 2)

    # Pattern: bare number followed by ft/feet
    m = re.search(r'(\d+\.?\d*)\s*(ft|feet|\')', text, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Fallback: first bare number in string
    m = re.search(r'(\d+\.?\d*)', text)
    if m:
        return float(m.group(1))

    return None


# =============================================================================
# STEP 4 — Compute pixels_per_foot
# =============================================================================

def compute_pixels_per_foot(line_width_px: int, real_world_feet: float) -> float:
    """
    Core calculation: how many pixels = 1 foot in this drawing?
    Everything else in the project depends on this number.
    """
    return line_width_px / real_world_feet


# =============================================================================
# STEP 5 — Visualise result (saves annotated image so you can inspect it)
# =============================================================================

def save_annotated(img_color: np.ndarray, best: tuple, pixels_per_foot: float, out_path: str):
    """Draw a box around the detected scale bar and label it."""
    x, y, w, h = best
    annotated = img_color.copy()

    # Draw bounding box in bright green
    cv2.rectangle(annotated, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 220, 0), 2)

    # Label
    label = f"Scale bar | {w}px = {pixels_per_foot:.1f}px/ft"
    cv2.putText(annotated, label, (x, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

    cv2.imwrite(out_path, annotated)
    print(f"[5] Annotated image saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def detect_scale_bar(image_path: str) -> dict:
    """
    Full pipeline. Returns a result dict with:
      - pixels_per_foot: the key ratio for all downstream measurements
      - scale_bar_px:    width of detected scale bar in pixels
      - real_world_ft:   the feet value read from the label
      - ocr_text:        raw OCR output for debugging
      - bbox:            (x, y, w, h) of the scale bar in the image
    """
    # Load
    img_color, img_gray = load_image(image_path)
    binary = preprocess(img_gray)

    # Detect candidate lines
    candidates = find_horizontal_lines(binary, img_color.shape)
    if not candidates:
        print("⚠️  No horizontal line candidates found. Try a higher-res scan.")
        return {}

    # Try each candidate until we find one with a readable scale label
    result = {}
    for i, (x, y, w, h) in enumerate(candidates[:5]):  # try top 5 candidates
        print(f"\n[3] Testing candidate {i+1}: bbox=({x},{y},{w},{h})")

        ocr_text = read_scale_label(img_gray, x, y, w, h)
        print(f"    OCR text: '{ocr_text}'")

        real_feet = parse_scale_from_text(ocr_text)
        print(f"    Parsed feet value: {real_feet}")

        if real_feet and real_feet > 0:
            ppf = compute_pixels_per_foot(w, real_feet)
            result = {
                "pixels_per_foot": round(ppf, 4),
                "scale_bar_px": w,
                "real_world_ft": real_feet,
                "ocr_text": ocr_text,
                "bbox": (x, y, w, h),
            }
            print(f"\n[4] ✅ Scale detected!")
            print(f"    Scale bar width : {w} px")
            print(f"    Real-world length: {real_feet} ft")
            print(f"    pixels_per_foot : {ppf:.4f}")
            break

    if not result:
        # Fallback: use the widest candidate and ask user to confirm scale
        x, y, w, h = max(candidates, key=lambda c: c[2])
        print("\n[4] ⚠️  Could not auto-read scale label.")
        print(f"    Best candidate line: {w}px wide at ({x},{y})")
        print("    → Set MANUAL_SCALE_FEET below and re-run.")
        result = {"pixels_per_foot": None, "scale_bar_px": w, "bbox": (x, y, w, h)}

    # Save annotated output image
    out_path = image_path.replace(".jpg", "").replace(".jpeg", "") + "_annotated.jpg"
    save_annotated(img_color, result["bbox"], result.get("pixels_per_foot") or 0, out_path)

    return result


# ── Manual override ───────────────────────────────────────────────────────────
# If OCR fails, set this to the real-world length of the scale bar in feet,
# then re-run. The script will use this instead of trying to read it.
MANUAL_SCALE_FEET = None   # e.g. set to 10.0 if you know the bar = 10 ft


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "elevation.jpg"

    result = detect_scale_bar(image_path)

    # Manual override if OCR failed
    if MANUAL_SCALE_FEET and result.get("scale_bar_px"):
        result["real_world_ft"] = MANUAL_SCALE_FEET
        result["pixels_per_foot"] = compute_pixels_per_foot(
            result["scale_bar_px"], MANUAL_SCALE_FEET
        )
        print(f"\n[manual] pixels_per_foot overridden → {result['pixels_per_foot']:.4f}")

    print("\n── Final Result ──────────────────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # This is what the rest of your pipeline will consume:
    if result.get("pixels_per_foot"):
        ppf = result["pixels_per_foot"]
        print(f"\n  ✅ Ready for measurement extraction.")
        print(f"     Example: a bounding box 150px wide = {150/ppf:.1f} ft")
