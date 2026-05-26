"""
Auto-annotator: generates YOLO-format .txt labels for each captured image.

Strategy per image
------------------
1. HSV color segmentation using per-class dominant hue ranges.
2. Morphological clean-up → find largest contour → bounding box.
3. If no valid contour found, fall back to multi-scale template matching
   using the reference PNG for that class.
4. Write YOLO label: class_id  cx_norm  cy_norm  w_norm  h_norm

Outputs
-------
- A .txt label file alongside every .png in each class folder.
- data.yaml        → YOLO training config (version-agnostic).
- train.txt / val.txt → 80/20 image-path lists for train/val split.
"""

import os
import random
import numpy as np
import cv2

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__),
                         'signals', 'from_static_images')
TMPL_DIR  = BASE_DIR           # reference PNGs live here

# ── Class registry (alphabetical → class IDs match data.yaml) ────────────────
CLASSES = [
    'forward_arrow',
    'forward_to_left_arrow',
    'forward_to_right',
    'give_way_signal',
    'stop_sign',
    'worker1_signal',
]
CLASS_ID = {name: idx for idx, name in enumerate(CLASSES)}

# ── Per-class HSV color ranges (list of (lo, hi) tuples) ─────────────────────
# Each sign has a dominant hue in the simulator; broad S/V thresholds keep
# this robust to lighting variation.
HSV_RANGES = {
    'stop_sign': [
        ((0,   60, 60), (12,  255, 255)),
        ((158, 60, 60), (180, 255, 255)),
    ],
    'give_way_signal': [
        ((0,   60, 60), (25,  255, 255)),
        ((155, 60, 60), (180, 255, 255)),
    ],
    'forward_arrow':         [((95,  50, 50), (135, 255, 255))],
    'forward_to_left_arrow': [((95,  50, 50), (135, 255, 255))],
    'forward_to_right':      [((95,  50, 50), (135, 255, 255))],
    'worker1_signal': [                             # red triangle warning sign
        ((0,   60, 60), (12,  255, 255)),
        ((158, 60, 60), (180, 255, 255)),
    ],
}

# ── Annotation parameters ─────────────────────────────────────────────────────
MIN_AREA         = 300      # px² — ignore tiny blobs
MAX_AREA_FRAC    = 0.35     # reject color bbox if it covers > this fraction of frame
MAX_ASPECT       = 4.0      # w/h or h/w must be < this
BBOX_PAD         = 4        # pixels of padding around detected bbox
TMPL_THRESH      = 0.30     # template-match fallback acceptance threshold
TMPL_SCALES      = [0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
VAL_SPLIT        = 0.20     # fraction of images held out for validation


# ─────────────────────────────────────────────────────────────────────────────
def _color_bbox(img: np.ndarray, class_name: str):
    """Return (x, y, w, h) from HSV color segmentation, or None."""
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask  = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in HSV_RANGES.get(class_name, []):
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

    # morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_AREA:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT:
        return None

    img_area = img.shape[0] * img.shape[1]
    if (w * h) / img_area > MAX_AREA_FRAC:
        return None   # too large → trigger template fallback

    return (x, y, w, h)


def _template_bbox(img: np.ndarray, class_name: str):
    """Return (x, y, w, h) via multi-scale template matching, or None."""
    tmpl_path = os.path.join(TMPL_DIR, f'{class_name}.png')
    if not os.path.exists(tmpl_path):
        return None
    tmpl = cv2.imread(tmpl_path)
    if tmpl is None:
        return None

    h_img, w_img = img.shape[:2]
    best_val, best_rect = -1.0, None

    for scale in TMPL_SCALES:
        th = int(tmpl.shape[0] * scale)
        tw = int(tmpl.shape[1] * scale)
        if th < 8 or tw < 8 or th > h_img or tw > w_img:
            continue
        resized = cv2.resize(tmpl, (tw, th))
        res     = cv2.matchTemplate(img, resized, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(res)
        if maxval > best_val:
            best_val  = maxval
            best_rect = (maxloc[0], maxloc[1], tw, th)

    if best_val < TMPL_THRESH or best_rect is None:
        return None
    return best_rect


def _to_yolo(x, y, w, h, img_w, img_h):
    """Convert pixel bbox to YOLO normalised format."""
    x  = max(0, x - BBOX_PAD)
    y  = max(0, y - BBOX_PAD)
    w  = min(img_w - x, w + 2 * BBOX_PAD)
    h  = min(img_h - y, h + 2 * BBOX_PAD)
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return cx, cy, w / img_w, h / img_h


def annotate_class(class_name: str):
    """Annotate all images in one class folder. Returns (ok, fail) counts."""
    class_dir = os.path.join(BASE_DIR, class_name)
    if not os.path.isdir(class_dir):
        return 0, 0

    class_id = CLASS_ID[class_name]
    images   = sorted(f for f in os.listdir(class_dir) if f.lower().endswith('.png'))
    ok = fail = 0

    for fname in images:
        img_path = os.path.join(class_dir, fname)
        img      = cv2.imread(img_path)
        if img is None:
            fail += 1
            continue

        ih, iw = img.shape[:2]

        # Strategy 1: color segmentation
        bbox = _color_bbox(img, class_name)
        src  = 'color'

        # Strategy 2: template matching fallback
        if bbox is None:
            bbox = _template_bbox(img, class_name)
            src  = 'template'

        if bbox is None:
            print(f'  FAIL  {class_name}/{fname}')
            fail += 1
            continue

        cx, cy, wn, hn = _to_yolo(*bbox, iw, ih)

        # guard against degenerate boxes
        if wn <= 0 or hn <= 0 or cx <= 0 or cy <= 0:
            print(f'  DEGENERATE  {class_name}/{fname}')
            fail += 1
            continue

        label_path = os.path.splitext(img_path)[0] + '.txt'
        with open(label_path, 'w') as f:
            f.write(f'{class_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n')
        ok += 1

    return ok, fail


def write_split_lists(all_image_paths: list):
    """Write train.txt and val.txt with 80/20 random split (relative paths)."""
    random.seed(42)
    shuffled = list(all_image_paths)
    random.shuffle(shuffled)
    cut  = int(len(shuffled) * (1 - VAL_SPLIT))
    train_imgs = shuffled[:cut]
    val_imgs   = shuffled[cut:]

    for fname, paths in [('train.txt', train_imgs), ('val.txt', val_imgs)]:
        out = os.path.join(BASE_DIR, fname)
        with open(out, 'w') as f:
            f.write('\n'.join(paths) + '\n')
    print(f'\nTrain/val split: {len(train_imgs)} / {len(val_imgs)} images')
    print(f'  → {BASE_DIR}/train.txt')
    print(f'  → {BASE_DIR}/val.txt')


def write_data_yaml():
    """Write data.yaml for YOLO training (version-agnostic format)."""
    yaml_path  = os.path.join(BASE_DIR, 'data.yaml')
    names_lines = '\n'.join(f'  {i}: {n}' for i, n in enumerate(CLASSES))
    content = (
        f'# YOLO dataset config — compatible with ultralytics YOLOv5/v8/v9/v10/v11\n'
        f'path: {BASE_DIR}\n'
        f'train: train.txt\n'
        f'val: val.txt\n'
        f'\n'
        f'nc: {len(CLASSES)}\n'
        f'names:\n'
        f'{names_lines}\n'
    )
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f'\ndata.yaml → {yaml_path}')


def main():
    print('=== YOLO Auto-Annotator ===\n')
    total_ok = total_fail = 0
    all_images = []

    for cls in CLASSES:
        ok, fail = annotate_class(cls)
        total_ok   += ok
        total_fail += fail
        status = '✓' if fail == 0 else f'✗ {fail} failed'
        print(f'  {cls:<28s}: {ok:3d} labelled  {status}')

        # collect relative image paths for train/val lists
        cls_dir = os.path.join(BASE_DIR, cls)
        if os.path.isdir(cls_dir):
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith('.png'):
                    all_images.append(os.path.join(cls, f))

    print(f'\nTotal: {total_ok} labelled, {total_fail} failed')

    write_split_lists(all_images)
    write_data_yaml()

    print('\nDone. To train (example with ultralytics):')
    print(f'  yolo train model=<any>.pt data={BASE_DIR}/data.yaml imgsz=320')


if __name__ == '__main__':
    main()
