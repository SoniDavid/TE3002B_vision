"""
SignalDetector — traffic sign detection pipeline (actividad 2.6)

Pre-filter  : Laplacian-variance blur detection.
              Blurry frames (score < BLUR_THRESH) are skipped entirely.

Approach B  : YOLO inference via the ultralytics API (any compatible version).
              Loaded from YOLO_MODEL_PATH; skipped silently if unavailable.

Public API
----------
    detector = SignalDetector(debug=True)
    detections = detector.detect(img)
    # img: BGR numpy array (320×240 recommended)
    # returns list of dicts:
    #   {'class': str, 'conf': float, 'bbox': (x,y,w,h), 'source': str}
    # detector.blur_score   → float  (Laplacian variance of last frame)
    # detector.is_blurry    → bool   (True  → detection was skipped)
    # detector.debug_frame  → BGR image with pipeline visualisation
"""

import os
import time
import numpy as np
import cv2

# ── Configuration ─────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(__file__)
TMPL_DIR        = os.path.join(_HERE, 'signals', 'from_static_images')
YOLO_MODEL_PATH = os.path.join(_HERE, 'best.pt')

# Unified class schema (matches merged_dataset training)
CLASSES = [
    'stop',         # 0
    'workers',      # 1
    'go_straight',  # 2
    'turn_right',   # 3
    'turn_left',    # 4
    'give_way',     # 5
]

# Normalize legacy/source-specific class labels to the unified schema.
CLASS_ALIASES = {
    'stop_sign': 'stop',
    'worker1_signal': 'workers',
    'forward_arrow': 'go_straight',
    'forward_to_right': 'turn_right',
    'forward_to_left_arrow': 'turn_left',
    'give_way_signal': 'give_way',
}

# ── Blur pre-filter (Laplacian variance) ──────────────────────────────────────
# Frames below this score are too blurry for reliable detection and are skipped.
# Typical values: <30 very blurry, 30-80 moderate, >80 sharp.
BLUR_THRESH = 50.0

# ── YOLO inference ────────────────────────────────────────────────────────────
YOLO_CONF = 0.55
YOLO_IOU  = 0.45

# ── Debug colours (BGR) ───────────────────────────────────────────────────────
CLR_TMPL  = (0,   200,   0)     # green  — template match
CLR_YOLO  = (255, 100,   0)     # blue   — YOLO
CLR_BOTH  = (0,   200, 200)     # yellow — fused
CLR_BLUR  = (0,    80, 200)     # orange — blurry frame


# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    """IoU between two (x,y,w,h) boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


class SignalDetector:
    def __init__(self, debug: bool = False, tmpl_enabled: bool = True,
                 parallel: bool = False, yolo_device: str = 'cuda',
                 allow_cpu_fallback: bool = False):
        self.debug        = debug
        self.tmpl_enabled = False
        self.parallel     = False
        self.yolo_device  = yolo_device
        self.allow_cpu_fallback = allow_cpu_fallback
        self.debug_frame  = None
        self.blur_score   = 0.0
        self.is_blurry    = False
        self.last_timing  = {
            'blur_ms': 0.0,
            'tmpl_ms': 0.0,
            'yolo_ms': 0.0,
            'fuse_ms': 0.0,
            'total_ms': 0.0,
        }

        # Load YOLO (optional — skipped gracefully if model or lib missing)
        self._yolo   = None
        self._device = 'cpu'
        try:
            from ultralytics import YOLO as _YOLO
            if os.path.exists(YOLO_MODEL_PATH):
                import torch
                req = str(self.yolo_device).lower()
                if req == 'cuda':
                    if torch.cuda.is_available():
                        self._device = 0
                    elif self.allow_cpu_fallback:
                        self._device = 'cpu'
                        print('[SignalDetector] WARNING: CUDA unavailable, falling back to CPU')
                    else:
                        raise RuntimeError(
                            'CUDA requested but unavailable. Fix CUDA/Torch setup '
                            'or run with CPU override (--cpu).'
                        )
                elif req == 'cpu':
                    self._device = 'cpu'
                else:
                    self._device = req
                self._yolo   = _YOLO(YOLO_MODEL_PATH)
                print(f'[SignalDetector] YOLO loaded  device={self._device}')
                model_names = getattr(self._yolo, 'names', None)
                if isinstance(model_names, dict):
                    aligned = all(
                        self._normalize_class(str(name)) == CLASSES[idx]
                        for idx, name in model_names.items() if idx < len(CLASSES)
                    )
                    if not aligned:
                        print('[SignalDetector] WARNING: model class names differ from '
                              'expected schema; applying alias normalization')
            else:
                print(f'[SignalDetector] YOLO model not found at {YOLO_MODEL_PATH} — '
                      'Approach B disabled')
        except ImportError:
            print('[SignalDetector] ultralytics not installed — Approach B disabled')

    # ── Pre-filter: Laplacian variance blur detection ─────────────────────────
    def _blur_score(self, img: np.ndarray) -> float:
        """
        Estimate frame sharpness via Laplacian variance.
        High variance → sharp edges present → not blurry.
        Low variance  → few edges           → blurry / motion-blurred.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _normalize_class(self, cls_name: str) -> str:
        return CLASS_ALIASES.get(cls_name, cls_name)

    def _yolo_detect_timed(self, img: np.ndarray):
        t0 = time.perf_counter()
        det = self._yolo_detect(img)
        return det, (time.perf_counter() - t0) * 1000.0

    # ── Approach B: YOLO ──────────────────────────────────────────────────────
    def _yolo_detect(self, img: np.ndarray):
        """Return highest-confidence YOLO detection or None."""
        if self._yolo is None:
            return None
        results   = self._yolo(img, conf=YOLO_CONF, iou=YOLO_IOU,
                               verbose=False, device=self._device)
        best_det  = None
        best_conf = -1.0
        for r in results:
            for box in r.boxes:
                conf     = float(box.conf[0])
                cls_id   = int(box.cls[0])
                if hasattr(r, 'names') and isinstance(r.names, dict):
                    raw_name = str(r.names.get(cls_id, cls_id))
                else:
                    raw_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
                cls_name = self._normalize_class(raw_name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if conf > best_conf:
                    best_conf = conf
                    best_det  = {
                        'class':  cls_name,
                        'conf':   conf,
                        'bbox':   (x1, y1, x2-x1, y2-y1),
                        'source': 'yolo',
                    }
        return best_det

    # ── Fusion (YOLO-only mode) ───────────────────────────────────────────────
    def _fuse(self, tmpl_det, yolo_det):
        if yolo_det is None:
            return []
        return [yolo_det]

    # ── Debug frame ───────────────────────────────────────────────────────────
    def _build_debug(self, img, tmpl_det, yolo_det, final_dets):
        h, w = img.shape[:2]

        def _annotate(canvas, det, color):
            if det is None:
                cv2.putText(canvas, 'N/A', (8, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
                return
            x, y, bw, bh = det['bbox']
            cv2.rectangle(canvas, (x, y), (x+bw, y+bh), color, 2)
            label = f"{det['class']} {det['conf']:.2f}"
            cv2.putText(canvas, label, (x, max(y-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
            cv2.putText(canvas, label, (x, max(y-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        panel_b = img.copy()
        _annotate(panel_b, yolo_det, CLR_YOLO)
        cv2.putText(panel_b, 'YOLO', (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_YOLO, 1)

        top_row = panel_b

        # Info bar: blur score + final detection
        bar = np.zeros((26, top_row.shape[1], 3), np.uint8)

        blur_clr = CLR_BLUR if self.is_blurry else (0, 180, 0)
        blur_txt = f'blur={self.blur_score:.1f}' \
                   + (' [SKIP]' if self.is_blurry else '')
        cv2.putText(bar, blur_txt, (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, blur_clr, 1)

        if final_dets:
            det = final_dets[0]
            clr = CLR_BOTH if det['source'] == 'both' else \
                  CLR_YOLO if det['source'] == 'yolo' else CLR_TMPL
            det_txt = f"[{det['source'].upper()}] {det['class']}  " \
                      f"conf={det['conf']:.2f}"
        else:
            det_txt = 'no detection'
            clr     = (80, 80, 80)
        cv2.putText(bar, det_txt, (top_row.shape[1]//2 + 6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)

        self.debug_frame = np.vstack([top_row, bar])

    # ── Main entry point ──────────────────────────────────────────────────────
    def detect(self, img: np.ndarray):
        """
        Pre-filter with Laplacian blur check, then run YOLO.

        Parameters
        ----------
        img : np.ndarray — BGR image (320×240 recommended)

        Returns
        -------
        list of dicts: class, conf, bbox (x,y,w,h), source='yolo'
        Empty list if frame is too blurry (check self.is_blurry / self.blur_score).
        """
        total_t0 = time.perf_counter()

        # ── Blur pre-filter ───────────────────────────────────────────────────
        blur_t0 = time.perf_counter()
        self.blur_score = self._blur_score(img)
        self.last_timing['blur_ms'] = (time.perf_counter() - blur_t0) * 1000.0
        self.is_blurry  = self.blur_score < BLUR_THRESH
        self.last_timing['tmpl_ms'] = 0.0
        self.last_timing['yolo_ms'] = 0.0
        self.last_timing['fuse_ms'] = 0.0

        if self.is_blurry:
            if self.debug:
                self._build_debug(img, None, None, [])
            self.last_timing['total_ms'] = (time.perf_counter() - total_t0) * 1000.0
            return []

        # ── Detection ─────────────────────────────────────────────────────────
        tmpl_det = None
        yolo_det, yolo_ms = self._yolo_detect_timed(img)
        self.last_timing['yolo_ms'] = yolo_ms

        fuse_t0 = time.perf_counter()
        final    = self._fuse(tmpl_det, yolo_det)
        self.last_timing['fuse_ms'] = (time.perf_counter() - fuse_t0) * 1000.0

        if self.debug:
            self._build_debug(img, tmpl_det, yolo_det, final)

        self.last_timing['total_ms'] = (time.perf_counter() - total_t0) * 1000.0

        return final
