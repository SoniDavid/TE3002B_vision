# Actividad 2.4 — Center-Line Detection for Autonomous Track Following

## Overview

This module implements a vision-based center-line detector for an autonomous robot
navigating a track in a 3D Gaussian Splatting simulator. The camera is mounted on
the robot and streams 320×240 BGR frames over gRPC. The detector outputs the pixel
coordinates `(cx, cy)` of the track's center dashed line, which are then used as
the error signal for steering control.

The final implementation is `actividad_2_04.py`.

The previous best version was named `actividad_2_04_otsu_v4.py`, and it is now
consolidated as `actividad_2_04.py`.

The simulator client and control loop are in `test_simulator.py`.

---

## Pipeline (`CenterLineDetector.detect_center_line`)

```
Frame (320×240 BGR)
  │
  S1  ROI crop — bottom 1/3 of the frame (y > 2h/3)
  │   Only the floor is relevant; sky and walls are discarded.
  │
  S2  Grayscale conversion
  │
  S3  Gaussian blur  5×5, σ=1.4
  │   Smooths JPEG compression noise before thresholding.
  │
  S4  Canny edge detection  (low=30, high=90)
  │   Used later in S7 for the asymmetry classifier.
  │
  S5  Otsu threshold  THRESH_BINARY_INV + THRESH_OTSU
  │   The track lines are dark on a light beige surface.
  │   Otsu automatically finds the optimal threshold per frame.
  │   Inversion makes lines white blobs on a black background.
  │
  S6  Morphological OPEN  3×3 kernel
  │   Removes single-pixel noise that survived thresholding.
  │
  S7  Smart contour selection  →  raw (cx, cy, area)
  │   See section below.
  │
  S8  Velocity gate
  │   If the detection came from a *tiny* blob (area < SIGNIFICANT_AREA)
  │   AND the jump exceeds MAX_JUMP px, hold the previous position.
  │   Detections from real track lines (area ≥ SIGNIFICANT_AREA) bypass
  │   the gate so the tracker can re-acquire after a turn.
  │
  S9  Median filter  K=3 frames
      Eliminates residual frame-level oscillation.
      prev_cx is updated from the smoothed value.
```

---

## S7 — Smart Contour Selection

### Contour classification by area

Track contours are split into two classes based on area:

| Class | Area range | Meaning |
|---|---|---|
| **Significant** | 500 – 4000 px² | Real track lines (center line ~1500–2500, boundary ~2000–3800) |
| **Tiny** | 50 – 499 px² | Engraving decorations, noise fragments |

Only significant blobs participate in the search window and position decisions.
Tiny blobs are a fallback of last resort when no significant blob exists.

### Warmup phase (first 8 frames)

The search window is disabled for the first 8 frames. Instead, the significant
blob closest to the image horizontal center (`w/2 = 160`) is selected. The robot
starts facing the center line, so this always anchors `prev_cx` correctly before
the window engages.

The velocity gate is also disabled during warmup so that the median can
self-correct if frames 0–1 contain merged blobs.

### Search window (normal operation)

After warmup, a `±WINDOW_HALF (20 px)` window around `prev_cx` filters candidates:

1. **Significant blobs in window** → use them for selection (n=1/2/3+ logic below).
2. **No significant blob in window** (line moved away or only noise inside) →
   jump directly to the globally nearest significant blob.
   This is the **turn-recovery path**: when the center line exits the window
   during a curve, the tracker re-acquires it immediately instead of chasing
   engraving fragments.
3. **No significant blob anywhere** → fall back to tiny blobs in window.

### Selection by candidate count

| Count | Strategy |
|---|---|
| 0 | Hold `prev_cx` or use image center |
| 1 | Use directly |
| 2 | **Canny edge-asymmetry classifier**: measure edge density in 21×10 px strips at ±25 px from each centroid. Symmetric edges → center line (bright track both sides). Asymmetric → boundary (floor on one side). Falls back to bilateral brightness when edge signal is too sparse. Inconclusive → score by distance to image center and `prev_cx`. |
| ≥3 | Sort by x-position, pick the **middle** index. The center line always lies between the two boundaries. |

---

## Temporal Smoothing

Two stages are combined:

**Velocity gate (S8):** Prevents noise blobs from entering the median history.
Applied only to tiny detections (`area < SIGNIFICANT_AREA`). Real track lines
bypass the gate so recovery from fast turns is not blocked.

**Median filter (S9, K=3):** The final output is the median of the last 3 raw
(gated) cx values. Eliminates single-frame outliers without perceptible lag
(≤2 frames at 40 fps). `prev_cx` is set from the smoothed value so the window
always tracks the stable estimate.

---

## Parameters

| Constant | Default | Effect |
|---|---|---|
| `MIN_AREA` | 50 px² | Minimum blob area; smaller blobs are discarded entirely |
| `MAX_AREA` | 4000 px² | Maximum blob area; larger merged regions are discarded |
| `SIGNIFICANT_AREA` | 500 px² | Threshold separating real lines from noise fragments |
| `WINDOW_HALF` | 20 px | Half-width of the tracking search window |
| `SAMPLE_OFFSET` | 25 px | Distance from centroid to Canny sample strip |
| `ASYM_THR` | 0.40 | Edge asymmetry threshold for center/boundary classification |
| `BRIGHT_THR` | 120 | Grayscale brightness threshold for bilateral fallback |
| `MAX_JUMP` | 40 px | Gate threshold (applies to tiny-blob detections only) |
| `MEDIAN_K` | 3 frames | Median filter window length |
| `WARMUP_FRAMES` | 8 frames | Frames before the search window engages |

**Tuning guidance:**
- Narrow turns losing the line → increase `WINDOW_HALF` slightly (25–30).
- Engraving blobs still causing drift → increase `SIGNIFICANT_AREA` (600–800).
- Detection too jumpy on straights → decrease `MAX_JUMP` or increase `MEDIAN_K`.

---

## Debug Mode

```python
detector = CenterLineDetector(debug=True)
```

Every call populates `detector.debug_frame` with a 7-tile horizontal strip:

```
S1:ROI | S2:gray | S3:blur | S4:Canny | S5:Otsu | S6:open | S7:result
```

The result tile shows:
- **Green contours** — significant blobs (real track lines)
- **Cyan contours** — tiny blobs (noise/engravings)
- **Orange rectangle** — current search window
- **Red cross** — detected center position
- `sig=N cx=C` overlay

`test_simulator.py` writes both the annotated main video and the debug strip
video (`debug_mode{mode}_{timestamp}.avi`) automatically.

---

## Running

### Environment requirement

For correct execution, dependencies must come from:

`/home/soni/Documents/classes/IRS_6to/TE3002B_vision/environment_3DGs.yaml`

If needed, create or update the Conda environment from that file before running:

```bash
conda env create -f /home/soni/Documents/classes/IRS_6to/TE3002B_vision/environment_3DGs.yaml
# or, if the environment already exists
conda env update -f /home/soni/Documents/classes/IRS_6to/TE3002B_vision/environment_3DGs.yaml --prune
```

```bash
# Mode 2 — simulator drives, we observe and detect
python3 test_simulator.py 2

# Mode 0 — our control loop steers the robot
python3 test_simulator.py 0

# Press Q to stop; recordings saved automatically
```

---

