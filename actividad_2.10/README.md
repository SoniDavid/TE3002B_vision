# Actividad 2.10 — Stereo Vision + ICP 3D Reconstruction

## Overview

This activity builds a stereo camera from two smartphones, generates depth maps via stereo disparity, reconstructs two XYZ point clouds (PLY format), and merges them using ICP (Iterative Closest Point) with Open3D.

**Pipeline:**
```
Two phones (stereo rig)
    │
    ├─ 01_stereo_calibrate.py   → calibration.npz
    │
    ├─ 02_stereo_reconstruct.py  (View 1)  → view1.ply
    │
    │  [shift rig ~15 cm horizontally]
    │
    ├─ 02_stereo_reconstruct.py  (View 2)  → view2.ply
    │
    └─ 03_icp_registration.py   → merged.ply
```

---

## Dependencies

```bash
pip install opencv-python numpy open3d
```

Python 3.10+ recommended.

---

## Step 1 — Print a Checkerboard

Print a checkerboard pattern (9×6 inner corners by default). Measure the actual side length of one square in metres (e.g. 2.5 cm = 0.025 m). Keep this measurement for the calibration command.

---

## Step 2 — Record Calibration Videos

Mount both phones on the stereo rig (fixed together). Each phone records its **own video independently** — no synchronisation needed.

- Hold the rig still
- One person holds/records **left phone video** while moving the checkerboard in front of both cameras, covering many angles and distances
- Then record **right phone video** the same way (or a teammate does it simultaneously — both options work)
- Aim for ~30–60 seconds, moving the board into different orientations and corners of the frame
- Name the files `left_calib.mp4` and `right_calib.mp4`

```bash
python 01_stereo_calibrate.py \
    --left left_calib.mp4 \
    --right right_calib.mp4 \
    --pattern 9x6 \
    --square 0.025 \
    --frame-step 10 \
    --output calibration.npz
```

**Check:** Reprojection RMS should be < 1.0 px. If it's higher, recapture with more varied poses.

> **`--pattern`**: inner corners, e.g. `9x6` for a 10×7 square board.
> **`--square`**: side of one square in metres.
> **`--frame-step`**: sample every N frames from each video (increase if too many pairs are found and it's slow).

---

## Step 3 — Capture View 1

Place the stereo rig facing the scene. Take a left image and a right image.

- For **static scenes** (no movement): you can capture them sequentially — take left first, then right, without moving the rig.
- For **scenes with movement**: capture both images simultaneously (e.g. use a remote shutter, or have two people tap at the same time).

Name the files `left1.png` and `right1.png`.

```bash
python 02_stereo_reconstruct.py \
    --left left1.png \
    --right right1.png \
    --calib calibration.npz \
    --output view1.ply \
    --max-depth 5.0
```

This also saves `view1_disparity.png` — open it to verify the disparity map looks reasonable (bright = closer, dark = farther).

---

## Step 4 — Shift and Capture View 2

Move the **entire stereo rig** ~10–20 cm horizontally (keep the same vertical height and orientation as much as possible). Take another stereo pair.

Name the files `left2.png` and `right2.png`.

```bash
python 02_stereo_reconstruct.py \
    --left left2.png \
    --right right2.png \
    --calib calibration.npz \
    --output view2.ply \
    --max-depth 5.0
```

---

## Step 5 — ICP Registration

```bash
python 03_icp_registration.py \
    --source view2.ply \
    --target view1.ply \
    --output merged.ply \
    --voxel 0.01
```

**What to expect:**
- RANSAC fitness > 0.3 and ICP fitness > 0.5 indicate a good alignment.
- The final RMSE should be in the millimetre range for a well-calibrated rig.

---

## Step 6 — MeshLab Screenshots

Open each PLY in MeshLab:

1. `File → Import Mesh` → select `view1.ply`
2. Rotate to a good angle showing the scene depth
3. `File → Save Screenshot` (or use your OS screenshot tool)
4. Repeat for `view2.ply` and `merged.ply`

---

## Without Calibration (rough mode)

If you skip calibration, provide the baseline and focal length manually:

```bash
python 02_stereo_reconstruct.py \
    --left left1.png --right right1.png \
    --baseline 0.12 \
    --focal 1400 \
    --output view1.ply
```

- `--baseline`: distance between the two camera centres in metres (measure with a ruler)
- `--focal`: focal length in pixels (≈ image_width × 1.2 for most smartphone rear cameras)

> Disparity quality without calibration will be noticeably lower.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Calibration RMS > 1.0 | Recapture videos with more varied checkerboard angles and distances |
| Disparity map is mostly black | Images may not be rectified; reduce `--max-depth` or check calibration |
| ICP fitness < 0.1 | The two views don't overlap enough; shift rig less, or check PLY files in MeshLab first |
| "No valid 3D points" | Check that left/right images are the same resolution and have good texture |
| Open3D RANSAC is slow | Increase `--voxel` to 0.02 or 0.03 to reduce the number of downsampled points |

---

## File Summary

| File | Purpose |
|---|---|
| `01_stereo_calibrate.py` | Stereo calibration from two independent videos |
| `02_stereo_reconstruct.py` | Stereo pair → disparity → PLY |
| `03_icp_registration.py` | ICP alignment of two views → merged PLY |
| `calibration.npz` | Calibration output (generated) |
| `view1.ply` / `view2.ply` | Per-view point clouds (generated) |
| `merged.ply` | ICP-merged reconstruction (generated) |
