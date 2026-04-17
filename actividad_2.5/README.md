# Actividad 2.5 - Traffic Light State Detection

## Overview

This activity implements a computer-vision detector that classifies a traffic
light state as:

- `red`
- `yellow`
- `green`
- `none`

The final detector is implemented in `actividad_2_05.py` inside
`TrafficLightDetection.detect_state(image)`.

The module is designed for two workflows:

- Online evaluation with the simulator (`test_simulator.py`)
- Offline scoring against the graded reference video (`test_against_video.py`)

---

## Files

- `actividad_2_05.py`: final two-stage detector (housing + color classification)
- `actividad_2_05_v1.py`: previous variant kept for comparison
- `test_simulator.py`: gRPC client to run the detector in the simulator and record output
- `test_against_video.py`: offline evaluator for `A01571777.mp4` with score/accuracy report
- `docs/clase6.pdf`: lecture or support material

---

## Detection Pipeline (`TrafficLightDetection.detect_state`)

Input frame (`320x240`, BGR)

1. ROI crop: keep top `2/3` of the image (where traffic lights appear).
2. Convert ROI to HSV and extract value channel `V`.
3. Stage 1 - Housing detection:
   - Threshold dark pixels (`V < HOUSING_V_MAX`).
   - Morphological close to connect housing structure.
   - Keep contours that match expected traffic-light housing geometry
     (area + tall aspect ratio).
   - Validate that the candidate contains at least one bright bulb-level pixel.
4. Stage 2 - Color scoring:
   - If housing exists, classify color only inside housing (plus small padding).
   - If not, fallback to full ROI.
   - Build HSV masks for red/yellow/green.
   - Apply morphology open and contour quality filtering (area, circularity,
     shape, brightness).
   - Score each color by largest valid blob area.
5. Decision:
   - Choose color with largest area.
   - If best area `< DETECT_THRESH`, output `none`.

---

## Main Parameters

Defined in `TrafficLightDetection`:

- Color HSV ranges:
  - `RED_LO1/RED_HI1`, `RED_LO2/RED_HI2`
  - `YEL_LO/YEL_HI`
  - `GRN_LO/GRN_HI`
- Blob filters:
  - `MIN_AREA`
  - `MIN_CIRCULARITY`
  - `MIN_MEAN_V`
  - `DETECT_THRESH`
- Housing filters:
  - `HOUSING_V_MAX`
  - `HOUSING_MIN_AREA`
  - `HOUSING_ASPECT_MIN`, `HOUSING_ASPECT_MAX`
  - `HOUSING_BULB_V_MIN`
  - `HOUSING_PAD`

If detections are unstable, tune these constants before modifying logic.

---

## Environment Setup

From repository root, create/update the Conda environment:

```bash
conda env create -f environment_3DGS.yaml
# or, if already created
conda env update -f environment_3DGS.yaml --prune
```

The environment includes:

- Python 3.10
- NumPy
- OpenCV
- gRPC + protobuf dependencies

Activate your environment before running the scripts.

---

## Run in Simulator

`test_simulator.py` connects via gRPC to the simulator interface, runs
`TrafficLightDetection`, overlays state/FPS/camera pose, and records videos.

```bash
cd actividad_2.5
python3 test_simulator.py 1
```

Argument:

- `1` is the default mode used if no argument is provided.

Controls in window:

- `W/S`: move forward/back in x
- `A/D`: strafe in y
- `R/F`: move up/down in z
- Arrow keys: pitch/yaw
- `0`: reset camera pose
- `Q`: quit

Generated outputs:

- `recording_YYYYMMDD_HHMMSS.avi`: main annotated stream
- `debug_YYYYMMDD_HHMMSS.avi`: debug pipeline view (when `debug=True`)

---

## Run Offline Video Evaluation

`test_against_video.py` reads the reference state from the overlay text in
`A01571777.mp4`, runs the detector frame-by-frame, and computes score metrics.

Expected file:

- `actividad_2.5/A01571777.mp4`

Run:

```bash
cd actividad_2.5
python3 test_against_video.py
```

Scoring used by the script:

- Correct frame: `+10`
- Incorrect frame: `-25`

Printed report includes:

- Frames processed
- Correct detections and accuracy (%)
- Final score
- Confusion matrix (`reference -> detected`)

---

## Notes

- `test_simulator.py` currently imports simulator stubs using an absolute path:
  `/home/soni/Documents/classes/IRS_6to/TE3002B_vision/3DGS_Simulator/Interface`.
  If your workspace path changes, update that line.
- For grading-style checks, use `test_against_video.py` because it is fully
  deterministic for the provided video.