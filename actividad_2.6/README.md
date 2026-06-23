# Traffic Sign Detection Pipeline (Actividad 2.6)

This project currently uses:
- Laplacian-variance blur filtering (pre-filter)
- YOLO detection (single detector path)

Template matching was removed from runtime because it was too slow for your target realtime scenario.

## Current Model

The detector loads this model by default:
- `best.pt` at the workspace root

Configured in:
- `actividad_2_06.py`

## What Was Causing Slowness

A benchmark run showed the detector is currently using CPU, not GPU:
- `YOLO loaded device=cpu`
- Average YOLO inference: ~120.82 ms/frame
- Effective FPS: ~8.24
- Source video FPS: 40

So the bottleneck is YOLO inference speed on CPU, not blur filtering.

## Run Detection

CUDA is now required by default.
If CUDA is unavailable, the script exits with an error instead of silently using CPU.

Default run:

```bash
python3 test_against_video.py
```

With input file:

```bash
python3 test_against_video.py recording_mode2_20260423_113645.avi
```

With benchmark stats:

```bash
python3 test_against_video.py --benchmark
```

Allow CPU fallback (only if needed):

```bash
python3 test_against_video.py --cpu
```

Preview mode:

```bash
python3 test_against_video.py --show
```

## Output Organization

Generated outputs are now written to:
- `outputs/videos/`

This keeps the workspace root cleaner.

## Project Layout (Important Paths)

- `actividad_2_06.py`: detector logic (blur + YOLO)
- `test_against_video.py`: video evaluation script
- `train_yolo.py`: YOLO training launcher
- `signals/merged_dataset/data.yaml`: class mapping and dataset definition
- `best.pt`: latest trained model in use
- `outputs/videos/`: generated detection/debug videos

## Class Names (Unified)

The active class schema is:
- `stop`
- `workers`
- `go_straight`
- `turn_right`
- `turn_left`
- `give_way`

## Recommended Next Performance Steps

1. Keep CUDA runtime/device configuration stable so YOLO runs on GPU.
2. If GPU is not available, try a lighter model (for example, a nano variant).
3. Reduce input size only if detection quality stays acceptable.

## Cleanup Notes

Old generated `detected_*.avi` and `debug_*.avi` files in project root were removed to reclaim space and reduce clutter. Input recordings were kept.
The `frames_mode2_*` folders and associated archive were also removed in deep cleanup.
