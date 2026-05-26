"""
Stereo camera calibration from two smartphone videos.

Each phone records its own video of a checkerboard independently — no sync needed.
The i-th valid frame from the left video is paired with the i-th valid frame from
the right video (valid = checkerboard detected). Both phones should cover similar
checkerboard poses during recording.

Usage:
    python 01_stereo_calibrate.py \
        --left left_calib.mp4 --right right_calib.mp4 \
        --pattern 9x6 --square 0.025 \
        --frame-step 10 --output calibration.npz
"""

import argparse
import sys
import cv2
import numpy as np


def extract_corners(video_path: str, pattern: tuple[int, int], frame_step: int):
    """Extract valid checkerboard corners from every `frame_step`-th frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    frame_idx = 0
    valid_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern, None)
            if found:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                valid_frames.append((gray.shape[::-1], corners_refined))
        frame_idx += 1

    cap.release()
    print(f"  {video_path}: {frame_idx} frames read, {len(valid_frames)} with corners")
    return valid_frames


def main():
    parser = argparse.ArgumentParser(description="Stereo calibration from two videos")
    parser.add_argument("--left",       required=True, help="Left camera calibration video")
    parser.add_argument("--right",      required=True, help="Right camera calibration video")
    parser.add_argument("--pattern",    default="9x6", help="Checkerboard inner corners WxH (default 9x6)")
    parser.add_argument("--square",     type=float, default=0.025, help="Square size in metres (default 0.025)")
    parser.add_argument("--frame-step", type=int, default=10, help="Sample every N frames (default 10)")
    parser.add_argument("--output",     default="calibration.npz", help="Output calibration file")
    args = parser.parse_args()

    w, h = map(int, args.pattern.lower().split("x"))
    pattern = (w, h)

    # World coordinates for one checkerboard pose
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * args.square

    print("Extracting corners from left video...")
    left_frames = extract_corners(args.left, pattern, args.frame_step)
    print("Extracting corners from right video...")
    right_frames = extract_corners(args.right, pattern, args.frame_step)

    # Pair by index: i-th valid left ↔ i-th valid right
    n_pairs = min(len(left_frames), len(right_frames))
    if n_pairs < 10:
        print(f"[WARNING] Only {n_pairs} pairs found — results may be inaccurate. "
              "Try reducing --frame-step or showing more checkerboard poses.")
    if n_pairs == 0:
        sys.exit("[ERROR] No valid pairs found. Check video paths and checkerboard pattern.")

    print(f"Using {n_pairs} pairs for calibration.")

    obj_points  = [objp] * n_pairs
    left_pts    = [left_frames[i][1]  for i in range(n_pairs)]
    right_pts   = [right_frames[i][1] for i in range(n_pairs)]
    image_size  = left_frames[0][0]   # (width, height)

    # Individual camera calibrations (provides good initial K, D estimates)
    print("Calibrating left camera...")
    _, K1, D1, _, _ = cv2.calibrateCamera(obj_points, left_pts, image_size, None, None)
    print("Calibrating right camera...")
    _, K2, D2, _, _ = cv2.calibrateCamera(obj_points, right_pts, image_size, None, None)

    # Stereo calibration
    print("Running stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC  # keep individual calibrations, only solve R/T
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, left_pts, right_pts,
        K1, D1, K2, D2,
        image_size, flags=flags
    )
    print(f"Stereo reprojection RMS: {rms:.4f} px", end="")
    if rms > 1.0:
        print("  [WARNING] RMS > 1.0 px — consider recapturing with more varied poses")
    else:
        print("  [OK]")

    baseline_m = np.linalg.norm(T)
    print(f"Baseline: {baseline_m*100:.1f} cm")

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        alpha=0,          # crop to valid pixels
        newImageSize=(0, 0)
    )

    np.savez(args.output,
             K1=K1, D1=D1, K2=K2, D2=D2,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             image_size=np.array(image_size))

    print(f"Calibration saved → {args.output}")
    print(f"  fx={K1[0,0]:.1f}  fy={K1[1,1]:.1f}  cx={K1[0,2]:.1f}  cy={K1[1,2]:.1f}")


if __name__ == "__main__":
    main()
