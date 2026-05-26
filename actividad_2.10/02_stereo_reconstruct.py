"""
Stereo reconstruction: left+right image pair → disparity map → XYZ point cloud → PLY.

Run once per view:
    python 02_stereo_reconstruct.py \
        --left left1.png --right right1.png \
        --calib calibration.npz \
        --output view1.ply

    python 02_stereo_reconstruct.py \
        --left left2.png --right right2.png \
        --calib calibration.npz \
        --output view2.ply

Without calibration (rough mode):
    python 02_stereo_reconstruct.py \
        --left left1.png --right right1.png \
        --baseline 0.12 --focal 1400 \
        --output view1.ply
"""

import argparse
import sys
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  PLY export (same format as actividad_2.9/clase14.py)                       #
# --------------------------------------------------------------------------- #

def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Save a coloured point cloud as ASCII PLY."""
    assert len(points) == len(colors)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


# --------------------------------------------------------------------------- #
#  Calibration helpers                                                         #
# --------------------------------------------------------------------------- #

def load_calibration(calib_path: str):
    data = np.load(calib_path)
    return (data["K1"], data["D1"], data["K2"], data["D2"],
            data["R1"], data["R2"], data["P1"], data["P2"],
            data["Q"], tuple(data["image_size"].tolist()))


def build_Q_from_estimates(focal: float, baseline: float, cx: float, cy: float):
    """
    Build the Q (disparity-to-depth) matrix for a simplified rectified stereo pair.

    Q maps [u, v, d, 1]^T → [X, Y, Z, W]^T where X/W, Y/W, Z/W are 3-D coords.
    """
    Q = np.float64([
        [1,  0,   0,   -cx],
        [0,  1,   0,   -cy],
        [0,  0,   0, focal],
        [0,  0, -1/baseline, 0]
    ])
    return Q


# --------------------------------------------------------------------------- #
#  Rectification                                                               #
# --------------------------------------------------------------------------- #

def rectify_pair(img_l, img_r, K1, D1, K2, D2, R1, R2, P1, P2, image_size):
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    rect_l = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR)
    return rect_l, rect_r


# --------------------------------------------------------------------------- #
#  Disparity                                                                   #
# --------------------------------------------------------------------------- #

def compute_disparity(rect_l, rect_r):
    """StereoSGBM tuned for indoor scenes at ~0.5–3 m depth."""
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    num_disp = 128   # must be divisible by 16; increase for closer baselines
    block    = 11    # odd number; larger = smoother but less detail
    P1       = 8  * 3 * block ** 2
    P2       = 32 * 3 * block ** 2

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
    return disp


def save_disparity_vis(disp: np.ndarray, path: str) -> None:
    valid = disp > 0
    if not valid.any():
        return
    norm = np.zeros_like(disp, dtype=np.uint8)
    cv2.normalize(disp, norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=valid.astype(np.uint8))
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    cv2.imwrite(path, colored)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Stereo → PLY reconstruction")
    parser.add_argument("--left",      required=True, help="Left rectified image")
    parser.add_argument("--right",     required=True, help="Right rectified image")
    parser.add_argument("--calib",     default=None,  help="calibration.npz from 01_stereo_calibrate.py")
    parser.add_argument("--baseline",  type=float, default=0.12,
                        help="Baseline in metres (used when --calib is not provided, default 0.12)")
    parser.add_argument("--focal",     type=float, default=None,
                        help="Focal length in px (used when --calib is not provided; auto-estimated from image width if omitted)")
    parser.add_argument("--max-depth", type=float, default=5.0,
                        help="Discard points farther than this (metres, default 5.0)")
    parser.add_argument("--output",    default="view1.ply", help="Output PLY file")
    args = parser.parse_args()

    img_l = cv2.imread(args.left)
    img_r = cv2.imread(args.right)
    if img_l is None:
        sys.exit(f"[ERROR] Cannot read: {args.left}")
    if img_r is None:
        sys.exit(f"[ERROR] Cannot read: {args.right}")

    if img_l.shape != img_r.shape:
        sys.exit("[ERROR] Left and right images must have the same resolution.")

    h, w = img_l.shape[:2]

    # --- Load or estimate calibration ---
    if args.calib is not None:
        print(f"Loading calibration from {args.calib}")
        K1, D1, K2, D2, R1, R2, P1, P2, Q, image_size = load_calibration(args.calib)

        if (w, h) != tuple(image_size):
            print(f"[WARNING] Image size {(w,h)} differs from calibration size {tuple(image_size)}. "
                  "Resizing images to match calibration.")
            img_l = cv2.resize(img_l, tuple(image_size))
            img_r = cv2.resize(img_r, tuple(image_size))
            w, h = image_size

        print("Rectifying images...")
        rect_l, rect_r = rectify_pair(img_l, img_r, K1, D1, K2, D2, R1, R2, P1, P2, (w, h))
    else:
        print("[INFO] No calibration file — using estimated parameters (rough mode).")
        focal = args.focal if args.focal is not None else w * 1.2
        cx, cy = w / 2.0, h / 2.0
        Q = build_Q_from_estimates(focal, args.baseline, cx, cy)
        print(f"  focal={focal:.1f} px  baseline={args.baseline*100:.1f} cm  cx={cx:.1f}  cy={cy:.1f}")
        rect_l, rect_r = img_l, img_r  # assume already roughly rectified

    # --- Disparity ---
    print("Computing disparity (SGBM)...")
    disp = compute_disparity(rect_l, rect_r)

    disp_vis_path = args.output.replace(".ply", "_disparity.png")
    save_disparity_vis(disp, disp_vis_path)
    print(f"Disparity visualization → {disp_vis_path}")

    # --- Reproject to 3D ---
    print("Reprojecting to 3D...")
    points_3d = cv2.reprojectImageTo3D(disp, Q)   # shape (H, W, 3)
    color_rgb  = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)

    # Build validity mask
    valid = (
        (disp > 0) &
        np.isfinite(points_3d[:, :, 2]) &
        (points_3d[:, :, 2] > 0) &
        (points_3d[:, :, 2] < args.max_depth)
    )

    pts  = points_3d[valid]
    cols = color_rgb[valid]

    print(f"Valid points: {len(pts):,}")
    if len(pts) == 0:
        sys.exit("[ERROR] No valid 3D points. Check calibration, disparity parameters, and image pair.")

    print(f"Saving PLY → {args.output}")
    save_ply(args.output, pts, cols)
    print("Done.")


if __name__ == "__main__":
    main()
