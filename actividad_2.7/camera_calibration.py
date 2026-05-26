import glob
import json
import os
import sys
import numpy as np
import cv2

# ── chessboard config 
INNER_COLS   = 7          # inner corners per row
INNER_ROWS   = 5          # inner corners per column
SQUARE_SIZE  = 1.0        # arbitrary units (1 = 1 square side)

IMAGES_DIR   = 'images_usb_cam_calibration'
SUBPIX_CRIT  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

#  prepare the 3-D object point template (same for every image) 
objp = np.zeros((INNER_ROWS * INNER_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:INNER_COLS, 0:INNER_ROWS].T.reshape(-1, 2) * SQUARE_SIZE

obj_points = []   # 3-D points in world space
img_points = []   # 2-D points in image plane
used_files = []

image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, '*.jpg')) +
                     glob.glob(os.path.join(IMAGES_DIR, '*.png')))

if not image_paths:
    print(f"No images found in '{IMAGES_DIR}/'")
    sys.exit(1)

print(f"Processing {len(image_paths)} images...")

for path in image_paths:
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (INNER_COLS, INNER_ROWS))
    if not ret:
        print(f"  [skip] {os.path.basename(path)}")
        continue

    corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRIT)

    obj_points.append(objp)
    img_points.append(corners_sub)
    used_files.append(path)
    print(f"  [ok]   {os.path.basename(path)}")

if len(obj_points) < 5:
    print(f"\nToo few valid images ({len(obj_points)}). Need at least 5.")
    sys.exit(1)

h, w = cv2.imread(used_files[0]).shape[:2]

print(f"\nCalibrating with {len(obj_points)} images ({w}x{h})...")

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, (w, h), None, None
)

k1, k2 = dist[0, 0], dist[0, 1]

#  per-image reprojection errors 
errors = []
for i, (objp_i, imgp_i, rvec, tvec) in enumerate(zip(obj_points, img_points, rvecs, tvecs)):
    projected, _ = cv2.projectPoints(objp_i, rvec, tvec, K, dist)
    err = cv2.norm(imgp_i, projected, cv2.NORM_L2) / len(projected)
    errors.append(err)

rmse = np.sqrt(np.mean(np.array(errors) ** 2))

#  results 
print("\ Camera Matrix K ")
print(f"  fx = {K[0,0]:.4f}   fy = {K[1,1]:.4f}")
print(f"  cx = {K[0,2]:.4f}   cy = {K[1,2]:.4f}")
print(f"\n  K =\n{K}")
print(f"\ Distortion coefficients ")
print(f"  k1 = {k1:.6f}")
print(f"  k2 = {k2:.6f}")
print(f"  Full dist vector: {dist}")
print(f"\ Reprojection error ")
for i, (path, err) in enumerate(zip(used_files, errors)):
    print(f"  {os.path.basename(path):30s}  {err:.4f} px")
print(f"\n  RMSE = {rmse:.4f} px  (cv2.calibrateCamera RMS = {rms:.4f} px)")

#  undistort a sample image and show it 
sample_path = used_files[len(used_files) // 2]
sample      = cv2.imread(sample_path)
undistorted = cv2.undistort(sample, K, dist)

side_by_side = np.hstack([sample, undistorted])

font  = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.6
cv2.putText(side_by_side, 'Original',    (10, 28),     font, scale, (0, 0, 200),   2)
cv2.putText(side_by_side, 'Undistorted', (w + 10, 28), font, scale, (0, 180, 0),   2)
cv2.putText(side_by_side,
            f'RMSE={rmse:.4f}px  k1={k1:.4f}  k2={k2:.4f}',
            (10, h - 12), font, 0.45, (255, 255, 255), 1)

params = {
    'K':    K.tolist(),
    'dist': dist.tolist(),
    'rms':  float(rms),
    'rmse': float(rmse),
}
with open('camera_params.json', 'w') as f:
    json.dump(params, f, indent=2)
print(f"\nParameters saved to: camera_params.json")
print("  Load with: import json, numpy as np")
print("             data = json.load(open('camera_params.json'))")
print("             K, dist = np.array(data['K']), np.array(data['dist'])")

cv2.imshow('Calibration result', side_by_side)
cv2.imwrite('undistorted_sample.jpg', undistorted)
print(f"\nSample: {os.path.basename(sample_path)}")
print("Undistorted image saved to: undistorted_sample.jpg")
print("Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
