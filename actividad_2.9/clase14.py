import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

# 1. Load the K matrix from the CSV file
K = []
with open("camera_intrinsics.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row and not row[0].startswith("#"):
            K.append([float(v) for v in row])
K = np.array(K)
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

# 2. Load the aligned color image
color_image = cv2.imread("aligned_color.png")
color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

depth_image = cv2.imread("aligned_depth_raw.png", cv2.IMREAD_UNCHANGED)

# 3. 3D Coordinate Calculation
height, width = depth_image.shape

# Pixel coordinate grids
u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

# Depth in meters (RealSense stores millimeters in 16-bit PNG)
Z = depth_image.astype(np.float32) / 1000.0

# Pinhole deprojection (same formula as stereo triangulation):
#   X = (u - cx) * Z / fx
#   Y = (v - cy) * Z / fy
X = (u_coords - cx) * Z / fx
Y = (v_coords - cy) * Z / fy

# Keep only valid (non-zero depth) points
valid = Z > 0
points_xyz = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
colors_rgb = color_rgb[valid]

print(f"Valid points: {len(points_xyz):,}")

# 4. Display the 3D result
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

step = 6
sub_pts = points_xyz[::step]
sub_clr = colors_rgb[::step]
mask = sub_pts[:, 2] < 3.0  # clip noise beyond 3 m


# Remap to natural camera orientation:
#   matplotlib X  = camera X  (left → right)
#   matplotlib Y  = camera Z  (depth, goes into screen)
#   matplotlib Z  = -camera Y (up, because image Y is positive-downward)
ax.scatter(
    sub_pts[mask, 0],
    sub_pts[mask, 2],
    -sub_pts[mask, 1],
    c=sub_clr[mask] / 255.0,
    s=1,
    depthshade=False
)
ax.set_xlabel("X (m)")
ax.set_ylabel("Z / depth (m)")
ax.set_zlabel("Y (m)")
ax.view_init(elev=5, azim=0)
ax.set_title("RealSense D435i — 3D Point Cloud")
plt.tight_layout()
plt.savefig("pointcloud_3d.png", dpi=150)
plt.show()

# 5. Export to PLY format (for MeshLab)
ply_path = "output.ply"
with open(ply_path, "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(points_xyz)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb):
        f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

print(f"PLY saved → {ply_path}")
