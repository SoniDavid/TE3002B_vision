"""
ICP point cloud registration using Open3D.

Aligns view2.ply (source) onto view1.ply (target) and saves the merged
reconstruction as merged.ply.

Pipeline:
  1. Load both PLYs
  2. Voxel downsample for speed
  3. FPFH features + RANSAC global registration (coarse alignment)
  4. Point-to-plane ICP refinement
  5. Merge transformed source + full-resolution target
  6. Save merged.ply

Usage:
    python 03_icp_registration.py \
        --source view2.ply --target view1.ply \
        --output merged.ply \
        --voxel 0.01
"""

import argparse
import sys
import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def load_pcd(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        sys.exit(f"[ERROR] No points in {path}. Check the PLY file.")
    print(f"  {path}: {len(pcd.points):,} points")
    return pcd


def preprocess(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """Downsample, estimate normals, compute FPFH features."""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 10, max_nn=30)
    )
    down.orient_normals_towards_camera_location()   # consistent normal direction

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 50, max_nn=100)
    )
    return down, fpfh


def global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size: float):
    """FPFH + RANSAC for coarse alignment."""
    dist_thresh = voxel_size * 1.5
    print(f"  RANSAC distance threshold: {dist_thresh:.4f} m")

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500)
    )
    return result


def icp_refine(src_down, tgt_down, init_transform, voxel_size: float):
    """Point-to-plane ICP for fine alignment."""
    dist_thresh = voxel_size * 5

    result = o3d.pipelines.registration.registration_icp(
        src_down, tgt_down,
        max_correspondence_distance=dist_thresh,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=100
        )
    )
    return result


def merge_clouds(src_full: o3d.geometry.PointCloud,
                 tgt_full: o3d.geometry.PointCloud,
                 transform: np.ndarray) -> o3d.geometry.PointCloud:
    """Apply transform to source, then combine with target (full resolution)."""
    src_transformed = o3d.geometry.PointCloud(src_full)
    src_transformed.transform(transform)
    merged = tgt_full + src_transformed
    return merged


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="ICP stereo view registration")
    parser.add_argument("--source",  default="view2.ply", help="Source PLY (view to align)")
    parser.add_argument("--target",  default="view1.ply", help="Target PLY (reference view)")
    parser.add_argument("--output",  default="merged.ply", help="Output merged PLY")
    parser.add_argument("--voxel",   type=float, default=0.01,
                        help="Voxel size in metres for downsampling (default 0.01)")
    args = parser.parse_args()

    print("Loading point clouds...")
    src_full = load_pcd(args.source)
    tgt_full = load_pcd(args.target)

    print(f"\nPreprocessing (voxel size = {args.voxel} m)...")
    src_down, src_fpfh = preprocess(src_full, args.voxel)
    tgt_down, tgt_fpfh = preprocess(tgt_full, args.voxel)
    print(f"  Source downsampled: {len(src_down.points):,} points")
    print(f"  Target downsampled: {len(tgt_down.points):,} points")

    print("\nGlobal registration (FPFH + RANSAC)...")
    ransac_result = global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, args.voxel)
    print(f"  Fitness:  {ransac_result.fitness:.4f}  (1.0 = perfect overlap)")
    print(f"  RMSE:     {ransac_result.inlier_rmse:.6f} m")
    if ransac_result.fitness < 0.1:
        print("  [WARNING] Low fitness — the two views may overlap too little, "
              "or calibration/disparity quality is poor.")

    print("\nICP refinement (point-to-plane)...")
    icp_result = icp_refine(src_down, tgt_down, ransac_result.transformation, args.voxel)
    print(f"  Fitness:  {icp_result.fitness:.4f}")
    print(f"  RMSE:     {icp_result.inlier_rmse:.6f} m")
    print(f"  Transformation:\n{np.round(icp_result.transformation, 4)}")

    print("\nMerging full-resolution clouds...")
    merged = merge_clouds(src_full, tgt_full, icp_result.transformation)
    print(f"  Merged cloud: {len(merged.points):,} points")

    o3d.io.write_point_cloud(args.output, merged, write_ascii=True)
    print(f"\nMerged cloud saved → {args.output}")
    print("Open in MeshLab: File → Import Mesh → select merged.ply")


if __name__ == "__main__":
    main()
