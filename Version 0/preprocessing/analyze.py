import open3d as o3d
import numpy as np

def print_point_cloud_summary(pcd):
    points = np.asarray(pcd.points)

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    print("[SUMMARY] Point cloud statistics:")
    print(f"  - Total points: {len(points)}")
    print(f"  - X range: {min_xyz[0]:.6f} to {max_xyz[0]:.6f}")
    print(f"  - Y range: {min_xyz[1]:.6f} to {max_xyz[1]:.6f}")
    print(f"  - Z range: {min_xyz[2]:.6f} to {max_xyz[2]:.6f}")

    if pcd.has_colors():
        print("  - Colors: Present")
    else:
        print("  - Colors: Not present")

def print_extreme_points(pcd):
    points = np.asarray(pcd.points)

    # get the index of the extreme points
    x_min_idx = np.argmin(points[:, 0])
    x_max_idx = np.argmax(points[:, 0])
    y_min_idx = np.argmin(points[:, 1])
    y_max_idx = np.argmax(points[:, 1])
    z_min_idx = np.argmin(points[:, 2])
    z_max_idx = np.argmax(points[:, 2])

    print("[INFO] Extreme points (actual points in point cloud):")
    print(f"  - X min: {points[x_min_idx]}")
    print(f"  - X max: {points[x_max_idx]}")
    print(f"  - Y min: {points[y_min_idx]}")
    print(f"  - Y max: {points[y_max_idx]}")
    print(f"  - Z min: {points[z_min_idx]}")
    print(f"  - Z max: {points[z_max_idx]}")

if __name__ == "__main__":
    # ✅ Correct way: load PLY file into a PointCloud object first
    pcd = o3d.io.read_point_cloud("pointcloud_voxel_0.0.ply")
    print_point_cloud_summary(pcd)
    print_extreme_points(pcd)