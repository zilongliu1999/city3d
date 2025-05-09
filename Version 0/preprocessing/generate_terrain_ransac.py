import os
import numpy as np
import open3d as o3d
from datetime import datetime

# === Step 1: Set up paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = "../../data/Yuehai"
output_base = os.path.join(script_dir, "output")

now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H%M%S")

output_folder = os.path.join(output_base, date_str)
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, f"total_ground_RANSAC_{time_str}.txt")

# === Step 2: RANSAC parameters ===
ransac_distance_threshold = 0.3     # max point-to-plane distance
min_ground_plane_ratio = 0.1        # min ratio of points for a plane to be considered
max_planes_to_try = 5               # max number of planes to extract

all_ground_points = []

txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
for file_name in txt_files:
    input_path = os.path.join(input_folder, file_name)
    raw_data = np.loadtxt(input_path)
    xyz = raw_data[:, :3]

    # Create point cloud from XYZ
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    best_plane = None
    best_plane_z = np.inf
    best_plane_inliers = []

    remaining_pcd = pcd
    for _ in range(max_planes_to_try):
        # Fit a plane using RANSAC
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=ransac_distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        [a, b, c, d] = plane_model

        # Extract inlier points (potential ground)
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        inlier_xyz = np.asarray(inlier_cloud.points)

        # Check Z coordinate of this plane (lower is more likely to be ground)
        mean_z = np.mean(inlier_xyz[:, 2])
        point_ratio = len(inliers) / len(xyz)

        if point_ratio > min_ground_plane_ratio and mean_z < best_plane_z:
            best_plane = plane_model
            best_plane_inliers = inliers
            best_plane_z = mean_z

        # Remove inliers and continue searching
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    if best_plane_inliers:
        ground_points = raw_data[best_plane_inliers, :]
        all_ground_points.append(ground_points)
        print(f"Processed: {file_name} - Ground points: {len(ground_points)} - Mean Z: {best_plane_z:.2f}")
    else:
        print(f"Processed: {file_name} - No suitable ground plane found.")

# === Step 3: Merge and save results ===
if all_ground_points:
    total_ground = np.vstack(all_ground_points)
    np.savetxt(output_file, total_ground, fmt="%.6f")
    print(f"\n✅ Saved {len(total_ground)} ground points to: {output_file}")
else:
    print("⚠️ No ground points extracted from any file.")

# === Step 4: Visualize ground points ===
if all_ground_points:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(total_ground[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(total_ground[:, 3:6] / 255.0)
    o3d.visualization.draw_geometries([pcd], window_name="RANSAC Ground Points")
