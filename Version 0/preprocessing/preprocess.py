import os
import open3d as o3d
import numpy as np
import time


def load_and_merge_txt_folder(folder_path, use_rgb=True):
    merged_points = []
    merged_colors = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                data = np.loadtxt(file_path)

                if data.shape[1] < 6:
                    print(f"[WARN] Skipping {filename}: less than 6 columns")
                    continue

                # Extract x, y, z
                merged_points.append(data[:, :3])

                # Extract r, g, b
                if use_rgb:
                    merged_colors.append(data[:, 3:6] / 255.0)  # normalize to 0~1

            except Exception as e:
                print(f"[ERROR] Failed to read {filename}: {e}")

    if not merged_points:
        raise ValueError(f"No valid point cloud files found in: {folder_path}")

    # Merge all points and colors
    all_points = np.vstack(merged_points)
    all_colors = np.vstack(merged_colors) if use_rgb and merged_colors else None

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    if all_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

    print(f"[INFO] Merged {len(merged_points)} files, total points: {len(all_points)}")
    return pcd

def downsample(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"[INFO] Downsampled to {len(pcd_down.points)} points with voxel size {voxel_size}")
    return pcd_down

def estimate_normals(pcd, radius=0.5, max_nn=30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    print(f"[INFO] Normals estimated with radius={radius} and max_nn={max_nn}")
    return pcd

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    pcd_clean = pcd.select_by_index(ind)
    print(f"[INFO] Removed outliers: from {len(pcd.points)} to {len(pcd_clean.points)} points")
    return pcd_clean

def save_point_cloud(pcd, voxel_size, output_folder="output"):
    import os

    os.makedirs(output_folder, exist_ok=True)  # Create output directory if it doesn't exist

    # Construct filename with one decimal place (e.g., voxel_0.2.ply)
    filename = f"pointcloud_voxel_{voxel_size:.1f}.ply"
    file_path = os.path.join(output_folder, filename)

    # Save as PLY format
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"[INFO] Saved point cloud to: {file_path}")

if __name__ == "__main__":

    voxel_size = 0.0001
    folder_path = r"D:\Desktop\3d model\city3d\data\Yuehai"

    print("[INFO] Loading and merging point cloud files...")
    start = time.time()
    pcd = load_and_merge_txt_folder(folder_path)
    print(f"[INFO] Merging done in {time.time() - start:.2f}s")

    print("[INFO] Starting voxel downsampling...")
    start = time.time()
    pcd = downsample(pcd, voxel_size)
    print(f"[INFO] Downsampling done in {time.time() - start:.2f}s")

    print("[INFO] Starting normal estimation...")
    start = time.time()
    pcd = estimate_normals(pcd)
    print(f"[INFO] Normal estimation done in {time.time() - start:.2f}s")

    print("[INFO] Starting outlier removal...")
    start = time.time()
    pcd = remove_outliers(pcd)
    print(f"[INFO] Outlier removal done in {time.time() - start:.2f}s")

    print("[INFO] Starting outlier removal...")
    start = time.time()
    pcd = remove_outliers(pcd)
    print(f"[INFO] Outlier removal done in {time.time() - start:.2f}s")

    print("[INFO] Starting visualization...")
    o3d.visualization.draw_geometries([pcd], window_name="Merged Point Cloud")
    save_point_cloud(pcd, voxel_size)
