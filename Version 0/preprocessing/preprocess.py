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

                merged_points.append(data[:, :3])
                if use_rgb:
                    merged_colors.append(data[:, 3:6] / 255.0)

            except Exception as e:
                print(f"[ERROR] Failed to read {filename}: {e}")

    if not merged_points:
        raise ValueError(f"No valid point cloud files found in: {folder_path}")

    all_points = np.vstack(merged_points)
    all_colors = np.vstack(merged_colors) if use_rgb and merged_colors else None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    if all_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

    print(f"[INFO] Merged {len(merged_points)} files, total points: {len(all_points)}")
    return pcd

def save_point_cloud(pcd, output_folder="output", name="merged_output.ply"):
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, name)
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"[INFO] Saved point cloud to: {file_path}")

if __name__ == "__main__":
    folder_path = "../../data/Yuehai"

    print("[INFO] Loading and merging point cloud files...")
    start = time.time()
    pcd = load_and_merge_txt_folder(folder_path)
    print(f"[INFO] Done in {time.time() - start:.2f}s")

    # visualize
    o3d.visualization.draw_geometries([pcd], window_name="Merged Point Cloud")

    # save the output
    save_point_cloud(pcd, output_folder="output", name="merged_output.ply")
