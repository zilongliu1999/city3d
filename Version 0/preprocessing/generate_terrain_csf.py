import os
import numpy as np
import CSF  # Assuming the CSF Python bindings are correctly configured
import open3d as o3d
from datetime import datetime

# === Step 1: Setup input/output paths ===
input_folder = "../data/Yuehai"  # <-- change to your own path

# Get current date and time
now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H%M%S")  # HHMMSS format

# Create output folder like: output/2025-05-07/
output_base = "output"
output_folder = os.path.join(output_base, date_str)
os.makedirs(output_folder, exist_ok=True)

# Output filename with timestamp
# Output file path (indicating CSF-generated)
output_file = os.path.join(output_folder, f"total_ground_CSF_{time_str}.txt")

# === Step 2: Load files and extract ground points ===
txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
all_ground_points = []

for file_name in txt_files:
    input_path = os.path.join(input_folder, file_name)

    # Load the original point cloud (X Y Z R G B ...)
    raw_data = np.loadtxt(input_path)  # Assumes 9 columns
    xyz = raw_data[:, :3]

    # Run CSF filtering
    csf = CSF.CSF()
    csf.setPointCloud(xyz.tolist())
    csf.params.cloth_resolution = 0.5
    csf.params.bSloopSmooth = False

    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    # Extract ground points based on indices
    ground_indices = list(ground)
    ground_points = raw_data[ground_indices, :]
    all_ground_points.append(ground_points)

    print(f"Processed: {file_name} - Found {len(ground_indices)} ground points.")

# === Step 3: Merge and save result ===
total_ground = np.vstack(all_ground_points)
np.savetxt(output_file, total_ground, fmt="%.6f")
print(f"\nSaved {len(total_ground)} ground points to: {output_file}")

# === Step 4: Visualize using Open3D ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(total_ground[:, :3])
pcd.colors = o3d.utility.Vector3dVector(total_ground[:, 3:6] / 255.0)
o3d.visualization.draw_geometries([pcd], window_name="Ground Points")
