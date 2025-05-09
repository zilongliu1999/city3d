import os
import numpy as np
import open3d as o3d
import subprocess
from datetime import datetime
import json

# === Step 1: Setup paths ===

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input folder relative to script location
input_folder = os.path.join(script_dir, "..", "data", "Yuehai")

# Create dated output folder under "output/yyyy-mm-dd"
now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H%M%S")

output_base = os.path.join(script_dir, "output")
output_folder = os.path.join(output_base, date_str)
os.makedirs(output_folder, exist_ok=True)

# Output file name (includes method and timestamp)
output_file = os.path.join(output_folder, f"total_ground_SMRF_{time_str}.txt")

# Temporary files used for PDAL pipeline
temp_xyz = os.path.join(script_dir, "temp_input.xyz")
temp_ground = os.path.join(script_dir, "temp_ground.xyz")
pipeline_file = os.path.join(script_dir, "smrf_pipeline.json")

# List to store all ground points from all files
all_ground_points = []

# === Step 2: Process each TXT file ===
txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

for file_name in txt_files:
    input_path = os.path.join(input_folder, file_name)

    # Load the original point cloud (expects 9 columns: X Y Z R G B feather1-3)
    raw_data = np.loadtxt(input_path)
    xyz = raw_data[:, :3]

    # Save XYZ to temporary file for PDAL input
    np.savetxt(temp_xyz, xyz, fmt="%.6f")

    # Create PDAL SMRF pipeline JSON
    pipeline = [
        {
            "type": "readers.text",
            "filename": temp_xyz,
            "header": "X Y Z"
        },
        {
            "type": "filters.smrf",
            "slope": 0.2,
            "window": 16.0,
            "threshold": 0.45,
            "scalar": 1.25
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"
        },
        {
            "type": "writers.text",
            "filename": temp_ground,
            "keep_unspecified": "false"
        }
    ]

    with open(pipeline_file, "w") as f:
        json.dump(pipeline, f)

    # Run PDAL pipeline
    subprocess.run(["pdal", "pipeline", pipeline_file], check=True)

    ground_xyz = np.loadtxt(temp_ground, delimiter=",", skiprows=1)

    if ground_xyz.ndim == 1:
        ground_xyz = ground_xyz.reshape(1, -1)

    ground_xyz = ground_xyz[:, :3]

    # Match XYZ from original data to retrieve full 9-column ground points
    dtype = np.dtype([('x', float), ('y', float), ('z', float)])
    xyz_view = xyz.view(dtype)
    ground_view = ground_xyz.view(dtype)
    mask = np.isin(xyz_view, ground_view).all(1)
    ground_points = raw_data[mask]

    all_ground_points.append(ground_points)
    print(f"Processed: {file_name} - Ground points: {len(ground_points)}")

# === Step 3: Merge and save results ===
total_ground = np.vstack(all_ground_points)
np.savetxt(output_file, total_ground, fmt="%.6f")
print(f"\nSaved {len(total_ground)} total ground points to: {output_file}")

# === Step 4: Visualize using Open3D ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(total_ground[:, :3])
pcd.colors = o3d.utility.Vector3dVector(total_ground[:, 3:6] / 255.0)
o3d.visualization.draw_geometries([pcd], window_name="SMRF Ground Points")

# === Step 5: Cleanup temporary files ===
os.remove(temp_xyz)
os.remove(temp_ground)
os.remove(pipeline_file)
