import numpy as np
import open3d as o3d
import os
from tkinter import Tk, filedialog

def visualize_txt_point_cloud(txt_path):
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"❌ File not found: {txt_path}")

    print(f"🔍 Loading point cloud from: {txt_path}")
    data = np.loadtxt(txt_path)

    if data.shape[1] < 6:
        raise ValueError("❌ File must have at least 6 columns (X Y Z R G B)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0)

    print(f"✅ Loaded {len(pcd.points)} points.")
    o3d.visualization.draw_geometries([pcd], window_name="TXT Point Cloud Viewer")

if __name__ == "__main__":
    # Hide main tkinter root window
    Tk().withdraw()
    print("📂 Please select a .txt point cloud file (with X Y Z R G B columns)...")
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])

    if file_path:
        visualize_txt_point_cloud(file_path)
    else:
        print("❌ No file selected. Exiting.")
