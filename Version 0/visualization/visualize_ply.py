import open3d as o3d
import os


def view_and_capture_ply(ply_path, output_image_path=None, show_normals=False):
    """
    Load and visualize a PLY file. Optionally save a screenshot.

    Parameters:
        ply_path (str): Path to the .ply file.
        output_image_path (str): Optional. Path to save the screenshot (.png).
        show_normals (bool): Whether to display normals as arrows.
    """
    if not os.path.exists(ply_path):
        print(f"[ERROR] File not found: {ply_path}")
        return

    print(f"[INFO] Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    if show_normals:
        print("[INFO] Estimating and showing normals...")
        pcd.estimate_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Viewer", width=1280, height=720)
    vis.add_geometry(pcd)

    # 设置渲染参数
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0

    if show_normals:
        render_opt.show_normal = True

    vis.poll_events()
    vis.update_renderer()

    if output_image_path:
        vis.capture_screen_image(output_image_path)
        print(f"[INFO] Screenshot saved to: {output_image_path}")

    vis.run()
    vis.destroy_window()
if __name__ == "__main__":
    view_and_capture_ply(
        ply_path="pointcloud_voxel_0.0.ply",
        output_image_path="pointcloud_voxel_0.0.ply",
        show_normals=False  # show the direction of normal
    )