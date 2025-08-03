import numpy as np
import open3d
import sys

def intensity_to_rgb(intensity, cmap_name='turbo'):
    """把 0~1 的 intensity 转成 RGB，使用 Open3D 的颜色映射"""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(intensity)[:, :3]
    return rgb.astype(np.float64)

def draw_pointcloud(points, point_colors=None, draw_origin=True):
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    ptc_file = sys.argv[-1]
    pts = np.fromfile(ptc_file, dtype=np.float32).reshape(-1, 4) # x, y, z, intensity

    # 根据点云intensity来确定点云颜色
    intensity = pts[:, 3]
    # print(intensity.shape)
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    point_colors = intensity_to_rgb(intensity)
    draw_pointcloud(pts, point_colors=point_colors)