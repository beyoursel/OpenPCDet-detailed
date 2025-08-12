import numpy as np
from pathlib import Path
import open3d
from visual_utils import open3d_vis_utils as V
import sys
from pcdet.utils import object3d_kitti, calibration_kitti, box_utils

ROOT_PATH = Path("/media/taole/mydisk/DL_PROJECT/OpenPCDet-detailed/data/kitti/training")

def get_label(idx):
    label_file = ROOT_PATH / 'label_2' / ('%s.txt' % idx)
    assert label_file.exists()
    return object3d_kitti.get_objects_from_label(label_file)

def get_calib(idx):
    calib_file = ROOT_PATH / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)


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
    # ptc_file = sys.argv[-1]
    # pts = np.fromfile(ptc_file, dtype=np.float32).reshape(-1, 4) # x, y, z, intensity

    ptc_folder = Path("/media/taole/mydisk/DL_PROJECT/OpenPCDet-detailed/data/kitti/training/velodyne")
    ptc_list = sorted(list(ptc_folder.glob(f'*.bin')))
    for index in range(len(ptc_list)):
        if index % 10 != 0:
            continue
        # index = 10
        sample_id = '{:06d}'.format(index)
        print(sample_id)
        obj_list = get_label(sample_id)
        calib = get_calib(sample_id)

        annotations = {}
        annotations['name'] = np.array([obj.cls_type for obj in obj_list])
        annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
        annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
        annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
        annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0) # 2d box
        annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
        annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
        annotations['score'] = np.array([obj.score for obj in obj_list]) # 相对于相机坐标系x轴的偏航角
        annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
        num_gt = len(annotations['name'])
        index_object = list(range(num_objects)) + [-1] * (num_gt - num_objects) # num_objects为剔除DontCare的目标数量
        annotations['index'] = np.array(index_object, dtype=np.int32)

        loc = annotations['location'][:num_objects] # 根据num_objects取前num_objects的目标，这要求DontCare都排在后
        dims = annotations['dimensions'][:num_objects]
        rots = annotations['rotation_y'][:num_objects]
        loc_lidar = calib.rect_to_lidar(loc)
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2 
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)

        ptc_file = ptc_list[index]
        ptcs = np.fromfile(ptc_file, dtype=np.float32).reshape(-1, 4)


        # 根据点云intensity来确定点云颜色
        intensity = ptcs[:, 3]
        # print(intensity.shape)
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        point_colors = intensity_to_rgb(intensity)
        # draw_pointcloud(ptcs, point_colors=point_colors)
        V.draw_scenes(points=ptcs, gt_boxes=gt_boxes_lidar)