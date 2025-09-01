import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from pcdet.utils import object3d_kitti, calibration_kitti, box_utils

ROOT_PATH = Path("/media/taole/mydisk/DL_PROJECT/OpenPCDet-detailed/data/kitti/training")

Color = Tuple[int, int, int]

CLASS_COLORS: Dict[str, Color] = {
    'Car':        (0, 255, 0),     # 绿色
    'Pedestrian': (0, 255, 255),   # 黄色
    'Cyclist':    (255, 0, 0),     # 蓝色
    'Van':        (0, 128, 255),   # 橙色
    'Truck':      (0, 0, 255),     # 红色
    'Person_sitting': (255, 255, 0),  # 青色
    'Tram':       (128, 128, 128), # 灰色
    'Misc':       (255, 255, 255), # 白色
    'DontCare':   (0, 0, 0)        # 黑色
}

DEFAULT_COLOR = (0, 255, 0)  # 兜底绿色

def draw_boxes_2d(img: np.ndarray,
               boxes: List[Tuple[str, int, int, int, int]],
               thickness: int = 2,
               font_scale: float = 0.6) -> np.ndarray:
    """
    在图像上绘制 2D 检测框与类别标签。

    Args:
        img: H×W×3 的 BGR numpy 数组，**原地修改并返回**。
        boxes: [(label, x1, y1, x2, y2), ...]，坐标为整数像素。
        color: 框及文字背景颜色 (B,G,R)。
        thickness: 矩形线宽。
        font_scale: 字体大小。

    Returns:
        同一张已画好框的图像。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 选择颜色
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        # 画矩形
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 计算文字尺寸
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

        # 文字背景填充
        cv2.rectangle(img,
                      (x1, y1 - th - 4),
                      (x1 + tw, y1),
                      color, -1)

        # 写文字
        cv2.putText(img, label, (x1, y1 - 2),
                    font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def get_label(idx):
    label_file = ROOT_PATH / 'label_2' / ('%s.txt' % idx)
    assert label_file.exists()
    return object3d_kitti.get_objects_from_label(label_file)

def get_calib(idx):
    calib_file = ROOT_PATH / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)


def get_fov_flag(pts_rect, img_shape, calib):
    """
    Args:
        pts_rect:
        img_shape:
        calib:

    Returns:

    """
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag

def plot_box2d(img, boxes, label_list, thickness: int = 2,
               font_scale: float = 0.6) -> np.ndarray:

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(boxes.shape[0]):
        box = boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        label = label_list[i]

        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        # cv2.rectangle(img,
        #               (x1, y1 - th - 4),
        #               (x1 + tw, y1),
        #               color, -1)

        cv2.putText(img, label, (x1, y1 - 2),
                    font, font_scale, color, 1, cv2.LINE_AA)
    return img

def plot_bbox3d_in_image(img, boxes_corner, label_list=None):
    """
    boxes_corner: [N, 8, 2]
    img: raw image
    kitti: box3d
             4-------- 5
           /|         /|
          7 -------- 6 .
          | |        | |
          . 0 -------- 1
          |/         |/
          3 -------- 2

    front: 0-1-4-5
    bottom: 0-1-2-3
    up: 4-5-6-7
    """
    # 1. 转int
    boxes_corner = boxes_corner.astype(np.int32)

    # 2. 定义 12 条边的连接关系（KITTI 顺序）
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),   # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),   # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)]   # 竖直

    img_h, img_w = img.shape[:2] # (h, w, 3)

    # 生成掩码：8 个点全部在 (0<=x<w, 0<=y<h) 内
    mask = np.all(
            (boxes_corner[..., 0] >= 0) &
            (boxes_corner[..., 0] < img_w) &
            (boxes_corner[..., 1] >= 0) &
            (boxes_corner[..., 1] < img_h),
            axis=-1)

    valid_boxes = boxes_corner[mask]   # 只保留 8 个点都在图内的框
    label_list = np.array(label_list)
    label_list = label_list[mask]

    # 3. 画框
    for i in range(valid_boxes.shape[0]):               # 遍历每个物体
        corners = valid_boxes[i]

        if label_list is not None:
            color = CLASS_COLORS[label_list[i]]
        else:
            color = (0, 255, 0)
        for i, j in edges:
            cv2.line(img,
                    tuple(corners[i]),
                    tuple(corners[j]),
                    color=color,
                    thickness=2)

        cv2.line(img, tuple(corners[1]), tuple(corners[4]), color=color, thickness=2)
        cv2.line(img, tuple(corners[0]), tuple(corners[5]), color=color, thickness=2)

    return img

def plot_lidar_in_image(img, ptc_img, ptc_depth):

    import matplotlib.cm as cm
    cmap = cm.get_cmap("turbo")
    rgb = cmap(ptc_depth)[:, :3]
    h, w, _ = img.shape
  
    for i, point in enumerate(ptc_img):
        x, y = point

        if (x < 0 or x > w or y < 0 or y > h):
            continue
        color = rgb[i]
        cv2.circle(img, (int(x), int(y)), radius=1, color=(int(color[0]*255), int(color[1]*255), int(color[2]*255)), thickness=-1) 
    return img

if __name__ == "__main__":
    img_folder = Path("/media/taole/mydisk/DL_PROJECT/OpenPCDet-detailed/data/kitti/training/image_2")
    img_list = list(img_folder.glob(f'*.png'))
    for index in range(len(img_list)):
        if index % 10 != 0:
            continue
        # index = 10
        sample_id = '{:06d}'.format(index)
        print(sample_id)
        img_file = ROOT_PATH / 'image_2' / ('%s.png' % str(sample_id))
        img = cv2.imread(img_file)
        labels = get_label(sample_id)
        calib = get_calib(sample_id)

        ptc_file = ROOT_PATH / 'velodyne' / ('%s.bin' % str(sample_id))
        ptcs = np.fromfile(ptc_file, dtype=np.float32).reshape(-1, 4)
        pts_rect = calib.lidar_to_rect(ptcs[:, 0:3])
        fov_flag = get_fov_flag(pts_rect, img.shape, calib)
        ptcs = ptcs[fov_flag]


        ptcs_xyz = ptcs[:, :3]
        ptcs_to_img, ptc_depth = calib.lidar_to_img(ptcs_xyz)
        ptc_depth = (ptc_depth - ptc_depth.min()) / (ptc_depth.max() - ptc_depth.min() + 1e-6)
        img = plot_lidar_in_image(img, ptcs_to_img, ptc_depth)

        # plot 3d gt bbox in image
        corners_cam_list = []
        label_list = []
        for label in labels:
            corners_cam = label.generate_corners3d()
            corners_cam_list.append(corners_cam)
            label_list.append(label.cls_type)

        corners_cam_overall = np.stack(corners_cam_list, axis=0)
        boxes, boxes_corner = calib.corners3d_to_img_boxes(corners_cam_overall) # [N, 4], [N, 8, 2]
        img = plot_bbox3d_in_image(img, boxes_corner, label_list=label_list)
        # img = plot_box2d(img, boxes, label_list)
        # # plot 2d gt bbox in image
        # annos = []
        # for label in labels:
        #     anno = (label.cls_type, *label.box2d)
        #     annos.append(anno)
        # img_rendering = draw_boxes_2d(img, annos)
        cv2.imshow('kitti', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()