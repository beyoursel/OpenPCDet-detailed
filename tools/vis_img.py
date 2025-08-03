import cv2 
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from pcdet.utils import object3d_kitti

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

def draw_boxes(img: np.ndarray,
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


if __name__ == "__main__":
    img_folder = Path("/media/taole/mydisk/DL_PROJECT/OpenPCDet-detailed/data/kitti/training/image_2")
    img_list = list(img_folder.glob(f'*.png'))
    for index in range(len(img_list)):
        # index = 7
        sample_id = '{:06d}'.format(index)
        print(sample_id)
        img_file = ROOT_PATH / 'image_2' / ('%s.png' % str(sample_id))
        img = cv2.imread(img_file)
        labels = get_label(sample_id)
        annos = []
        for label in labels:
            anno = (label.cls_type, *label.box2d)
            annos.append(anno)
        img_rendering = draw_boxes(img, annos)
        cv2.imshow('kitti', img_rendering)
        cv2.waitKey(0)
        cv2.destroyAllWindows()