# kitti数据集介绍

[CSDN之kitti-3D数据集解析](https://blog.csdn.net/qq_16137569/article/details/118873033)

## label

|字段|	字段长度|	单位|	含义|
|--------|-----------|-------|--------------|
|Type	|1	|-	|目标类型|
|Truncated|	1	|-	|目标截断程度：0~1之间的浮点数,表示目标距离图像边界的程度|
|Occluded	|1	|-	|目标遮挡程度：0~3之间的整数, 0：完全可见 1：部分遮挡 2：大部分遮挡 3：未知|
|Alpha|	1|	弧度	|目标观测角：[−pi, pi]|
|Bbox|	4|	像素	|目标2D检测框位置：左上顶点和右下顶点的像素坐标|
|Dimensions|	3	|米|	3D目标尺寸：高、宽、长|
|Location|	3|	米	|目标3D框上底面中心坐标：(x, y,z)，相机坐标系|
|Rotation_y| 1	|弧度|	目标朝向角：[−pi, pi]|

annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
在label.txt中DontCare类别都排在最后

## 传感器坐标系

• Camera: x = right, y = down, z = forward
• Velodyne: x = forward, y = left, z = up
• GPS/IMU: x = forward, y = left, z = up