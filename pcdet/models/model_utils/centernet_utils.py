# This file is modified from https://github.com/tianweiy/CenterPoint

import torch
import torch.nn.functional as F
import numpy as np
import numba


def gaussian_radius(height, width, min_overlap=0.5):
    """
    Args: 高斯半径决定了目标对周围区域的影响范围
        height: (N)
        width: (N)
        min_overlap:
    Returns:
    参考https://zhuanlan.zhihu.com/p/96856635
    """
    # 预测框两个角点在GT框的两个角点以r为半径的圆内，如何确定半径r，保证预测框与真值框的IOU大于一个阈值
    """
    1.一角点在真值框内,一角点在真值框外
    最小IOU在预测框两个角点分别和和半径r的圆相外切和相内切时取得(例如可以固定某一角点在x方向不变,变动y方向观察相交、相并面积的变化情况)
    因此我们只需要考虑“预测的框和GTbox两个角点以r为半径的圆一个边内切,一个边外切
    min_overlap =(h-r)*(w-r)/(2*h*w-(h-r)*(w-r)) --> r
    整理为r的一元二次方程: r^2 - (h+w)*r + (1-min_overlap)*h*w / (1+min_overlap) =0
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2 # cornerNet修改为r1  = (b1 - sq1) / (2 * a1)

    """
    2.两角点均在真值框内
    最小IOU在预测框和半径r圆相切获取
    min_overlap =(h-2*r)*(w-2*r)/(h*w) --> r
    整理为r的一元二次方程: 4*r^2 - 2*(h+w)*r + (1-min_overlap)*h*w =0
    """
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2 # cornernet修改为r2  = (b2 - sq2) / (2 * a2)

    """
    3.两角点均在真值框外
    最小IOU在预测框和半径r相外切时取得,只需要考虑 预测的框和GTbox两个角点以r为半径的圆外切
    min_overlap =(h*w)*(w+2*r)/(h+2*r) --> r
    整理为r的一元二次方程: 4*min_overlap*r^2 + 2*min_overlap*(h+w)*r + (min_overlap-1)*h*w =0
    """
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2 # cornernet修改为 r3  = (b3 + sq3) / (2 * a3)
    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape] # get radius
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0 # h.max()=1, 若gaussian2d内的值过小，则设置为0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1, valid_mask=None):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2] # heatmap shape

    left, right = min(x, radius), min(width - x, radius + 1) # 限制高斯渲染不超过feature map的边界，radius+1考虑到索引时最右侧取不到
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right] # 需要渲染的局部heatmap
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device).float() # 赋值

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        if valid_mask is not None:
            cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
            masked_gaussian = masked_gaussian * cur_valid_mask.float()
        # 下面k=1相当于没有变化
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap) # 比较masked_heatmap和masked_gaussian * k对应位置的值，取最大值作为输出
    return heatmap


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def gaussian3D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap_voxels(heatmap, distances, radius, k=1):
    diameter = 2 * radius + 1
    sigma = diameter / 6
    masked_gaussian = torch.exp(- distances / (2 * sigma * sigma))

    torch.max(heatmap, masked_gaussian, out=heatmap)

    return heatmap


@numba.jit(nopython=True)
def circle_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) # 扩展ind的最后一个维度和feat一致
    feat = feat.gather(1, ind) # 在第二维度进行索引，输出的shape和ind一致，out[i][j][k] = feat[i][ind[i][j][k]][k]
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous() # 2, 3为feature map维度，1为特征维数
    feat = feat.view(feat.size(0), -1, feat.size(3)) # 展平feature map
    feat = _gather_feat(feat, ind) # ind也是feature map展平后的索引
    return feat


def _topk(scores, K=40):
    batch, num_class, height, width = scores.size()
    # topk_scores为每个batch和每个channel前K个最大值，topk_inds为每个batch在展平维度上的索引位置
    topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), K) # 将2,3维度展平, 即feature map的h, w维度

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float() # 还原feature map上的位置
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K) # 所有类别中(num_class * K)选取topk
    topk_classes = (topk_ind // K).int() # 得到class id
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K) # 得到每个batch的topk在feature map上flatten的索引
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K) # 获得在feature map上对应的y坐标
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K) # 获得在feature map上的x坐标

    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


def decode_bbox_from_heatmap(heatmap, rot_cos, rot_sin, center, center_z, dim,
                             point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, iou=None, K=100,
                             circle_nms=False, score_thresh=None, post_center_limit_range=None):
    batch_size, num_class, _, _ = heatmap.size()

    if circle_nms:
        # TODO: not checked yet
        assert False, 'not checked yet'
        heatmap = _nms(heatmap)

    scores, inds, class_ids, ys, xs = _topk(heatmap, K=K) # select topk
    center = _transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
    rot_sin = _transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
    rot_cos = _transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
    center_z = _transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)

    angle = torch.atan2(rot_sin, rot_cos) # 单独的sin或cos无法唯一确定角度, atan2的输出在[-pi, pi]之间连续
    xs = xs.view(batch_size, K, 1) + center[:, :, 0:1] # xs为整数部分，center为小数部分
    ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]

    xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0] # decoder到实际世界坐标
    ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

    box_part_list = [xs, ys, center_z, dim, angle]
    if vel is not None:
        vel = _transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
        box_part_list.append(vel)

    if iou is not None:
        iou = _transpose_and_gather_feat(iou, inds).view(batch_size, K)

    final_box_preds = torch.cat((box_part_list), dim=-1)
    final_scores = scores.view(batch_size, K)
    final_class_ids = class_ids.view(batch_size, K)

    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2) # 仅输出在post_center_limit_range内的目标

    if score_thresh is not None:
        mask &= (final_scores > score_thresh) # 输出满足score_thresh的目标

    ret_pred_dicts = []
    for k in range(batch_size):
        cur_mask = mask[k]
        cur_boxes = final_box_preds[k, cur_mask]
        cur_scores = final_scores[k, cur_mask]
        cur_labels = final_class_ids[k, cur_mask]

        if circle_nms:
            assert False, 'not checked yet'
            centers = cur_boxes[:, [0, 1]]
            boxes = torch.cat((centers, scores.view(-1, 1)), dim=1)
            keep = _circle_nms(boxes, min_radius=min_radius, post_max_size=nms_post_max_size)

            cur_boxes = cur_boxes[keep]
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]

        ret_pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels
        })

        if iou is not None:
            ret_pred_dicts[-1]['pred_iou'] = iou[k, cur_mask]
    return ret_pred_dicts

def _topk_1d(scores, batch_size, batch_idx, obj, K=40, nuscenes=False):
    # scores: (N, num_classes)
    topk_score_list = []
    topk_inds_list = []
    topk_classes_list = []

    for bs_idx in range(batch_size):
        batch_inds = batch_idx==bs_idx
        if obj.shape[-1] == 1 and not nuscenes:
            score = scores[batch_inds].permute(1, 0)
            topk_scores, topk_inds = torch.topk(score, K)
            topk_score, topk_ind = torch.topk(obj[topk_inds.view(-1)].squeeze(-1), K) #torch.topk(topk_scores.view(-1), K)
        else:
            score = obj[batch_inds].permute(1, 0)
            topk_scores, topk_inds = torch.topk(score, min(K, score.shape[-1]))
            topk_score, topk_ind = torch.topk(topk_scores.view(-1), min(K, topk_scores.view(-1).shape[-1]))
            #topk_score, topk_ind = torch.topk(score.reshape(-1), K)

        topk_classes = (topk_ind // K).int()
        topk_inds = topk_inds.view(-1).gather(0, topk_ind)
        #print('topk_inds', topk_inds)

        if not obj is None and obj.shape[-1] == 1:
            topk_score_list.append(obj[batch_inds][topk_inds])
        else:
            topk_score_list.append(topk_score)
        topk_inds_list.append(topk_inds)
        topk_classes_list.append(topk_classes)

    topk_score = torch.stack(topk_score_list)
    topk_inds = torch.stack(topk_inds_list)
    topk_classes = torch.stack(topk_classes_list)

    return topk_score, topk_inds, topk_classes

def gather_feat_idx(feats, inds, batch_size, batch_idx):
    feats_list = []
    dim = feats.size(-1)
    _inds = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), dim)

    for bs_idx in range(batch_size):
        batch_inds = batch_idx==bs_idx
        feat = feats[batch_inds]
        feats_list.append(feat.gather(0, _inds[bs_idx]))
    feats = torch.stack(feats_list)
    return feats

def decode_bbox_from_voxels_nuscenes(batch_size, indices, obj, rot_cos, rot_sin,
                            center, center_z, dim, vel=None, iou=None, point_cloud_range=None, voxel_size=None, voxels_3d=None,
                            feature_map_stride=None, K=100, score_thresh=None, post_center_limit_range=None, add_features=None):
    batch_idx = indices[:, 0]
    spatial_indices = indices[:, 1:]
    scores, inds, class_ids = _topk_1d(None, batch_size, batch_idx, obj, K=K, nuscenes=True)

    center = gather_feat_idx(center, inds, batch_size, batch_idx)
    rot_sin = gather_feat_idx(rot_sin, inds, batch_size, batch_idx)
    rot_cos = gather_feat_idx(rot_cos, inds, batch_size, batch_idx)
    center_z = gather_feat_idx(center_z, inds, batch_size, batch_idx)
    dim = gather_feat_idx(dim, inds, batch_size, batch_idx)
    spatial_indices = gather_feat_idx(spatial_indices, inds, batch_size, batch_idx)

    if not add_features is None:
        add_features = [gather_feat_idx(add_feature, inds, batch_size, batch_idx) for add_feature in add_features]

    if not isinstance(feature_map_stride, int):
        feature_map_stride = gather_feat_idx(feature_map_stride.unsqueeze(-1), inds, batch_size, batch_idx)

    angle = torch.atan2(rot_sin, rot_cos)
    xs = (spatial_indices[:, :, -1:] + center[:, :, 0:1]) * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = (spatial_indices[:, :, -2:-1] + center[:, :, 1:2]) * feature_map_stride * voxel_size[1] + point_cloud_range[1]
    #zs = (spatial_indices[:, :, 0:1]) * feature_map_stride * voxel_size[2] + point_cloud_range[2] + center_z

    box_part_list = [xs, ys, center_z, dim, angle]

    if not vel is None:
        vel = gather_feat_idx(vel, inds, batch_size, batch_idx)
        box_part_list.append(vel)

    if not iou is None:
        iou = gather_feat_idx(iou, inds, batch_size, batch_idx)
        iou = torch.clamp(iou, min=0, max=1.)

    final_box_preds = torch.cat((box_part_list), dim=-1)
    final_scores = scores.view(batch_size, K)
    final_class_ids = class_ids.view(batch_size, K)
    if not add_features is None:
        add_features = [add_feature.view(batch_size, K, add_feature.shape[-1]) for add_feature in add_features]

    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

    if score_thresh is not None:
        mask &= (final_scores > score_thresh)

    ret_pred_dicts = []
    for k in range(batch_size):
        cur_mask = mask[k]
        cur_boxes = final_box_preds[k, cur_mask]
        cur_scores = final_scores[k, cur_mask]
        cur_labels = final_class_ids[k, cur_mask]
        cur_add_features = [add_feature[k, cur_mask] for add_feature in add_features] if not add_features is None else None
        cur_iou = iou[k, cur_mask] if not iou is None else None

        ret_pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels,
            'pred_ious': cur_iou,
            'add_features': cur_add_features,
        })
    return ret_pred_dicts


def decode_bbox_from_pred_dicts(pred_dict, point_cloud_range=None, voxel_size=None, feature_map_stride=None):
    batch_size, _, H, W = pred_dict['center'].shape

    batch_center = pred_dict['center'].permute(0, 2, 3, 1).contiguous().view(batch_size, H*W, 2)  # (B, H, W, 2)
    batch_center_z = pred_dict['center_z'].permute(0, 2, 3, 1).contiguous().view(batch_size, H*W, 1)  # (B, H, W, 1)
    batch_dim = pred_dict['dim'].exp().permute(0, 2, 3, 1).contiguous().view(batch_size, H*W, 3)  # (B, H, W, 3)
    batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1).permute(0, 2, 3, 1).contiguous().view(batch_size, H*W, 1)  # (B, H, W, 1)
    batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1).permute(0, 2, 3, 1).contiguous().view(batch_size, H*W, 1)  # (B, H, W, 1)
    batch_vel = pred_dict['vel'].permute(0, 2, 3, 1).contiguous().view(batch_size, H*W, 2) if 'vel' in pred_dict.keys() else None

    angle = torch.atan2(batch_rot_sin, batch_rot_cos)  # (B, H*W, 1)

    ys, xs = torch.meshgrid([torch.arange(0, H, device=batch_center.device, dtype=batch_center.dtype),
                             torch.arange(0, W, device=batch_center.device, dtype=batch_center.dtype)])
    ys = ys.view(1, H, W).repeat(batch_size, 1, 1)
    xs = xs.view(1, H, W).repeat(batch_size, 1, 1)
    xs = xs.view(batch_size, -1, 1) + batch_center[:, :, 0:1]
    ys = ys.view(batch_size, -1, 1) + batch_center[:, :, 1:2]

    xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

    box_part_list = [xs, ys, batch_center_z, batch_dim, angle]
    if batch_vel is not None:
        box_part_list.append(batch_vel)

    box_preds = torch.cat((box_part_list), dim=-1).view(batch_size, H, W, -1)

    return box_preds
