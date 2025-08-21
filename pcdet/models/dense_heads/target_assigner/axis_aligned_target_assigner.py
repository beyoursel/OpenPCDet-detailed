import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils


class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)
        # self.separate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        # if self.seperate_multihead:
        #     rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
        #     self.gt_remapping = {}
        #     for rpn_head_cfg in rpn_head_cfgs:
        #         for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
        #             self.gt_remapping[name] = idx + 1

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """

        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1] # 最后一维为class_id
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1 # 去掉padding的空gt,为了使得格式对齐，以最大gt数量分配空间
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)  # 过滤的当前class的mask
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]
                    anchors = anchors.view(-1, anchors.shape[-1]) # flatten to (N, 7)
                    selected_classes = cur_gt_classes[mask]

                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                )
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                } # view shape
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1) # 上述代码有些冗余

            bbox_targets.append(target_dict['box_reg_targets']) # 存储每一帧处理的结果
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        bbox_targets = torch.stack(bbox_targets, dim=0) # stack all frame

        cls_labels = torch.stack(cls_labels, dim=0)
        reg_weights = torch.stack(reg_weights, dim=0)
        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights

        }
        return all_targets_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):

        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        # 1. 初始化标签和目标框ID
        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # 2. 计算anchor和目标gt的iou
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])

            # NOTE: The speed of these two versions depends the environment and the number of anchors
            # anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            # 3. 为每个anchor选择iou最大的gt
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1) # shape: (num_anchor) 每个anchor与目标框iou最大的目标框索引
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax] # 取出每个anchor与gt最大的iou

            # gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            # 4. 为每个目标框选择与之最大重叠的anchor
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0) # 每个目标框与锚点重叠度最大的锚点的索引
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)] # 取出每个gt最大的iou
            empty_gt_mask = gt_to_anchor_max == 0 # 若最大iou为0，则该anchor也为父样本
            gt_to_anchor_max[empty_gt_mask] = -1

            # 5. 为每个anchor分配标签，与gt之间iou值最大的anchor
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0] # 找到所有anchor与目标框iou最大值相等的索引（存在多个anchor与gt之间的iou值相等）
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap] # 获得对应的gt索引，每个anchor还是分配与其iou值最大的gt
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] # 给具有最大iou的anchor分配对应gt label
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int() # 存放gt_id 未用到

            # 6. 为正样本分配标签
            pos_inds = anchor_to_gt_max >= matched_threshold # 若anchor与gt之间最大的iou值满足正样本阈值
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            # 7. 为负样本分配标签
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0] # 小于负样本阈值的为负样本，在正负样本阈值之前的anchor不做处理
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0] # 前景anchor

        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size) # sample_size为正负样本总和，而pos_fraction为正样本的比例
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1 # 正样本数量过多，强制设置为-1，即忽略该anchor
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))] # 若bg_inds的数量超过设置的负样本数量，则从bg_inds采样num_bg个负样本
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0 # 背景label都置为0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] # 具有最大iou的anchor，代码重复

        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :] # 前景anchor对应的gt_boxes，即网络需要在anchor的基础上拟合gt_boxes
            fg_anchors = anchors[fg_inds, :] # 前景anchor
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors) # 编码network需要预测的目标值

        reg_weights = anchors.new_zeros((num_anchors,)) # 仅前景权重为1

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        return ret_dict
