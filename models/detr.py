# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # 分类
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 回归
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.query_embed 类似于传统目标检测里面的anchor 这里设置了100个  [100,256]
        # nn.Embedding 等价于 nn.Parameter
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss   # True

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # out: list{0: tensor=[bs,2048,19,26] + mask=[bs,19,26]}  经过backbone resnet50 block5输出的结果
        # pos: list{0: [bs,256,19,26]}  位置编码
        features, pos = self.backbone(samples)

        # src: Tensor [bs,2048,19,26]
        # mask: Tensor [bs,19,26]
        src, mask = features[-1].decompose()
        assert mask is not None

        # 数据输入transformer进行前向传播
        # self.input_proj(src) [bs,2048,19,26]->[bs,256,19,26]
        # mask: False的区域是不需要进行注意力计算的
        # self.query_embed.weight  类似于传统目标检测里面的anchor 这里设置了100个
        # pos[-1]  位置编码  [bs, 256, 19, 26]
        # hs: [6, bs, 100, 256]
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # 分类 [6个decoder, bs, 100, 256] -> [6, bs, 100, 92(类别)]
        outputs_class = self.class_embed(hs)
        # 回归 [6个decoder, bs, 100, 256] -> [6, bs, 100, 4]
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:   # True
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        # dict: 3
        # 0 pred_logits 分类头输出[bs, 100, 92(类别数)]
        # 1 pred_boxes 回归头输出[bs, 100, 4]
        # 3 aux_outputs list: 5  前5个decoder层输出 5个pred_logits[bs, 100, 92(类别数)] 和 5个pred_boxes[bs, 100, 4]
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes     # 数据集类别数
        self.matcher = matcher             # HungarianMatcher()  匈牙利算法 二分图匹配
        self.weight_dict = weight_dict     # dict: 18  3x6  6个decoder的损失权重   6*(loss_ce+loss_giou+loss_bbox)
        self.eos_coef = eos_coef           # 0.1
        self.losses = losses               # list: 3  ['labels', 'boxes', 'cardinality']
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef   # tensro: 92   前91=1  92=eos_coef=0.1
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        outputs：'pred_logits'=[bs, 100, 92] 'pred_boxes'=[bs, 100, 4] 'aux_outputs'=5*([bs, 100, 92]+[bs, 100, 4])
        targets：'boxes'=[3,4] labels=[3] ...
        indices： [3] 如：5,35,63  匹配好的3个预测框idx
        num_boxes：当前batch的所有gt个数
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # 分类：[bs, 100, 92类别]

        # idx tuple:2  0=[num_all_gt] 记录每个gt属于哪张图片  1=[num_all_gt] 记录每个匹配到的预测框的index
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 正样本+负样本  上面匹配到的预测框作为正样本 正常的idx  而100个中没有匹配到的预测框作为负样本(idx=91 背景类)
        target_classes[idx] = target_classes_o

        # 分类损失 = 正样本 + 负样本
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        # 日志 记录分类误差
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        # losses: 'loss_ce': 分类损失
        #         'class_error':Top-1精度 即预测概率最大的那个类别与对应被分配的GT类别是否一致  这部分仅用于日志显示 并不参与模型训练
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        outputs：'pred_logits'=[bs, 100, 92] 'pred_boxes'=[bs, 100, 4] 'aux_outputs'=5*([bs, 100, 92]+[bs, 100, 4])
        targets：'boxes'=[3,4] labels=[3] ...
        indices： [3] 如：5,35,63  匹配好的3个预测框idx
        num_boxes：当前batch的所有gt个数
        """
        assert 'pred_boxes' in outputs
        # idx tuple:2  0=[num_all_gt] 记录每个gt属于哪张图片  1=[num_all_gt] 记录每个匹配到的预测框的index
        idx = self._get_src_permutation_idx(indices)

        # [all_gt_num, 4]  这个batch的所有正样本的预测框坐标
        src_boxes = outputs['pred_boxes'][idx]
        # [all_gt_num, 4]  这个batch的所有gt框坐标
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # 计算GIOU损失
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # 'loss_bbox': L1回归损失   'loss_giou': giou回归损失
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # [num_all_gt]  记录每个gt都是来自哪张图片的 idx
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # 记录匹配到的预测框的idx
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
                      dict: 'pred_logits'=Tensor[bs, 100, 92个class]  'pred_boxes'=Tensor[bs, 100, 4]  最后一个decoder层输出
                             'aux_output'={list:5}  0-4  每个都是dict:2 pred_logits+pred_boxes 表示5个decoder前面层的输出
             targets: list of dicts, such that len(targets) == batch_size.   list: bs
                      每张图片包含以下信息：'boxes'、'labels'、'image_id'、'area'、'iscrowd'、'orig_size'、'size'
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # dict: 2   最后一个decoder层输出  pred_logits[bs, 100, 92个class] + pred_boxes[bs, 100, 4]
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 匈牙利算法  解决二分图匹配问题  从100个预测框中找到和N个gt框一一对应的预测框  其他的100-N个都变为背景
        # Retrieve the matching between the outputs of the last layer and the targets  list:1
        # tuple: 2    0=Tensor3=Tensor[5, 35, 63]  匹配到的3个预测框  其他的97个预测框都是背景
        #             1=Tensor3=Tensor[1, 0, 2]    对应的三个gt框
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)   # int 统计这整个batch的所有图片的gt总个数  3
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()   # 3.0

        # 计算最后层decoder损失  Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 计算前面5层decoder损失  累加到一起  得到最终的losses
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)   # 同样匈牙利算法匹配
                for loss in self.losses:   # 计算各个loss
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # 参加权重更新的损失：losses: 'loss_ce' + 'loss_bbox' + 'loss_giou'    用于log日志: 'class_error' + 'cardinality_error'
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
                     0 pred_logits 分类头输出[bs, 100, 92(类别数)]
                     1 pred_boxes 回归头输出[bs, 100, 4]
                     2 aux_outputs list: 5  前5个decoder层输出 5个pred_logits[bs, 100, 92(类别数)] 和 5个pred_boxes[bs, 100, 4]
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # out_logits：[bs, 100, 92(类别数)]
        # out_bbox：[bs, 100, 4]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # [bs, 100, 92]  对每个预测框的类别概率取softmax
        prob = F.softmax(out_logits, -1)
        # prob[..., :-1]: [bs, 100, 92] -> [bs, 100, 91]  删除背景
        # .max(-1): scores=[bs, 100]  100个预测框属于最大概率类别的概率
        #           labels=[bs, 100]  100个预测框的类别
        scores, labels = prob[..., :-1].max(-1)

        # cxcywh to xyxy  format   [bs, 100, 4]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates  bs张图片的宽和高
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]  # 归一化坐标 -> 绝对位置坐标(相对于原图的坐标)  [bs, 100, 4]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        # list: bs    每个list都是一个dict  包括'scores'  'labels'  'boxes'三个字段
        # scores = Tensor[100,]  这张图片预测的100个预测框概率分数
        # labels = Tensor[100,]  这张图片预测的100个预测框所属类别idx
        # boxes = Tensor[100, 4] 这张图片预测的100个预测框的绝对位置坐标(相对这张图片的原图大小的坐标)
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    # 搭建backbone resnet + PositionEmbeddingSine
    backbone = build_backbone(args)

    # 搭建transformer
    transformer = build_transformer(args)

    # 搭建整个DETR模型
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    # 是否需要额外的分割任务
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    # HungarianMatcher()  二分图匹配
    matcher = build_matcher(args)

    # 损失权重
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:   # 分割任务  False
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:   # 辅助损失  每个decoder都参与计算损失  True
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]

    # 定义损失函数
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    # 定义后处理
    postprocessors = {'bbox': PostProcess()}

    # 分割
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
