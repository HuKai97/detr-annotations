# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    Absolute pos embedding, Sine.  没用可学习参数  不可学习  定义好了就固定了
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats    # 128维度  x/y  = d_model/2
        self.temperature = temperature        # 常数 正余弦位置编码公式里面的10000
        self.normalize = normalize            # 是否对向量进行max规范化   True
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            # 这里之所以规范化到2*pi  因为位置编码函数的周期是[2pi, 20000pi]
            scale = 2 * math.pi  # 规范化参数 2*pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors   # [bs, 2048, 19, 26]  预处理后的 经过backbone 32倍下采样之后的数据  对于小图片而言多余部分用0填充
        mask = tensor_list.mask   # [bs, 19, 26]  用于记录矩阵中哪些地方是填充的（原图部分值为False，填充部分值为True）
        assert mask is not None
        not_mask = ~mask   # True的位置才是真实有效的位置

        # 考虑到图像本身是2维的 所以这里使用的是2维的正余弦位置编码
        # 这样各行/列都映射到不同的值 当然有效位置是正常值 无效位置会有重复值 但是后续计算注意力权重会忽略这部分的
        # 而且最后一个数字就是有效位置的总和，方便max规范化
        # 计算此时y方向上的坐标  [bs, 19, 26]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # 计算此时x方向的坐标    [bs, 19, 26]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # 最大值规范化 除以最大值 再乘以2*pi 最终把坐标规范化到0-2pi之间
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)   # 0 1 2 .. 127
        # 2i/2i+1: 2 * (dim_t // 2)  self.temperature=10000   self.num_pos_feats = d/2
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)   # 分母

        pos_x = x_embed[:, :, :, None] / dim_t   # 正余弦括号里面的公式
        pos_y = y_embed[:, :, :, None] / dim_t   # 正余弦括号里面的公式
        # x方向位置编码: [bs,19,26,64][bs,19,26,64] -> [bs,19,26,64,2] -> [bs,19,26,128]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # y方向位置编码: [bs,19,26,64][bs,19,26,64] -> [bs,19,26,64,2] -> [bs,19,26,128]
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # concat: [bs,19,26,128][bs,19,26,128] -> [bs,19,26,256] -> [bs,256,19,26]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        # [bs,256,19,26]  dim=1时  前128个是y方向位置编码  后128个是x方向位置编码
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    可以发现整个类其实就是初始化了相应shape的位置编码参数，让后通过可学习的方式学习这些位置编码参数
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        # nn.Embedding  相当于 nn.Parameter  其实就是初始化函数
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]   # 特征图h w
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)   # 初始化x方向位置编码
        y_emb = self.row_embed(j)   # 初始化y方向位置编码
        # concat x y 方向位置编码
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    """
    创建位置编码
    args: 一系列参数  args.hidden_dim: transformer中隐藏层的维度   args.position_embedding: 位置编码类型 正余弦sine or 可学习learned
    """
    # N_steps = 128 = 256 // 2  backbone输出[bs,256,25,34]  256维度的特征
    # 而传统的位置编码应该也是256维度的, 但是detr用的是一个x方向和y方向的位置编码concat的位置编码方式  这里和ViT有所不同
    # 二维位置编码   前128维代表x方向位置编码  后128维代表y方向位置编码
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        # [bs,256,19,26]  dim=1时  前128个是y方向位置编码  后128个是x方向位置编码
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
