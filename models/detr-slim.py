# 纯吊包版DETR
import torch
from torch import nn
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        # backbone = resnet50 除掉average pool和fc层  只保留conv1 - conv5_x
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        # 1x1卷积降维 2048->256
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        # 6层encoder + 6层decoder    hidden_dim=256  nheads多头注意力机制 8头   num_encoder_layers=num_decoder_layers=6
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        # 分类头
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # 回归头
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        # 位置编码  encoder输入
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        # query pos编码  decoder输入
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

    def forward(self, inputs):
        x = self.backbone(inputs)    # [1,3,800,1066] -> [1,2048,25,34]
        h = self.conv(x)             # [1,2048,25,34] -> [1,256,25,34]
        H, W = h.shape[-2:]          # H=25  W=34
        # pos = [850,1,256]  self.col_embed = [50,128]  self.row_embed[:H]=[50,128]
        pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                        self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                        ], dim=-1).flatten(0, 1).unsqueeze(1)
        # encoder输入  decoder输入
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()


detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
detr.eval()
inputs = torch.randn(1, 3, 800, 1066)
logits, bboxes = detr(inputs)
print(logits.shape)   # torch.Size([100, 1, 92])
print(bboxes.shape)   # torch.Size([100, 1, 4])
