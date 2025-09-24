import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoV2SegHead(nn.Module):
    def __init__(self, in_channels=768, mid_channels=256, num_classes=2, out_size=256):
        super(DinoV2SegHead, self).__init__()
        self.out_size = out_size

        self.bn = nn.SyncBatchNorm(in_channels)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        """
        x: B x C x H x W 来自DINOv2 backbone的特征图
        输出: B x num_classes x 256 x 256 的分割图
        """
        feat = self.bn(x)
        feat = self.dropout(feat)
        output = self.classifier(feat)
        output = F.interpolate(output, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        
        return x
