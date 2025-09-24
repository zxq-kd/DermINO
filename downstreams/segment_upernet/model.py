import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models
import sys
sys.path.append("/data/zxq/DermINO/pretrain")
from dinov2.eval.setup import setup_and_build_model
from PIL import Image
from io import BytesIO
from mmseg.models.decode_heads import UPerHead
import torch.nn.functional as F

dino_backbones = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14
    },
}


class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class conv_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(conv_head, self).__init__()
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 64, (3,3), padding=(1,1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3,3), padding=(1,1)),
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        return x



class newSegHead(nn.Module):
    def __init__(self, in_channels=768, mid_channels=256, num_classes=2, out_size=224, model_type=None):
        super(newSegHead, self).__init__()
        self.out_size = out_size

        norm_cfg = dict(type='SyncBN', requires_grad=True)

        in_channels = [768, 768, 768, 768]
        mid_channels = 768

        self.uper_head = UPerHead(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=mid_channels,
            dropout_ratio=0.0,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
            )

    def forward(self, x):
        """
        x: B x C x H x W 来自DINOv2 backbone的特征图
        输出: B x num_classes x 256 x 256 的分割图
        """
        output = self.uper_head(x)
        output = F.interpolate(output, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        
        return output



def convert_vit_outputs_to_upernet_input(hidden_states, selected_indices=[3, 6, 9, 12], image_size=224, patch_size=16):
    """
    将 ViT 的中间层输出 [B, 197, C] 提取并转换为 UPerHead 输入格式 [B, C, H, W]

    Args:
        hidden_states (tuple): backbone(x, output_hidden_states=True).hidden_states 输出
        selected_indices (list): 要提取的层索引
        image_size (int): 输入图像尺寸（假设正方形）
        patch_size (int): patch 大小

    Returns:
        List[Tensor]: [x1, x2, x3, x4] → 每个为 [B, C, H, W]
    """
    B, _, C = hidden_states[0].shape
    num_patches = (image_size // patch_size) ** 2
    H = W = image_size // patch_size

    features = []
    for idx in selected_indices:
        x = hidden_states[idx]              
        x = x.permute(0, 2, 1).reshape(B, C, H, W) 
        features.append(x)

    return features  # 可用于 uper_head(features)

class Segmentor(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s', head = 'conv', backbones = dino_backbones, args=None):
        super(Segmentor, self).__init__()
        self.heads = {
            'conv':conv_head
        }

        self.model_type = args.model_type

        self.backbone, autocast_dtype = setup_and_build_model(args)
        self.backbone.eval()
        self.embedding_size = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
      
        self.num_classes = num_classes
        self.head = newSegHead(in_channels=self.embedding_size, mid_channels=256, num_classes=2, out_size=224, model_type=args.model_type)

    def forward(self, x):

        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        
        with torch.no_grad():
            x = self.backbone.get_intermediate_layers(
                x=x.cuda(),
                n=[2,5,8,11],
                return_class_token=False,
                norm=True
            )
            x = [
                input.transpose(1, 2).reshape(input.size(0), input.size(2), int(input.size(1) ** 0.5), int(input.size(1) ** 0.5))
                for input in x
            ]
            
        x = self.head(x)
        return x



