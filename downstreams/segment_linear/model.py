import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models
import sys
sys.path.append("/data/zxq/nature/nature_data/dinov2-patch-main")
from dinov2.eval.setup import setup_and_build_model
from PIL import Image
from io import BytesIO
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




class Classifier(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s', head = 'linear', backbones = dino_backbones):
        super(Classifier, self).__init__()
        self.heads = {
            'linear':linear_head
        }
        self.backbones = dino_backbones
        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.backbone.eval()
        self.head = self.heads[head](self.backbones[backbone]['embedding_size'],num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        #breakpoint()
        x = self.head(x)
        return x


def read_image(image_path):
    """读取图像并返回二进制数据"""
    with open(image_path, "rb") as f:
        return f.read()
def tensor_to_base64(tensor):
    tensor = tensor.permute(1, 2, 0).mul(255).byte()  # (3, 224, 224) -> (224, 224, 3)
    image = Image.fromarray(tensor.numpy())  # 转为 PIL Image
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # 保存到 buffer
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class newSegHead(nn.Module):
    def __init__(self, in_channels=768, mid_channels=256, num_classes=2, out_size=224):
        super(newSegHead, self).__init__()
        self.out_size = out_size

        self.bn = nn.SyncBatchNorm(in_channels)
        self.dropout = nn.Dropout2d(p=0.1)
        

        # 最终分类层
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
        
        return output


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
        self.head = newSegHead(in_channels=self.embedding_size, mid_channels=256, num_classes=2, out_size=224)

    def forward(self, x):

        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        
        with torch.no_grad():
            x = self.backbone.forward_features(x.cuda())
            x = x['x_norm_patchtokens'] 
            x = x.permute(0,2,1)
            x = x.reshape(batch_size,self.embedding_size,int(mask_dim[0]),int(mask_dim[1]))
       
        x = self.head(x)

        return x



