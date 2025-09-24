import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageCaptionDataset(Dataset):
    """
    CSV 文件中包含两列：skincap_file_path 和 caption_en
    """
    def __init__(self, csv_file, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file, header=0)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dataset_folder = "/data/zxq/DermINO/datasets/caption/skin_cap_images/"
        row = self.data.iloc[idx]
        image_path = row['id']
        
        caption = row['caption_en']+' </s>'
        if not os.path.exists(dataset_folder+image_path):
            raise FileNotFoundError(f"Image file not found: {dataset_folder+image_path}")
        image = Image.open(dataset_folder+image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, caption

def get_train_transform(image_size=224):
    """
    为皮肤疾病图像设计的增强训练预处理流程，
    先调整到较大尺寸并进行变换，然后裁剪到目标尺寸以避免白边
    """
    padding_factor = 1.4  # 增加20%的尺寸作为缓冲
    temp_size = int(image_size * padding_factor)
    
    transform = T.Compose([
        T.Resize((temp_size, temp_size)),  # 先调整到较大尺寸
        T.RandomHorizontalFlip(p=0.5),  # 水平翻转不会改变疾病特征
        T.RandomVerticalFlip(p=0.1),  # 轻微垂直翻转
        T.RandomRotation(degrees=25),   # 轻微旋转，模拟拍摄角度变化
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # 轻微颜色变化
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # 轻微缩放和平移
        T.CenterCrop(image_size),  # 裁剪到目标尺寸，避免边缘问题
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return transform

def get_val_transform(image_size=224):
    """
    验证时图像预处理流程：仅 resize、转换为 Tensor 以及归一化。
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return transform