from enum import Enum
from typing import Any, Dict, List, Tuple, Callable, Optional
from PIL import Image
from fastai.vision.all import Path, get_image_files, verify_images
import torch
from dinov2.data.datasets.extended import ExtendedVisionDataset
from io import BytesIO
import io
import numpy as np
from torchvision.transforms import ToPILImage
import lz4.frame
from pathlib import Path


class Med_dataset(ExtendedVisionDataset):
    def __init__(self, root: str, verify: bool = False, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, num_classes=None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        self.num_classes = num_classes
        image_paths = get_image_files(self.root)    

        invalid_images = set()
        if verify:
            invalid_images = set(verify_images(image_paths))
        self.image_paths = [p for p in image_paths]


    def get_image_data(self, index: int) -> bytes:
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert("RGB")
        return img
        
    def get_target(self, index: int) -> Any:
        return 0
    
    def get_name(self, index: int):
        return str(self.image_paths[index]).split("/")[-1].split(".")[0]

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        image = self.get_image_data(index)
        target = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        target = int(str(self.image_paths[index]).split('/')[-1].split('.')[0].split('_')[-1])

        return image, target
    
