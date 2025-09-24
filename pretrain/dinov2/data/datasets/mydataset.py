from enum import Enum
from typing import Any, Dict, List, Tuple, Callable, Optional
from PIL import Image
from fastai.vision.all import Path, get_image_files, verify_images
import csv
from dinov2.data.datasets.extended import ExtendedVisionDataset
import ast
import logging
import dinov2.distributed as distributed
logger = logging.getLogger("dinov2")

def updata_list(lst):
    if not lst: # empty, no target
        a = None
    else:
        a = [0] * 602

        for index in lst:   # one hot for multi class
            if 0 <= index < 602:
                a[index] = 1

    return a

    
class MyDataset(ExtendedVisionDataset):
    def __init__(self, root: str, verify: bool = False, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, num_classes=None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        self.root = Path(root).expanduser()

        image_paths = get_image_files(self.root)    #
        invalid_images = set()
        if verify:
            invalid_images = set(verify_images(image_paths))

        self.image_paths = [p for p in image_paths if p not in invalid_images]
        self.label_path = "/data/zxq/DermINO/extra_files/new_all.csv"

        self.image_list, self.label_list, self.mask_list = [], [], []    
        with open(self.label_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  
            for row in reader:
                self.image_list.append(row[0])  
                self.label_list.append(row[1])  
                self.mask_list.append(row[2])  


    def get_image_data(self, index: int) -> bytes:
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert("RGB")
        return img
        
    def get_target(self, index: int) -> Any:
        return 0

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        
        path = self.image_paths[index]
        image_name = str(path).split('/')[-1]
        try:
            new_index = self.image_list.index(image_name)
            image_label = ast.literal_eval(self.label_list[new_index])  # str to object
            target = updata_list(image_label)   # multi class label 
            if target == None:
                mask = None
            else:
                mask = self.mask_list[new_index]    # datasets index

        except ValueError:
            target = None
            mask = None
        
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target, mask
    
