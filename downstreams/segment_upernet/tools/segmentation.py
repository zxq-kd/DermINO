from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from model import Segmentor
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import torchmetrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(pred, mask):
    pred_cls = torch.argmax(pred, dim=1)  # (B, H, W)
    label_cls = torch.argmax(mask, dim=1)  # (B, H, W)

    accu = torch.mean((pred_cls == label_cls).float())
    classes = torch.unique(torch.cat((pred_cls.flatten(), label_cls.flatten())))
    miou_sum, dice_sum = 0.0, 0.0
    total_tp = total_fp = total_fn = 0
    valid_classes = 0

    for cls in classes:
        tp = torch.sum((pred_cls == cls) & (label_cls == cls))
        fp = torch.sum((pred_cls == cls) & (label_cls != cls))
        fn = torch.sum((pred_cls != cls) & (label_cls == cls))
        iou = 1.0 if (tp + fp + fn) == 0 else tp / (tp + fp + fn)
        dice = 1.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
        miou_sum += iou
        dice_sum += dice
        total_tp += tp
        total_fp += fp
        total_fn += fn
        valid_classes += 1

    miou = miou_sum / valid_classes if valid_classes > 0 else 0.0
    dice = dice_sum / valid_classes if valid_classes > 0 else 0.0
    jac = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 1.0
    return accu.item(), miou.cpu(), jac, dice


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes, img_transform=None, mask_transform=None, images=None, dataset_type=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes
        self.dataset_type = dataset_type

        if images is None:
            self.images = [img for img in os.listdir(img_dir)]
        else:
            self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx].split(".")[0]
        img_path = os.path.join(self.img_dir, self.images[idx])
        if self.dataset_type == "ph2_data":
            mask_path = os.path.join(self.mask_dir, img_name + "_lesion.bmp")
        elif self.dataset_type == "skin_cancer":
            img_name = img_name.split("_")[:-1]
            img_name = "_".join(img_name)
            mask_path = os.path.join(self.mask_dir, img_name + "_contour.png")  
        elif self.dataset_type == "seg_2016":
            mask_path = os.path.join(self.mask_dir, img_name + "_Segmentation.png")  
        elif self.dataset_type == "seg_2017" or self.dataset_type == "seg_2018":
            mask_path = os.path.join(self.mask_dir, img_name + "_segmentation.png")                     

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:

            mask = self.mask_transform(mask)

        bin_mask = torch.zeros(self.num_classes, mask.shape[1], mask.shape[2])

        # Ensure mask is a torch tensor and is in the same device as bin_mask
        mask = torch.from_numpy(np.array(mask)).to(bin_mask.device)
        
        # Convert mask to type float for comparison
        mask = mask.float()

        for i in range(self.num_classes):
            bin_mask[i] = (mask == i).float()  # Ensure resulting mask is float type
        
        #breakpoint()
        return image, bin_mask, mask_path



def train(model, train_loader, criterion, optimizer, scheduler, epoch, logger, train_loss_epoch_list, train_accu_epoch_list, train_miou_epoch_list):

    model.train()
    loop = tqdm(train_loader, total=len(train_loader))
    running_loss = 0
    correct = 0

    for batch_idx, (data, target, mask_path) in enumerate(loop):
        # print(batch_idx) 
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        acc, miou, jaccard, dice = compute_metrics(output, target)
        logger.info(f"train: Accuracy: {acc:.4f}, mIoU: {miou:.4f}, jaccard: {jaccard:.4f}, dice: {dice:.4f}")
        #print()      

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 打印当前学习率
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}, Learning Rate: {lr}")
        #print()

        running_loss += loss.item()

        train_loss_epoch_list.append(loss.item())
        train_accu_epoch_list.append(acc)
        train_miou_epoch_list.append(miou)

        #_, predicted = torch.max(output.data, 1)
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss = loss.item())
    #print()
    logger.info(f'\nTrain set: Average loss: {running_loss/len(train_loader):.4f}')
    return train_loss_epoch_list, train_accu_epoch_list, train_miou_epoch_list



def validation(model, criterion, valid_loader, logger, val_loss_epoch_list, val_accu_epoch_list, val_miou_epoch_list, val_jac_epoch_list, val_dice_epoch_list):
    model.eval()
    running_loss = 0
    correct = 0

    with torch.no_grad():
        loop = tqdm(valid_loader, total=len(valid_loader))
        for data, target, mask_path in loop:
            data, target = data.to(device), target.to(device)
            output = model(data)

            acc, miou, jaccard, dice = compute_metrics(output, target)
            logger.info(f"val: Accuracy: {acc:.4f}, mIoU: {miou:.4f}, jaccard: {jaccard:.4f}, dice: {dice:.4f}")
            if criterion == None:
                val_loss_epoch_list = []
            else:
                loss = criterion(output, target)
                running_loss += loss.item()
                val_loss_epoch_list.append(loss.item())

            val_accu_epoch_list.append(acc)
            val_miou_epoch_list.append(miou.numpy())
            val_jac_epoch_list.append(jaccard.cpu().numpy())
            val_dice_epoch_list.append(dice.cpu().numpy())

    if criterion != None:
        logger.info(f'\nValidation set: Average loss: {running_loss/len(valid_loader):.4f}')

    return val_loss_epoch_list, val_accu_epoch_list, val_miou_epoch_list, val_jac_epoch_list, val_dice_epoch_list
