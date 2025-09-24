"""dermino-fl: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import List, Tuple
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import torch.nn as nn
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd
import sys 
sys.path.append("/data/zxq/DermINO/pretrain")
from dinov2.models.vision_transformer import vit_base

NUM_CLASSES = 2


# 1. Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f))
        ]
        self.labels = [self._get_label_from_filename(f) for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _get_label_from_filename(filename):
        """Extracts label from filename (e.g., '154372_1.jpg' -> 1)."""
        try:
            # The label is the number after the last underscore
            return int(os.path.splitext(filename)[0].split("_")[-1])
        except (ValueError, IndexError):
            return 0  # Default label for files with no label

class LinearClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes=1000):
        super().__init__()
        in_dim = embed_dim
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
    def forward(self, feats):
        return self.linear(feats)

def get_net(backbone, embedding_size, model_type) -> torch.nn.Module:

    classifier_head = LinearClassifier(embed_dim=embedding_size, num_classes=NUM_CLASSES)   
    class ViTWithCustomHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            with torch.no_grad():
                feats = self.backbone(x)
            return self.head(feats)

    model = ViTWithCustomHead(backbone, classifier_head)
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    return model    
    



# 3. Data Loading
def load_data(dataset_name: str, dataset_dir=None, model_type=None) -> Tuple[DataLoader, DataLoader]:
    """Loads the specified dataset."""

    train_dir = os.path.join(dataset_dir, dataset_name, "train_data")
    test_dir = os.path.join(dataset_dir, dataset_name, "test_data")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    trainset = ImageDataset(root_dir=train_dir, transform=transform)
    testset = ImageDataset(root_dir=test_dir, transform=transform)

    train_batch_size = 32 if len(trainset) >= 32 else len(trainset)
    test_batch_size = 32 if len(testset) >= 32 else len(testset)

    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=test_batch_size)

    return trainloader, testloader

# 4. Training and Testing
def train(net, trainloader, epochs, config, device, logger=None, lr=None): #TODO config
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.head.parameters(), lr=lr) # learning rate
    min_lr = 1e-6
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    net.to(device)
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if logger != None:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(trainloader)}, Lr: {current_lr}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(trainloader)}, Lr: {current_lr}")


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    y_true, y_pred, y_scores = [], [], []
    net.to(device)
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            # Use softmax to get probabilities for ROC AUC
            probs = torch.nn.functional.softmax(outputs, dim=1)
            # Store scores for the positive class (class 1)
            if probs.shape[1] > 1:
                y_scores.extend(probs[:, 1].cpu().numpy())
            else: # for single class output
                y_scores.extend(probs[:, 0].cpu().numpy())


    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Calculate ROC AUC score
    roc_auc = 0.0
    # Ensure there are samples from more than one class to calculate ROC AUC
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            roc_auc = 0.0 # Should not happen if checks are correct
            
    return loss / len(testloader.dataset), accuracy, f1, roc_auc



def test_bootstrap(net, testloader, device, num_bootstrap=1000, dataset_name=None, logger=None):
    """Validate the model on the test set using bootstrap resampling and save npy scores."""
    y_true, y_pred, y_scores = [], [], []
    net.to(device)
    net.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            probs = torch.nn.functional.softmax(outputs, dim=1)
            if probs.shape[1] > 1:
                y_scores.extend(probs[:, 1].cpu().numpy())
            else:
                y_scores.extend(probs[:, 0].cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    n = len(y_true)

    f1_scores = []
    aurocs = []

    for _ in tqdm(range(num_bootstrap), desc="Bootstrapping"):
        indices = np.random.choice(n, n, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        y_scores_sample = y_scores[indices]

        f1 = f1_score(y_true_sample, y_pred_sample, average="macro", zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true_sample, y_scores_sample)
        except ValueError:
            roc_auc = 0.0

        f1_scores.append(f1)
        aurocs.append(roc_auc)

    def summarize(metric_list):
        mean = np.mean(metric_list)
        std = np.std(metric_list)
        return mean, mean - std, mean + std

    f1_mean, f1_minus, f1_plus = summarize(f1_scores)
    auc_mean, auc_minus, auc_plus = summarize(aurocs)

    results = pd.DataFrame([{
        'dataset': dataset_name if dataset_name else 'unknown',
        'f1_mean': f1_mean, 'f1_mean-std': f1_minus, 'f1_mean+std': f1_plus,
        'auc_mean': auc_mean, 'auc_mean-std': auc_minus, 'auc_mean+std': auc_plus
    }])

    logger.info(results)

    return f1_mean, auc_mean



# 5. Weight Helpers
def get_weights(net: torch.nn.Module) -> List[np.ndarray]:
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: torch.nn.Module, weights: List[np.ndarray]) -> None:
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    msg = net.load_state_dict(state_dict, strict=True)
    print(msg)
