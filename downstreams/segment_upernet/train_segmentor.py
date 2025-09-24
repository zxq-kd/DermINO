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
from tools.segmentation import SegmentationDataset, train, validation
import sys
sys.path.append("/data/zxq/DermINO/pretrain")
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.metrics import MetricType, build_metric
from typing import List, Optional
import argparse
import random
import logging
import shutil

def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--test-datasets",
        dest="test_dataset_strs",
        type=str,
        nargs="+",
        help="Test datasets, none to reuse the validation dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--val-class-mapping-fpath",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="class_nums",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="model-type",
    )
    parser.add_argument(
        "--multi-class",
        type=bool,
        default=False,
        help="multi-class",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        help="dataset-type",
    )
    parser.add_argument(
        "--output-num",
        type=str,
        default=None,
        help="output-num",
    )
    parser.add_argument(
        "--train-epoch",
        type=str,
        default=None,
        help="train-epoch",
    )
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        test_dataset_strs=None,
        epochs=50,
        batch_size=256,
        num_workers=18,
        epoch_length=10,
        save_checkpoint_frequency=20,
        eval_period_iterations=10,
        learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
    )
    return parser


def main(args):
    
    # 初始化 Logger
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "val_predicted"), exist_ok=True)
    

    logging.basicConfig(
        filename=output_dir + "/metrics.log",  # 日志文件名
        filemode="w",            # 追加模式
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO        # 记录 INFO 级别以上的日志
    )

    logger = logging.getLogger()

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    train_dataset_dir = args.train_dataset_str
    test_dataset_str = args.test_dataset_str
    epochs = args.epochs
    num_class = args.num_class
    batch_size = args.batch_size
    lr = args.lr
    dataset_type = args.dataset_type

    train_dataset = SegmentationDataset(img_dir=os.path.join(train_dataset_dir, "imgs"), mask_dir=os.path.join(train_dataset_dir, "labels"), num_classes = num_class, img_transform=img_transform, mask_transform=mask_transform, dataset_type=dataset_type)
    valid_dataset = SegmentationDataset(img_dir=os.path.join(test_dataset_str, "imgs"), mask_dir=os.path.join(test_dataset_str, "labels"), num_classes = num_class, img_transform=img_transform, mask_transform=mask_transform, dataset_type=dataset_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) #, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4) #, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Segmentor(2, args=args)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  

    criterion = torch.nn.CrossEntropyLoss()

    train_loss_epoch_list = []
    train_accu_epoch_list = []
    train_miou_epoch_list = []
    
    val_loss_epoch_list = []
    val_accu_epoch_list = []
    val_miou_epoch_list = []
    val_jac_epoch_list = []
    val_dice_epoch_list = []

    for epoch in range(epochs):
        train_loss_epoch_list, train_accu_epoch_list, train_miou_epoch_list = train(model, train_loader, criterion, optimizer, scheduler, epoch, logger, train_loss_epoch_list, train_accu_epoch_list, train_miou_epoch_list)
        val_loss_epoch_list, val_accu_epoch_list, val_miou_epoch_list, val_jac_epoch_list, val_dice_epoch_list = validation(model, criterion, valid_loader, logger, val_loss_epoch_list, val_accu_epoch_list, val_miou_epoch_list, val_jac_epoch_list, val_dice_epoch_list)
        if epoch == 0:
            length = len(val_accu_epoch_list)


    avg_accu = round(sum(val_accu_epoch_list[-length:]) * 100 / len(val_accu_epoch_list[-length:]), 2)
    avg_miou = round(sum(val_miou_epoch_list[-length:]) * 100 / len(val_miou_epoch_list[-length:]), 2)
    avg_jac = round(sum(val_jac_epoch_list[-length:]) * 100 / len(val_jac_epoch_list[-length:]), 2)
    avg_dice = round(sum(val_dice_epoch_list[-length:]) * 100 / len(val_dice_epoch_list[-length:]), 2)

    logger.info(f'\n avg_accu in val: {avg_accu}')
    logger.info(f'\n avg_miou in val: {avg_miou}')
    logger.info(f'\n avg_jac in val: {avg_jac}')
    logger.info(f'\n avg_dice in val: {avg_dice}')
  
    torch.save(model.state_dict(), output_dir + '/segmentation_model.pt')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    description = "DINOv2 segmentation evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    set_seed(0)

    train_epoch = args.train_epoch
    args.model_type = None

    args.train_dataset_str = f"/data/zxq/DermINO/datasets/segment/{args.dataset_type}/seg_train"
    args.test_dataset_str = f"/data/zxq/DermINO/datasets/segment/{args.dataset_type}/seg_test"

    args.epochs = 20
    args.num_class = 2
    args.batch_size = 64
    args.lr = 1e-3

    args.output_dir = os.path.join("/data/zxq/DermINO/downstreams/segment_upernet/output_dir", args.dataset_type) 
    args.config_file = f"/data/zxq/DermINO/checkpoint/dermino/config.yaml"
    args.pretrained_weights = f"/data/zxq/DermINO/checkpoint/dermino/teacher_checkpoint.pth"  

    sys.exit(main(args))
