# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import torch
from torch import nn
from torchmetrics import MetricCollection

from dinov2.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from scipy.special import softmax

logger = logging.getLogger("dinov2")


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


def compute_macro_metrics(all_preds: np.ndarray, all_targets: np.ndarray):
    """
    all_preds: (n, m) 预测分数
    all_targets: (n,) 整型标签
    """
    n, m = all_preds.shape
    # ---- Macro F1 ----
    pred_labels = np.argmax(all_preds, axis=1)
    macro_f1 = f1_score(all_targets, pred_labels, average="macro")

    softmax_score = softmax(all_preds, axis=1)
    # ---- Macro AUROC ----
    # one-vs-rest 需要将标签二值化
    if m == 2:  # 二分类
        try:
            macro_auroc = roc_auc_score(all_targets, softmax_score[:, 1], multi_class='ovr', average='macro')
        except ValueError:
            macro_auroc = float("nan")
    else:       # 多分类
        # y_true_bin = label_binarize(all_targets, classes=np.arange(m))
        try:
            macro_auroc = roc_auc_score(all_targets, softmax_score,
                                        average="macro", multi_class="ovr")
        except ValueError:
            macro_auroc = float("nan")

    return macro_f1, macro_auroc

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    all_preds = []
    all_targets = []

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        outputs = model(samples.to(device)) #85*2  
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            
            metric_inputs = postprocessors[k](outputs, targets)

            all_preds.append(metric_inputs["preds"].detach().cpu())
            all_targets.append(metric_inputs["target"].detach().cpu())

            metric.update(**metric_inputs)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    macro_f1, macro_auroc = compute_macro_metrics(all_preds, all_targets)
    logger.info(f"macro_f1: {macro_f1}; macro_auroc: {macro_auroc}")

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats

@torch.inference_mode()
def evaluate_knn(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    clf=None,
    raw_model=None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    all_features = []
    all_targets = []

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        outputs = model(samples.to(device)) #85*2  
        test_feature = raw_model(samples.to(device))
        all_features.append(test_feature.detach().cpu())
        all_targets.append(targets.detach().cpu())
        
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            
            metric_inputs = postprocessors[k](outputs, targets)

            metric.update(**metric_inputs)

    
    all_features = torch.cat(all_features).numpy()
    all_targets = torch.cat(all_targets).numpy()

    y_pred = clf.predict(all_features)
    macro_f1 = f1_score(all_targets, y_pred, average='macro')

    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(all_targets)
    y_pred_prob = clf.predict_proba(all_features)

    try:
        if y_test_bin.shape[1] == 1:
            macro_auc = roc_auc_score(all_targets, y_pred_prob[:, 1])
        else:
            macro_auc = roc_auc_score(y_test_bin, y_pred_prob, average='macro', multi_class='ovr')
    except Exception as e:
        macro_auc = 0.0
        exit()

    logger.info(f"macro_f1: {macro_f1}; macro_auc: {macro_auc}")

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dataset, batch_size, num_workers, gather_on_cpu=False):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels
