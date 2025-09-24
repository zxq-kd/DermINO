"""dermino: A Flower / PyTorch app."""

import logging
from typing import Dict, List, Optional, Tuple
import random
import flwr as fl
import torch
from flwr.common import EvaluateRes, Metrics, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common.parameter import ndarrays_to_parameters
from typing import Union
from collections import OrderedDict

import flwr as fl
from flwr.common import Parameters, FitRes
import numpy as np
import os      

import sys
sys.path.append("/data/zxq/DermINO/downstreams/fl-master")

from dermino.args import get_args_parser
from dermino.task import (
    get_net,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
    test_bootstrap,
)

sys.path.append("/data/zxq/DermINO/pretrain")
from dinov2.eval.setup import setup_and_build_model

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 2,
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1"] for num_examples, m in metrics]
    roc_aucs = [num_examples * m["roc_auc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    aggregated_metrics = {
        "accuracy": sum(accuracies) / sum(examples),
        "f1": sum(f1_scores) / sum(examples),
        "roc_auc": sum(roc_aucs) / sum(examples),
    }
    print(f"Round aggregated metrics: {aggregated_metrics}")

    # Log aggregated metrics
    logger.info(
        f"Round: AGGREGATED | "
        f"Accuracy: {aggregated_metrics['accuracy']:.4f} | "
        f"F1: {aggregated_metrics['f1']:.4f} | "
        f"ROC AUC: {aggregated_metrics['roc_auc']:.4f}"
    )

    return aggregated_metrics

class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy to log individual client metrics."""

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and log individual client metrics."""
        # Log individual client metrics
        for client_proxy, evaluate_res in results:
            cid = client_proxy.cid
            # Pop the dataset_name to use it for logging and remove it for aggregation
            dataset_name = evaluate_res.metrics.pop("dataset_name", f"client_{cid}")
            loss = evaluate_res.loss
            metrics = evaluate_res.metrics  # Now this dict doesn't have dataset_name

            log_message = (
                f"Round: {server_round} | Client: {dataset_name} ({cid}) | "
                f"Loss: {loss:.4f} | Accuracy: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | ROC AUC: {metrics['roc_auc']:.4f}"
            )
            logger.info(log_message)

        # Call the parent class's aggregate_evaluate method
        return super().aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model to disk
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_round_{server_round}.pth'))

        return aggregated_parameters, aggregated_metrics


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        self.dataset_name = DATASETS[int(cid)]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = model.to(self.device)
        
        self.trainloader, self.testloader = load_data(self.dataset_name, args.dataset_dir, args.model_type)
        

    def get_parameters(self, config):
        print(f"[Client {self.dataset_name}] get_parameters")
        return get_weights(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.dataset_name}] fit, config: {config}")
        set_weights(self.net, parameters)
        train(self.net, self.trainloader, epochs=config["local_epochs"], config=config, device=self.device, lr=start_lr)
        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.dataset_name}] evaluate, config: {config}")
        set_weights(self.net, parameters)
        loss, accuracy, f1, roc_auc = test(self.net, self.testloader, device=self.device)
        print(
            f"[Client {self.dataset_name}] Test Loss: {loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}"
        )
        return (
            float(loss),
            len(self.testloader.dataset),
            {
                "accuracy": float(accuracy),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
                "dataset_name": self.dataset_name,  # Add dataset_name to metrics
            },
        )


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client instance for a given client ID."""
    return FlowerClient(cid)


def set_seed(seed: int = 114514):
    # Python 内置随机数
    random.seed(seed)

    # Numpy 随机数
    np.random.seed(seed)

    # PyTorch 随机数（CPU 和 CUDA）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU

    # 确保 cudnn 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    description = "fl downstream"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()


    args.model_type = None
    model_type = args.model_type

    args.config_file = f"/data/zxq/DermINO/checkpoint/dermino/config.yaml"
    args.pretrained_weights = f"/data/zxq/DermINO/checkpoint/dermino/teacher_checkpoint.pth" 
    args.output_dir = f"/data/zxq/DermINO/downstreams/fl-master/output_dir"
    args.dataset_dir = "/data/zxq/DermINO/datasets/fl"

    backbone, autocast_dtype = setup_and_build_model(args)
    embedding_size = 768

    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)

    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    #**************************************************
    PROJECT_ROOT = "/data/zxq/DermINO/pretrain/" 
    ANOTHER_PATH = "/data/zxq/DermINO/downstreams/fl-master/"
    num_rounds = 2

    start_lr = 2e-4
    down_stream_lr = 5e-3
    #**************************************************

    model = get_net(backbone, embedding_size, args.model_type).to("cuda:0")
    set_seed(114514)
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, "fl_metrics.log")
    with open(log_file, "w") as f:
        pass

    logger = logging.getLogger("fl_logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    DATASETS = ["med_node","ph2_data", "fit_data"]
    CLIENTS = 3

    # Define strategy
    strategy = CustomFedAvg(  # Use the custom strategy
        fraction_fit=1.0,  # Sample all clients for training
        fraction_evaluate=1.0,  # Sample all clients for evaluation
        min_fit_clients=CLIENTS,
        min_evaluate_clients=CLIENTS,
        min_available_clients=CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate custom metrics
        on_fit_config_fn=fit_config,  # Custom configuration function
        initial_parameters=fl.common.ndarrays_to_parameters(get_weights(model)),
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 4, "num_gpus": 0.25} if torch.cuda.is_available() else None,
        ray_init_args = {
            "runtime_env": {
                "working_dir": PROJECT_ROOT,
                "env_vars": {
                    "PYTHONPATH": f"{PROJECT_ROOT}:{ANOTHER_PATH}"
                },
            },
            "ignore_reinit_error": True,
        }
    )

    logger.info("\n=== Federated training finished ===")

    num_rounds = 2
    pth_path = os.path.join(args.output_dir, f'model_round_{num_rounds}.pth')
    state_dict = torch.load(pth_path)
    msg = model.load_state_dict(state_dict)
    DEVICE_SINGLE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for ds_name in DATASETS:
        trainloader, testloader = load_data(dataset_name = ds_name, dataset_dir = args.dataset_dir, model_type = args.model_type)
        _ , _ = test_bootstrap(model, testloader, device=DEVICE_SINGLE ,num_bootstrap=1000, dataset_name=ds_name, logger=logger)
