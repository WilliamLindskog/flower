"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import xgboost as xgb
from flwr.common.logger import log
from logging import INFO
from flwr.common import NDArrays, Scalar, Parameters
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from treesXnets.utils import test
from pandas import DataFrame
from treesXnets.constants import TARGET, TASKS
from treesXnets.utils import TabularDataset
from hydra.utils import instantiate
from treesXnets.tree_utils import BST_PARAMS
from treesXnets.constants import TARGET

def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    print(f"Total number of samples: {total_num}")
    # eval_metrics is a list of tuples (num_samples, metrics) where metrics is a dict
    # Get the metric name with is the only key in the dict
    metric_name = list(eval_metrics[0][1].keys())[0]
    metric_aggregated = (
        sum([metrics[metric_name] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {f"{metric_name}": metric_aggregated}
    return metrics_aggregated

def get_evaluate_fn(
    dataset_name: str,
    testdata: FederatedDataset,
    device: torch.device,
    model: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
               Optional[Tuple[float, Dict[str, Scalar]]] ]
    The centralized evaluation function.
    """

    if not model._target_ == "xgboost.Booster": 
        def evaluate(
            server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            # pylint: disable=unused-argument
            """Use the entire Emnist test set for evaluation."""
            net = instantiate(model)
            params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            net.to(device)

            # Get dataloader
            dataset = TabularDataset(DataFrame(testdata), TARGET[dataset_name])
            testloader = DataLoader(dataset, batch_size=64, shuffle=False)

            task = TASKS[dataset_name]
            if task == "classification":
                loss, accuracy = test(net, testloader, device=device, task=task, evaluate=True)
                return loss, {"accuracy": accuracy}
            loss, r2 = test(net, testloader, device=device, task=task, evaluate=True)
            return loss, {"r2": r2}
    else:
        def evaluate(
            server_round: int, parameters: Parameters, config: Dict[str, Scalar]
        ):
            params = BST_PARAMS[dataset_name]
            # If at the first round, skip the evaluation
            if server_round == 0:
                return 0, {}
            else:
                bst = xgb.Booster(params=params)
                for para in parameters.tensors:
                    para_b = bytearray(para)

                # Load global model
                bst.load_model(para_b)

                # Get test data
                dataset = DataFrame(testdata)
                X_test = dataset.drop(columns=[TARGET[dataset_name]])
                y_test = dataset[TARGET[dataset_name]]
                test_data = xgb.DMatrix(X_test, label=y_test)

                # Get metric name
                print("Metric name is: ", params["eval_metric"])
                print("---------------------------------------------------------")
                if params["eval_metric"] == "auc":
                    metric_name = "auc"
                elif params["objective"] == "reg:squarederror":
                    metric_name = "rmse"
                else:
                    raise NotImplementedError

                # Run evaluation
                eval_results = bst.eval_set(
                    evals=[(test_data, "server_test")],
                    iteration=bst.num_boosted_rounds() - 1,
                )
                metric = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
                log(INFO, f"{metric_name} = {metric} at round {server_round}")

                return 0, {f"{metric_name}": metric}

    return evaluate
