"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from typing import Dict, Callable, Optional, List, Tuple, Union
from flwr.common import Scalar


def get_strategy(strategy_name: str) -> str:
    """ Get strategy to use for aggregation. """
    if strategy_name == "fedavg":
        return "flwr.server.strategy.fedavg.FedAvg"
    elif strategy_name == "fedprox": 
        return "flwr.server.strategy.fedprox.FedProx"
    elif strategy_name == "xgboost":
        return "flwr.server.strategy.fedxgb_bagging.FedXgbBagging"
    elif strategy_name == "fedxgb_nn_avg":
        return "flwr.server.strategy.fedxgb_nn_avg.FedXgbNnAvg"
    else: 
        raise ValueError("Unknown strategy name.")
    
def get_evaluate_config() -> Callable:
    """ Get evaluate config. """

    def evaluate_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration for evaluation."""
        return {"epoch": 1,"batch_size": 64,}
    
    return evaluate_config


def agg_metrics_train(metrics: List[Tuple[int, Dict[str, int]]]) -> Dict[str, Scalar]:
    # Collect all the FL Client metrics and sum them
    # for each tuple in metrics, the second element is a dictionary with a key "total_bytes"
    # this value should be extracted from each tuple and summed
    agg_metrics = {}
    tmp_dict = {}
    for i, metric in enumerate(metrics):
        tmp_dict[i] = metric[1]["total_bytes"]

    agg_metrics["total_bytes"] = sum(tmp_dict.values())
    return agg_metrics