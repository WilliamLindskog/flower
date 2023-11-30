"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

from typing import Dict, Callable
from flwr.common import Scalar

def get_strategy(strategy_name: str) -> str:
    """ Get strategy to use for aggregation. """
    if strategy_name == "fedavg":
        return "flwr.server.strategy.fedavg.FedAvg"
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