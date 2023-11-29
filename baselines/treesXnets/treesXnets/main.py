"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import flwr as fl
import hydra
from omegaconf import DictConfig, OmegaConf
from os.path import join
import pickle
import functools
from pandas import DataFrame

from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig

from treesXnets.dataset import load_data
from treesXnets.server import get_evaluate_fn, eval_config, evaluate_metrics_aggregation, FL_Server
from treesXnets.constants import TASKS
from treesXnets.utils import plot_metric_from_history, empty_dir, modify_config
from treesXnets.strategy import get_strategy

from flwr.server.server import Server
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg, FedXgbBagging, FedXgbNnAvg
from pathlib import Path

from typing import Dict
from flwr.common import Scalar
from sklearn.model_selection import train_test_split


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 0. Empty tmp dir and make appropriate changes to the config
    empty_dir(Path("./treesXnets/tmp"))
    cfg = modify_config(cfg)

    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    cfg.task = TASKS[cfg.dataset.name]
    df, cfg.dataset = load_data(cfg.dataset, cfg.task)

    # Get train and test data
    test_frac = cfg.dataset.test_frac
    df_train, df_test = train_test_split(df, test_size=test_frac, random_state=cfg.seed)

    # 3. Define your clients
    client_fn = call(cfg.client_fn, df=df_train, cfg=cfg)

    # 4. Define your strategy
    device = cfg.device
    evaluate_fn = get_evaluate_fn(df_test, device, cfg)
    cfg.strategy._target_ = get_strategy(cfg.strategy_name)

    if cfg.model_name != 'xgboost':
        strategy = instantiate(cfg.strategy, evaluate_fn=evaluate_fn)
    #elif cfg.model_name.lower() == 'glxgb':
    #    # Configure the strategy
    #    def fit_config(server_round: int) -> Dict[str, Scalar]:
    #        print(f"Configuring round {server_round}")
    #        return {
    #            "num_iterations": cfg.num_local_rounds,
    #            "batch_size": cfg.batch_size,
    #        }
    #    strategy = FedXgbNnAvg(
    #        fraction_fit=1.0,
    #        fraction_evaluate=1.0, 
    #        min_fit_clients=cfg.num_clients,
    #        min_evaluate_clients=cfg.num_clients,
    #        min_available_clients=cfg.num_clients,  # all clients should be available
    #        on_fit_config_fn=fit_config,
    #        on_evaluate_config_fn=(lambda r: {"batch_size": cfg.batch_size}),
    #        evaluate_fn=get_evaluate_fn(cfg.dataset.name, fds.load_full("test"), device, cfg.model),
    #        accept_failures=False,
    #    )
    #else:
    #    strategy = FedXgbBagging(
    #        evaluate_function=get_evaluate_fn(cfg.dataset.name, fds.load_full("test"), device, cfg.model) if cfg.centralized_eval else None,
    #        fraction_fit=1.0,
    #        min_fit_clients=cfg.num_clients,
    #        min_available_clients=cfg.num_clients,
    #        min_evaluate_clients=cfg.num_clients if not cfg.centralized_eval else 0,
    #        fraction_evaluate=1.0 if not cfg.centralized_eval else 0.0,
    #        on_evaluate_config_fn=eval_config,
    #        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation
    #        if not cfg.centralized_eval
    #        else None,
    #    )

    server = Server(strategy=strategy, client_manager=SimpleClientManager()) 

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        server=server,
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
    )

    print(history)

    # 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    print(save_path)
    with open(join(save_path, "history.pkl"), "wb") as f_ptr:
        pickle.dump(history, f_ptr)

    file_suffix: str = (
        f"_{cfg.strategy_name}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_lr={cfg.learning_rate}"
    )

    plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
        metric_type='centralized' if cfg.centralized_eval else 'decentralized',
        regression=True if cfg.model.output_dim == 1 else False,
    )


if __name__ == "__main__":
    main()
