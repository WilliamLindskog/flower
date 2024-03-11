"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import flwr as fl
import hydra
import time
from omegaconf import DictConfig, OmegaConf
from os.path import join
import pickle
import numpy as np
import functools

from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig

from treesXnets.dataset import load_data
from treesXnets.server import get_evaluate_fn, eval_config, FL_Server,serverside_eval
from treesXnets.constants import TASKS, TARGET
from treesXnets.utils import (
    plot_metric_from_history, empty_dir, modify_config, train, test,
    partition_to_dataloader
)
from treesXnets.strategy import get_strategy, agg_metrics_train

from flwr.server.server import Server
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy.fedxgb_bagging import FedXgbBagging
from flwr.server.strategy.fedxgb_nn_avg import FedXgbNnAvg
from pathlib import Path

from typing import Dict
from flwr.common import Scalar
from sklearn.model_selection import train_test_split
from treesXnets.tree_utils import get_dataloader, TreeDataset

import os
os.environ["CURL_CA_BUNDLE"]=""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ignore warnings
import warnings
# warnings.filterwarnings("ignore")

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
    df_test = df_test.drop(columns=["ID"])

    if cfg.federated:
        # 3. Define your clients
        client_fn = call(cfg.client_fn, df=df_train, cfg=cfg)

        # 4. Define your strategy
        device = cfg.device
        evaluate_fn = get_evaluate_fn(df_test, device, cfg)
        cfg.strategy._target_ = get_strategy(cfg.strategy_name)
        
        if cfg.model_name.lower() == 'xgboost':
            strategy = FedXgbBagging(
                evaluate_function=get_evaluate_fn(df_test, device, cfg) if cfg.centralized_eval else None,
                fraction_fit=1.0,
                min_fit_clients=cfg.num_clients,
                min_available_clients=cfg.num_clients,
                min_evaluate_clients=cfg.num_clients if not cfg.centralized_eval else 0,
                fraction_evaluate=1.0 if not cfg.centralized_eval else 0.0,
                on_evaluate_config_fn=eval_config,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation
                if not cfg.centralized_eval
                else None,
            )
        elif cfg.model_name.lower() == 'glxgb':
            # Configure the strategy
            def fit_config(server_round: int) -> Dict[str, Scalar]:
                print(f"Configuring round {server_round}")
                return {
                    "num_iterations": 2,
                    "batch_size": 64,
                }

            X_test, y_test = df_test.drop(columns=TARGET[cfg.dataset.name]).to_numpy(), df_test[TARGET[cfg.dataset.name]].to_numpy()
            X_test.flags.writeable, y_test.flags.writeable = True, True
            testset = TreeDataset(np.array(X_test, copy=True), np.array(y_test, copy=True))
            testloader = get_dataloader(testset, partition='test', batch_size=len(testset))

            strategy = FedXgbNnAvg(
                fraction_fit=1.0,
                fraction_evaluate=0.0, 
                min_fit_clients=cfg.num_clients,
                min_evaluate_clients=cfg.num_clients,
                min_available_clients=cfg.num_clients,  # all clients should be available
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=(lambda r: {"batch_size": cfg.batch_size}),
                evaluate_fn=functools.partial(
                    serverside_eval,
                    cfg=cfg,
                    testloader=testloader,
                ),
                accept_failures=False,
            )
        else:
            strategy = instantiate(
                cfg.strategy, 
                evaluate_fn=evaluate_fn, 
                fit_metrics_aggregation_fn=agg_metrics_train
            )

        if cfg.model_name.lower() != 'glxgb':
            server = Server(strategy=strategy, client_manager=SimpleClientManager()) 
        else:
            server = FL_Server(strategy=strategy, client_manager=SimpleClientManager())

        # 5. Start Simulation
        # measure time
        start_time = time.time()
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
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Time elapsed: {time_elapsed:.2f} seconds")

        print(history)

        # 6. Save your results
        save_path = HydraConfig.get().runtime.output_dir
        print(save_path)
        with open(join(save_path, "history.pkl"), "wb") as f_ptr:
            pickle.dump(history, f_ptr)
        with open(join(save_path, "time.txt"), "w") as f_ptr:
            f_ptr.write(str(time_elapsed))

        file_suffix: str = (
            f"_{cfg.strategy_name}"
            f"_C={cfg.num_clients}"
            f"_B={cfg.batch_size}"
            f"_E={cfg.num_epochs}"
            f"_R={cfg.num_rounds}"
            f"_lr={cfg.lr}"
        )

        plot_metric_from_history(
            history,
            save_path,
            (file_suffix),
            metric_type='centralized' if cfg.centralized_eval else 'decentralized',
            regression=True if cfg.model.output_dim == 1 else False,
            model_name=cfg.model_name,
        )
    else: 
        model = instantiate(cfg.model)
        model.to(cfg.device)
        df_train = df_train.drop(columns=["ID"])
        trainloader = partition_to_dataloader(df_train, cfg.dataset.name, cfg.batch_size,)
        testloader = partition_to_dataloader(df_test, cfg.dataset.name,test=True, batch_size=len(df_test))

        # train model
        train(model, trainloader, cfg, central=True)

        # test model
        loss, metrics = test(model, testloader, cfg.device, cfg.task)
        print(f"Test loss: {loss:.4f}")
        print(f"Test metrics: {metrics}")



if __name__ == "__main__":
    main()
