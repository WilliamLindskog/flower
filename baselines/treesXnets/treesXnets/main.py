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

from hydra.utils import call, instantiate
from hydra.core.hydra_config import HydraConfig

from treesXnets.dataset import load_data
from treesXnets.server import gen_evaluate_fn
from treesXnets.constants import TASKS
from treesXnets.utils import plot_metric_from_history

from flwr.server.server import Server
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    cfg.task = TASKS[cfg.dataset.name]
    fds, cfg.dataset = load_data(cfg.dataset, cfg.task)

    # 3. Define your clients
    client_fn = call(cfg.client_fn, fds=fds, cfg=cfg)

    # 4. Define your strategy
    device = cfg.device
    evaluate_fn = gen_evaluate_fn(
        cfg.dataset.name,
        fds.load_full("test"), 
        device, 
        cfg.model
    )
    if cfg.strategy_name == 'fedavg':
        cfg.strategy._target_ = "flwr.server.strategy.FedAvg"
    else: 
        raise NotImplementedError
    strategy = instantiate(cfg.strategy, evaluate_fn=evaluate_fn)
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
        metric_type='centralized',
        regression=True if cfg.model.output_dim == 1 else False,
    )


if __name__ == "__main__":
    main()
