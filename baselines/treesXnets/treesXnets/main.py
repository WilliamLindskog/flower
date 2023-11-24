"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import flwr as fl
import hydra
from omegaconf import DictConfig, OmegaConf

from hydra.utils import call, instantiate

from treesXnets.dataset import load_data
from treesXnets.server import gen_evaluate_fn
from treesXnets.constants import TASKS

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
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir


if __name__ == "__main__":
    main()
