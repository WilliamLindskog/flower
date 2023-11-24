"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from omegaconf import DictConfig
from typing import Iterable, Tuple, List

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    NaturalIdPartitioner, SizePartitioner, LinearPartitioner,
    SquarePartitioner, ExponentialPartitioner
)
from treesXnets.constants import TARGET, TASKS

from treesXnets.dataset_preparation import get_partitioner

def load_data(cfg: DictConfig, task: str) -> FederatedDataset:
    """Return the dataloaders for the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config for the dataset.
    task : str
        The task type of the dataset.

    Returns
    -------
    Federated Dataset.
    """
    # Get dataset name
    dataset_name = cfg.name

    # Get partitioner
    partitioner = get_partitioner(cfg.partition, cfg.id_col)

    fds = FederatedDataset(
        dataset = dataset_name,
        partitioners = {
            "train" : partitioner(cfg.num_clients)
        },
    )

    # Get number of input features and classes
    cfg.num_input = len(fds.load_partition(0).features)-1
    if task == "regression":
        cfg.num_classes = 1
    else: 
        raise NotImplementedError
    
    return fds, cfg