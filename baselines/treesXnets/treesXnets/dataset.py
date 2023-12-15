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
from pandas import DataFrame, concat

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    NaturalIdPartitioner, SizePartitioner, LinearPartitioner,
    SquarePartitioner, ExponentialPartitioner
)
from treesXnets.constants import TARGET, TASKS, NUM_CLASSES
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
    # Get dataset name and partitioner
    dataset_name = cfg.name
    partitioner = get_partitioner(cfg.partition, cfg.id_col)

    with open('./token.txt', 'r') as f:
        token = f.read()

    fds = FederatedDataset(
        dataset = 'inria-soda/tabular-benchmark',
        subset = dataset_name,
        partitioners = {"train" : partitioner(cfg.num_clients)},
        token = token,
    )

    df_list = []
    for client_id in range(cfg.num_clients):
        df = DataFrame(fds.load_partition(client_id))
        df["ID"] = client_id
        df_list.append(df)
    df = DataFrame(concat(df_list, ignore_index=True))

    # Get number of input features and classes
    cfg.num_input = len(df.columns)-2
    cfg.num_classes = NUM_CLASSES[dataset_name]
    if cfg.num_classes > 1:
        # ensure that target value starts at 0
        if df[TARGET[dataset_name]].min() != 0:
            df[TARGET[dataset_name]] -= df[TARGET[dataset_name]].min()
    return df, cfg