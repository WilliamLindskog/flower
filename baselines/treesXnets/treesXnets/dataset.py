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
    SquarePartitioner, ExponentialPartitioner, # ShardPartitioner, DirichletPartitioner
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

    # Get token
    with open('./token.txt', 'r') as f:
        token = f.read()

    fds = _get_federated_data(
        dataset_name=dataset_name,
        hf_dataset=cfg.hf_dataset,
        partitioner=partitioner,
        token=token,
        num_clients=cfg.num_clients,
    )
    print(f"Loaded {dataset_name} dataset with {cfg.num_clients} clients")
    quit()

    df_list = []
    for client_id in range(cfg.num_clients):
        df = DataFrame(fds.load_partition(client_id))
        df["ID"] = client_id
        df_list.append(df)
    df = DataFrame(concat(df_list, ignore_index=True))

    # Get number of input features and classes
    cfg.num_input = len(df.columns)-2
    cfg.num_classes = NUM_CLASSES[dataset_name]

    # if nans drop 
    if df.isnull().values.any():
        print(len(df))
        df = df.dropna()
        print(len(df))

    if cfg.num_classes > 1:
        # if values in target are not integers, convert them to category
        if not isinstance(df[TARGET[dataset_name]].dtype, int):
            df[TARGET[dataset_name]] = df[TARGET[dataset_name]].astype("category").cat.codes
        # ensure that target value starts at 0
        if df[TARGET[dataset_name]].min() != 0:
            df[TARGET[dataset_name]] -= df[TARGET[dataset_name]].min()
    return df, cfg


def _get_federated_data(
    dataset_name: str,
    hf_dataset: str = "inria-soda/tabular-benchmark",
    partitioner: str = "iid",
    token: str = None,
    num_clients: int = 10,
) -> FederatedDataset:
    """Return the data for the federated dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    hf_dataset : str
        The name of the Hugging Face dataset.
    partitioner : str
        The name of the partitioner.
    token : str
        The token for the dataset.
    num_clients : int
        The number of clients.

    Returns
    -------
    Tuple[Iterable[Tuple[int, DataFrame]], int]
        The data for the federated dataset and the number of input features.
    """
    # Get partitioner
    partitioner = get_partitioner(partitioner)

    # Load dataset
    if not isinstance(partitioner, ShardPartitioner): 
        fds = FederatedDataset(
            dataset=hf_dataset, subset=dataset_name,
            partitioners={"train": partitioner(num_clients)}, token=token,
        )
    else: 
        fds = FederatedDataset(
            dataset=hf_dataset, subset=dataset_name,
            partitioners={"train": partitioner(
                num_partitions=num_clients, partition_by=TARGET[dataset_name], 
                shard_size=100, keep_incomplete_shard=True)}, 
                token=token,
        )

    return fds