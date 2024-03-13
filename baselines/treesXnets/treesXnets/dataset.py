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
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    NaturalIdPartitioner, SizePartitioner, LinearPartitioner,
    SquarePartitioner, ExponentialPartitioner, # ShardPartitioner, DirichletPartitioner
)
from treesXnets.constants import TARGET, TASKS, NUM_CLASSES
from treesXnets.dataset_preparation import get_partitioner, _label_partition, _check_data_gen, _gen_leaf_data, _create_df
from treesXnets.partitioner.shard_partitioner import ShardPartitioner

PATHS = {
    'femnist' : 'treesXnets/leaf/data/femnist',
    'synthetic' : 'treesXnets/leaf/data/synthetic',
}

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

    if cfg.name in ["femnist", "synthetic"]:
        df = _download_data(cfg.name, cfg.fraction)
        cfg.num_input = len(df.columns)-2
        cfg.num_classes = NUM_CLASSES[cfg.name]
        # set column "user" to "ID"
        df.rename(columns={"user": "ID"}, inplace=True)
        return df, cfg
    else:
        fds = _get_federated_data(hf_dataset="inria-soda/tabular-benchmark",config=cfg,)
    

    dataset_name = cfg.name
    df_list = []
    for client_id in range(cfg.num_clients):
        df = DataFrame(fds.load_partition(client_id))
        df["ID"] = client_id
        df_list.append(df)
    df = DataFrame(concat(df_list, ignore_index=True))

    if cfg.partition == "label":
        df = _label_partition(df, cfg.num_clients, "ID", TARGET[dataset_name])

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
    config: DictConfig,
    hf_dataset: str = "inria-soda/tabular-benchmark",
) -> FederatedDataset:
    """Return the data for the federated dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    hf_dataset : str
        The name of the Hugging Face dataset.
    config : DictConfig
        An omegaconf object that stores the hydra config for the dataset.
    num_clients : int
        The number of clients.

    Returns
    -------
    Tuple[Iterable[Tuple[int, DataFrame]], int]
        The data for the federated dataset and the number of input features.
    """

    # Get dataset name and partitioner
    dataset_name = config.name
    partitioner_name = config.partition
    partitioner = get_partitioner(config.partition, config.id_col)
    num_clients = config.num_clients

    # Load dataset
    fds = FederatedDataset(
        dataset=hf_dataset, subset=dataset_name,
        partitioners={"train": partitioner(num_clients)},
    )

    return fds

def _download_data(dataset_name="emnist", fraction=None):
    """Download the requested dataset. Currently supports cifar10, mnist, and fmnist.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training dataset, the test dataset.
    """
    # Set femnist and data path
    FL_BENCH_ROOT = Path(__file__).parent.parent
    data_root = FL_BENCH_ROOT / Path(PATHS[dataset_name])
    data_dir = data_root / 'data'
    data_path = data_dir / f'{dataset_name}.csv'

    if not _check_data_gen(data_dir):
        _gen_leaf_data(data_root, dataset_name)

    # Create femnist df if not exists
    if not data_path.exists():
        _create_df(data_dir, tag=f'{dataset_name}')

    # Read data
    data = pd.read_csv(data_path)

    if fraction is not None:
        data = data.sample(frac=fraction)
        data = data.reset_index(drop=True)

    # remove columns with 95% of the same value
    if dataset_name == 'femnist':
        print("Number of columns: ", len(data.columns))
        for col in data.columns:
            if data[col].value_counts(normalize=True).values[0] > 0.95:
                data.drop(col, axis=1, inplace=True)
        print("Number of columns after removing columns with 95% of the same value: ", len(data.columns))

    # encode user column
    user_encoder = {user: i for i, user in enumerate(data['user'].unique())}
    data['user'] = data['user'].apply(lambda x: user_encoder[x])

    return data
    # Create train and test set
    trainset, testset = train_test_split(data, test_size=0.1, random_state=42)