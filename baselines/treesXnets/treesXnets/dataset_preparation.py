"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
# import hydra
# from hydra.core.hydra_config import HydraConfig
# from hydra.utils import call, instantiate
# from omegaconf import DictConfig, OmegaConf


# @hydra.main(config_path="conf", config_name="base", version_base=None)
# def download_and_preprocess(cfg: DictConfig) -> None:
#     """Does everything needed to get the dataset.

#     Parameters
#     ----------
#     cfg : DictConfig
#         An omegaconf object that stores the hydra config.
#     """

#     ## 1. print parsed config
#     print(OmegaConf.to_yaml(cfg))

#     # Please include here all the logic
#     # Please use the Hydra config style as much as possible specially
#     # for parts that can be customised (e.g. how data is partitioned)

# if __name__ == "__main__":

#     download_and_preprocess()

from flwr_datasets.partitioner import (
    ExponentialPartitioner,
    LinearPartitioner,
    NaturalIdPartitioner,
    SizePartitioner,
    SquarePartitioner,
    IidPartitioner,
)

import numpy as np
import pandas as pd
from typing import Callable, Optional

import random
import json

from treesXnets.partitioner.shard_partitioner import ShardPartitioner

from typing import Callable, Optional
import pandas as pd
from pathlib import Path
import os
import subprocess

def get_partitioner(partition_name: str, id_col: Optional[str] = None) -> Callable:
    """Return the partitioner function given its name.

    Parameters
    ----------
    partition_name : str
        The name of the partitioner.

    Returns
    -------
    Callable
        The partitioner function.
    """
    if partition_name == "natural":
        assert id_col is not None, "id_col must be provided for NaturalIdPartitioner"
        return NaturalIdPartitioner
    elif partition_name == "size":
        return SizePartitioner
    elif partition_name == "linear":
        return LinearPartitioner
    elif partition_name == "square":
        return SquarePartitioner
    elif partition_name == "exponential":
        return ExponentialPartitioner
    elif partition_name in ["iid", "label", "gaussian"]:
        return IidPartitioner
    else:
        raise ValueError(f"Partitioner {partition_name} not found.")
    
def _label_partition(
        df: pd.DataFrame, num_clients: int, id_col: str, target_col: str,
        num_allotted: Optional[int] = 1
        ) -> pd.DataFrame:
    """Partition the dataset by the label.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be partitioned.
    num_clients : int
        The number of clients.
    id_col : str
        The name of the column that contains the client id.
    target_col : str
        The name of the column that contains the target.
    num_allotted : int, optional
        The number of labels allotted to each client, by default 1.

    Returns
    -------
    pd.DataFrame
        The partitioned dataset.
    """
    assert num_allotted > 0, "num_allotted must be greater than 0"
    assert num_allotted <= df[target_col].nunique(), "num_allotted must be less than or equal to the number of unique labels"
    seed = random.randint(0, 1000)
    prng = np.random.default_rng(seed)

    targets = df[target_col].values
    num_classes = len(set(targets))
    times = [0 for _ in range(num_classes)]
    contains = []

    for i in range(num_clients):
        current = [i % num_classes]
        times[i % num_classes] += 1
        j = 1
        while j < num_allotted:
            index = prng.choice(num_classes, 1)[0]
            if index not in current:
                current.append(index)
                times[index] += 1
                j += 1
        contains.append(current)
    idx_clients = [[] for _ in range(num_clients)]
    for i in range(num_classes):
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if i in contains[j]:
                idx_clients[j] += idx_k_split[ids].tolist()
                ids += 1
    for i in range(num_clients):
        prng.shuffle(idx_clients[i])
    for i in range(num_clients):
        df.loc[idx_clients[i], id_col] = i
        
    return df

def _gaussian_noise_partition(
        df: pd.DataFrame, target_col: str,
        sigma: float = 0.1
    ) -> pd.DataFrame:
    """    Specifically, given user-defined noise level σ, we add noises
            xˆ ∼ Gau(σ · i/N) for Party Pi
            , where Gau(σ · i/N) is
            a Gaussian distribution with mean 0 and variance σ · i/N.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be partitioned.
    num_clients : int
        The number of clients.
    id_col : str
        The name of the column that contains the client id.
    target_col : str
        The name of the column that contains the target.
    sigma : float, optional
        The standard deviation of the Gaussian noise, by default 0.1.

    Returns
    -------
    pd.DataFrame
        The partitioned dataset.
    """

    # add noise to each feature (i.e. not the target)
    # also, only add noise to continuous features and not categorical ones (i.e. values other than int or floats that end with .0)
    for col in df.columns:
        
        if col not in [target_col, "ID"] and df[col].dtype in [int, float] and not str(df[col].dtype).endswith(".0"):
            df[col] += np.random.normal(0, sigma, len(df))

    return df

    

def _gen_leaf_data(data_root: Path, dataset_name: str) -> None:
    """ Generate leaf data.
    
    Parameters
    ----------
    data_root : Path
        Path to data root.
    dataset_name : str
        Name of dataset.
        
    Returns
    -------
    None
    """

    cwd = os.getcwd()
    os.chdir(data_root)

    # run subprocess to generate data
    if dataset_name == 'femnist':
        subprocess.run(
            [   
                
                'bash', 
                './preprocess.sh',
                '-s', "niid", 
                '--sf', "0.2",
                '-k', "128",
                '-t', 'sample'
            ]
        )
    else:
        subprocess.run(
            [   
                'python', 
                './main.py',
                '-num-tasks', "3000", 
                '-num-classes', "10",
                '-num-dim', "20",
            ]
        )

        subprocess.run(
            [   
                'bash', 
                './preprocess.sh',
                '--sf', "1.0", 
                '-k', "128",
                '-t', "sample",
            ]
        )

    os.chdir(cwd)

def _check_data_gen(data_path: Path) -> bool:
    """ Check if data is already generated.
    
    Parameters
    ----------
    data_path : Path
        Path to data.

    Returns
    -------
    bool
        True if data is already generated, False otherwise.
    """
    return (data_path / 'train').exists() and (data_path / 'test').exists()

def _create_df(data_dir: Path, tag: str = None) -> None:
    """ Create femnist df.
    
    Parameters
    ----------
    data_path : Path
        Path to data.
    femnist_data_path : Path
        Path to femnist data.
        
    Returns
    -------
    None
    """

    train_path, test_path = data_dir / 'train', data_dir / 'test'
    df_list = []
    for _, path in enumerate([train_path, test_path]):
        for file in path.glob('*'):
            print(file)
            with open(file) as f:
                data = json.load(f)
                users = data['users']
                for user in users:
                    user_data = {'x': data['user_data'][user]['x'], 'y': data['user_data'][user]['y']}
                    for i in range(len(user_data['x'])):
                        user_data['x'][i] = np.array(user_data['x'][i])
                        for j in range(len(user_data['x'][i])):
                            user_data[f'x_{j}'] = user_data['x'][i][j]
                        user_data['y'][i] = np.array(user_data['y'][i])
                        df_temp = pd.DataFrame({k: [v] for k, v in user_data.items() if k not in ['x', 'y']})
                        df_temp['y'] = user_data['y'][i]
                        df_temp['user'] = user
                        df_list.append(df_temp)
    # name df based on n
    df = pd.concat(df_list)
    data_path = data_dir / f'{tag}.csv'
    df.to_csv(data_path, index=False)