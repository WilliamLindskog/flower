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

from treesXnets.partitioner.shard_partitioner import ShardPartitioner

from typing import Callable, Optional
import pandas as pd

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
    elif partition_name in ["iid", "label"]:
        return IidPartitioner
    #elif partition_name == "label":
    #    return ShardPartitioner
    elif partition_name == "dirichlet":
        return DirichletPartitioner
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

    return df
    