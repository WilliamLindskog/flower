import argparse

from datasets import Dataset, DatasetDict, concatenate_datasets
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)

from typing import Union, Any, List, Tuple, Optional, Dict, OrderedDict
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from treesXnets.utils import TabularDataset
from pandas import DataFrame
from torch import tensor

from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from flwr.common import NDArray
import torch.nn as nn
from torchmetrics import Accuracy, MeanSquaredError
import flwr as fl
from tqdm import tqdm
from treesXnets.models import CNN


BST_PARAMS = {
    "imodels/credit-card" : {
        "objective": "binary:logistic",
        "eta": 0.01,  # Learning rate
        "max_depth": 6,
        "eval_metric": "rmse",
        "nthread": 16,
        "num_parallel_tree": 1,
        "subsample": 1.0,
        "tree_method": "hist",
    }
}

CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}


def client_args_parser():
    """Parse arguments to define experimental settings on client side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-partitions", default=10, type=int, help="Number of partitions."
    )
    parser.add_argument(
        "--partitioner-type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types.",
    )
    parser.add_argument(
        "--node-id",
        default=0,
        type=int,
        help="Node ID used for the current client.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed used for train/test splitting."
    )
    parser.add_argument(
        "--test-fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting.",
    )
    parser.add_argument(
        "--centralised-eval",
        action="store_true",
        help="Conduct centralised evaluation (True), or client evaluation on hold-out data (False).",
    )

    args = parser.parse_args()
    return args


def server_args_parser():
    """Parse arguments to define experimental settings on server side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool-size", default=2, type=int, help="Number of total clients."
    )
    parser.add_argument(
        "--num-rounds", default=5, type=int, help="Number of FL rounds."
    )
    parser.add_argument(
        "--num-clients-per-round",
        default=2,
        type=int,
        help="Number of clients participate in training each round.",
    )
    parser.add_argument(
        "--num-evaluate-clients",
        default=2,
        type=int,
        help="Number of clients selected for evaluation.",
    )
    parser.add_argument(
        "--centralised-eval",
        action="store_true",
        help="Conduct centralised evaluation (True), or client evaluation on hold-out data (False).",
    )

    args = parser.parse_args()
    return args



def instantiate_partitioner(partitioner_type: str, num_partitions: int):
    """Initialise partitioner based on selected partitioner type and number of
    partitions."""
    partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
        num_partitions=num_partitions
    )
    return partitioner


def train_test_split(partition: Dataset, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data: DataFrame, target: str) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    x = data.drop(target, axis=1)
    y = data[target]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def resplit(dataset: DatasetDict) -> DatasetDict:
    """Increase the quantity of centralised test samples from 500K to 1M."""
    return DatasetDict(
        {
            "train": dataset["train"].select(
                range(0, dataset["train"].num_rows - 500_000)
            ),
            "test": concatenate_datasets(
                [
                    dataset["train"].select(
                        range(
                            dataset["train"].num_rows - 500_000,
                            dataset["train"].num_rows,
                        )
                    ),
                    dataset["test"],
                ]
            ),
        }
    )

def plot_xgbtree(tree: Union[XGBClassifier, XGBRegressor], n_tree: int) -> None:
    """Visualize the built xgboost tree."""
    xgb.plot_tree(tree, num_trees=n_tree)
    plt.rcParams["figure.figsize"] = [50, 10]
    plt.show()



def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )


def construct_tree(
    dataset: Dataset, label: NDArray, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset."""
    if tree_type == "classification":
        if np.unique(label).shape[0] == 2:
            tree = xgb.XGBClassifier(
                objective="binary:logistic",
                learning_rate=0.1,
                max_depth=8,
                n_estimators=n_estimators,
                subsample=0.8,
                colsample_bylevel=1,
                colsample_bynode=1,
                colsample_bytree=1,
                alpha=5,
                gamma=5,
                num_parallel_tree=1,
                min_child_weight=1,
            )
        else:
            tree = xgb.XGBClassifier(
                objective="multi:softmax",
                learning_rate=0.1,
                max_depth=8,
                n_estimators=n_estimators,
                subsample=0.8,
                colsample_bylevel=1,
                colsample_bynode=1,
                colsample_bytree=1,
                alpha=5,
                gamma=5,
                num_parallel_tree=1,
                min_child_weight=1,
            )

    elif tree_type == "regression":
        tree = xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.1,
            max_depth=8,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            alpha=5,
            gamma=5,
            num_parallel_tree=1,
            min_child_weight=1,
        )

    tree.fit(dataset, label)
    return tree


def construct_tree_from_loader(
    dataset_loader: DataLoader, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset loader."""
    for dataset in dataset_loader:
        data, label = dataset[0], dataset[1]
    return construct_tree(data, label, n_estimators, tree_type)


def single_tree_prediction(
    tree: Union[XGBClassifier, XGBRegressor], n_tree: int, dataset: NDArray
) -> Optional[NDArray]:
    """Extract the prediction result of a single tree in the xgboost tree
    ensemble."""
    # How to access a single tree
    # https://github.com/bmreiniger/datascience.stackexchange/blob/master/57905.ipynb
    num_t = len(tree.get_booster().get_dump())
    if n_tree > num_t:
        print(
            "The tree index to be extracted is larger than the total number of trees."
        )
        return None

    return tree.predict(  # type: ignore
        dataset, iteration_range=(n_tree, n_tree + 1), output_margin=True
    )


def tree_encoding(  # pylint: disable=R0914
    trainloader: DataLoader,
    client_trees: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    client_tree_num: int,
    client_num: int,
) -> Optional[Tuple[NDArray, NDArray]]:
    """Transform the tabular dataset into prediction results using the
    aggregated xgboost tree ensembles from all clients."""
    if trainloader is None:
        return None

    for local_dataset in trainloader:
        x_train, y_train = local_dataset[0], local_dataset[1]

    x_train_enc = np.zeros((x_train.shape[0], client_num * client_tree_num))
    x_train_enc = np.array(x_train_enc, copy=True)

    temp_trees: Any = None
    if isinstance(client_trees, list) is False:
        temp_trees = [client_trees[0]] * client_num
    elif isinstance(client_trees, list) and len(client_trees) != client_num:
        temp_trees = [client_trees[0][0]] * client_num
    else:
        cids = []
        temp_trees = []
        for i, _ in enumerate(client_trees):
            temp_trees.append(client_trees[i][0])  # type: ignore
            cids.append(client_trees[i][1])  # type: ignore
        sorted_index = np.argsort(np.asarray(cids))
        temp_trees = np.asarray(temp_trees)[sorted_index]

    for i, _ in enumerate(temp_trees):
        for j in range(client_tree_num):
            x_train_enc[:, i * client_tree_num + j] = single_tree_prediction(
                temp_trees[i], j, x_train
            )

    x_train_enc32: Any = np.float32(x_train_enc)
    y_train32: Any = np.float32(y_train)

    x_train_enc32, y_train32 = torch.from_numpy(
        np.expand_dims(x_train_enc32, axis=1)  # type: ignore
    ), torch.from_numpy(
        np.expand_dims(y_train32, axis=-1)  # type: ignore
    )
    return x_train_enc32, y_train32

class TreeDataset(Dataset):
    def __init__(self, df, target: str, use_tensor: bool = False):
        if use_tensor:
            self.df = df
            self.target = target
        else:
            self.df = df.drop(target, axis=1).values
            self.df = torch.tensor(self.df, dtype=torch.float32)
            self.target = df[target].values
            self.target = torch.tensor(self.target, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int):
        label = self.target[idx]
        data = self.df[idx, :]
        sample = {0: data, 1: label}
        return sample

def train_tree(
    task_type: str,
    net: CNN,
    trainloader: DataLoader,
    device: torch.device,
    num_iterations: int,
    log_progress: bool = True,
) -> Tuple[float, float, int]:
    # Define loss and optimizer
    if task_type == "BINARY":
        criterion = nn.BCELoss()
    elif task_type == "REG":
        criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

    def cycle(iterable):
        """Repeats the contents of the train loader, in case it gets exhausted in 'num_iterations'."""
        while True:
            for x in iterable:
                yield x

    # Train the network
    net.train()
    total_loss, total_result, n_samples = 0.0, 0.0, 0
    pbar = (
        tqdm(iter(cycle(trainloader)), total=num_iterations, desc=f"TRAIN")
        if log_progress
        else iter(cycle(trainloader))
    )

    # Unusually, this training is formulated in terms of number of updates/iterations/batches processed
    # by the network. This will be helpful later on, when partitioning the data across clients: resulting
    # in differences between dataset sizes and hence inconsistent numbers of updates per 'epoch'.
    for i, data in zip(range(num_iterations), pbar):
        tree_outputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(tree_outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collected training loss and accuracy statistics
        total_loss += loss.item()
        n_samples += labels.size(0)

        if task_type == "classification":
            acc = Accuracy(task="binary")(outputs, labels.type(torch.int))
            total_result += acc * labels.size(0)
        elif task_type == "regression":
            mse = MeanSquaredError()(outputs, labels.type(torch.int))
            total_result += mse * labels.size(0)

        if log_progress:
            if task_type == "classification":
                pbar.set_postfix(
                    {
                        "train_loss": total_loss / n_samples,
                        "train_acc": total_result / n_samples,
                    }
                )
            elif task_type == "regression":
                pbar.set_postfix(
                    {
                        "train_loss": total_loss / n_samples,
                        "train_mse": total_result / n_samples,
                    }
                )
    if log_progress:
        print("\n")

    return total_loss / n_samples, total_result / n_samples, n_samples


def test_tree(
    task_type: str,
    net: CNN,
    testloader: DataLoader,
    device: torch.device,
    log_progress: bool = True,
    num_classes: Optional[int] = 2,
) -> Tuple[float, float, int]:
    """Evaluates the network on test data."""
    if num_classes is None:
        raise ValueError("num_classes must be specified for classification tasks.")
    elif num_classes == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    total_loss, total_result, n_samples = 0.0, 0.0, 0
    net.eval()
    with torch.no_grad():
        pbar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in pbar:
            tree_outputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(tree_outputs)

            # Collected testing loss and accuracy statistics
            total_loss += criterion(outputs, labels).item()
            n_samples += labels.size(0)

            if task_type == "classification":
                acc = Accuracy(task="binary")(
                    outputs.cpu(), labels.type(torch.int).cpu()
                )
                total_result += acc * labels.size(0)
            elif task_type == "regression":
                mse = MeanSquaredError()(outputs.cpu(), labels.type(torch.int).cpu())
                total_result += mse * labels.size(0)

    if log_progress:
        print("\n")

    return total_loss / n_samples, total_result / n_samples, n_samples

def tree_encoding_loader(
    dataloader: DataLoader,
    batch_size: int,
    client_trees: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    client_tree_num: int,
    client_num: int,
) -> DataLoader:
    encoding = tree_encoding(dataloader, client_trees, client_tree_num, client_num)
    if encoding is None:
        return None
    data, labels = encoding
    tree_dataset = TreeDataset(data, labels, use_tensor=True)
    return get_dataloader(tree_dataset, "tree", batch_size)

def accuracy(preds: np.ndarray, target: np.ndarray) -> float:
    """Computes the accuracy for multiple binary predictions"""
    return np.sum(preds == target) / len(preds)

def r2_metric(preds: np.ndarray, target: np.ndarray) -> float:
    """Computes the R2 score for multiple predictions"""
    # calculate the r2 metric without using sklearn
    ss_res = np.sum((target - preds) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - (ss_res / ss_tot)

def f1_metric(preds: np.ndarray, target: np.ndarray) -> float:
    """Computes the F1 score for multiple binary predictions"""
    # calculate the f1 metric without using sklearn
    tp = np.sum((preds == 1) & (target == 1))
    fp = np.sum((preds == 1) & (target == 0))
    fn = np.sum((preds == 0) & (target == 1))
    return tp / (tp + 0.5 * (fp + fn))

def mae_metric(preds: np.ndarray, target: np.ndarray) -> float:
    """Computes the MAE score for multiple predictions"""
    return np.mean(np.abs(preds - target))