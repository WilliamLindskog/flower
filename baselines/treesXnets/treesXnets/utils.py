"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Callable, Optional, OrderedDict
from sklearn.metrics import (
    r2_score, accuracy_score, mean_squared_error, f1_score, roc_auc_score,
    mean_absolute_error
)
from pandas import DataFrame
from omegaconf import DictConfig
from torch.optim import Adam, Optimizer, SGD
from flwr.server.history import History
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
from types import ModuleType
from torch.utils.data import DataLoader
from treesXnets.constants import TARGET

import os

def test(
    net: nn.Module, 
    testloader: DataLoader, 
    device: torch.device, 
    task: str = "classification", 
    evaluate: bool = False
) -> Tuple[float, float]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to evaluate.
    testloader : DataLoader
        The test set dataloader object.
    device : torch.device
        The device on which to evaluate the network.

    Returns
    -------
    Tuple[float, float]
        The loss and accuracy of the network on the test set.
    """
    # Get critertion
    criterion = _get_criterion(task=task)

    net.eval()
    loss = 0.0
    with torch.no_grad():
        # since it's a test loop, out batch size is the whole test set
        data, target = next(iter(testloader))
        data, target = data.to(device), target.to(device)
        target = target.unsqueeze(1) if task == "regression" else target.long()
        output = net(data)
        loss += criterion(output, target).item()
        metrics = _get_scores(task, target, output)
    return loss, metrics

def _get_scores(task, target, output):
    """ Get scores for regression or classification task. """
    if task in ["multi", "binary"]:
        output = F.softmax(output, dim=1).argmax(dim=1)
        acc = accuracy_score(target.cpu().numpy(), output.cpu().numpy())
        if task == "binary":
            auc_score = roc_auc_score(target.cpu().numpy(), output.cpu().numpy())
            metrics = {"accuracy": acc, "auc": auc_score}
        else:
            f1 = f1_score(target.cpu().numpy(), output.cpu().numpy(), average='weighted')
            metrics = {"accuracy": acc, "f1": f1}
    else:
        r2 = r2_score(target.cpu().numpy(), output.cpu().numpy())
        mse = mean_squared_error(target.cpu().numpy(), output.cpu().numpy())
        metrics = {"r2": r2, "mse": mse}
    return metrics
    
def train(
    net: nn.Module,
    trainloader: DataLoader,
    cfg: DictConfig,
    central: bool = False,
) -> None:
    # pylint: disable=too-many-arguments
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The training set dataloader object.
    cfg:
        DictConfig with configuration parameters for training. 

    Returns
    -------
    None
    """
    if cfg.strategy_name in ['fedavg','fedprox']:
        # Get training parameters
        criterion = _get_criterion(cfg.task)
        lr, wd = cfg.lr, cfg.wd
        optimizer = SGD(net.parameters(), lr=lr, weight_decay=wd)
        # Train model
        net.train()
        num_epochs = cfg.num_rounds if central else cfg.num_epochs
        for _ in range(num_epochs):
            net = _train_one_epoch(net, trainloader, cfg.device, criterion, optimizer, cfg.task)
    else:
        raise NotImplementedError


def _train_one_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer,
    task: str,
) -> nn.Module:
    """Train the network on the training set for one epoch."""
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        target = target.unsqueeze(1) if task == "regression" else target.long()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return net
    
def _get_criterion(task: str) -> Callable:
    """ Get criterion/loss function for training. """
    if task in ["binary", "multi"]:
        criterion = nn.CrossEntropyLoss()
    elif task == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError("Task not supported")
    return criterion

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].

    References:

        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].

    References:

        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            return ReGLU()
        elif module_type == 'GEGLU':
            return GEGLU()
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(
                    f'Failed to construct the module {module_type} with the arguments {args}'
                ) from err
            return cls(*args)
    else:
        return module_type(*args)

class TabularDataset(torch.utils.data.Dataset):
    """Dataset for tabular data.

    Parameters
    ----------
    df : DataFrame  
        The dataframe containing the data.
    target : str
        The name of the target column.
    """
    def __init__(self, df: DataFrame, target: str):
        self.data = df.drop(columns=[target]).values
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.target = df[target].values
        self.target = torch.tensor(self.target, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
def scale_data(dataframe: DataFrame, target: str, test: bool = False) -> DataFrame:
    """Scale the data that is not the target column are are continuous. """

    # Encode categorical columns
    for col in dataframe.columns:
        if dataframe[col].dtype == "object":
            dataframe[col] = dataframe[col].astype("category")
            dataframe[col] = dataframe[col].cat.codes

    if not test:
        for col in dataframe.columns:
            if col != target and len(dataframe[col].unique()) > 50:
                dataframe[col] = dataframe[col].astype("float32")
                dataframe[col] = dataframe[col] / dataframe[col].max()

    return dataframe
    
def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
    metric_type: Optional[str] = "centralized",
    regression: Optional[bool] = False,
    model_name: Optional[str] = "mlp",
) -> None:
    """Plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    print(metric_dict)

    if regression:
        # get mse and r2 values from metric_dict
        m1, m2 = "mse", "r2"
        _, values_m1 = zip(*metric_dict[m1])
        _, values_m2 = zip(*metric_dict[m2])
        values_m1 = tuple(x for x in values_m1)
        values_m2 = tuple(x for x in values_m2)
    else:
        m1, m2 = "accuracy", "auc"
        _, values_m1 = zip(*metric_dict[m1])
        _, values_m2 = zip(*metric_dict[m2])
        values_m1 = tuple(x for x in values_m1)
        values_m2 = tuple(x for x in values_m2)



    if metric_type == "centralized":
        rounds_loss, values_loss = zip(*hist.losses_centralized)
        # make tuple of normal floats instead of tensors
        values_loss = tuple(x for x in values_loss)
    else:
        # let's extract decentralized loss (main metric reported in FedProx paper)
        rounds_loss, values_loss = zip(*hist.losses_distributed)
        # make tuple of normal floats instead of tensors


    _, axs = plt.subplots(nrows=3, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    if metric_type == "centralized":
        if model_name =='xgboost':
            axs[1].plot(np.asarray(rounds_loss[1:]), np.asarray(values))
        else:
            axs[1].plot(np.asarray(rounds_loss), np.asarray(values_m1))    
            axs[2].plot(np.asarray(rounds_loss), np.asarray(values_m2))
    else:
        axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    axs[0].set_ylabel("Loss")

    if regression:
        axs[1].set_ylabel("MSE")
        axs[2].set_ylabel("R2")
    else:
        axs[1].set_ylabel("Accuracy")
        axs[2].set_ylabel("AUC")

    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()

    # get distributed fit metrics from history

    metric_dict = (hist.metrics_distributed_fit)
    _, total_bytes = zip(*metric_dict["total_bytes"])
    total_bytes = tuple(x for x in total_bytes)
    # sum total bytes
    total_bytes = sum(total_bytes)
    print(total_bytes)
    # store in txt
    with open(Path(save_plot_path) / Path(f"{metric_type}_total_bytes{suffix}.txt"), "wb") as f:
        f.write(str(total_bytes).encode())

def empty_dir(path: Path) -> None:
    """Empty a directory.

    Parameters
    ----------
    path : Path
        The directory to empty.
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)

def modify_config(cfg: DictConfig) -> DictConfig:
    """Modify the config file to add the correct paths."""
    # Get model target
    cfg.model._target_ = _get_model_target(cfg.model_name)
    # Get strategy target
    if cfg.model_name == "xgboost":
        cfg.strategy._target_ = "flwr.server.strategy.fedxgb_bagging.FedXgbBagging"
        cfg.strategy_name = "xgboost"
    elif cfg.model_name == "glxgb":
        cfg.strategy._target_ = "flwr.server.strategy.fedxgb_nn_avg.FedXgbNnAvg"
        cfg.strategy_name = "fedxgb_nn_avg"

    return cfg

def _get_model_target(model_name: str) -> str:
    if model_name == "mlp":
        return "treesXnets.models.MLP"
    elif model_name == "cnn":
        return "treesXnets.models.CNN"
    elif model_name == "resnet":
        return "treesXnets.models.ResNet"
    elif model_name == "xgboost":
        return "xgboost.Booster"
    elif model_name == "glxgb":
        return "treesXnets.models.CNN"
    else:
        raise ValueError("Unknown model name.")    

def update_model_state(model, parameters, device):
    """ Update the model using the parameters."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)

def partition_to_dataloader(
        dataset, 
        dataset_name: str, 
        batch_size: int = 64, 
        tag: str = "tabular", 
        test: bool = False
):
    """ Convert a HuggingFace partition of dataset to a PyTorch DataLoader.
    
    Parameters
    ----------
    dataset : Dataset
        The dataset to convert.
    dataset_name : str
        The name of the dataset.
    batch_size : int
        The batch size.
    tag : str
        The tag of the dataset.
    test : bool
        Whether the dataset is a test dataset.

    """
    if test: 
        batch_size = len(dataset)
    if tag == "tabular":
        return _partition_to_tabular(dataset, dataset_name, batch_size)
    else:
        raise ValueError(f"Unknown dataset tag {tag}.")

def _partition_to_tabular(
        dataset, 
        dataset_name: str, 
        batch_size: int = 64, 
    ):
    """ Convert a HuggingFace partition of dataset to a PyTorch DataLoader.

    Parameters
    ----------
    dataset : Dataset
        The dataset to convert.
    dataset_name : str
        The name of the dataset.
    batch_size : int
        The batch size.
    test : bool
        Whether the dataset is a test dataset.
    """
    dataset = scale_data(dataset, TARGET[dataset_name], test)
    dataset = TabularDataset(dataset, TARGET[dataset_name])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)