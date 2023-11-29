"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Callable, Optional
from sklearn.metrics import r2_score
from pandas import DataFrame
from omegaconf import DictConfig
from torch.optim import Adam, Optimizer
from flwr.server.history import History
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
from types import ModuleType

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
    if task == "classification":
        criterion = nn.CrossEntropyLoss(reduction="sum")
    elif task == "regression":
        criterion = nn.MSELoss(reduction="sum")
    else:
        raise ValueError("Task not supported")
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            if task == "regression":
                target = target.unsqueeze(1)
            output = net(data)
            loss += criterion(output, target).item()
            if task == "classification":
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
            else:
                predicted = output
            total += target.size(0)
    loss = loss / total
    if task == "classification":
        acc = correct / total
        return loss, acc
    else:
        r2 = r2_score(target.cpu().numpy(), predicted.cpu().numpy())
        return loss, r2
    
def train(
    net: nn.Module,
    trainloader: DataLoader,
    cfg: DictConfig
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
    if cfg.strategy_name == 'fedavg':
        # Get training parameters
        criterion = _get_criterion(cfg.task)
        lr, wd = cfg.lr, cfg.wd
        optimizer = Adam(net.parameters(), lr=lr, weight_decay=wd)
        # Train model
        net.train()
        for _ in range(cfg.num_epochs):
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
        if task == "regression":
            target = target.unsqueeze(1)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return net
    
def _get_criterion(task: str) -> Callable:
    """ Get criterion/loss function for training. """
    if task == "classification":
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
    
def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
    metric_type: Optional[str] = "centralized",
    regression: Optional[bool] = False,
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
        _, values = zip(*metric_dict["rmse"])
        values = tuple(x for x in values)
    else:
        _, values = zip(*metric_dict["accuracy"])

    if metric_type == "centralized":
        rounds_loss, values_loss = zip(*hist.losses_centralized)
        # make tuple of normal floats instead of tensors
        values_loss = tuple(x for x in values_loss)
    else:
        # let's extract decentralized loss (main metric reported in FedProx paper)
        rounds_loss, values_loss = zip(*hist.losses_distributed)
        # make tuple of normal floats instead of tensors


    _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    if metric_type == "centralized":
        axs[1].plot(np.asarray(rounds_loss[1:]), np.asarray(values))    
    else:
        axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    axs[0].set_ylabel("Loss")

    if regression:
        axs[1].set_ylabel("RMSE")
    else:
        axs[1].set_ylabel("Accuracy")

    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()

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