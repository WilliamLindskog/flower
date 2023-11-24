"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Callable
from sklearn.metrics import r2_score
from pandas import DataFrame
from omegaconf import DictConfig
from torch.optim import Adam, Optimizer

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