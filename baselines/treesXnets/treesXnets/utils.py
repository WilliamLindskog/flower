"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
from sklearn.metrics import r2_score
from pandas import DataFrame

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