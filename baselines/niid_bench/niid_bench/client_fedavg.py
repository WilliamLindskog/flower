"""Defines the client class and support functions for FedAvg."""

from typing import Callable, Dict, List, OrderedDict

import flwr as fl
import torch
import os
import pickle
from flwr.common import Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from niid_bench.models.aggregation import test, train_fedavg
from niid_bench.models.tabnet import TabNet
import numpy as np
from niid_bench.utils import untangle_dataloader, initialize_tabnet

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


# pylint: disable=too-many-instance-attributes
class FlowerClientFedAvg(fl.client.NumPyClient):
    """Flower client implementing FedAvg."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        task: str = "classification",
        batch_size: int = 128,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.task = task
        self.batch_size = batch_size

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for FedAvg."""
        self.set_parameters(parameters)
        train_fedavg(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            self.task
        )
        final_p_np = self.get_parameters({})
        return final_p_np, len(self.trainloader), {}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        if self.task == "classification":
            loss, acc = test(self.net, self.valloader, self.device)
            return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}
        else:
            loss, r2 = test(self.net, self.valloader, self.device, task=self.task)
            return float(loss), len(self.valloader.dataset), {"r2": float(r2)}


# pylint: disable=too-many-arguments
def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    num_epochs: int,
    learning_rate: float,
    model: DictConfig,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
) -> Callable[[str], FlowerClientFedAvg]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the FedAvg flower clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    momentum : float
        The momentum for SGD optimizer of clients
    weight_decay : float
        The weight decay for SGD optimizer of clients

    Returns
    -------
    Callable[[str], FlowerClientFedAvg]
        The client function that creates the FedAvg flower clients
    """

    def client_fn(cid: str) -> FlowerClientFedAvg:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model._target_ == "niid_bench.models.tabnet.TabNet":
            net = instantiate(model)
        else:
            net = instantiate(model).to(device)
        task = model.task
        batch_size = model.batch_size

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientFedAvg(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
            task,
            batch_size,
        )

    return client_fn
