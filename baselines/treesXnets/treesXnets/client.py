"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Callable, Dict, OrderedDict, Tuple
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
from flwr_datasets import FederatedDataset
import torch
import flwr as fl
from treesXnets.utils import test, TabularDataset, train
from pandas import DataFrame
from treesXnets.constants import TARGET

from flwr.common import Scalar

class FlowerClient(fl.client.NumPyClient):
    """Flower client implementing FedAvg."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: torch.nn.Module,
        fds: FederatedDataset,
        device: torch.device,
        cid: str,
        cfg: DictConfig,
    ) -> None:
        self.net = net
        self.fds = fds
        self.device = device
        self.cid = int(cid)
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.task = cfg.task

        # Get dataloaders 
        self.trainloader, self.testloader = self._load_data()

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
        train(self.net, self.trainloader, self.cfg)
        final_p_np = self.get_parameters({})
        return final_p_np, len(self.trainloader), {}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        if self.task == "classification":
            loss, acc = test(self.net, self.testloader, self.device)
            return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}
        else:
            loss, r2 = test(self.net, self.testloader, self.device, task=self.task)
            return float(loss), len(self.testloader.dataset), {"r2": float(r2)}
        
    def _load_data(self,) -> Tuple[DataLoader, DataLoader]:
        """Return the dataloader for the client."""
        # Get client partition
        partition = self.fds.load_partition(self.cid)

        # Divide partition into train and test
        partition_train_test = partition.train_test_split(test_size=0.2)
        trainset, testset = partition_train_test["train"], partition_train_test["test"]
        dataset_name = self.cfg.dataset_name

        train_data = TabularDataset(DataFrame(trainset), TARGET[dataset_name])
        test_data = TabularDataset(DataFrame(testset), TARGET[dataset_name])

        trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return trainloader, testloader

def gen_client_fn(
    fds: FederatedDataset,
    cfg: DictConfig,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the FedAvg flower clients.

    Parameters
    ----------
    fds : FederatedDataset
        The federated dataset object that contains the data, can be partitioned. 
    cfg : DictConfig
        An omegaconf object that stores the hydra config for the model.
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

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(cfg.model).to(device)

        return FlowerClient(net, fds, device, cid, cfg.client)

    return client_fn