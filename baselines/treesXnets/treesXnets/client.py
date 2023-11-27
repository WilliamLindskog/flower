"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Callable, Dict, OrderedDict, Tuple, Union
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
from flwr_datasets import FederatedDataset
import torch
import flwr as fl
from treesXnets.utils import test, TabularDataset, train
from pandas import DataFrame
from treesXnets.constants import TARGET
from logging import INFO
from hydra.utils import instantiate


from flwr.common import (
    Code, EvaluateIns, EvaluateRes, FitIns, FitRes,
    GetParametersIns, GetParametersRes, Parameters, Status,
)

from flwr.common import Scalar
from flwr.common.logger import log
from treesXnets.tree_utils import BST_PARAMS
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

class NetClient(fl.client.NumPyClient):
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
    

# Define Flower client
class XgbClient(fl.client.Client):
    def __init__(
            self, 
            fds: FederatedDataset, 
            device: str, 
            cid: str, 
            cfg: DictConfig
        ) -> None:
        self.config = None
        self.fds = fds
        self.device = device
        self.cid = int(cid)
        self.cfg = cfg
        self.params = BST_PARAMS[cfg.dataset_name]
        self.bst = None

        # Load data
        self.train_dmatrix, self.test_dmatrix = self._load_data()
        self.num_train = self.train_dmatrix.num_row()
        self.num_test = self.test_dmatrix.num_row()
        self.num_local_round = self.cfg.num_local_round

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(code=Code.OK,message="OK",),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            self.bst.update(self.train_dmatrix, self.bst.num_boosted_rounds())

        # Extract the last N=num_local_round trees for sever aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - self.num_local_round : self.bst.num_boosted_rounds()
        ]

        return bst
    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # First round local training
            log(INFO, "Start training at round 1")
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.test_dmatrix, "test"), (self.train_dmatrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            bst = self._local_boost()

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        # Save model in tmp folder
        bst.save_model(f"./treesXnets/tmp/{self.cid}.txt")

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load local model into booster
        #self.bst = xgb.Booster(self.params)
        #self.bst.load_model(f"./treesXnets/tmp/{self.cid}.txt")

        eval_results = self.bst.eval_set(
            evals=[(self.test_dmatrix, "test")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )

        metric = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        global_round = ins.config["global_round"]
        if self.params["eval_metric"] == 'auc':
            metric_name = "AUC"
        elif self.params["eval_metric"] == 'rmse':
            metric_name = "RMSE"
        log(INFO, f"{metric_name} = {metric} at round {global_round}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_test,
            metrics={f"{metric_name}": metric},
        )

    def _load_data(self,) -> Tuple[DataLoader, DataLoader]:
        """Return the dataloader for the client."""
        # Get client partition
        partition = self.fds.load_partition(self.cid, split="train")

        # Divide partition into train and test
        partition_train_test = partition.train_test_split(test_size=0.2)
        trainset, testset = partition_train_test["train"], partition_train_test["test"]
        dataset_name = self.cfg.dataset_name

        # Set train and test data
        train_data, test_data = DataFrame(trainset), DataFrame(testset)
        X_train, y_train = train_data.drop(TARGET[dataset_name], axis=1), train_data[TARGET[dataset_name]]
        X_test, y_test = test_data.drop(TARGET[dataset_name], axis=1), test_data[TARGET[dataset_name]]

        # Set DMatrix
        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        valid_dmatrix = xgb.DMatrix(X_test, label=y_test)

        return train_dmatrix, valid_dmatrix

def get_client_fn(
    fds: FederatedDataset,
    cfg: DictConfig,
) -> Callable[[str], Union[NetClient, XgbClient]]:  # pylint: disable=too-many-arguments
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

    def client_fn(cid: str) -> Union[NetClient, XgbClient]:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if cfg.model_name != 'xgboost': 
            model = instantiate(cfg.model).to(device)


        if cfg.model_name != 'xgboost': 
            return NetClient(model, fds, device, cid, cfg.client)
        return XgbClient(fds, device, cid, cfg.client)

    return client_fn