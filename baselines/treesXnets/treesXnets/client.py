"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Callable, Dict, OrderedDict, Tuple, Union, List, Type
from torch.utils.data import DataLoader, Dataset
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
from treesXnets.utils import test, TabularDataset, _train_one_epoch, scale_data
from treesXnets.models import CNN
from treesXnets.tree_utils import (
    tree_encoding_loader, TreeDataset, construct_tree_from_loader,
    train_tree, test_tree, mae_metric, r2_metric, f1_metric, accuracy, transform_dataset_to_dmatrix
)
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split

from flwr.common import (
    Code, EvaluateIns, EvaluateRes, FitIns, FitRes,
    GetParametersIns, GetParametersRes, Parameters, Status, parameters_to_ndarrays,
    ndarrays_to_parameters, GetPropertiesIns, GetPropertiesRes
)

from flwr.common import Scalar
from flwr.common.logger import log
from treesXnets.tree_utils import BST_PARAMS
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

class NetClient(fl.client.NumPyClient):
    """Flower client implementing FedAvg."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: torch.nn.Module,
        df: DataFrame,
        device: torch.device,
        cid: str,
        cfg: DictConfig,
    ) -> None:
        self.net = net
        self.device = device
        self.cid = int(cid)
        self.df = df[df.ID == self.cid]
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.task = cfg.task

        # Get dataloaders 
        self.trainloader, self.testloader = _load_data(self.df, cfg, self.batch_size,)

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
        loss, metrics = test(self.net, self.testloader, device=self.device, task=self.task, evaluate=True)
        return float(loss), len(self.testloader), metrics

# Define Flower client
class XgbClient(fl.client.Client):
    def __init__(
            self, 
            model,
            df: DataFrame, 
            device: str, 
            cid: str, 
            cfg: DictConfig,
        ) -> None:
        self.config = None
        self.model = model
        self.cid = int(cid)
        self.df = df[df["ID"] == self.cid]
        self.device = device
        self.cfg = cfg
        self.params = dict(cfg.xgboost)
        self.bst = None

        # Load data
        self.train_dmatrix, self.test_dmatrix = _load_data(self.df, cfg, cfg.batch_size, tag="dmatrix")
        self.num_train = self.train_dmatrix.num_row()
        self.num_test = self.test_dmatrix.num_row()
        self.num_local_round = self.cfg.num_local_round

        if cfg.num_classes == 1:
            self.params["objective"] = "reg:squarederror"
        elif cfg.num_classes == 2:
            self.params["objective"] = "binary:logistic"
        else:
            self.params["objective"] = "multi:softmax"


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
            if self.cfg.num_classes == 1:
                self.params["eval_metric"] = 'mae'# [mae_metric, r2_metric]
                self.metric_name = "mae"
            elif self.cfg.num_classes == 2:
                self.params["eval_metric"] = 'error'
                self.metric_name = "error"
            else:
                self.params["eval_metric"] = 'merror'
                self.metric_name = "merror"
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.train_dmatrix, "train"), (self.test_dmatrix, "test")],
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

        return FitRes(
            status=Status(code=Code.OK,message="OK",),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        eval_results = self.bst.eval_set(
            evals=[(self.test_dmatrix, "test")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )

        metric = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        global_round = ins.config["global_round"]
        log(INFO, f"{self.metric_name} = {metric} at round {global_round}")

        return EvaluateRes(
            status=Status(code=Code.OK,message="OK",),
            loss=0.0,
            num_examples=self.num_test,
            metrics={f"{self.metric_name}": metric},
        )

    
class GLXGBClient(fl.client.Client):
    def __init__(
        self, 
        net, 
        fds: DataFrame, 
        device: str, 
        cid: str, 
        cfg: DictConfig
    ):
        """
        Creates a client for training `network.Net` on tabular dataset.
        """
        self.task = cfg.task
        self.cid = cid
        self.fds = fds
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.client_tree_num = cfg.client_tree_num
        self.num_clients = cfg.num_clients
        self.properties = {"tensor_type": "numpy.ndarray"}
        self.log_progress = False

        # instantiate model
        self.net = net

        # determine device
        self.device = device

        # load data
        self.trainloader_original, self.testloader_original = self._load_data()
        self.tree = construct_tree_from_loader(self.trainloader, self.client_tree_num, self.task)

    def _load_data(self,) -> Tuple[DataLoader, DataLoader]:
        """Return the dataloader for the client."""
        # Get client partition
        partition = self.fds.load_partition(self.cid)

        # Divide partition into train and test
        partition_train_test = partition.train_test_split(test_size=0.2)
        trainset, testset = partition_train_test["train"], partition_train_test["test"]
        dataset_name = self.cfg.dataset_name
        train_set, test_set = DataFrame(trainset), DataFrame(testset)
        X_train, y_train = train_set.drop(TARGET[dataset_name], axis=1), train_set[TARGET[dataset_name]]
        X_test, y_test = test_set.drop(TARGET[dataset_name], axis=1), test_set[TARGET[dataset_name]]

        train_data = TreeDataset(X_train.to_numpy(), y_train.to_numpy())
        test_data = TreeDataset(X_test.to_numpy(), y_test.to_numpy())

        trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return trainloader, testloader

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(properties=self.properties)

    def get_parameters(
        self, ins: GetParametersIns
    ) -> Tuple[
        GetParametersRes, Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]
    ]:
        return [
            GetParametersRes(
                status=Status(Code.OK, ""),
                parameters=ndarrays_to_parameters(self.net.get_weights()),
            ),
            (self.tree, int(self.cid)),
        ]

    def set_parameters(
        self,
        parameters: Tuple[
            Parameters,
            Union[
                Tuple[XGBClassifier, int],
                Tuple[XGBRegressor, int],
                List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
            ],
        ],
    ) -> Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ]:
        self.net.set_weights(parameters_to_ndarrays(parameters[0]))
        return parameters[1]

    def fit(self, fit_params: FitIns) -> FitRes:
        # Process incoming request to train
        num_iterations = self.cfg.num_local_round
        batch_size = self.batch_size
        aggregated_trees = self.set_parameters(fit_params.parameters)

        if type(aggregated_trees) is list:
            print("Client " + self.cid + ": recieved", len(aggregated_trees), "trees")
        else:
            print("Client " + self.cid + ": only had its own tree")
        self.trainloader = tree_encoding_loader(
            self.trainloader_original,
            batch_size,
            aggregated_trees,
            self.client_tree_num,
            self.client_num,
        )
        self.testloader = tree_encoding_loader(
            self.testloader_original,
            batch_size,
            aggregated_trees,
            self.client_tree_num,
            self.client_num,
        )

        # num_iterations = None special behaviour: train(...) runs for a single epoch, however many updates it may be
        num_iterations = num_iterations or len(self.trainloader)

        # Train the model
        print(f"Client {self.cid}: training for {num_iterations} iterations/updates")
        self.net.to(self.device)
        train_loss, train_result, num_examples = train_tree(
            self.task_type,
            self.net,
            self.trainloader,
            device=self.device,
            num_iterations=num_iterations,
            log_progress=self.log_progress,
        )
        print(
            f"Client {self.cid}: training round complete, {num_examples} examples processed"
        )

        # Return training information: model, number of examples processed and metrics
        if self.task == "classification":
            return FitRes(
                status=Status(Code.OK, ""),
                parameters=self.get_parameters(fit_params.config),
                num_examples=num_examples,
                metrics={"loss": train_loss, "accuracy": train_result},
            )
        elif self.task == "regression":
            return FitRes(
                status=Status(Code.OK, ""),
                parameters=self.get_parameters(fit_params.config),
                num_examples=num_examples,
                metrics={"loss": train_loss, "mse": train_result},
            )

    def evaluate(self, eval_params: EvaluateIns) -> EvaluateRes:
        # Process incoming request to evaluate
        self.set_parameters(eval_params.parameters)

        # Evaluate the model
        self.net.to(self.device)
        loss, result, num_examples = test_tree(
            self.task_type,
            self.net,
            self.valloader,
            device=self.device,
            log_progress=self.log_progress,
        )

        # Return evaluation information
        if self.task_type == "classification":
            print(
                f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, accuracy={result:.4f}"
            )
            return EvaluateRes(
                status=Status(Code.OK, ""),
                loss=loss,
                num_examples=num_examples,
                metrics={"accuracy": result},
            )
        elif self.task_type == "regression":
            print(
                f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, mse={result:.4f}"
            )
            return EvaluateRes(
                status=Status(Code.OK, ""),
                loss=loss,
                num_examples=num_examples,
                metrics={"mse": result},
            )

def get_client_fn(
    df: DataFrame,
    cfg: DictConfig,
) -> Callable[[str], Union[NetClient, XgbClient]]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the FedAvg flower clients.

    Parameters
    ----------
    fds : FederatedDataset
        The federated dataset object that contains the data, can be partitioned. 
    cfg : DictConfig
        An omegaconf object that stores the hydra config for the model.
    Returns
    -------
    Callable[[str], FlowerClient]
        The client function that creates the flower clients
    """

    def client_fn(cid: str) -> Union[NetClient, XgbClient]:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if cfg.model_name != 'xgboost': 
            model = instantiate(cfg.model).to(device)
        else: 
            model = None

        client = _get_client_class(cfg.model_name)
        return client(model, df, device, cid, cfg.client)

    return client_fn

def _get_client_class(model_name: str) -> Callable:
    """Get the client class based on the model name."""
    if model_name == "cnn":
        return GLXGBClient
    elif model_name == "xgboost":
        return XgbClient
    elif model_name in ["mlp", "resnet"]:
        return NetClient
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
def _load_data(df, cfg, batch_size, tag: str = 'tabular') -> Tuple[DataLoader, DataLoader]:
        """Return the dataloader for the client."""
        df = df.drop(columns=['ID'])
        dataset_name = cfg.dataset_name
        trainset, testset = train_test_split(df, test_size=0.2, random_state=42)

        # Scale data
        trainset, testset = scale_data(trainset, TARGET[dataset_name]), scale_data(testset, TARGET[dataset_name])

        # Get dataset class
        dataset_class = _get_dataset_class(tag)
        train_data = dataset_class(trainset, TARGET[dataset_name])
        test_data = dataset_class(testset, TARGET[dataset_name])

        if tag == 'dmatrix':
            return train_data, test_data
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_data, batch_size=len(testset), shuffle=False)

        return trainloader, testloader

def _get_dataset_class(tag: str = 'tabular') -> Callable:
    """Get the dataset class based on the tag."""
    if tag == "tabular":
        return TabularDataset
    elif tag == "tree":
        return TreeDataset
    elif tag == "dmatrix":
        return transform_dataset_to_dmatrix
    else:
        raise ValueError(f"Unknown dataset tag: {tag}")