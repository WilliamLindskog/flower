"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""
import flwr as fl
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Union, List
from omegaconf import DictConfig
from hydra.utils import instantiate
import timeit
from treesXnets.utils import update_model_state

import torch
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from flwr.common.logger import log
from logging import INFO, DEBUG
from flwr.common import NDArrays, Scalar, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.server import evaluate_clients, fit_clients
from flwr.server.client_proxy import ClientProxy
# Flower client
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    GetPropertiesRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from treesXnets.utils import test, TabularDataset, partition_to_dataloader
from pandas import DataFrame
from treesXnets.constants import TARGET, TASKS
from treesXnets.models import CNN
from hydra.utils import instantiate
from treesXnets.tree_utils import BST_PARAMS, tree_encoding_loader, TreeDataset
from treesXnets.constants import TARGET

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]

def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    print(f"Total number of samples: {total_num}")
    # eval_metrics is a list of tuples (num_samples, metrics) where metrics is a dict
    # Get the metric name with is the only key in the dict
    metric_name = list(eval_metrics[0][1].keys())[0]
    metric_aggregated = (
        sum([metrics[metric_name] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {f"{metric_name}": metric_aggregated}
    return metrics_aggregated

class FL_Server(fl.server.Server):
    """Flower server."""

    def __init__(
        self, *, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy
        self.max_workers: Optional[int] = None

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)

        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[
            Optional[
                Tuple[
                    Parameters,
                    Union[
                        Tuple[XGBClassifier, int],
                        Tuple[XGBRegressor, int],
                        List[
                            Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]
                        ],
                    ],
                ]
            ],
            Dict[str, Scalar],
            FitResultsAndFailures,
        ]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        NN_aggregated: Parameters
        trees_aggregated: Union[
            Tuple[XGBClassifier, int],
            Tuple[XGBRegressor, int],
            List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
        ]
        metrics_aggregated: Dict[str, Scalar]
        aggregated, metrics_aggregated = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        NN_aggregated, trees_aggregated = aggregated[0], aggregated[1]

        if type(trees_aggregated) is list:
            print("Server side aggregated", len(trees_aggregated), "trees.")
        else:
            print("Server side did not aggregate trees.")

        return (
            [NN_aggregated, trees_aggregated],
            metrics_aggregated,
            (results, failures),
        )

    def _get_initial_parameters(
        self, timeout: Optional[float]
    ) -> Tuple[Parameters, Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]]:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res_tree = random_client.get_parameters(ins=ins, timeout=timeout)
        print("----------------------------------")
        print(get_parameters_res_tree)
        print("----------------------------------")
        parameters = [get_parameters_res_tree[0].parameters, get_parameters_res_tree[1]]
        log(INFO, "Received initial parameters from one random client")

        return parameters

def get_evaluate_fn(
    testdata: FederatedDataset,
    device: torch.device,
    cfg: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.
    cfg : DictConfig
        The configuration.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
               Optional[Tuple[float, Dict[str, Scalar]]] ]
    The centralized evaluation function.
    """

    # Get model configuration, model name and dataset name
    model, model_name, dataset_name = cfg.model, cfg.model_name.lower(), cfg.dataset.name

    # Get the test function
    if model_name in ['mlp', 'resnet']: 
        def evaluate(
            server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            # pylint: disable=unused-argument
            """Use the entire Emnist test set for evaluation."""
            net = instantiate(model)
            net = update_model_state(net, parameters_ndarrays, device=device)

            # Get dataloader and return scores
            testloader = partition_to_dataloader(testdata, dataset_name, batch_size=len(testdata), test=True)
            task = TASKS[dataset_name]
            loss, metrics = test(net, testloader, device=device, task=task, evaluate=True)
            return loss, metrics
        
    elif model._target_ == "treesXnets.models.CNN":
        def evaluate(
            server_round: int,
            parameters: Tuple[
                Parameters,
                Union[
                    Tuple[XGBClassifier, int],
                    Tuple[XGBRegressor, int],
                    List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
                ],
            ],
            config: Dict[str, Scalar],
        ) -> Tuple[float, Dict[str, float]]:
            """An evaluation function for centralized/serverside evaluation over the entire test set."""
            net = instantiate(model)

            net.set_weights(parameters_to_ndarrays(parameters[0]))
            net.to(device)

            test_set = DataFrame(testdata)
            X_test, y_test = test_set.drop(TARGET[dataset_name], axis=1), test_set[TARGET[dataset_name]]
            test_data = TreeDataset(X_test.to_numpy(), y_test.to_numpy())
            testloader = DataLoader(test_data, batch_size=model.batch_size, shuffle=False)

            trees_aggregated = parameters[1]
            testloader = tree_encoding_loader(
                testloader, model.batch_size, trees_aggregated, model.client_tree_num, 
                model.num_clients
            )
            loss, result, _ = test(
                model.task, model, testloader, device=device, log_progress=False
            )

            if model.task == "classification":
                print(
                    f"Evaluation on the server: test_loss={loss:.4f}, test_accuracy={result:.4f}"
                )
                return loss, {"accuracy": result}
            elif model.task == "regression":
                print(f"Evaluation on the server: test_loss={loss:.4f}, test_mse={result:.4f}")
                return loss, {"mse": result}

    else:
        def evaluate(
            server_round: int, parameters: Parameters, config: Dict[str, Scalar]
        ):
            params = BST_PARAMS[dataset_name]
            # If at the first round, skip the evaluation
            if server_round == 0:
                return 0, {}
            else:
                bst = xgb.Booster(params=params)
                for para in parameters.tensors:
                    para_b = bytearray(para)

                # Load global model
                bst.load_model(para_b)

                # Get test data
                dataset = DataFrame(testdata)
                X_test = dataset.drop(columns=[TARGET[dataset_name]])
                y_test = dataset[TARGET[dataset_name]]
                test_data = xgb.DMatrix(X_test, label=y_test)

                # Get metric name
                print("Metric name is: ", params["eval_metric"])
                print("---------------------------------------------------------")
                if params["eval_metric"] == "auc":
                    metric_name = "auc"
                elif params["objective"] == "reg:squarederror":
                    metric_name = "rmse"
                else:
                    raise NotImplementedError

                # Run evaluation
                eval_results = bst.eval_set(
                    evals=[(test_data, "server_test")],
                    iteration=bst.num_boosted_rounds() - 1,
                )

                metric = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
                log(INFO, f"{metric_name} = {metric} at round {server_round}")

                # Get loss and metric
                

                return 0, {f"{metric_name}": metric}

    return evaluate
