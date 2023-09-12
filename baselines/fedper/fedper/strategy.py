"""FL server strategies."""
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy

from fedper.constants import Algorithms
from fedper.implemented_models.mobile_model import MobileNetModelSplit
from fedper.implemented_models.resnet_model import ResNetModelSplit
from fedper.models import ModelSplit


class ServerInitializationStrategy(Strategy):
    """Server FL Parameter Initialization strategy implementation."""

    def __init__(
        self,
        model_split_class: Union[
            Type[MobileNetModelSplit], Type[ModelSplit], Type[ResNetModelSplit]
        ],
        create_model: Callable[[], nn.Module],
        config: Dict[str, Any],
        algorithm: str = Algorithms.FEDAVG.value,
        has_fixed_head: bool = False,
        initial_parameters: Optional[Parameters] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        min_available_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_fit_clients: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.algorithm = algorithm
        self.on_fit_config_fn = on_fit_config_fn
        self.initial_parameters = initial_parameters
        self.min_available_clients = min_available_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_fit_clients = min_fit_clients
        # self.evaluate_fn = evaluate_fn
        self.model = model_split_class(
            model=create_model(), has_fixed_head=has_fixed_head
        )

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the (global) model parameters.

        Args:
            client_manager: ClientManager. The client manager which holds all currently
                connected clients.

        Returns
        -------
            If parameters are returned, then the server will treat these as the
            initial global model parameters.
        """
        initial_parameters: Optional[Parameters] = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if initial_parameters is None and self.model is not None:
            if self.algorithm == Algorithms.FEDPER.value:
                initial_parameters_use = [
                    val.cpu().numpy() for _, val in self.model.body.state_dict().items()
                ]
            else:  # FedAvg
                initial_parameters_use = [
                    val.cpu().numpy() for _, val in self.model.state_dict().items()
                ]

        if isinstance(initial_parameters_use, list):
            initial_parameters = ndarrays_to_parameters(initial_parameters_use)
        return initial_parameters


class AggregateFullStrategy(ServerInitializationStrategy):
    """Full model aggregation strategy implementation."""

    def __init__(self, save_path: Path = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        if save_path is not None:
            self.save_path = save_path / "models"
            self.save_path.mkdir(parents=True, exist_ok=True)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        Args:
            server_round: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently
                connected clients.

        Returns
        -------
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """
        # Same as superclass method but adds the head

        # Parameters and config
        config: Dict[Any, Any] = {}

        weights = parameters_to_ndarrays(parameters)

        parameters = ndarrays_to_parameters(weights)

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if server_round >= 0:
            clients = client_manager.sample(
                num_clients=self.min_available_clients,
                min_num_clients=self.min_evaluate_clients,
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate received local parameters, set global model parameters and save.

        Args:
            server_round: The current round of federated learning.
            results: Successful updates from the previously selected and configured
                clients. Each pair of `(ClientProxy, FitRes)` constitutes a
                successful update from one of the previously selected clients. Not
                that not all previously selected clients are necessarily included in
                this list: a client might drop out and not submit a result. For each
                client that did not submit an update, there should be an `Exception`
                in `failures`.
            failures: Exceptions that occurred while the server was waiting for client
                updates.

        Returns
        -------
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        agg_params, agg_metrics = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if agg_params is not None:
            # Update Server Model
            parameters = parameters_to_ndarrays(agg_params)
        model_keys = [
            k
            for k in self.model.state_dict().keys()
            if k.startswith("_body") or k.startswith("_head")
        ]
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.set_parameters(state_dict)

        if self.save_path is not None:
            # Save Model
            torch.save(self.model, self.save_path / f"model-ep_{server_round}.pt")

        return agg_params, agg_metrics


class AggregateBodyStrategy(ServerInitializationStrategy):
    """Body Aggregation strategy implementation."""

    def __init__(self, save_path: Path = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        if save_path is not None:
            self.save_path = save_path / "models"
            self.save_path.mkdir(parents=True, exist_ok=True)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Args:
            server_round: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all
                currently connected clients.

        Returns
        -------
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """
        # Same as superclass method but adds the head

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        weights = parameters_to_ndarrays(parameters)

        # Add head parameters to received body parameters
        weights.extend(
            [val.cpu().numpy() for _, val in self.model.head.state_dict().items()]
        )

        parameters = ndarrays_to_parameters(weights)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        clients = client_manager.sample(
            num_clients=self.min_available_clients, min_num_clients=self.min_fit_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        Args:
            server_round: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently
                connected clients.

        Returns
        -------
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """
        # Same as superclass method but adds the head

        # Parameters and config
        config: Dict[Any, Any] = {}

        weights = parameters_to_ndarrays(parameters)

        # Add head parameters to received body parameters
        weights.extend(
            [val.cpu().numpy() for _, val in self.model.head.state_dict().items()]
        )

        parameters = ndarrays_to_parameters(weights)

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if server_round >= 0:
            clients = client_manager.sample(
                num_clients=self.min_available_clients,
                min_num_clients=self.min_evaluate_clients,
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:
        """Aggregate received local parameters, set global model parameters and save.

        Args:
            server_round: The current round of federated learning.
            results: Successful updates from the previously selected and configured
                clients. Each pair of `(ClientProxy, FitRes)` constitutes a
                successful update from one of the previously selected clients. Not
                that not all previously selected clients are necessarily included in
                this list: a client might drop out and not submit a result. For each
                client that did not submit an update, there should be an `Exception`
                in `failures`.
            failures: Exceptions that occurred while the server was waiting for client
                updates.

        Returns
        -------
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        agg_params, agg_metrics = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if agg_params is not None:
            parameters = parameters_to_ndarrays(agg_params)
        model_keys = [
            k for k in self.model.state_dict().keys() if k.startswith("_body")
        ]
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.set_parameters(state_dict)

        if self.save_path is not None:
            # Save Model
            torch.save(self.model, self.save_path / f"model-ep_{server_round}.pt")

        return agg_params, agg_metrics


class StoreHistoryStrategy(Strategy):
    """Server FL history storage per round strategy implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hist: Dict[str, Dict[int, Any]] = {
            "trn": defaultdict(dict),
            "tst": defaultdict(dict),
        }


class StoreMetricsStrategy(StoreHistoryStrategy):
    """Server FL metrics storage per round strategy implementation."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate the received local parameters and store the training aggregated.

        Args:
            server_round: The current round of federated learning.
            results: Successful updates from the previously selected and configured
                clients. Each pair of `(ClientProxy, FitRes)` constitutes a
                successful update from one of the previously selected clients. Not
                that not all previously selected clients are necessarily included in
                this list: a client might drop out and not submit a result. For each
                client that did not submit an update, there should be an `Exception`
                in `failures`.
            failures: Exceptions that occurred while the server was waiting for client
                updates.

        Returns
        -------
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        aggregates = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )

        self.hist["trn"][server_round] = {
            k.cid: {"num_examples": v.num_examples, **v.metrics} for k, v in results
        }

        return aggregates

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate the received local parameters and store the test aggregated.

        Args:
            server_round: The current round of federated learning.
            results: Successful updates from the
                previously selected and configured clients. Each pair of
                `(ClientProxy, FitRes` constitutes a successful update from one of the
                previously selected clients. Not that not all previously selected
                clients are necessarily included in this list: a client might drop out
                and not submit a result. For each client that did not submit an update,
                there should be an `Exception` in `failures`.
            failures: Exceptions that occurred while the server
                was waiting for client updates.

        Returns
        -------
            Optional `float` representing the aggregated evaluation result. Aggregation
            typically uses some variant of a weighted average.
        """
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round=server_round, results=results, failures=failures
        )
        _ = aggregated_metrics  # Avoid unused variable warning
        self.hist["tst"][server_round] = {
            k.cid: {"num_examples": v.num_examples, "loss": v.loss, **v.metrics}
            for k, v in results
        }

        # Weigh accuracy of each client by number of examples used
        accuracies: List[float] = []
        for _, res in results:
            accuracy: float = float(res.metrics["accuracy"])
            accuracies.append(accuracy)
        print(f"Round {server_round} accuracies: {accuracies}")

        # Aggregate and print custom metric
        averaged_accuracy = sum(accuracies) / len(accuracies)
        print(f"Round {server_round} accuracy averaged: {averaged_accuracy}")
        return aggregated_loss, {"accuracy": averaged_accuracy}


class StoreSelectedClientsStrategy(StoreHistoryStrategy):
    """Server FL selected clients storage per training/evaluation round strategy."""

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training and save the selected clients.

        Args:
            server_round: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently
                connected clients.

        Returns
        -------
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """
        result = super().configure_fit(
            server_round=server_round,
            parameters=parameters,
            client_manager=client_manager,
        )

        if server_round not in self.hist["trn"].keys():
            self.hist["trn"][server_round] = {}

        self.hist["trn"][server_round]["selected_clients"] = [
            client.cid for client, _ in result
        ]

        # Return client/config pairs
        return result

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation and save the selected clients.

        Args:
            server_round: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently
                connected clients.

        Returns
        -------
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """
        result = super().configure_evaluate(
            server_round=server_round,
            parameters=parameters,
            client_manager=client_manager,
        )

        if server_round not in self.hist["tst"].keys():
            self.hist["tst"][server_round] = {}

        self.hist["tst"][server_round]["selected_clients"] = [
            client.cid for client, _ in result
        ]

        # Return client/config pairs
        return result
