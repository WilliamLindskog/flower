"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from treesXnets.utils import test
from pandas import DataFrame
from treesXnets.constants import TARGET, TASKS
from treesXnets.utils import TabularDataset

def gen_evaluate_fn(
    dataset_name: str,
    testdata: FederatedDataset,
    device: torch.device,
    model: DictConfig,
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

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
               Optional[Tuple[float, Dict[str, Scalar]]] ]
    The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire Emnist test set for evaluation."""
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        # Get dataloader
        dataset = TabularDataset(DataFrame(testdata), TARGET[dataset_name])
        testloader = DataLoader(dataset, batch_size=64, shuffle=False)

        task = TASKS[dataset_name]
        if task == "classification":
            loss, accuracy = test(net, testloader, device=device, task=task, evaluate=True)
            return loss, {"accuracy": accuracy}
        loss, r2 = test(net, testloader, device=device, task=task, evaluate=True)
        return loss, {"r2": r2}

    return evaluate
