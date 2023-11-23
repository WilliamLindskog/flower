"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

from omegaconf import DictConfig

from niid_bench.constants import NUM_FEATURES, NUM_CLASSES

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from flwr.server.history import History
from torch.utils.data import DataLoader, Dataset

def prepare_model_config(config: DictConfig) -> DictConfig:
    """Prepare the model configuration.

    Parameters
    ----------
    config : DictConfig
        The configuration of the model

    Returns
    -------
    DictConfig
        The configuration of the model, with some adjusted parameters
    """
    dataset_name, model_name = config.dataset_name, config.model_name

    config.model.input_dim = NUM_FEATURES[dataset_name]
    config.model.output_dim = NUM_CLASSES[dataset_name]
    config.model._target_ = _set_model_target(model_name)
    config.task = _set_task(config.model.output_dim)
    
    return config

def _set_model_target(model_name: str) -> str:
    """Set the target of the model.

    Parameters
    ----------
    model_name : str
        The name of the model

    Returns
    -------
    str
        The target of the model
    """
    model_name = model_name.lower()

    if model_name == 'cnn':
        return 'niid_bench.models.cnn.CNN'
    elif model_name == 'mlp':
        return 'niid_bench.models.mlp.MLP'
    elif model_name == 'resnet':
        return 'niid_bench.models.resnet.ResNet.make_baseline'
    elif model_name == 'tabnet':
        return 'niid_bench.models.tabnet.TabNet'
    elif model_name == 'xgboost':
        pass
    else:
        raise NotImplementedError(f"Model {model_name} not supported")

def _set_task(output_dim: int) -> str:
    """Set the task of the model.

    Parameters
    ----------
    output_dim : int
        The output dimension of the model

    Returns
    -------
    str
        The task of the model
    """
    if output_dim == 1:
        return 'regression'
    else:
        return 'classification'
    
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

    if regression:
        _, values = zip(*metric_dict["r2"])
        values = tuple(x.item() for x in values)
    else:
        _, values = zip(*metric_dict["accuracy"])

    if metric_type == "centralized":
        rounds_loss, values_loss = zip(*hist.losses_centralized)
        # make tuple of normal floats instead of tensors
        values_loss = tuple(x for x in values_loss)
    else:
        # let's extract decentralized loss (main metric reported in FedProx paper)
        rounds_loss, values_loss = zip(*hist.losses_distributed)


    _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    if metric_type == "centralized":
        axs[1].plot(np.asarray(rounds_loss), np.asarray(values))    
    else:
        axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    axs[0].set_ylabel("Loss")

    if regression:
        axs[1].set_ylabel("R2")
    else:
        axs[1].set_ylabel("Accuracy")

    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()

