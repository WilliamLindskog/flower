"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

import torch.nn as nn
import torch.nn.functional as F
from typing import List, Type

from torch import Tensor
from types import ModuleType

from treesXnets.utils import _make_nn_module

SMALL = [32, 64, 128, 64]
MEDIUM = [32, 64, 128, 256, 128, 64]
LARGE = [32, 64, 128, 256, 512, 256, 128, 64]

def _set_hidden_layers(model_size: str) -> List[int]:
    """Set hidden layers."""
    if model_size == 'small':
        return SMALL
    elif model_size == 'medium':
        return MEDIUM
    elif model_size == 'large':
        return LARGE
    else:
        raise NotImplementedError

class MLP(nn.Module):
    """MLP model."""
        
    def __init__(self, **kwargs):
        """Initialize the model."""
        super().__init__()
        # Set model architecture
        self._set_model_architecture(**kwargs)

        # Set layers 
        self.model = nn.Sequential()
        for layer in range(self.num_layers):
            if layer == 0:
                self.model.add_module(f'fc{layer}', nn.Linear(self.input_dim, self.hidden_dims[layer]))
                self.model.add_module(f'act{layer}', nn.ReLU())
            else:
                self.model.add_module(f'fc{layer}', nn.Linear(self.hidden_dims[layer-1], self.hidden_dims[layer]))
                self.model.add_module(f'act{layer}', nn.ReLU())

        self.model.add_module(f'fc{self.num_layers}', nn.Linear(self.hidden_dims[-1], self.output_dim))

    def forward(self, x):
        """ Forward pass. """
        return self.model(x)

    def _set_model_architecture(self, **kwargs):
        """Set model architecture."""
        # Model architecture
        self.input_dim = kwargs['input_dim']
        self.hidden_dims = _set_hidden_layers(kwargs['model_size'])
        self.output_dim = kwargs['output_dim']
        self.num_layers = len(self.hidden_dims)


class ResNet(nn.Module):
    """The ResNet model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)

                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)

          Head: (in) -> Norm -> Activation -> Linear -> (out)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float = 0.0,
            dropout_second: float = 0.0,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        self.input_dim = kwargs['input_dim']
        self.hidden_dims = _set_hidden_layers(kwargs['model_size'])
        self.num_blocks = len(self.hidden_dims)
        self.output_dim = kwargs['output_dim']

        normalization, activation = 'ReLU', 'ReLU'
        self.first_layer = nn.Linear(self.input_dim, self.hidden_dims[0], bias=True)   
        self.blocks = nn.Sequential()
        for i in range(self.num_blocks - 1):
            if i == 0:
                self.blocks.add_module(
                    f'block_{i}',
                    ResNet.Block(
                        d_main=self.hidden_dims[i],
                        d_hidden=self.hidden_dims[i+1],
                        bias_first=True,
                        bias_second=True,
                        dropout_first=0.0,
                        dropout_second=0.0,
                        normalization=normalization,
                        activation=activation,
                        skip_connection=True,
                    ),
                )
            else: 
                self.blocks.add_module(
                    f'block_{i}',
                    ResNet.Block(
                        d_main=self.blocks[-1].linear_second.out_features,
                        d_hidden=self.hidden_dims[i+1],
                        bias_first=True,
                        bias_second=True,
                        dropout_first=0.0,
                        dropout_second=0.0,
                        normalization=normalization,
                        activation=activation,
                        skip_connection=True,
                    ),
                )

        self.head = ResNet.Head(
            d_in=self.blocks[-1].linear_second.out_features,
            d_out=self.output_dim,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x