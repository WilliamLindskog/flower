import torch.nn as nn

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
        self.hidden_dims = kwargs['hidden_dims']
        self.output_dim = kwargs['output_dim']
        self.num_layers = len(self.hidden_dims)