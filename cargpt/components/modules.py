import torch.nn as nn
from typing_extensions import override


class SequentialModule(nn.Module):
    # torchvision mlp? check
    def __init__(self, layer_dims):
        super().__init__()

        layers = []
        in_dim = layer_dims[0]

        for out_dim in layer_dims[1:-1]:
            layers.extend((nn.Linear(in_dim, out_dim, bias=True), nn.ReLU()))
            in_dim = out_dim  # Update input dimension for the next layer

        # Final output layer without ReLU
        layers.append(nn.Linear(in_dim, layer_dims[-1], bias=False))

        self.model = nn.Sequential(*layers)

    @override
    def forward(self, x):
        return self.model(x)
