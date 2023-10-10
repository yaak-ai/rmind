from torch import nn


class Sequential(nn.Sequential):
    def forward(self, *args, **kwargs):
        """Allows passing arbitrary args/kwargs to the first module."""
        first, *rest = self._modules.values()
        x = first(*args, **kwargs)

        for module in rest:
            x = module(x)

        return x
