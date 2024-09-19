from collections.abc import Callable

from tensordict import TensorDict
from torch.nn import Module


class CallableToTensorDictModule(Module):
    def __init__(self, callable: Callable):
        super().__init__()
        self.callable = callable

    def forward(self, tensordict: TensorDict) -> TensorDict:
        return tensordict.apply(self.callable)
