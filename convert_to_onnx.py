from dataclasses import dataclass
import time

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from loguru import logger
from tensordict.nn import InteractionType, NormalParamExtractor, set_interaction_type
from tensordict.nn import ProbabilisticTensorDictModule as Prob
from tensordict.nn import TensorDictModule
from tensordict.nn import TensorDictSequential
from tensordict import TensorDict
import tensordict
from torch import distributions as dists
from torch import nn
from torch.export import export
from torch.nn import ModuleDict


@dataclass
class InputDataClass:
    td: TensorDict


@dataclass
class OutputDataClass:
    res_td: TensorDict


class VanillaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.md = ModuleDict({"x": nn.Linear(4, 8)})

    def forward(self, input: InputDataClass) -> OutputDataClass:
        res_td = input.td.named_apply(lambda k, v: self.md[k](v))
        return OutputDataClass(res_td=res_td)


from torch.utils._pytree import register_pytree_node

register_pytree_node(
    tensordict._td.TensorDict,
    tensordict._td.TensorDict.tree_flatten,  # flatten function
    tensordict._td.TensorDict.tree_unflatten,  # unflatten function
    serialized_type_name="TensorDict",  # Add this argument
)
# torch.export.register_dataclass(
#     InputDataClass, serialized_type_name="tensordict.TensorDict"
# )
# torch.export.register_dataclass(
#     OutputDataClass, serialized_type_name="tensordict.TensorDict"
# )


t = TensorDict({"x": torch.randn(2, 4)}, batch_size=[2])
input_sample = InputDataClass(td=t)

model_vanilla = VanillaModel()
exp = torch.export.export(model_vanilla, args=(input_sample,))
torch.export.save(exp, "model.pt2")
# model_vanilla(t)
# model_tensordict = TensorDictModule(MyModel(), in_keys=["x"], out_keys=["y"])


breakpoint()

# model_vanilla(x)
# model_tensordict(x=x)

# model_export_vanilla = export(model_vanilla, args=(x,))

# try:
#     model_export_tensordict = export(model_tensordict, args=(), kwargs={"x": x})
#     model_export_tensordict.module()(x=x)
# except:
#     print("Cant export TensorDictModel")


# model_export = export(model, args=(), kwargs={"x": x})

# model = Seq(
#     # 1. A small network for embedding
#     Mod(nn.Linear(3, 4), in_keys=["x"], out_keys=["hidden"]),
#     Mod(nn.ReLU(), in_keys=["hidden"], out_keys=["hidden"]),
#     Mod(nn.Linear(4, 4), in_keys=["hidden"], out_keys=["latent"]),
#     # 2. Extracting params
#     Mod(NormalParamExtractor(), in_keys=["latent"], out_keys=["loc", "scale"]),
#     # 3. Probabilistic module
#     Prob(
#         in_keys=["loc", "scale"], out_keys=["sample"], distribution_class=dists.Normal
#     ),
# )

# x = torch.randn(1, 3)
# print(model(x=x))

# from torch.export import export

# model_export = export(model, args=(), kwargs={"x": x})
# breakpoint()
# torch_model = MyModel()

# torch_input = torch.randn(1, 1, 32, 32)
# try:
#     torch_model(torch_input)
# except:
#     raise ValueError("Wrong cant inference")
# onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)


# logger.info("Saving model")
# onnx_program.save("my_image_classifier.onnx")


# logger.info("Loading onnx")
# onnx_model = onnx.load("my_image_classifier.onnx")
# onnx.checker.check_model(onnx_model)
# logger.info("Onnx is sucessfully checked")
