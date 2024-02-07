from math import prod

from jaxtyping import Float
from torch import Tensor, nn


class DinoEncoder(nn.Module):
    def __init__(self, *, dino: nn.Module, freeze: bool = True) -> None:
        super().__init__()

        self.dino = dino.requires_grad_(not freeze).train(not freeze)

    def forward(
        self,
        frames: Float[Tensor, "*b c1 h1 w1"],
    ) -> Float[Tensor, "*b h2 w2 c2"]:
        *B, _, H1, W1 = frames.shape

        frames = frames.view(prod(B), -1, H1, W1)
        feat = self.dino.forward_features(frames)
        tokens = feat["x_norm_patchtokens"]

        H2 = H1 // self.dino.patch_size
        W2 = W1 // self.dino.patch_size
        return tokens.view(*B, H2, W2, -1)
