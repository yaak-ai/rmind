import torch
from dall_e import load_model, unmap_pixels  # pyright: ignore
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F
from typing_extensions import override


class DVAETokens(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(
        self,
        probs: Float[Tensor, "b c1 h w"],
        tokens_shift: int,
        embeddings: nn.Module,
    ) -> tuple[Float[Tensor, "b c2 h w"], Int[Tensor, "b h w"]]:
        tokens = torch.argmax(probs.detach(), dim=1)
        tokens += tokens_shift
        x = embeddings(tokens)
        x = rearrange(x, "b h w d -> b d h w")

        return x, tokens


# https://github.com/openai/DALL-E/tree/master
class DalleDVAEEncoder(nn.Module):
    def __init__(self, *, enc_weights: str, freeze: bool = True) -> None:
        super().__init__()

        self.enc = load_model(enc_weights).requires_grad_(not freeze).train(not freeze)

    @override
    def forward(
        self,
        x: Float[Tensor, "b c1 h1 w1"],
    ) -> Float[Tensor, "b c2 h2 w2"]:
        logits = self.enc(x)
        return F.softmax(logits, dim=1)


class DalleDVAEDecoder(torch.nn.Module):
    def __init__(
        self, *, dec_weights: str, vocab_size: int, freeze: bool = True
    ) -> None:
        super().__init__()

        self.dec = load_model(dec_weights).requires_grad_(not freeze).train(not freeze)
        self.vocab_size = vocab_size

    @override
    def forward(
        self,
        z_logits: Float[Tensor, "b c h1 w1"],
    ) -> Float[Tensor, "b c2 h2 w2"]:
        z = torch.argmax(z_logits, dim=1)
        z_one_hot = (
            nn.functional.one_hot(z, num_classes=self.vocab_size)
            .permute(0, 3, 1, 2)
            .float()
        )

        x_stats = self.dec(z_one_hot).float()
        return unmap_pixels(torch.sigmoid(x_stats[:, :3]))

    def reconstruct(self, z):
        z_one_hot = (
            nn.functional.one_hot(z, num_classes=self.vocab_size)
            .permute(0, 3, 1, 2)
            .float()
        )

        x_stats = self.dec(z_one_hot).float()
        return unmap_pixels(torch.sigmoid(x_stats[:, :3]))
