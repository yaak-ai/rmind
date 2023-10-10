import torch
from dall_e import load_model, unmap_pixels
from jaxtyping import Float
from torch import Tensor, nn


# https://github.com/openai/DALL-E/tree/master
class dalleDVAE(torch.nn.Module):
    def __init__(self, dec_weights: str, vocab_size: int, freeze: bool = True) -> None:
        super().__init__()

        self.dec = load_model(dec_weights).requires_grad_(not freeze).train(not freeze)
        self.vocab_size = vocab_size

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
