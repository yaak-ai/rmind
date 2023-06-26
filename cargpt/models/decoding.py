import torch
from dall_e import load_model, unmap_pixels
from jaxtyping import Float
from torch import Tensor, nn


# https://github.com/openai/DALL-E/tree/master
class dalleDVAE(torch.nn.Module):
    def __init__(self, dec_weights: str, vocab_size: int, freeze: bool = True):
        super().__init__()

        self.dec = load_model(dec_weights)
        self.vocab_size = vocab_size

        if freeze:
            self.requires_grad_(False)
            self.dec.eval()

    def forward(
        self, z_logits: Float[Tensor, "b c h1 w1"]
    ) -> Float[Tensor, "b c2 h2 w2"]:
        z = torch.argmax(z_logits, dim=1)
        z_one_hot = (
            nn.functional.one_hot(z, num_classes=self.vocab_size)
            .permute(0, 3, 1, 2)
            .float()
        )

        x_stats = self.dec(z_one_hot).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))

        return x_rec

    def reconstruct(self, z):
        z_one_hot = (
            nn.functional.one_hot(z, num_classes=self.vocab_size)
            .permute(0, 3, 1, 2)
            .float()
        )

        x_stats = self.dec(z_one_hot).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))

        return x_rec
