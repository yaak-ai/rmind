from typing import Any, List
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from pytorch_lightning.utilities.parsing import AttributeDict


np.set_printoptions(suppress=True)


class Frames(pl.LightningModule):
    hparams: AttributeDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(self.hparams.inference_model)
        self.model.eval()
        self.decoder = instantiate(self.hparams.image_decoder)

    def get_future_frames(self, batch: Any) -> List[np.ndarray]:
        sample = self.model.prepare_batch(batch)

        b, clips, c, h, w = sample["frames"].shape
        episode, labels, _, episode_values = self.model._make_episode(sample)
        b, seqlen, d = episode.shape
        future_images = []

        with torch.no_grad():
            for timestep in range(self.hparams.future_timestep):  # type: ignore
                # first image token
                # add local timestep token
                image_tokens = []
                for image_token_index in range(self.hparams.patch_row * self.hparams.patch_col):  # type: ignore
                    pred, _ = self.model.forward(episode=episode)
                    last_token = torch.argmax(F.softmax(pred[:, [-1]], dim=2), dim=2)
                    embedding = self.model.sensor_embedding(last_token)
                    last_token -= self.model.hparams.tokens_shift["ImageEncoder"]
                    image_tokens.append(last_token)
                    row = (
                        torch.tensor(
                            [image_token_index // self.hparams.patch_col],  # type: ignore
                            device=embedding.device,
                        )
                        .view(1, 1)
                        .repeat(b, 1)
                    )
                    col = (
                        torch.tensor(
                            [image_token_index - row * self.hparams.patch_col],
                            device=embedding.device,
                        )
                        .view(1, 1)
                        .repeat(b, 1)
                    )
                    # add patch row / col position
                    embedding += self.model.patch_row(row) + self.model.patch_col(col)
                    # add local timestep
                    local_pos = (
                        torch.tensor([image_token_index], device=embedding.device)
                        .view(1, 1)
                        .repeat(b, 1)
                    )
                    embedding += self.model.local_position(local_pos)
                    # global timstep
                    global_pos = (
                        torch.tensor([timestep], device=embedding.device)
                        .view(1, 1)
                        .repeat(b, 1)
                    )
                    embedding += self.model.global_position(global_pos)
                    episode = torch.cat([episode, embedding], 1)

                image_tokens = torch.cat(image_tokens, 1).view(
                    b, self.hparams.patch_row, self.hparams.patch_col  # type: ignore
                )
                image_tokens[image_tokens < 0] = 0
                image = self.decoder.reconstruct(image_tokens)
                x_rec = T.ToPILImage(mode="RGB")(image[0].detach().cpu())
                x_rec.save(f"timestep_{timestep:03}.png")

                future_images.append(image[0].detach().cpu().numpy())

                for metadata_token_index in range(len(self.model.metadata_keys)):
                    # add local time step if metadata
                    pass
                # add sep token
                for action_key_index in range(len(self.model.action_keys)):
                    # add action position if action
                    pass

                # update sample and build a new episode
                episode, labels, _, episode_values = self.model._make_episode(sample)

        # computed_actions.append(detokenized_actions)

        return future_images

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> List[np.ndarray]:
        future_images = self.get_future_frames(batch)

        return future_images
