from typing import Any, List

import more_itertools as mit
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import tqdm
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
        self.logging = self.hparams.logging

    def get_future_frames(self, batch: Any) -> List[np.ndarray]:
        with torch.no_grad():
            future_images = []
            frames = mit.one(batch["frames"].values())
            b, clips, c, h, w = frames.shape
            episode, labels, _, episode_values, _ = self.model._make_episode(
                batch, is_training=True
            )
            b, seqlen, d = episode.shape

            max_length = labels.shape[1]

            output = self.model.gpt(
                inputs_embeds=episode,
                return_dict=True,
                use_cache=True,
                output_hidden_states=True,
            )
            logits = output["logits"]
            past_key_values = output["past_key_values"]

            last_token = torch.argmax(F.softmax(logits[:, [-1]], dim=2), dim=2)
            # last_token = labels[:, [0]]
            # TODO: My humor is DRY unlike my code
            # Lot of duplicated code below, i beg your pardon upfront.
            torchvision.utils.save_image(frames[0], "inputs.png")

            global_timestep_embedding = self.model.sensor_embedding(
                torch.tensor([clips], device=episode.device).view(1, 1)
            ).view(1, 1, d)

            for timestep in tqdm.tqdm(range(self.hparams.future_timestep), ascii=True, unit="img"):  # type: ignore
                past_key_values = [
                    [key[:, :, -max_length:], value[:, :, -max_length:]]
                    for key, value in past_key_values
                ]
                # history = episode[:, :timestep_size]
                # first image token
                # add local timestep token
                image_tokens = []
                metadata_action_tokens = []

                for image_token_index in range(self.hparams.patch_row * self.hparams.patch_col):  # type: ignore
                    embedding = self.model.sensor_embedding(last_token)
                    last_token -= self.model.hparams.tokens_shift["ImageEncoder"]
                    image_tokens.append(last_token)

                    if self.model.hparams.have_position_encoding.patch:  # type: ignore[union-attr]
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
                        embedding += self.model.patch_row(row) + self.model.patch_col(
                            col
                        )

                    if self.model.hparams.have_position_encoding.local:  # type: ignore[union-attr]
                        # add local timestep
                        local_pos = (
                            torch.tensor([image_token_index], device=embedding.device)
                            .view(1, 1)
                            .repeat(b, 1)
                        )
                        embedding += self.model.local_position(local_pos)
                    if self.model.hparams.have_position_encoding.global_pos:  # type: ignore[union-attr]
                        # global timstep
                        embedding += global_timestep_embedding

                    output = self.model.gpt(
                        inputs_embeds=embedding,
                        return_dict=True,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = output["logits"]
                    past_key_values = output["past_key_values"]
                    last_token = torch.argmax(F.softmax(logits[:, [-1]], dim=2), dim=2)

                image_tokens = torch.cat(image_tokens, 1).view(
                    b, self.hparams.patch_row, self.hparams.patch_col  # type: ignore
                )
                image_tokens[image_tokens < 0] = 0
                image = self.decoder.reconstruct(image_tokens)
                x_rec = T.ToPILImage(mode="RGB")(image[0].detach().cpu())
                x_rec.save(f"timestep_{timestep:03}.png")
                future_images.append(
                    [
                        (image.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(
                            np.uint8
                        ),
                        None,
                    ]
                )

                for idx, metadata_key in enumerate(self.model.metadata_keys):
                    # add local time step if metadata
                    embedding = self.model.sensor_embedding(last_token)
                    metadata_action_tokens.append(last_token)
                    if self.model.hparams.have_position_encoding.local:  # type: ignore[union-attr]
                        # add local timestep
                        local_pos = (
                            torch.tensor(
                                [self.hparams.patch_row * self.hparams.patch_col + idx],  # type: ignore[union-attr]
                                device=embedding.device,
                            )
                            .view(1, 1)
                            .repeat(b, 1)
                        )
                        embedding += self.model.local_position(local_pos)
                    if self.model.hparams.have_position_encoding.global_pos:  # type: ignore[union-attr]
                        embedding += global_timestep_embedding
                    output = self.model.gpt(
                        inputs_embeds=embedding,
                        return_dict=True,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = output["logits"]
                    past_key_values = output["past_key_values"]
                    last_token = torch.argmax(F.softmax(logits[:, [-1]], dim=2), dim=2)

                # Here last token is SEP
                for idx, action_key in enumerate(["sep"] + self.model.action_keys):
                    # add action position if action
                    embedding = self.model.sensor_embedding(last_token)
                    metadata_action_tokens.append(last_token)
                    if self.model.hparams.have_position_encoding.action:  # type: ignore[union-attr]
                        # add action position
                        action_position = (
                            self.model.action_position(
                                torch.tensor([0], device=embedding.device)
                                .view(1, 1)
                                .repeat(b, 1)
                            )
                            if action_key != "sep"
                            else 0
                        )
                        embedding += action_position
                    # global timstep
                    if self.model.hparams.have_position_encoding.global_pos:  # type: ignore[union-attr]
                        embedding += global_timestep_embedding
                    output = self.model.gpt(
                        inputs_embeds=embedding,
                        return_dict=True,
                        output_hidden_states=True,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = output["logits"]
                    past_key_values = output["past_key_values"]
                    last_token = torch.argmax(F.softmax(logits[:, [-1]], dim=2), dim=2)

        return future_images[0]

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> List[np.ndarray]:
        future_images = self.get_future_frames(batch)

        return future_images
