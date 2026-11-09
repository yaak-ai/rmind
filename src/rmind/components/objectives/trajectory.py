"""Autoregressive FAST trajectory head.

The target is the ego trajectory over the next `num_poses` steps from the last timestep,
tokenized with the fitted FAST tokenizer (rounded DCT coefficients of the poses, then BPE;
`rmind.components.fast_trajectory`). A causal transformer decoder cross-attends to the last
world latent W(T-1) and predicts the token sequence one token at a time (next-token
cross-entropy); `predict` decodes greedily and returns the poses.
"""

from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import final, override

import numpy as np
import torch
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import Module

from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.fast_trajectory import FASTTrajectoryTokenizer
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
)

IGNORE = -100
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../src/rmind


class _OutputBias(Module):
    """Output bias of the tied head (its weight is the token embedding)."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n))


@final
class TrajectoryObjective(Objective):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        decoder: InstanceOf[Module],
        tokenizer_dir: str,
        dim: int,
        target: tuple[str, ...] = ("input", "context", "trajectory_target"),
        key_norm: InstanceOf[Module] | None = None,
        latent_index: int = -1,
        max_tokens: int = 32,
        loss: InstanceOf[Module] | None = None,
        decode_on_validation: bool = True,
        tie_embeddings: bool = True,
        num_samples: int = 0,
    ) -> None:
        super().__init__()
        path = Path(tokenizer_dir)
        self.fast = FASTTrajectoryTokenizer.load(
            path if path.is_absolute() else _PACKAGE_ROOT / path
        )
        vocab = self.fast.vocab_size
        # special ids after the BPE vocabulary
        self.bos, self.eos, self.pad = vocab, vocab + 1, vocab + 2
        self.decoder = decoder
        self.key_norm: Module | None = key_norm
        self.token_embedding = nn.Embedding(vocab + 3, dim)
        # small init (GPT-2 style): with tying this matrix also produces the logits, and the
        # default N(0, 1) would start them ~sqrt(dim) times too large
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # module parameters named weight / bias, so SelectiveAdamW can group them
        self.position = nn.Embedding(max_tokens, dim)
        nn.init.normal_(self.position.weight, std=0.02)
        self.final_norm = nn.LayerNorm(dim)
        # output over the same vocab + 3 ids as the input embedding. BOS / PAD are never targets,
        # so training pushes their logits down, and generation masks them out. With
        # tie_embeddings the output layer is the token-embedding matrix (plus its own bias).
        self.tie_embeddings = tie_embeddings
        self.head: Module = (
            _OutputBias(vocab + 3) if tie_embeddings else nn.Linear(dim, vocab + 3)
        )
        self.target = target
        self.latent_index = latent_index
        self.max_tokens = max_tokens
        # token loss on the non-padding positions; default cross-entropy (e.g. FocalLoss instead)
        self.losses: Module = loss if loss is not None else nn.CrossEntropyLoss()
        # in eval mode, also decode greedily to report position errors in metres and keep
        # target / prediction for the W&B trajectory logger (training steps skip this)
        self.decode_on_validation = decode_on_validation
        # predict(): 0 = greedy decoding; N > 0 = the most likely of the greedy decode and N - 1 samples (by total
        # log-probability; candidates not decoding to exactly P x 3 values are skipped). Stats of the last
        # predict() go to `last_decode_stats`.
        self.num_samples = num_samples
        self.last_decode_stats: dict | None = None

    def _key(self, latent: Tensor) -> Tensor:
        key = latent[:, self.latent_index]  # (b, n, d) world latent of one timestep
        return self.key_norm(key) if self.key_norm is not None else key

    def _logits(self, key: Tensor, tokens: Tensor) -> Tensor:
        """(b, L) input token ids -> (b, L, vocab + 3) next-token logits."""
        x = self.token_embedding(tokens) + self.position.weight[: tokens.shape[1]]
        h = self.final_norm(self.decoder(x, key, key))
        weight = (
            self.token_embedding.weight if self.tie_embeddings else self.head.weight
        )
        return torch.nn.functional.linear(h, weight, self.head.bias)

    def _encode(self, poses: Tensor) -> tuple[Tensor, Tensor]:
        """(b, P, 3) poses -> decoder inputs [BOS t1 .. tL] and targets [t1 .. tL EOS], padded."""
        ids = self.fast.encode(poses.detach().float().cpu().numpy())
        n = min(max(len(i) for i in ids) + 1, self.max_tokens)
        inputs = torch.full((len(ids), n), self.pad, dtype=torch.long)
        targets = torch.full((len(ids), n), IGNORE, dtype=torch.long)
        for row, seq in enumerate(ids):
            kept = seq[: n - 1]
            inputs[row, : len(kept) + 1] = torch.tensor([self.bos, *kept])
            targets[row, : len(kept) + 1] = torch.tensor([*kept, self.eos])
        return inputs.to(poses.device), targets.to(poses.device)

    @override
    def compute_metrics(
        self, *, episode: Episode, embedding: Tensor, latent: Tensor | None = None
    ) -> Metrics:
        assert latent is not None  # noqa: S101
        poses = episode.get(self.target)[
            :, -1
        ]  # (b, P, 3) trajectory from the last timestep
        inputs, targets = self._encode(poses)
        logits = self._logits(self._key(latent), inputs)
        valid = targets != IGNORE
        loss = self.losses(logits[valid], targets[valid])
        metrics: dict = {"loss": {"tokens": loss}}
        with torch.no_grad():
            metrics["token_accuracy"] = (
                (logits.argmax(-1) == targets)[valid].float().mean()
            )
            if self.decode_on_validation and not self.training:
                predicted, _ = self.generate(latent)
                err = (predicted[..., :2] - poses[..., :2].float()).norm(
                    dim=-1
                )  # (b, P) metres
                metrics["ade_m"] = err.mean()
                metrics["fde_m"] = err[:, -1].mean()
                metrics["_artifacts"] = {
                    "target": poses.float(),
                    "predicted": predicted,
                }
            else:
                # training: no batch decode; the W&B logger decodes only the samples it plots
                metrics["_artifacts"] = {
                    "target": poses.float().detach(),
                    "key": self._key(latent).detach(),
                }
        return metrics  # ty:ignore[invalid-return-type]

    @torch.no_grad()
    def generate(self, latent: Tensor) -> tuple[Tensor, list[list[int]]]:
        """Greedy decoding: (b, P, 3) poses and the predicted token ids (without EOS)."""
        return self.generate_from_key(self._key(latent))

    @torch.no_grad()
    def generate_from_key(self, key: Tensor) -> tuple[Tensor, list[list[int]]]:
        """Greedy decoding from the (b, n, d) cross-attention keys (the normalized world latent)."""
        b = key.shape[0]
        tokens = torch.full((b, 1), self.bos, dtype=torch.long, device=key.device)
        done = torch.zeros(b, dtype=torch.bool, device=key.device)
        for _ in range(self.max_tokens - 1):
            logits = self._logits(key, tokens)[:, -1]
            logits[:, [self.bos, self.pad]] = float("-inf")  # never emitted
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
            tokens = torch.cat([tokens, nxt[:, None]], 1)
            done |= nxt == self.eos
            if done.all():
                break
        ids = [
            row[: row.index(self.eos)] if self.eos in row else row
            for row in tokens[:, 1:].tolist()
        ]
        poses = torch.from_numpy(self.fast.decode(ids).astype(np.float32)).to(
            key.device
        )
        return poses, ids

    @torch.no_grad()
    def sample_from_key(
        self, key: Tensor, num_samples: int, temperature: float = 1.0
    ) -> tuple[list[list[list[int]]], Tensor]:
        """`num_samples` token sequences per (b, n, d) key: (b, N) id lists (without EOS) and their total
        log-probabilities (b, N) under the model; sequence 0 of each sample is the greedy decode.

        The BPE segmentation of a trajectory is deterministic, so a sequence's log-probability is the
        log-probability of its trajectory (no length normalization).
        """
        b, n = key.shape[0], num_samples
        device = key.device
        keys = key.repeat_interleave(n, 0)
        greedy = torch.zeros(b, n, dtype=torch.bool, device=device)
        greedy[:, 0] = True
        greedy = greedy.reshape(-1)
        tokens = torch.full((b * n, 1), self.bos, dtype=torch.long, device=device)
        logprob = torch.zeros(b * n, device=device)
        done = torch.zeros(b * n, dtype=torch.bool, device=device)
        for _ in range(self.max_tokens - 1):
            logits = self._logits(keys, tokens)[:, -1].float()
            logits[:, [self.bos, self.pad]] = float("-inf")  # never emitted
            logp = logits.log_softmax(-1)
            sampled = torch.multinomial((logits / temperature).softmax(-1), 1)[:, 0]
            nxt = torch.where(greedy, logits.argmax(-1), sampled)
            nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
            logprob += torch.where(done, 0.0, logp.gather(1, nxt[:, None])[:, 0])
            tokens = torch.cat([tokens, nxt[:, None]], 1)
            done |= nxt == self.eos
            if done.all():
                break
        rows = tokens[:, 1:].tolist()
        ids = [row[: row.index(self.eos)] if self.eos in row else row for row in rows]
        return [ids[i * n : (i + 1) * n] for i in range(b)], logprob.reshape(b, n)

    @torch.no_grad()
    def generate_most_likely(
        self, latent: Tensor, num_samples: int
    ) -> tuple[Tensor, dict]:
        """(b, P, 3) poses of the most likely of the greedy decode and `num_samples - 1` samples, plus stats
        (greedy / best log-probability, how many candidates were valid, lateral spread of their end points)."""
        ids, logprob = self.sample_from_key(self._key(latent), num_samples)
        b, n = logprob.shape
        n_values = self.fast.num_poses * len(self.fast.gammas)
        vocab = {i: t for t, i in self.fast._fitted()[0].get_vocab().items()}  # noqa: SLF001
        lengths = np.array([
            [sum(len(vocab.get(t, "")) for t in seq) for seq in row] for row in ids
        ])
        poses = self.fast.decode([seq for row in ids for seq in row]).reshape(
            b, n, -1, 3
        )
        score = logprob.cpu().numpy().copy()
        score[
            lengths != n_values
        ] = -np.inf  # malformed: not exactly P x 3 coefficient values
        best = score.argmax(1)
        out = torch.from_numpy(poses[np.arange(b), best].astype(np.float32)).to(
            latent.device
        )
        lp = logprob.cpu().numpy()
        valid = np.isfinite(score)
        stats = {
            "greedy_logprob": lp[:, 0].tolist(),
            "best_logprob": lp[np.arange(b), best].tolist(),
            "best_index": best.tolist(),
            "n_valid": valid.sum(1).tolist(),
            "greedy_valid": valid[:, 0].tolist(),
            "end_x_right": [poses[i, valid[i], -1, 0].tolist() for i in range(b)],
            "end_logprob": [lp[i, valid[i]].tolist() for i in range(b)],
        }
        return out, stats

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
        latent: Tensor | None = None,
    ) -> TensorDict:
        assert latent is not None  # noqa: S101
        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
            if self.num_samples > 0:
                poses, self.last_decode_stats = self.generate_most_likely(
                    latent, self.num_samples
                )
            else:
                poses, _ = self.generate(latent)
            predictions[key] = Prediction(
                value=TensorDict({"trajectory": poses}, batch_size=[poses.shape[0]]),
                batch_size=[poses.shape[0]],
            )
        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            # needs the clip's future (TrajectoryTarget on a full-length clip)
            poses = episode.get(self.target)[:, -1].float()
            predictions[key] = Prediction(
                value=TensorDict({"trajectory": poses}, batch_size=[poses.shape[0]]),
                batch_size=[poses.shape[0]],
            )
        return TensorDict(predictions).auto_batch_size_(1)  # ty:ignore[invalid-argument-type]
