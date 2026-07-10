"""Unit tests for `JointPolicyObjective` (VQ-BeT joint code + offset head).

Covers the offset teacher-forcing fix: `_gather_offset` correctness, behavioral
regression of the refactored `_predict` against the verbatim pre-refactor
implementation, gradient routing of the teacher-forced offset loss, and
equivalence of `teacher_force_offset=False` with the old (sampled-code) loss.
"""

from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from einops import rearrange
from torch import Tensor
from torch.nn import L1Loss, Linear, Sequential
from torchvision.ops import MLP

from rmind.components.base import Modality
from rmind.components.containers import ModuleDict
from rmind.components.loss import FocalLoss
from rmind.components.nn import Identity
from rmind.components.norm import Scaler
from rmind.components.objectives.joint_policy import JointPolicyObjective
from rmind.components.vq import ResidualVQ
from rmind.models.action_tokenizer import ActionTokenizer

if TYPE_CHECKING:
    from rmind.components.episode import Episode

BATCH_SIZE = 2
EPISODE_LENGTH = 2  # timesteps in the stub episode; compute_metrics uses [:, -1]
ACTION_HORIZON = 3  # action-chunk timesteps
ACTION_FIELDS = 4  # gas_pedal, brake_pedal, steering_angle, turn_signal
ACTION_DIM = ACTION_HORIZON * ACTION_FIELDS
LATENT_DIM = 8
NUM_QUANTIZERS = 2
CODEBOOK_SIZE = 4
FEATURE_DIM = 16
CHUNK: tuple[str, ...] = ("input", "joint_actions")


def _make_tokenizer(device: torch.device) -> ActionTokenizer:
    """Tiny ActionTokenizer mirroring config/model/yaak/action_tokenizer/raw.yaml."""
    return ActionTokenizer(
        input_transform=Sequential(
            Identity(),
            ModuleDict(
                modules={
                    Modality.CONTINUOUS: Identity(),
                    Modality.DISCRETE: {
                        # turn_signal {0, 1, 2} -> {0.0, 0.5, 1.0}
                        "turn_signal": Scaler(in_range=(0.0, 2.0), out_range=(0.0, 1.0))
                    },
                }
            ),
        ),
        encoder=Linear(ACTION_DIM, LATENT_DIM),
        quantizer=ResidualVQ(
            dim=LATENT_DIM,
            codebook_size=CODEBOOK_SIZE,
            num_quantizers=NUM_QUANTIZERS,
            kmeans_init=False,
        ),
        decoder=Linear(LATENT_DIM, ACTION_DIM),
        targets={
            Modality.CONTINUOUS: {
                "gas_pedal": ("continuous", "gas_pedal"),
                "brake_pedal": ("continuous", "brake_pedal"),
                "steering_angle": ("continuous", "steering_angle"),
            },
            Modality.DISCRETE: {"turn_signal": ("discrete", "turn_signal")},
        },
    ).to(device)


def _make_objective(
    device: torch.device, *, sample_codes: bool, teacher_force_offset: bool
) -> JointPolicyObjective:
    return (
        JointPolicyObjective(
            tokenizer=_make_tokenizer(device),
            code_head=MLP(FEATURE_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE]),
            offset_head=MLP(
                FEATURE_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE * ACTION_DIM]
            ),
            losses=ModuleDict(modules={"code": FocalLoss(), "offset": L1Loss()}),
            chunk=CHUNK,
            sample_codes=sample_codes,
            teacher_force_offset=teacher_force_offset,
        )
        .to(device)
        .eval()
    )


class _EpisodeStub:
    """Minimal Episode stand-in: compute_metrics only calls episode.get(chunk).

    `_features` is monkeypatched on the objective instance, so the stub never
    has to provide embeddings; the loss-path math below is the real
    `compute_metrics`.
    """

    def __init__(self, chunk: Tensor) -> None:
        self._chunk = chunk

    def get(self, path: tuple[str, ...]) -> Tensor:
        assert path == CHUNK
        return self._chunk


def _stubbed_inputs(
    objective: JointPolicyObjective,
    device: torch.device,
    *,
    requires_grad: bool = False,
) -> tuple["Episode", Tensor, Tensor]:
    """Random chunk episode + leaf features wired into the objective."""
    features = torch.randn(
        BATCH_SIZE, FEATURE_DIM, device=device, requires_grad=requires_grad
    )
    # monkeypatch _features on the instance: the stub episode has no embeddings
    objective._features = (  # noqa: SLF001 # ty:ignore[invalid-assignment]
        lambda episode, embedding: features  # noqa: ARG005
    )
    episode = _EpisodeStub(
        torch.rand(
            BATCH_SIZE, EPISODE_LENGTH, ACTION_HORIZON, ACTION_FIELDS, device=device
        )
    )
    return cast("Episode", episode), features, torch.empty(0, device=device)


def _predict_reference(
    objective: JointPolicyObjective, features: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Verbatim pre-refactor `_predict` (origin/feat/action-tokenizer)."""
    quantizer = cast("ActionTokenizer", objective.tokenizer).quantizer
    g, c = quantizer.num_quantizers, quantizer.codebook_size

    code_logits = rearrange(objective.code_head(features), "b (g c) -> b g c", g=g, c=c)
    if objective.sample_codes:
        codes = rearrange(
            torch.multinomial(code_logits.softmax(dim=-1).reshape(-1, c), 1),
            "(b g) 1 -> b g",
            g=g,
        )
    else:
        codes = code_logits.argmax(dim=-1)

    offsets = rearrange(
        objective.offset_head(features), "b (g c a) -> b g c a", g=g, c=c
    )
    index = codes[..., None, None].expand(-1, -1, 1, offsets.shape[-1])
    offset = offsets.gather(2, index).squeeze(2).sum(dim=1)  # (b, action_dim)

    return code_logits, codes, offset


def _offset_loss_reference(
    objective: JointPolicyObjective, episode: Any, embedding: Tensor
) -> Tensor:
    """Pre-refactor `compute_metrics` offset-loss value (sampled-code recon).

    Mirrors the op order of the current `compute_metrics` up to the multinomial
    call so RNG consumption is identical under a shared seed; the focal losses
    are RNG-free and therefore omitted.
    """
    features = objective._features(episode, embedding)  # noqa: SLF001
    tokenizer = cast("ActionTokenizer", objective.tokenizer)

    with torch.no_grad():
        chunk = episode.get(objective.chunk)[:, -1]
        _ = tokenizer(chunk)  # target_codes: computed for op-order parity
        target = tokenizer._normalize(chunk.flatten(-2, -1))  # noqa: SLF001

    _, codes, offset = _predict_reference(objective, features)
    predicted_chunk = tokenizer.invert(codes) + offset
    return objective.losses["offset"](predicted_chunk, target)


def test_gather_offset() -> None:
    b, g, c, a = 2, 2, 4, 3
    offsets = torch.arange(b * g * c * a, dtype=torch.float32).reshape(b, g, c, a)
    codes = torch.tensor([[1, 3], [0, 2]])

    # offsets[b, g, c, a] == ((b*2 + g)*4 + c)*3 + a, hence:
    # b=0: offsets[0,0,1] + offsets[0,1,3] = [3,4,5] + [21,22,23] = [24,26,28]
    # b=1: offsets[1,0,0] + offsets[1,1,2] = [24,25,26] + [42,43,44] = [66,68,70]
    expected = torch.tensor([[24.0, 26.0, 28.0], [66.0, 68.0, 70.0]])

    gathered = JointPolicyObjective._gather_offset(offsets, codes)  # noqa: SLF001

    assert gathered.shape == (b, a)
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)


@pytest.mark.parametrize("sample_codes", [True, False], ids=["sampled", "argmax"])
def test_predict_regression_old_impl(
    device: torch.device,
    sample_codes: bool,  # noqa: FBT001
) -> None:
    objective = _make_objective(
        device, sample_codes=sample_codes, teacher_force_offset=True
    )
    features = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)

    torch.manual_seed(1234)
    code_logits_new, codes_new, offset_new = objective._predict(features)  # noqa: SLF001

    torch.manual_seed(1234)
    code_logits_old, codes_old, offset_old = _predict_reference(objective, features)

    torch.testing.assert_close(code_logits_new, code_logits_old, rtol=0, atol=0)
    torch.testing.assert_close(codes_new, codes_old, rtol=0, atol=0)
    torch.testing.assert_close(offset_new, offset_old, rtol=0, atol=0)


def test_gradients_teacher_forced(device: torch.device) -> None:
    objective = _make_objective(device, sample_codes=False, teacher_force_offset=True)
    episode, features, embedding = _stubbed_inputs(
        objective, device, requires_grad=True
    )

    metrics = objective.compute_metrics(episode=episode, embedding=embedding)
    losses = cast("dict[str, Tensor]", metrics["loss"])

    # offset loss: gradients flow to the offset head only
    losses["offset"].backward(retain_graph=True)

    offset_grads = [p.grad for p in objective.offset_head.parameters()]
    assert not any(g is None for g in offset_grads)
    assert sum(g.abs().sum().item() for g in offset_grads if g is not None) > 0
    assert all(p.grad is None for p in objective.code_head.parameters())
    assert all(p.grad is None for p in objective.tokenizer.parameters())
    assert not any(p.requires_grad for p in objective.tokenizer.parameters())
    assert features.grad is not None

    # focal losses: gradients flow to the code head only
    objective.zero_grad()
    features.grad = None

    focal = torch.stack([losses[f"code_{q}"] for q in range(NUM_QUANTIZERS)]).sum()
    focal.backward()

    code_grads = [p.grad for p in objective.code_head.parameters()]
    assert not any(g is None for g in code_grads)
    assert sum(g.abs().sum().item() for g in code_grads if g is not None) > 0
    assert all(p.grad is None for p in objective.offset_head.parameters())
    assert all(p.grad is None for p in objective.tokenizer.parameters())


def test_teacher_force_false_matches_old_loss(device: torch.device) -> None:
    objective = _make_objective(device, sample_codes=True, teacher_force_offset=False)
    episode, _, embedding = _stubbed_inputs(objective, device)

    torch.manual_seed(1234)
    metrics = objective.compute_metrics(episode=episode, embedding=embedding)

    torch.manual_seed(1234)
    expected = _offset_loss_reference(objective, episode, embedding)

    offset_loss = cast("dict[str, Tensor]", metrics["loss"])["offset"]
    torch.testing.assert_close(offset_loss, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "teacher_force_offset", [True, False], ids=["teacher_forced", "sampled"]
)
def test_metric_offset_sampled_recon(
    device: torch.device,
    teacher_force_offset: bool,  # noqa: FBT001
) -> None:
    objective = _make_objective(
        device, sample_codes=True, teacher_force_offset=teacher_force_offset
    )
    episode, _, embedding = _stubbed_inputs(objective, device)

    metrics = objective.compute_metrics(episode=episode, embedding=embedding)

    recon = cast("dict[str, Tensor]", metrics["metric"])["offset_sampled_recon"]
    assert isinstance(recon, Tensor)
    assert recon.ndim == 0
    assert not recon.requires_grad
    assert cast("dict[str, Tensor]", metrics["loss"])["offset"].requires_grad
