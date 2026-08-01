"""Unit tests for the optional max-speed conditioning token (map-context Arm M).

Covers the shared 13-token semantic vocabulary mapping (UNKNOWN/UNLIMITED/
WALK/nearest-with-tie-down; the tokenizer is float-only, so the data-side
join must translate OSM maxspeed=walk to a float <= 7, e.g. 7.0 as
`overpass.parse_maxspeed_value` does),
that models WITHOUT the token are byte-identical to pre-map-context
`PatchPolicy` (state dict + forward), the missing-input -> all-UNKNOWN path,
the inference-time `max_speed_override` seam, and that the train-time UNKNOWN
dropout is inactive in eval mode.
"""

import math

import torch
from hydra import compose, initialize
from torch.nn import Identity, L1Loss, Linear
from torchvision.ops import MLP

from rmind.components.containers import ModuleDict
from rmind.components.loss import FocalLoss
from rmind.components.map_context import (
    MAX_SPEED_NUM_SPECIAL,
    MAX_SPEED_UNKNOWN_ID,
    MAX_SPEED_UNLIMITED_ID,
    MAX_SPEED_VOCAB_KMH,
    MAX_SPEED_VOCAB_SIZE,
    MAX_SPEED_WALK_ID,
    MaxSpeedTokenizer,
)
from rmind.components.nn import Embedding
from rmind.components.norm import UniformBinner
from rmind.models.patch_policy import BlockCausalTransformer, PatchPolicy

from .test_patch_policy import (
    ACTION_DIM,
    BATCH_SIZE,
    CODEBOOK_SIZE,
    EPISODE_LENGTH,
    GOAL_DIM,
    IMAGE_DIM,
    NUM_PATCHES,
    NUM_QUANTIZERS,
    POLICY_DIM,
    SPEED_BINS,
    _GoalEncoderStub,
    _make_batch,
    _make_tokenizer,
)

CONFIG_PATH = "../config"


def _make_model(
    *, with_max_speed: bool = True, max_speed_dropout: float = 0.3
) -> PatchPolicy:
    tokens_per_frame = NUM_PATCHES + (2 if with_max_speed else 1)
    return PatchPolicy(
        input_transform=Identity(),
        image_encoder=Identity(),
        goal_encoder=_GoalEncoderStub(),
        patch_projection=Linear(IMAGE_DIM + GOAL_DIM, POLICY_DIM),
        speed_tokenizer=UniformBinner(range=(0.0, 130.0), bins=SPEED_BINS),
        speed_embedding=Embedding(SPEED_BINS, POLICY_DIM),
        max_speed_tokenizer=MaxSpeedTokenizer() if with_max_speed else None,
        max_speed_embedding=(
            Embedding(MAX_SPEED_VOCAB_SIZE, POLICY_DIM) if with_max_speed else None
        ),
        max_speed_dropout=max_speed_dropout,
        encoder=BlockCausalTransformer(
            dim_model=POLICY_DIM,
            num_layers=2,
            num_heads=2,
            max_sequence_length=EPISODE_LENGTH * tokens_per_frame,
        ),
        tokenizer=_make_tokenizer(),
        code_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE]),
        offset_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE * ACTION_DIM]),
        losses=ModuleDict(modules={"code": FocalLoss(), "offset": L1Loss()}),
        norm=torch.nn.LayerNorm(POLICY_DIM),
        sample_codes=False,
    ).eval()


def _batch_with_max_speed(value: float | None) -> dict:
    batch = _make_batch()
    if value is not None:
        batch["context"]["max_speed"] = torch.full(
            (BATCH_SIZE, EPISODE_LENGTH, 1), value
        )
    return batch


def test_tokenizer_vocabulary_mapping() -> None:
    tokenizer = MaxSpeedTokenizer()

    assert MAX_SPEED_VOCAB_SIZE == 13
    assert MAX_SPEED_VOCAB_KMH == (10, 20, 30, 50, 60, 70, 80, 100, 120, 130)

    # every vocabulary speed maps to its own id (3..12)
    ids = tokenizer(torch.tensor(MAX_SPEED_VOCAB_KMH))
    torch.testing.assert_close(
        ids, torch.arange(MAX_SPEED_NUM_SPECIAL, MAX_SPEED_VOCAB_SIZE)
    )

    cases = {
        math.nan: MAX_SPEED_UNKNOWN_ID,  # NaN -> UNKNOWN
        -1.0: MAX_SPEED_UNLIMITED_ID,  # autobahn maxspeed=none sentinel
        -5.0: MAX_SPEED_UNLIMITED_ID,  # any negative -> UNLIMITED
        0.0: MAX_SPEED_WALK_ID,  # <= 7 -> WALK
        5.0: MAX_SPEED_WALK_ID,  # override convention for Schrittgeschwindigkeit
        7.0: MAX_SPEED_WALK_ID,  # data-side maxspeed=walk translation (7.0)
        8.0: 3,  # > 7 -> nearest numeric class: 10
        40.0: 5,  # exact tie 30/50 snaps DOWN -> 30 (conservative)
        45.0: 6,  # nearest -> 50
        90.0: 9,  # exact tie 80/100 snaps DOWN -> 80
        110.0: 10,  # exact tie 100/120 snaps DOWN -> 100
        135.0: 12,  # > 130 clamps to 130
        500.0: 12,  # clamps to 130
    }
    out = tokenizer(torch.tensor(list(cases.keys())))
    torch.testing.assert_close(out, torch.tensor(list(cases.values())))


def test_tokenizer_preserves_shape() -> None:
    tokenizer = MaxSpeedTokenizer()
    for shape in [(2, 3), (2, 3, 1)]:
        out = tokenizer(torch.full(shape, 50.0))
        assert out.shape == shape
        assert out.dtype == torch.long


def test_baseline_model_unaffected() -> None:
    """No max-speed args (raw.yaml path) -> identical params and working forward."""
    baseline = _make_model(with_max_speed=False)

    assert baseline.max_speed_tokenizer is None
    assert baseline.max_speed_embedding is None
    assert not any("max_speed" in k for k in baseline.state_dict())

    features, _ = baseline._features(_make_batch())  # noqa: SLF001
    assert features.shape == (BATCH_SIZE, EPISODE_LENGTH, POLICY_DIM)
    # a batch CARRYING max_speed is ignored entirely by a baseline model
    torch.testing.assert_close(
        features, baseline._features(_batch_with_max_speed(30.0))[0]  # noqa: SLF001
    )


def test_forward_shapes_with_token() -> None:
    model = _make_model()
    features, chunk = model._features(_batch_with_max_speed(50.0))  # noqa: SLF001
    assert features.shape == (BATCH_SIZE, EPISODE_LENGTH, POLICY_DIM)
    assert chunk is not None

    output = model.forward(_batch_with_max_speed(50.0))
    assert output["policy", "joint_actions"].ndim == 3

    loss = model._compute_metrics(_batch_with_max_speed(50.0))[  # noqa: SLF001
        "policy", "loss"
    ].sum(reduce=True)
    assert loss.isfinite()


def test_missing_input_equals_unknown() -> None:
    model = _make_model()
    absent, _ = model._features(_batch_with_max_speed(None))  # noqa: SLF001
    all_nan, _ = model._features(_batch_with_max_speed(math.nan))  # noqa: SLF001
    torch.testing.assert_close(absent, all_nan)


def test_max_speed_input_changes_output() -> None:
    model = _make_model()
    low, _ = model._features(_batch_with_max_speed(30.0))  # noqa: SLF001
    high, _ = model._features(_batch_with_max_speed(100.0))  # noqa: SLF001
    assert not torch.allclose(low, high)


def test_override_replaces_input() -> None:
    model = _make_model()

    model.max_speed_override = 30.0
    thirty_a, _ = model._features(_batch_with_max_speed(50.0))  # noqa: SLF001
    thirty_b, _ = model._features(_batch_with_max_speed(130.0))  # noqa: SLF001
    thirty_c, _ = model._features(_batch_with_max_speed(None))  # noqa: SLF001
    # the override wins over whatever the batch carries (or doesn't)
    torch.testing.assert_close(thirty_a, thirty_b)
    torch.testing.assert_close(thirty_a, thirty_c)

    model.max_speed_override = 100.0
    hundred, _ = model._features(_batch_with_max_speed(50.0))  # noqa: SLF001
    assert not torch.allclose(thirty_a, hundred)

    model.max_speed_override = None
    none_features, _ = model._features(_batch_with_max_speed(None))  # noqa: SLF001
    unknown, _ = model._features(_batch_with_max_speed(math.nan))  # noqa: SLF001
    torch.testing.assert_close(none_features, unknown)


def test_dropout_only_in_train_mode() -> None:
    model = _make_model(max_speed_dropout=1.0)
    batch = _batch_with_max_speed(100.0)
    reference = batch["continuous"]["speed"]

    # eval: dropout inert, the token embeds the tokenized input
    token = model._max_speed_token(batch, reference=reference)  # noqa: SLF001
    expected_ids = model.max_speed_tokenizer(batch["context"]["max_speed"])
    torch.testing.assert_close(token, model.max_speed_embedding(expected_ids))

    # train + p=1.0: every frame's token becomes UNKNOWN
    model.train()
    token = model._max_speed_token(batch, reference=reference)  # noqa: SLF001
    unknown_ids = torch.full(
        (BATCH_SIZE, EPISODE_LENGTH, 1), MAX_SPEED_UNKNOWN_ID, dtype=torch.long
    )
    torch.testing.assert_close(token, model.max_speed_embedding(unknown_ids))


def test_mismatched_max_speed_args_rejected() -> None:
    try:
        _make_model_partial = PatchPolicy(
            input_transform=Identity(),
            image_encoder=Identity(),
            goal_encoder=_GoalEncoderStub(),
            patch_projection=Linear(IMAGE_DIM + GOAL_DIM, POLICY_DIM),
            speed_tokenizer=UniformBinner(range=(0.0, 130.0), bins=SPEED_BINS),
            speed_embedding=Embedding(SPEED_BINS, POLICY_DIM),
            max_speed_tokenizer=MaxSpeedTokenizer(),  # embedding missing
            encoder=BlockCausalTransformer(
                dim_model=POLICY_DIM,
                num_layers=1,
                num_heads=1,
                max_sequence_length=EPISODE_LENGTH * (NUM_PATCHES + 2),
            ),
            tokenizer=_make_tokenizer(),
            code_head=MLP(POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE]),
            offset_head=MLP(
                POLICY_DIM, [16, NUM_QUANTIZERS * CODEBOOK_SIZE * ACTION_DIM]
            ),
            losses=ModuleDict(modules={"code": FocalLoss(), "offset": L1Loss()}),
        )
    except ValueError:
        pass
    else:
        msg = f"expected ValueError, got {_make_model_partial}"
        raise AssertionError(msg)


def test_experiment_config_composes() -> None:
    """dinov2_dinowm_maxspeed resolves: token modules present, encoder sized
    for episode_length * (num_patches + 2)."""
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "train", overrides=["experiment=yaak/patch_policy/dinov2_dinowm_maxspeed"]
        )

    assert (
        cfg.model.max_speed_tokenizer._target_
        == "rmind.components.map_context.MaxSpeedTokenizer"
    )
    assert cfg.model.max_speed_embedding.num_embeddings == MAX_SPEED_VOCAB_SIZE
    assert cfg.model.max_speed_override is None
    assert cfg.model.encoder.max_sequence_length == 6 * (256 + 2)
    assert list(cfg.model.input_transform._args_[0].paths.context.max_speed) == [
        "data",
        "meta/MapContext/max_speed",
    ]
    assert "map_context" in cfg.wandb.tags
