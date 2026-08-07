from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from einops.layers.torch import Rearrange
from hydra import compose, initialize
from hydra.utils import instantiate
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import CenterCrop, Normalize, Resize, ToDtype

from rmind.components.drivor.backbone import RegisterViTBackbone
from rmind.components.drivor.ego_state import EgoStateEncoder
from rmind.components.drivor.loss import (
    WinnerTakesAllPoseLoss,
    winner_takes_all_pose_l1,
)
from rmind.components.drivor.trajectory_decoder import TrajectoryDecoderHead
from rmind.components.drivor.trajectory_target import (
    dead_reckon_future_trajectory,
    gnss_anchor_drift_m,
)
from rmind.components.nn import Identity
from rmind.components.transformer import CrossAttentionDecoder
from rmind.components.vq import ResidualVQ
from rmind.datamodules import GenericDataModule
from rmind.models.drivor import DrivoR
from rmind.models.waypoints_tokenizer import (
    WaypointsLatentTokenizer,
    WaypointsTokenizer,
)
from tests.conftest import make_batch

ROUTE_LATENT_DIM = 8
DIM_MODEL = 16
NUM_POSES = 10


# --- WaypointsTokenizer (ported from origin/feat/wpts-rvq), built locally with
# random-init weights -- no wandb artifact download needed for these tests. ---


@pytest.fixture(scope="module")
def waypoints_tokenizer() -> WaypointsTokenizer:
    return WaypointsTokenizer(
        input_transform=Identity(),
        encoder=nn.Sequential(
            nn.Linear(20, 16), nn.GELU(), nn.Linear(16, ROUTE_LATENT_DIM)
        ),
        quantizer=ResidualVQ(
            dim=ROUTE_LATENT_DIM, codebook_size=4, num_quantizers=2, kmeans_init=False
        ),
        decoder=nn.Sequential(
            nn.Linear(ROUTE_LATENT_DIM, 16), nn.GELU(), nn.Linear(16, 20)
        ),
        waypoints=("dummy",),
    )


@pytest.fixture(scope="module")
def route_tokenizer(
    waypoints_tokenizer: WaypointsTokenizer,
) -> WaypointsLatentTokenizer:
    return WaypointsLatentTokenizer(tokenizer=waypoints_tokenizer)


def test_waypoints_tokenizer_encode_shape(
    waypoints_tokenizer: WaypointsTokenizer,
) -> None:
    b = 2
    waypoints = torch.randn(b, 10, 2)
    z_q = waypoints_tokenizer.encode(waypoints)
    assert z_q.shape == (b, ROUTE_LATENT_DIM)


def test_waypoints_latent_tokenizer_shape(
    route_tokenizer: WaypointsLatentTokenizer,
) -> None:
    b = 2
    waypoints = torch.randn(b, 10, 2)
    out = route_tokenizer(waypoints)
    assert out.shape == (b, 1, ROUTE_LATENT_DIM)


# --- RegisterViTBackbone -- needs network access to download the pretrained
# DINOv3 checkpoint from HF Hub, same as `TimmBackbone` already does for the
# existing `episode_builder` fixture in `tests/conftest.py`. ---


@pytest.fixture(scope="module")
def register_backbone(device: torch.device) -> RegisterViTBackbone:
    return RegisterViTBackbone(
        img_size=[256, 256], num_registers=4, lora_rank=4, lora_alpha=4.0
    ).to(device)


def test_backbone_forward_shape(
    device: torch.device, register_backbone: RegisterViTBackbone
) -> None:
    x = torch.randn(2, 3, 256, 256, device=device)
    out = register_backbone(x)
    assert out.shape == (2, 4, register_backbone.model.embed_dim)


def test_backbone_freeze_contract(register_backbone: RegisterViTBackbone) -> None:
    for name, p in register_backbone.model.named_parameters():
        if "lora_" not in name:
            assert not p.requires_grad, name

    assert register_backbone.reg_token.requires_grad

    lora_param_names = [
        name
        for name, _ in register_backbone.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]
    assert lora_param_names
    for name, p in register_backbone.named_parameters():
        if name in lora_param_names:
            assert p.requires_grad, name


def test_backbone_timm_internals_contract(
    register_backbone: RegisterViTBackbone,
) -> None:
    model = register_backbone.model
    assert hasattr(model, "_pos_embed")
    assert hasattr(model, "norm_pre")
    assert hasattr(model, "norm")
    assert hasattr(model, "patch_embed")
    assert hasattr(model, "num_prefix_tokens")
    assert isinstance(model.blocks, ModuleList)
    for block in model.blocks:
        assert hasattr(block.attn, "num_prefix_tokens")
        assert (
            block.attn.num_prefix_tokens
            == model.num_prefix_tokens + register_backbone.num_registers
        )


# --- dead_reckon_future_trajectory / gnss_anchor_drift_m -----------------------


def test_dead_reckon_constant_velocity() -> None:
    t = 5
    speed_kmh = torch.full((1, t), 36.0)  # 10 m/s
    heading_deg = torch.zeros(1, t)
    time_stamp_s = torch.arange(t, dtype=torch.float32).unsqueeze(0)  # 1s steps

    position, heading = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_s=time_stamp_s,
        reference_index=0,
    )

    expected_x = torch.tensor([10.0, 20.0, 30.0, 40.0]) / 100.0
    assert torch.allclose(position[0, :, 0], expected_x, atol=1e-4)
    assert torch.allclose(position[0, :, 1], torch.zeros(4), atol=1e-4)
    assert torch.allclose(heading[0], torch.zeros(4), atol=1e-6)


def test_dead_reckon_heading_change() -> None:
    speed_kmh = torch.full((1, 3), 36.0)  # 10 m/s
    heading_deg = torch.tensor([[0.0, 90.0, 90.0]])
    time_stamp_s = torch.tensor([[0.0, 1.0, 2.0]])

    position, heading = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_s=time_stamp_s,
        reference_index=0,
    )

    # interval 0->1 uses heading@t0=0deg (rel. to ref=0deg) -> +x; interval 1->2
    # uses heading@t1=90deg (rel. to ref=0deg) -> +y.
    expected_position = torch.tensor([[10.0, 0.0], [10.0, 10.0]]) / 100.0
    assert torch.allclose(position[0], expected_position, atol=1e-4)

    expected_heading = torch.deg2rad(torch.tensor([90.0, 90.0]))
    assert torch.allclose(heading[0], expected_heading, atol=1e-4)


def test_gnss_anchor_drift_zero_when_consistent() -> None:
    speed_kmh = torch.full((1, 3), 36.0)  # 10 m/s
    heading_deg = torch.zeros(1, 3)
    time_stamp_s = torch.tensor([[0.0, 1.0, 2.0]])

    position, _ = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_s=time_stamp_s,
        reference_index=0,
    )

    # ego frame == world frame here (heading=0 throughout); dead-reckoned final
    # position is (20, 0) m, so a consistent GNSS trace has the same endpoint.
    gnss_xy = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]])
    drift = gnss_anchor_drift_m(
        dead_reckoned_position_normalized=position,
        gnss_xy=gnss_xy,
        heading_deg=heading_deg,
        reference_index=0,
    )
    assert torch.allclose(drift, torch.zeros(1), atol=1e-4)


def test_gnss_anchor_drift_nonzero_when_inconsistent() -> None:
    speed_kmh = torch.full((1, 3), 36.0)
    heading_deg = torch.zeros(1, 3)
    time_stamp_s = torch.tensor([[0.0, 1.0, 2.0]])

    position, _ = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_s=time_stamp_s,
        reference_index=0,
    )

    gnss_xy = torch.tensor([
        [[0.0, 0.0], [10.0, 0.0], [200.0, 0.0]]
    ])  # last fix way off
    drift = gnss_anchor_drift_m(
        dead_reckoned_position_normalized=position,
        gnss_xy=gnss_xy,
        heading_deg=heading_deg,
        reference_index=0,
    )
    assert drift.item() > 100.0  # noqa: PLR2004


# --- winner_takes_all_pose_l1 --------------------------------------------------


def test_winner_takes_all_pose_l1_selects_best_candidate() -> None:
    target_xy = torch.zeros(1, 2, 2)
    target_heading = torch.zeros(1, 2)
    input = torch.zeros(1, 3, 2, 3)
    input[:, 0] = 10.0
    input[:, 1] = 1.0
    input[:, 2] = 5.0

    loss, best_index, per_candidate = winner_takes_all_pose_l1(
        input, target_xy, target_heading
    )

    assert best_index.item() == 1
    assert per_candidate.shape == (1, 3)
    assert loss.item() > 0


def test_winner_takes_all_pose_l1_gradient_only_to_winner() -> None:
    target_xy = torch.zeros(1, 2, 2)
    target_heading = torch.zeros(1, 2)
    values = torch.tensor([10.0, 1.0, 5.0]).view(1, 3, 1, 1).expand(1, 3, 2, 3).clone()
    input = values.requires_grad_(True)  # noqa: FBT003

    loss, best_index, _ = winner_takes_all_pose_l1(input, target_xy, target_heading)
    assert best_index.item() == 1

    loss.backward()
    assert input.grad is not None
    assert torch.all(input.grad[:, 0] == 0)
    assert torch.all(input.grad[:, 2] == 0)
    assert torch.any(input.grad[:, 1] != 0)


def test_winner_takes_all_pose_l1_exact_match_is_zero() -> None:
    target_xy = torch.zeros(1, 2, 2)
    target_heading = torch.zeros(1, 2)
    input = torch.full((1, 4, 2, 3), 10.0)
    input[:, 2] = 0.0  # exact match

    loss, best_index, _ = winner_takes_all_pose_l1(input, target_xy, target_heading)
    assert best_index.item() == 2  # noqa: PLR2004
    assert loss.item() < 1e-6  # noqa: PLR2004


# --- full DrivoR model, built directly (not via Hydra/wandb) ------------------


@pytest.fixture(scope="module")
def drivor_model(
    device: torch.device,
    register_backbone: RegisterViTBackbone,
    route_tokenizer: WaypointsLatentTokenizer,
) -> DrivoR:
    image_preprocess = nn.Sequential(
        Rearrange("... h w c -> ... c h w"),
        CenterCrop([320, 576]),
        Resize([256, 256]),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
    register_projection = nn.Sequential(
        nn.LayerNorm(register_backbone.model.embed_dim),
        nn.Linear(register_backbone.model.embed_dim, DIM_MODEL),
    )
    ego_state_encoder = EgoStateEncoder(
        continuous_dim=4,
        num_turn_signal_classes=3,
        route_embedding_dim=ROUTE_LATENT_DIM,
        embedding_dim=DIM_MODEL,
        hidden_dim=DIM_MODEL,
    )
    trajectory_head = TrajectoryDecoderHead(
        decoder=CrossAttentionDecoder(
            dim_model=DIM_MODEL, num_layers=1, num_heads=2, hidden_layer_multiplier=1
        ),
        ego_state_encoder=ego_state_encoder,
        num_queries=8,
        dim_model=DIM_MODEL,
        num_poses=NUM_POSES,
        pose_dims=3,
    )

    return DrivoR(
        image_preprocess=image_preprocess,
        backbone=register_backbone,
        register_projection=register_projection,
        route_tokenizer=route_tokenizer,
        trajectory_head=trajectory_head,
        loss=WinnerTakesAllPoseLoss(),
        reference_timestep=0,
    ).to(device)


@pytest.fixture
def drivor_batch(device: torch.device) -> dict[str, Any]:
    return make_batch(device, b=2, t=11).to_dict(retain_none=False)


def test_drivor_forward_shape(
    drivor_model: DrivoR, drivor_batch: dict[str, Any]
) -> None:
    pred = drivor_model(drivor_batch)
    assert pred.shape == (2, 8, NUM_POSES, 3)


def test_drivor_route_tokenizer_frozen(drivor_model: DrivoR) -> None:
    for p in drivor_model.route_tokenizer.parameters():
        assert not p.requires_grad

    drivor_model.train()
    assert not drivor_model.route_tokenizer.training

    drivor_model.eval()
    assert not drivor_model.route_tokenizer.training


@pytest.fixture
def drivor_datamodule(device: torch.device) -> pl.LightningDataModule:
    dataset: TensorDict = make_batch(device, b=2, t=11).to_tensordict()  # ty:ignore[invalid-assignment]
    dataloader: DataLoader[Any] = DataLoader(
        dataset,  # ty:ignore[invalid-argument-type]
        batch_size=1,
        collate_fn=TensorDict.to_dict,  # ty:ignore[invalid-argument-type]
    )
    return GenericDataModule(train=dataloader, val=dataloader, predict=dataloader)


@pytest.fixture
def drivor_trainer(device: torch.device) -> pl.Trainer:
    return pl.Trainer(
        accelerator=device.type, devices=1, fast_dev_run=1, enable_progress_bar=False
    )


def test_drivor_fit(
    drivor_trainer: pl.Trainer,
    drivor_model: DrivoR,
    drivor_datamodule: pl.LightningDataModule,
) -> None:
    drivor_trainer.fit(drivor_model, datamodule=drivor_datamodule)


def test_drivor_predict(
    drivor_trainer: pl.Trainer,
    drivor_model: DrivoR,
    drivor_datamodule: pl.LightningDataModule,
) -> None:
    match drivor_trainer.predict(
        drivor_model, datamodule=drivor_datamodule, return_predictions=True
    ):
        case [TensorDict() as prediction]:
            pass

        case _:
            msg = "expected exactly one prediction"
            raise AssertionError(msg)

    match prediction.to_dict():
        case {"trajectory": {"prediction": Tensor() as pred}}:
            assert pred.shape[-2:] == (NUM_POSES, 3)

        case _:
            msg = "missing `trajectory.prediction` in prediction output"
            raise AssertionError(msg)


# --- checkpoint round-trip, via the real Hydra config -------------------------
#
# Unlike `drivor_model` above, this builds `DrivoR` from
# `config/model/yaak/drivor/raw.yaml` itself (small-dim overrides), so its
# hparams contain proper `HydraConfig`s and `load_from_checkpoint` can
# reconstruct the model from them alone -- mirrors `test_models.py`'s
# `model_yaak_control_transformer_raw`/`test_resume_from_checkpoint`. NOTE:
# this needs network access for BOTH the pretrained DINOv3 checkpoint (HF
# Hub, same dependency `tests/conftest.py`'s `episode_builder` fixture already
# has via `TimmBackbone`) AND the frozen `WaypointsTokenizer` artifact (W&B,
# `yaak/waypoints-tokenizer/model-gzxgumtf:v9`).


@pytest.fixture
def model_yaak_drivor_raw() -> DrivoR:
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            "model/yaak/drivor/raw",
            overrides=[
                "+num_registers=4",
                "+lora_rank=4",
                "+lora_alpha=4",
                "+vit_embedding_dim=384",
                "+route_embedding_dim=384",
                "+decoder_embedding_dim=16",
                "+decoder_num_layers=1",
                "+decoder_num_heads=2",
                "+num_queries=8",
                "+num_poses=10",
                "+heading_weight=0.1",
            ],
        )

    return instantiate(cfg.model.yaak.drivor)


def test_drivor_resume_from_checkpoint(
    drivor_trainer: pl.Trainer,
    model_yaak_drivor_raw: DrivoR,
    drivor_datamodule: pl.LightningDataModule,
    tmp_path: Path,
) -> None:
    drivor_trainer.fit(model_yaak_drivor_raw, datamodule=drivor_datamodule)
    ckpt_path = tmp_path / "model.ckpt"
    drivor_trainer.save_checkpoint(ckpt_path)
    model = model_yaak_drivor_raw.__class__.load_from_checkpoint(ckpt_path, strict=True)
    drivor_trainer.fit(model, datamodule=drivor_datamodule)
