"""Export the pallet perception models to an on-device bundle (ONNX + config + runtime).

    uv run python -m rmind.scripts.pallet_export_device --detector .../final.pt \
        --keypoints .../best.pt --out /nasa/team-space/nikita/pallet-detector/device/pallet_v1

Writes `detector.onnx` (DINOv3 encoder + box head, normalisation and sigmoids inside),
`keypoints.onnx` (HRNet-W18, normalisation inside, static batch `--kp-batch`),
`pipeline.json`, the runtime `device_runtime.py` and the numpy pose code it imports
(`rmind/pallet/{geometry,pose,pose_free}.py`). Checks ONNX against PyTorch with
onnxruntime when available. On the Jetson, build engines with
`trtexec --onnx=detector.onnx --saveEngine=detector.engine --fp16` (same for keypoints).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from structlog import get_logger
from torch import nn

from rmind.components.timm_backbone import TimmBackbone
from rmind.pallet.correspondence import NAMES
from rmind.pallet.entry import MEAN as KP_MEAN
from rmind.pallet.entry import STD as KP_STD
from rmind.pallet.geometry import EUR_KEYPOINTS
from rmind.pallet.keypoints import STRIDE, PalletKeypointNet
from rmind.scripts.pallet_proposer_head import ENCODER, LAYER, MEAN, STD
from rmind.scripts.pallet_proposer_multi import make_head

logger = get_logger(__name__)
PALLET = Path(__file__).resolve().parents[1] / "pallet"


class Detector(nn.Module):
    def __init__(self, enc: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.enc, self.head = enc, head
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        heat, size, offset, load = self.head(self.enc((image - self.mean) / self.std))
        return heat.sigmoid(), size, offset, load.sigmoid()


class Keypoints(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("mean", KP_MEAN.clone())
        self.register_buffer("std", KP_STD.clone())

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        return self.net((crops - self.mean) / self.std)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kp-batch", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dck = torch.load(args.detector, map_location="cpu", weights_only=False)
    res = tuple(dck["res"])
    enc = TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[res[1], res[0]])
    det = Detector(enc, make_head(dck["state_dict"])).eval()
    kck = torch.load(args.keypoints, map_location="cpu", weights_only=False)
    shape = tuple(kck.get("crop_shape") or (384, 384))
    net = PalletKeypointNet(kck["backbone"], num_keypoints=len(NAMES), pretrained=False)
    net.load_state_dict(kck["state_dict"])
    kp = Keypoints(net).eval()
    x_det = torch.rand(1, 3, res[1], res[0])
    x_kp = torch.rand(args.kp_batch, 3, shape[1], shape[0])
    torch.onnx.export(
        det,
        x_det,
        args.out / "detector.onnx",
        input_names=["image"],
        output_names=["heat", "size", "offset", "load"],
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        kp,
        x_kp,
        args.out / "keypoints.onnx",
        input_names=["crops"],
        output_names=["heatmaps"],
        opset_version=17,
        dynamo=False,
    )
    try:
        import onnxruntime as ort

        for name, model, x in (("detector", det, x_det), ("keypoints", kp, x_kp)):
            sess = ort.InferenceSession(
                str(args.out / f"{name}.onnx"), providers=["CPUExecutionProvider"]
            )
            got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
            with torch.no_grad():
                ref = model(x)
            ref = ref if isinstance(ref, tuple) else (ref,)
            err = max(
                float(np.abs(g - r.numpy()).max())
                for g, r in zip(got, ref, strict=True)
            )
            logger.info("onnx parity", model=name, max_abs_diff=err)
    except ImportError:
        logger.warning("onnxruntime not installed: parity not checked")
    cfg = {
        "detector": str(args.detector),
        "keypoint_model": str(args.keypoints),
        "detector_input": list(res),
        "box_threshold": 0.3,
        "max_boxes": 100,
        "min_kp_box_px": 80,
        "max_kp": 8,
        "kp_batch": args.kp_batch,
        "kp_confidence": 0.3,
        "max_pose_rms_px": 8.0,
        "crop_shape": list(shape),
        "crop_margin": kck.get("crop_margin", 0.3),
        "crop_max_aspect": 6.0,
        "stride": STRIDE,
        "keypoint_names": list(NAMES),
        "layout_m": [list(EUR_KEYPOINTS[n]) for n in NAMES],
    }
    (args.out / "pipeline.json").write_text(json.dumps(cfg, indent=1))
    shutil.copy(PALLET / "device_runtime.py", args.out / "device_runtime.py")
    dst = args.out / "rmind" / "pallet"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("geometry.py", "pose.py", "pose_free.py"):
        shutil.copy(PALLET / f, dst / f)
    for d in (args.out / "rmind", dst):
        (d / "__init__.py").write_text("")
    logger.info("bundle", out=str(args.out), crop_shape=shape, detector_input=res)


if __name__ == "__main__":
    main()
