"""Label Distribution Smoothing (LDS) loss weights for the flow policy.

Yang et al., "Delving into Deep Imbalanced Regression," ICML 2021. Here the
imbalanced regression target is the per-chunk maneuver intensity.

The flow MSE is flat-dominated: ~90% of frames are near-zero cruise, so a
mean-reverting field already nails them and the rare maneuvers never drive the
gradient. Worse, combined with the Gaussianize transform, uniform per-frame loss
allocates raw precision by data DENSITY (finest in cruise, coarsest in the
maneuver tail where the inverse Jacobian is ~30-100x) — the opposite of
importance for driving, where maneuver precision is the safety-critical part.

LDS reweights the loss from frequency toward importance:

- The chunk label is the peak |physical action| over the horizon, per model
  channel (longitudinal, steering). Peak-over-slots upweights the WHOLE chunk —
  including the near-zero lead-in slots — whenever it contains or approaches a
  maneuver (per-slot magnitude would miss the lead-in).
- The empirical label density is smoothed with a Gaussian kernel so adjacent
  maneuver intensities share statistical strength (the "LDS" step; raw inverse-
  frequency on a sparse tail gives spiky/undefined weights).
- Each chunk is weighted by (1 / smoothed_density)^alpha, capped (a near-empty
  extreme-maneuver bin must not get the runaway weight that amplifies sampling
  noise — cf. the mu-law and DCT sigma-floor failures), and mean-1 normalized
  over the EMPIRICAL distribution so the loss scale — hence LR/schedule — is
  unchanged.

alpha (aggressiveness; 0 = off/uniform, 0.5 = sqrt-inverse, 1 = full inverse-
frequency) and the cap are applied at LOAD, so they sweep without refitting. The
weight depends only on the clean target, so it is well-defined at every
flow-time; it multiplies the per-element flow loss, broadcast over slots.

flow_action_norm.py fits and writes {edges, emp, smooth, model_keys} under
the "lds" key of the stats JSON; this module only consumes them.
"""

import json
from pathlib import Path
from typing import Annotated, Self

import torch
from pydantic import ConfigDict, Field, validate_call
from structlog import get_logger
from torch import Tensor, nn

logger = get_logger(__name__)

_EPS = 1e-8


class LDSWeights(nn.Module):
    """Maps a per-chunk peak-intensity label to an LDS loss weight per channel.

    `edges` (C, nbins+1) are the per-channel histogram bin edges (physical model
    space, e.g. |longitudinal|, |steering|); `emp`/`smooth` (C, nbins) are the
    empirical and Gaussian-smoothed bin densities. The per-bin weight table is
    precomputed at init from (alpha, cap): (1/smooth)^alpha, capped, then divided
    by E_emp[w] per channel so weights average 1 over the training data.
    """

    edges: Tensor
    bin_weight: Tensor

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(  # noqa: PLR0913
        self,
        *,
        edges: Tensor,
        emp: Tensor,
        smooth: Tensor,
        model_keys: tuple[str, ...],
        alpha: Annotated[float, Field(ge=0)],
        cap: Annotated[float, Field(gt=1)],
    ) -> None:
        # alpha/cap ranges are enforced by @validate_call; the tensor-shape
        # consistency below is cross-field and stays imperative (cf. the
        # FlowActionDecoder constructor).
        super().__init__()
        edges, emp, smooth = edges.float(), emp.float(), smooth.float()
        if not (edges.ndim == emp.ndim == smooth.ndim == 2):  # noqa: PLR2004
            msg = "edges/emp/smooth must all be 2-D (channels, bins[+1])"
            raise ValueError(msg)
        if emp.shape != smooth.shape or edges.shape != (emp.shape[0], emp.shape[1] + 1):
            msg = (
                "shape mismatch: edges must be (C, nbins+1) and emp/smooth (C, nbins); "
                f"got edges={tuple(edges.shape)}, emp={tuple(emp.shape)}"
            )
            raise ValueError(msg)
        if len(model_keys) != emp.shape[0]:
            msg = f"model_keys has {len(model_keys)} entries, expected {emp.shape[0]}"
            raise ValueError(msg)

        self.model_keys = tuple(model_keys)
        self.alpha = float(alpha)
        self.cap = float(cap)

        # Per-bin weight: (1/smooth)^alpha, capped, then mean-1 normalized over
        # the empirical distribution (E_emp[w] == 1 per channel) so the loss
        # scale is preserved and LR/schedule transfer from the unweighted run.
        weight = (1.0 / smooth.clamp_min(_EPS)).pow(self.alpha).clamp_max(self.cap)
        mean = (emp * weight).sum(dim=1, keepdim=True).clamp_min(_EPS)
        weight /= mean

        # Non-persistent (fit from data, not learned); keeps checkpoints clean.
        self.register_buffer("edges", edges, persistent=False)
        self.register_buffer("bin_weight", weight, persistent=False)

    @classmethod
    def from_stats_file(cls, path: str | Path, *, alpha: float, cap: float) -> Self:
        stats = json.loads(Path(path).read_text(encoding="utf-8"))
        if "lds" not in stats:
            msg = f"no 'lds' section in stats file {path} (fit with +action_norm.lds=true)"
            raise KeyError(msg)
        lds = stats["lds"]
        return cls(
            edges=torch.tensor(lds["edges"]),
            emp=torch.tensor(lds["emp"]),
            smooth=torch.tensor(lds["smooth"]),
            model_keys=tuple(lds["model_keys"]),
            alpha=alpha,
            cap=cap,
        )

    @property
    def model_dim(self) -> int:
        return int(self.bin_weight.shape[0])

    def forward(self, label: Tensor) -> Tensor:
        """Peak-intensity label (..., C) -> per-channel loss weight (..., C).

        Labels are bucketized per channel into the fitted bins; values past the
        last edge land in the top (highest-weight, but capped) bin.

        Raises:
            ValueError: if the label's channel count != fitted `model_dim`.
        """
        if label.shape[-1] != self.model_dim:
            msg = f"label has {label.shape[-1]} channels, expected {self.model_dim}"
            raise ValueError(msg)
        out = torch.empty_like(label)
        nbins = self.bin_weight.shape[1]
        for c in range(self.model_dim):
            # Interior edges (edges[c, 1:-1]) define nbins buckets, indices 0..nbins-1.
            idx = torch.bucketize(label[..., c].contiguous(), self.edges[c, 1:-1])
            out[..., c] = self.bin_weight[c][idx.clamp_(min=0, max=nbins - 1)]
        return out


def build_lds_weights(
    stats_path: str | None, *, alpha: float, cap: float
) -> LDSWeights | None:
    """Hydra factory: LDS weights from a stats file's "lds" section, or None.

    None when no path is given OR the file has no "lds" section (a Gaussianize-
    only fit) — so a single shared action-norm file can back both this and the
    action transform, with LDS simply off when it wasn't fit. Keeps the (alpha,
    cap) knobs with the component, so the objective gets a ready module (or None).
    """
    if not stats_path:
        return None
    if "lds" not in json.loads(Path(stats_path).read_text(encoding="utf-8")):
        if alpha > 0:
            # Intent mismatch: LDS asked for (alpha>0) but the file wasn't fit
            # with it. Surface it rather than silently train without weighting.
            logger.warning(
                "lds_alpha > 0 but the action-norm stats have no 'lds' section; "
                "LDS weighting is OFF — refit with +action_norm.lds=true to enable",
                stats_path=stats_path,
                alpha=alpha,
            )
        return None
    return LDSWeights.from_stats_file(stats_path, alpha=alpha, cap=cap)
