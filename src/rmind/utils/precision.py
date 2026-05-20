import torch
from structlog import get_logger

logger = get_logger(__name__)


def auto_precision(requested: str) -> str:
    """Downgrade bf16-mixed to 16-mixed on CUDA GPUs without native bf16 support (e.g. Turing)."""
    if (
        requested == "bf16-mixed"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        logger.warning(
            "bf16 not supported on this GPU, downgrading precision",
            requested=requested,
            resolved="16-mixed",
        )
        return "16-mixed"
    return requested
