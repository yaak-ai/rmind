import numpy as np
import numpy.typing as npt

LFG_LABEL_SHAPE = (4, 16, 16)
LFG_LABEL_NBYTES = 4 * 16 * 16


def decode_lfg_label(data: bytes) -> npt.NDArray[np.uint8]:
    """Decode a packed LFG per-patch label blob written by scripts/lfg_label_drives.py.

    Layout is `(4, 16, 16)` uint8, C-order: seg_label, seg_purity, motion, confidence.

    Raises:
        ValueError: if the blob is not exactly `LFG_LABEL_NBYTES` long.
    """
    if len(data) != LFG_LABEL_NBYTES:
        msg = f"expected {LFG_LABEL_NBYTES} bytes, got {len(data)}"
        raise ValueError(msg)
    return np.frombuffer(data, dtype=np.uint8).reshape(LFG_LABEL_SHAPE)
