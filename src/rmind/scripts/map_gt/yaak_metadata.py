"""Minimal standalone parser for yaak ``metadata.log`` files.

Framing (mirrors rbyte.io.yaak.metadata.message_iterator): a 12-byte file
header (uint32 header_len=12, uint32 version=1, uint32 msg_header_len=8)
followed by messages, each prefixed by (uint32 msg_type_idx, uint32 msg_len).

Message type indices: 0=Gnss, 4=ImageMetadata, 6=DriveSessionInfo,
7=VehicleMotion, 8=VehicleState (protobuf, see vendored ``proto/``).
"""

from __future__ import annotations

import struct
import sys
from mmap import ACCESS_READ, mmap
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent / "proto"))

import can_pb2  # noqa: E402  (vendored)
import sensor_pb2  # noqa: E402  (vendored)

MSG_GNSS = 0
MSG_IMAGE_METADATA = 4
MSG_VEHICLE_MOTION = 7

GEAR_NAMES = {0: "P", 1: "R", 2: "N", 3: "D", 4: "B"}


def _ts_us(ts) -> int:
    return ts.seconds * 1_000_000 + ts.nanos // 1_000


def parse_metadata_log(
    path: str | Path, camera: str = "cam_front_left"
) -> dict[str, pl.DataFrame]:
    """Parse metadata.log into polars frames.

    Returns dict with keys:
      - frames: frame_idx (Int64), time_stamp_us (Int64)  [for `camera` only]
      - gnss:   time_stamp_us, latitude, longitude, speed_mps
      - motion: time_stamp_us, speed_kmh, gear (Utf8)
    All sorted by time_stamp_us.
    """
    frames: list[tuple[int, int]] = []
    gnss: list[tuple[int, float, float, float]] = []
    motion: list[tuple[int, float, str]] = []

    img = sensor_pb2.ImageMetadata()
    gns = sensor_pb2.Gnss()
    mot = can_pb2.VehicleMotion()

    with Path(path).open("rb") as fh, mmap(fh.fileno(), 0, access=ACCESS_READ) as f:
        header = struct.unpack("III", f.read(12))
        if header != (12, 1, 8):
            msg = f"unexpected metadata.log header: {header}"
            raise ValueError(msg)
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            msg_type, msg_len = struct.unpack("II", h)
            data = f.read(msg_len)
            if msg_type == MSG_IMAGE_METADATA:
                img.ParseFromString(data)
                if img.camera_name == camera:
                    frames.append((img.frame_idx, _ts_us(img.time_stamp)))
            elif msg_type == MSG_GNSS:
                gns.ParseFromString(data)
                gnss.append(
                    (_ts_us(gns.time_stamp), gns.latitude, gns.longitude, gns.speed)
                )
            elif msg_type == MSG_VEHICLE_MOTION:
                mot.ParseFromString(data)
                motion.append(
                    (_ts_us(mot.time_stamp), mot.speed, GEAR_NAMES.get(mot.gear, "?"))
                )

    return {
        "frames": pl.DataFrame(
            frames, schema={"frame_idx": pl.Int64, "time_stamp_us": pl.Int64},
            orient="row",
        ).sort("time_stamp_us"),
        "gnss": pl.DataFrame(
            gnss,
            schema={
                "time_stamp_us": pl.Int64,
                "latitude": pl.Float64,
                "longitude": pl.Float64,
                "speed_mps": pl.Float64,
            },
            orient="row",
        ).sort("time_stamp_us"),
        "motion": pl.DataFrame(
            motion,
            schema={"time_stamp_us": pl.Int64, "speed_kmh": pl.Float64, "gear": pl.Utf8},
            orient="row",
        ).sort("time_stamp_us"),
    }
