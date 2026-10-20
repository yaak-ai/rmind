"""Offline Rerun recording with synchronized cameras, recorded poses and FM samples."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def render_rerun(
    directory: Path, job: Path, arrays: Mapping[str, np.ndarray], cameras: Sequence[str]
) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    from rmind.scripts.eval_d12_fm import ACTIONS

    recording = rr.RecordingStream("d12-fm-openloop")
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*[
                rrb.Spatial2DView(origin=f"/cameras/{camera}", name=camera)
                for camera in cameras
            ]),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/world", name="Recorded position (metres)"),
                rrb.Vertical(*[
                    rrb.Spatial2DView(
                        origin=f"/chunks/{name}",
                        name=f"{name}: x=future seconds, y=-command",
                    )
                    for name in ACTIONS
                ]),
            ),
            rrb.TextDocumentView(
                origin="/status", name="Recording position / sample semantics"
            ),
            row_shares=[3, 5, 1],
        ),
        rrb.TimePanel(timeline="recording_seconds", expanded=True),
    )
    recording.save(directory / "openloop.rrd", default_blueprint=blueprint)
    colors = np.asarray(
        [
            [31, 119, 180],
            [255, 127, 14],
            [44, 160, 44],
            [214, 39, 40],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [127, 127, 127],
        ],
        dtype=np.uint8,
    )
    horizon = arrays["target"].shape[1]
    future = np.arange(horizon) / 30
    if "ego_xy" in arrays:
        segments = np.split(arrays["ego_xy"], np.flatnonzero(arrays["reset"])[1:])
        recording.log(
            "world/recorded_path",
            rr.LineStrips2D(segments, colors=[130, 130, 130]),
            static=True,
        )
    for name in ACTIONS:
        recording.log(
            f"chunks/{name}/bounds",
            rr.LineStrips2D(
                [
                    [[0, -1], [future[-1], -1]],
                    [[0, 0], [future[-1], 0]],
                    [[0, 1], [future[-1], 1]],
                    [[0.2, -1], [0.2, 1]],
                ],
                colors=[180, 180, 180],
            ),
            static=True,
        )
    for position, stamp in enumerate(arrays["seconds"]):
        recording.set_time("recording_seconds", duration=float(stamp))
        recording.set_time("fork_frame", sequence=int(arrays["frames"][position, 0]))
        for camera_index, camera in enumerate(cameras):
            frame = int(arrays["frames"][position, camera_index])
            recording.log(
                f"cameras/{camera}",
                rr.EncodedImage(
                    path=job / camera / "frames/256x144" / f"{frame:06d}.jpg"
                ),
            )
        if "ego_xy" in arrays:
            for key, label, color in (
                ("ego_xy", "vehicle", [20, 150, 255]),
                ("pallet_xy", "pickup", [30, 200, 80]),
                ("dropoff_xy", "dropoff", [255, 140, 20]),
            ):
                recording.log(
                    f"world/{label}",
                    rr.Points2D(
                        [arrays[key][position]],
                        colors=color,
                        radii=0.07,
                        labels=[label],
                    ),
                )
            heading = np.deg2rad(arrays["heading_deg"][position])
            center = arrays["ego_xy"][position]
            recording.log(
                "world/heading",
                rr.LineStrips2D(
                    [
                        [
                            center,
                            center + 0.4 * np.array([np.cos(heading), np.sin(heading)]),
                        ]
                    ],
                    colors=[20, 150, 255],
                ),
            )
        for action_index, name in enumerate(ACTIONS):
            for sample_id, chunk in enumerate(arrays["stream"][position]):
                recording.log(
                    f"chunks/{name}/sample_{sample_id}",
                    rr.LineStrips2D(
                        [np.column_stack([future, -chunk[:, action_index]])],
                        colors=colors[sample_id % len(colors)],
                        labels=[f"sample {sample_id}"],
                        show_labels=False,
                    ),
                )
            recording.log(
                f"chunks/{name}/recorded",
                rr.LineStrips2D(
                    [
                        np.column_stack([
                            future,
                            -arrays["target"][position, :, action_index],
                        ])
                    ],
                    colors=[0, 0, 0],
                    radii=0.008,
                ),
            )
        height = arrays["height"][position]
        duration = (
            (np.diff(arrays["recording_bounds_ns"])[0] / 1e9)
            if "recording_bounds_ns" in arrays
            else arrays["seconds"][-1]
        )
        recording.log(
            "status",
            rr.TextDocument(
                f"{job.name} | recording {stamp:.2f}/{duration:.2f}s | frame {arrays['frames'][position, 0]} | cache reset: {bool(arrays['reset'][position])}\nHeight {height[0] * 2.3:.2f} m, valid={bool(height[2])}. Black: recorded action chunk. Colors: individual Gaussian samples, NOT persistent mode identities. Recorded poses only; action chunks are NOT spatial rollouts. Scrub recording_seconds to inspect another replan."
            ),
        )
    recording.flush()
    recording.disconnect()
