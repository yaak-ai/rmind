"""Decode yaak per-drive ``osm.mcap`` files (protobuf-encoded, schema embedded).

Topics observed (all drives in the train list have this file):
  - osm/Way: the platform's map-matched way the vehicle is on, re-emitted
    every ~10 s with fields: locations (polyline), highway (enum: motorway=0,
    trunk=1, primary=2, ..., track=15), surface (enum), maxspeed (uint32,
    0 = untagged OR unlimited — proto3 cannot distinguish), lanes, length,
    start/end (google.protobuf.Timestamp — the time interval the vehicle is
    matched to this way), name.
  - osm/Turn: manoeuvre annotations (location, type enum, timestamp).
  - osm/CurriculumLineString / osm/CurriculumPoint: scenario annotations
    (TRAFFIC_LIGHTS, STOP, ROUNDABOUT, ... with time/geo anchors).

NOTE: osm/Way carries NO OSM way id and, because ``highway`` is a proto3 enum
with motorway=0, an absent highway tag is indistinguishable from motorway.
Overpass (see overpass.py) is therefore still needed for way ids, unlimited-vs-
unknown maxspeed and traffic_signals/stop nodes.

Message classes are built dynamically from the schema embedded in the mcap —
no generated code needed.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.reader import make_reader


def _message_class(pool: descriptor_pool.DescriptorPool, name: str, data: bytes):
    fds = descriptor_pb2.FileDescriptorSet.FromString(data)
    pending = list(fds.file)
    while pending:
        rest = []
        progressed = False
        for fd in pending:
            try:
                pool.Add(fd)
                progressed = True
            except Exception as exc:  # noqa: BLE001 — dependency order / dupes
                if "duplicate" in str(exc).lower() or "already" in str(exc).lower():
                    progressed = True
                else:
                    rest.append(fd)
        if not progressed:
            for fd in rest:
                pool.Add(fd)  # re-raise the real error
            break
        pending = rest
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(name))


def _ts_us(ts) -> int:
    return ts.seconds * 1_000_000 + ts.nanos // 1_000


def read_way_intervals(path: str | Path) -> pl.DataFrame:
    """Read osm/Way messages as time intervals.

    Returns columns: start_us, end_us (Int64), highway (Utf8 enum name),
    maxspeed_kmh (Int64, 0 = untagged-or-unlimited), lanes, name.
    Way messages are re-emitted with a growing ``end``; intervals are deduped
    by (start, highway, maxspeed, name) keeping the max end. Sorted by start_us.
    """
    pool = descriptor_pool.DescriptorPool()
    rows = []
    with Path(path).open("rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        classes = {}
        for ch in summary.channels.values():
            schema = summary.schemas[ch.schema_id]
            classes[ch.topic] = _message_class(pool, schema.name, schema.data)
        if "osm/Way" not in classes:
            return pl.DataFrame(
                schema={
                    "start_us": pl.Int64,
                    "end_us": pl.Int64,
                    "highway": pl.Utf8,
                    "maxspeed_kmh": pl.Int64,
                    "lanes": pl.Int64,
                    "name": pl.Utf8,
                }
            )
        way_cls = classes["osm/Way"]
        highway_enum = way_cls.DESCRIPTOR.fields_by_name["highway"].enum_type
        for _schema, channel, message in reader.iter_messages():
            if channel.topic != "osm/Way":
                continue
            way = way_cls()
            way.ParseFromString(message.data)
            rows.append(
                (
                    _ts_us(way.start),
                    _ts_us(way.end),
                    highway_enum.values_by_number[way.highway].name,
                    way.maxspeed,
                    way.lanes,
                    way.name,
                )
            )
    df = pl.DataFrame(
        rows,
        schema={
            "start_us": pl.Int64,
            "end_us": pl.Int64,
            "highway": pl.Utf8,
            "maxspeed_kmh": pl.Int64,
            "lanes": pl.Int64,
            "name": pl.Utf8,
        },
        orient="row",
    )
    return (
        df.group_by(["start_us", "highway", "maxspeed_kmh", "name"])
        .agg(pl.col("end_us").max(), pl.col("lanes").first())
        .select(["start_us", "end_us", "highway", "maxspeed_kmh", "lanes", "name"])
        .sort("start_us")
    )
