"""Overpass helper: fetch OSM ways + traffic_signals/stop nodes along a route.

One query per drive (polyline ``around:`` filter on the simplified map-matched
route — much smaller than a bbox for long drives). Raw responses cached under
``caches/map_gt/_overpass/<Vehicle>__<drive-id>.json``; a sleep between real
network calls keeps us polite.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import requests
from shapely.geometry import LineString

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
# overpass-api.de 406s the default python-requests User-Agent
USER_AGENT = "yaak-map-gt/0.1 (contact: max@yaak.ai)"
RETRY_STATUS = {429, 502, 503, 504}
MAX_ATTEMPTS = 6
_last_call: list[float] = [0.0]

MPH_TO_KMH = 1.609344

# German implicit-limit zone/type values -> km/h (-1.0 = unlimited)
_ZONE_VALUES = {
    "de:urban": 50.0,
    "de:rural": 100.0,
    "de:motorway": -1.0,
    "urban": 50.0,
    "rural": 100.0,
    "motorway": -1.0,
    "de:living_street": 7.0,
    "de:walk": 7.0,
}


def parse_maxspeed_value(value: str) -> float | None:
    """Parse an OSM maxspeed-ish value. -1.0 = explicitly unlimited, None = no info."""
    v = value.strip().lower()
    if not v:
        return None
    if v == "none":
        return -1.0
    if v in ("signals", "variable"):
        return None  # variable limit, no static assertion
    if v == "walk":
        return 7.0
    if v in _ZONE_VALUES:
        return _ZONE_VALUES[v]
    # "DE:zone30", "DE:zone:30", "zone:30", "zone30"
    for prefix in ("de:zone:", "de:zone", "zone:", "zone"):
        if v.startswith(prefix) and v[len(prefix) :].strip().isdigit():
            return float(v[len(prefix) :])
    if v.endswith("mph"):
        num = v[:-3].strip()
        try:
            return float(num) * MPH_TO_KMH
        except ValueError:
            return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_way_maxspeed(tags: dict[str, str]) -> tuple[float | None, str]:
    """Resolve the asserted legal limit from way tags.

    Returns (kmh or -1.0 or None, source) where source is one of
    'tag' (maxspeed=*), 'directional' (maxspeed:forward==backward),
    'zone' (zone:maxspeed / maxspeed:type / source:maxspeed), 'none'.
    Only what the map asserts — no env defaults here.
    """
    if "maxspeed" in tags:
        parsed = parse_maxspeed_value(tags["maxspeed"])
        if parsed is not None:
            return parsed, "tag"
    fwd, bwd = tags.get("maxspeed:forward"), tags.get("maxspeed:backward")
    if fwd is not None and fwd == bwd:
        parsed = parse_maxspeed_value(fwd)
        if parsed is not None:
            return parsed, "directional"
    for key in ("zone:maxspeed", "maxspeed:type", "source:maxspeed", "zone:traffic"):
        if key in tags:
            parsed = parse_maxspeed_value(tags[key])
            if parsed is not None:
                return parsed, "zone"
    return None, "none"


def _simplify_route(coords: list[list[float]], max_pts: int = 220) -> list[list[float]]:
    """Reduce a (lon, lat) polyline to <= max_pts points, keeping shape."""
    if len(coords) <= max_pts:
        return coords
    line = LineString(coords)
    for tol in (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3):
        simplified = list(line.simplify(tol).coords)
        if len(simplified) <= max_pts:
            return simplified
    step = math.ceil(len(simplified) / max_pts)
    return simplified[::step] + [simplified[-1]]


def fetch_route_osm(
    route_coords_lonlat: list[list[float]],
    cache_path: str | Path,
    radius_m: float = 40.0,
    sleep_s: float = 2.0,
    timeout_s: int = 300,
) -> dict:
    """Fetch highway=* ways and traffic_signals/stop nodes around the route.

    route_coords_lonlat: GeoJSON-style [[lon, lat], ...]. Returns the parsed
    Overpass JSON (list under 'elements'; ways carry 'geometry', nodes lat/lon).
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    coords = _simplify_route(route_coords_lonlat)
    poly = ",".join(f"{lat:.6f},{lon:.6f}" for lon, lat in coords)
    query = f"""[out:json][timeout:{timeout_s}];
(
  way["highway"](around:{radius_m:.0f},{poly});
  node["highway"="traffic_signals"](around:{radius_m:.0f},{poly});
  node["highway"="stop"](around:{radius_m:.0f},{poly});
);
out geom;"""

    payload = None
    for attempt in range(MAX_ATTEMPTS):
        wait = sleep_s - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            resp = requests.post(
                url,
                data={"data": query},
                timeout=timeout_s + 30,
                headers={"User-Agent": USER_AGENT},
            )
        except requests.RequestException:
            _last_call[0] = time.monotonic()
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(10.0 * (attempt + 1))
            continue
        _last_call[0] = time.monotonic()
        if resp.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS - 1:
            time.sleep(10.0 * (attempt + 1))
            continue
        resp.raise_for_status()
        payload = resp.json()
        break
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    return payload
