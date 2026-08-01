# Demonstrator compliance audit (map-GT sidecars)

Drives: 20, frames: 671140 (~10 Hz; frame share ~ time share).
Frames with known finite limit: 526352 / 671140.

## Sidecar coverage per drive

| drive | frames | max_speed known | road_class known |
|---|---|---|---|
| Niro096-HQ/2023-01-11--13-47-36 | 46387 | 97.1% | 100.0% |
| Niro101-HQ/2022-12-25--09-58-33 | 43170 | 63.6% | 97.1% |
| Niro102-HQ/2022-12-20--09-30-06 | 21627 | 93.2% | 100.0% |
| Niro102-HQ/2023-06-05--06-14-24 | 31942 | 97.0% | 100.0% |
| Niro102-HQ/2023-06-07--13-47-01 | 30107 | 87.1% | 100.0% |
| Niro103-HQ/2023-03-23--15-24-19 | 20905 | 76.4% | 100.0% |
| Niro104-HQ/2023-03-15--10-02-09 | 21709 | 86.5% | 100.0% |
| Niro104-HQ/2023-05-08--05-53-39 | 46682 | 76.4% | 100.0% |
| Niro104-HQ/2023-05-31--18-18-28 | 11799 | 78.3% | 99.7% |
| Niro106-HQ/2023-05-22--10-49-24 | 16751 | 78.9% | 100.0% |
| Niro107-HQ/2023-05-24--12-31-05 | 40815 | 69.9% | 100.0% |
| Niro109-HQ/2023-05-26--16-28-20 | 63089 | 78.0% | 97.9% |
| Niro109-HQ/2023-05-30--13-57-54 | 29514 | 84.5% | 97.8% |
| Niro109-HQ/2023-06-03--05-47-33 | 42392 | 88.1% | 100.0% |
| Niro111-HQ/2023-03-21--10-56-42 | 14208 | 92.3% | 100.0% |
| Niro111-HQ/2023-05-14--06-02-12 | 139045 | 81.9% | 100.0% |
| Niro111-HQ/2023-05-25--15-00-19 | 8594 | 83.6% | 100.0% |
| Niro116-HQ/2023-04-26--06-50-41 | 13625 | 99.0% | 100.0% |
| Niro128-HQ/2023-03-29--11-42-27 | 19656 | 99.4% | 100.0% |
| Niro130-HQ/2023-05-19--09-25-01 | 9123 | 96.6% | 100.0% |

## % of time over the asserted legal limit (all frames with known limit)

| env_class | frames | +0 | +3 | +5 | +10 |
|---|---|---|---|---|---|
| city | 313968 | 7.3% | 4.3% | 2.9% | 1.6% |
| motorway | 38790 | 4.0% | 2.6% | 1.7% | 1.6% |
| private | 9157 | 4.1% | 3.7% | 3.3% | 3.3% |
| rural | 164419 | 3.0% | 1.8% | 1.2% | 0.9% |
| ALL | 526334 | 5.6% | 3.4% | 2.3% | 1.4% |

## Same, moving frames only (speed > 5 km/h)

| env_class | frames | +0 | +3 | +5 | +10 |
|---|---|---|---|---|---|
| city | 259413 | 8.8% | 5.2% | 3.5% | 2.0% |
| motorway | 38304 | 4.1% | 2.6% | 1.7% | 1.6% |
| private | 4101 | 9.1% | 8.3% | 7.4% | 7.4% |
| rural | 161259 | 3.0% | 1.8% | 1.2% | 0.9% |
| ALL | 463077 | 6.4% | 3.8% | 2.6% | 1.6% |

## Unlimited (autobahn) segments

32462 frames (4.8% of all) on explicitly unlimited roads; 3.5% of them above the 130 km/h advisory (95th-pct speed 128 km/h).

## Stop behaviour near mapped nodes (min speed per approach, within 50 m)

No light-state GT yet — red and green approaches are pooled, so the
distribution mixes 'had to stop' and 'rolled through a green'.

- traffic light: 136 approaches; min-speed percentiles p10/p25/p50/p75/p90 = 0.0/0.0/17.4/40.9/45.9 km/h; 39.0% came (near-)to a stop (<5 km/h), 52.2% slowed below 20 km/h.
- stop sign: 27 approaches; min-speed percentiles p10/p25/p50/p75/p90 = 7.2/25.6/41.0/48.3/66.4 km/h; 11.1% came (near-)to a stop (<5 km/h), 22.2% slowed below 20 km/h.

## Interpretation

The demonstrators are strongly compliant: over the 20 audited drives they exceed
the asserted legal limit on 5.6% of frames at +0 tolerance, dropping to 1.4%
beyond +10 km/h, and on explicitly unlimited autobahn the 95th-percentile speed
is 128 km/h — essentially never above the 130 advisory. City driving is where
almost all exceedance lives (8.8% of moving time above the limit, but only 2.0%
more than 10 km/h over), i.e. small creep over 30/50 zones rather than gross
speeding; rural and motorway exceedance is ~3-4% and mostly within +5. This is
the BC ceiling: a policy that imitates this data perfectly would violate limits
on roughly 6% of moving time at zero tolerance and ~1.6% at +10, so any model
exceedance materially above those numbers is a modelling failure, not a data
artifact. Traffic-light approaches split into a clear stop mode (39% reach
<5 km/h — presumably red) and a pass-through mode, consistent with pooling red
and green phases; the stop-sign numbers (only 11% full stops over 27 approaches)
should be read with caution because OSM highway=stop nodes within 30 m of the
route can belong to cross-street lanes, so light-state / sign-orientation GT is
needed before either becomes a per-event metric.

### Method caveats

- max_speed is what the map asserts (OSM maxspeed tag / zone, else the
  platform's osm.mcap maxspeed when > 0); untagged roads stay NaN and are
  excluded from the tables. Coverage is 64-99% per drive (median ~87%).
- env_class city/rural split for non-residential roads uses the asserted
  limit (<=50 -> city) as a proxy for place polygons.
- Distances to signals/stop signs are measured along the map-matched route;
  routes that revisit the same road segment can alias, and cross-street nodes
  within 30 m of the route are counted.
- Limits are static: variable/conditional limits (signals, weather, time)
  are not modelled, which can overstate exceedance on gantry-controlled
  autobahn sections.
