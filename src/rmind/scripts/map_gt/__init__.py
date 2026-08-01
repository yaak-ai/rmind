"""Map-GT sidecar pipeline (traffic rules & environment awareness, phase 0/1).

Self-contained tooling: no rmind/torch imports. Runs with a plain venv:

    python3 -m venv .map_gt_venv
    .map_gt_venv/bin/pip install polars pyarrow numpy mcap protobuf requests shapely

Scripts (run as plain files, NOT as ``python -m rmind...`` — the rmind package
pulls in torch):

    python src/rmind/scripts/map_gt/build_sidecar.py --help
    python src/rmind/scripts/map_gt/audit.py --help

Outputs one parquet per drive at ``caches/map_gt/<Vehicle>/<drive-id>.parquet``
following the shared data contract (see build_sidecar.py docstring).
"""
