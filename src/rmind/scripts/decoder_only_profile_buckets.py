"""Bucket a `trtexec --dumpProfile` dump of the decoder step, two ways.

This exists to make one specific claim falsifiable. `docs/decoder_only_kv_cache.md`
§9 originally reported a bucket called "fused kernels containing the cache `Concat`"
at 68.2 ms / 34 % of the small arm at 64 frames, and concluded that a paged/in-place
attention plugin was the highest-value follow-up. That number was a **name-matching
artifact**, and the conclusion drawn from it does not survive measurement (§10).

TRT fuses and renames everything to `__myl_<Op><Op>...`, so a bucketing has to read
those concatenated op abbreviations -- and `Con` means *any* `Concat`. Three
unrelated ones occur per layer:

* the KV cache concat, `cat(past_k, k)`;
* `F.pad(cache_bias)`, a 1-row concat appending the own-frame keys' zero bias;
* RoPE's own `cat((-x2, x1))`.

Matching `Con` before the softmax pattern puts `__myl_RepRepConAddMaxSubExpSumDivMul`
-- the single most expensive kernel in the `_big`/64 profile, and a **softmax** --
into the `Concat` bucket.

    python -m rmind.scripts.decoder_only_profile_buckets decoder_big_n64.profile.txt

`naive` reproduces the old number; `corrected` splits it. Run it on the profile that
produced the original claim before trusting either.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from structlog import get_logger

logger = get_logger(__name__)

# trtexec profile row: Time(ms) Avg.(ms) Median(ms) Time(%) Layer
ROW = re.compile(r"\]\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\S.*?)\s*$")
SOFTMAX = re.compile(r"MaxSubExpSumDiv|SubExpDivMulSumMax|AddMaxSubExpSum")
GEMM = ("node_MatMul", "node_Gemm", "node_matmul", "node_linear")
# the K path: slice the layer out of the stacked cache, rope the new k, concat,
# transpose for the GEMM -- TRT fuses all of it into one copy
KV_K = re.compile(r"^__myl_SliResTra.*Con.*(ResTra|ResMovTra)")
KV_V = re.compile(r"^__myl_SliResTraResCon")


def rows(path: Path) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW.search(line)
        if m and m.group(5) not in {"Layer", "Total"}:
            out.append((float(m.group(3)), m.group(5)))
    return out


def _shared(name: str) -> str | None:
    if name.startswith(GEMM) or "+__mye" in name:
        return "matmul"
    if "_gemm_mha" in name:
        return "FUSED attention (_gemm_mha_v2)"
    if "scaled_dot_product_attention" in name:
        return "sdpa (the P.V GEMM)"
    if name.startswith("node_conv2d"):
        return "conv2d"
    if name.startswith("Reformatting"):
        return "input reformat (dtype/layout)"
    return None


def naive(name: str) -> str:
    """The original rule: any `Con` in the name is the cache `Concat`."""
    return _shared(name) or ("CONCAT-bucket" if "Con" in name else "softmax/other")


def corrected(name: str) -> str:
    """Softmax pattern wins over `Con`, and the KV copy is matched by shape."""
    if (bucket := _shared(name)) is not None:
        return bucket
    if SOFTMAX.search(name):
        return "softmax (incl. cache_bias pad)"
    if KV_K.match(name):
        return "KV copy: K path (slice+rope+cat+transpose)"
    if KV_V.match(name):
        return "KV copy: V path (slice+cat)"
    if name.startswith("__myl_Sli"):
        return "stacked-cache slice copy"
    return "other fused elementwise"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.profile:
        table = rows(path)
        total = sum(t for t, _ in table)
        logger.info(
            "profile", file=path.name, kernels=len(table), sum_median_ms=round(total, 1)
        )
        for label, rule in (("naive", naive), ("corrected", corrected)):
            agg: dict[str, float] = defaultdict(float)
            count: dict[str, int] = defaultdict(int)
            for t, name in table:
                agg[rule(name)] += t
                count[rule(name)] += 1
            for bucket, t in sorted(agg.items(), key=lambda kv: -kv[1]):
                logger.info(
                    label,
                    bucket=bucket,
                    ms=round(t, 1),
                    pct=round(100 * t / total, 1),
                    n=count[bucket],
                )


if __name__ == "__main__":
    main()
