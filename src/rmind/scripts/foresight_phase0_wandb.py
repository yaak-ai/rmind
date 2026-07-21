"""Log Phase-0 (D0-D4) + L2 content-probe results to wandb as ONE run.

Reads foresight_mm/results/{phase0_d0_d4.json,l2_content_probe.json} and
figs/*.png, logs headline scalars, tables, and images.

Run: uv run python src/rmind/scripts/foresight_phase0_wandb.py
Prints the run URL on success.
"""

import json
import sys
from pathlib import Path

ROOT = Path("/home/max/Code/rmind-fmm")
RESULTS = ROOT / "foresight_mm" / "results"


def main() -> None:
    import wandb

    d = json.loads((RESULTS / "phase0_d0_d4.json").read_text())
    l2 = json.loads((RESULTS / "l2_content_probe.json").read_text())

    run = wandb.init(
        entity="yaak",
        project="rmind",
        job_type="probe",
        tags=["foresight_mm", "phase0"],
        name="foresight-mm-phase0",
        config={"phase0": d["config"], "l2": l2["config"]},
    )

    d0 = d["D0"]["entropy_mean_pairwise"]
    d1, d2, d3, d4 = d["D1"], d["D2"], d["D3"], d["D4"]

    ent_overall = d0["overall"]["mean"]
    scalars = {
        "d0/entropy_H1": ent_overall[0],
        "d0/entropy_H5": ent_overall[4],
        "d0/entropy_growth_H5_over_H1": ent_overall[4] / ent_overall[0],
        "d0/floor_p5_H1": d0["low_entropy_floor_p5"][0],
        "d0/floor_p5_H5": d0["low_entropy_floor_p5"][4],
        "d0/spearman_c_cS_H1": d["D0"]["context_secondary_cS"]["spearman_vs_primary_per_H"][0],
        "d0/spearman_c_cS_H5": d["D0"]["context_secondary_cS"]["spearman_vs_primary_per_H"][4],
        "d1/median_r_top_quintile": d1["median_r_per_quintile"][4],
        "d1/spearman_entropy_r": d1["spearman_entropy_r"],
        "d2/norm_ratio_Q1": d2["norm_ratio_per_quintile"][0],
        "d2/norm_ratio_Q5": d2["norm_ratio_per_quintile"][4],
        "d2/residual_pc1_evr_Q5": d2["residual_pca_top_quintile"]["pc1_explained_variance_ratio"],
        "d2/residual_gmm2_better": int(d2["residual_pca_top_quintile"]["gmm2_better"]),
        "d3/fork_prevalence_H1": d3["fork_prevalence_H1"]["overall"],
        "d3/fork_prevalence_H5": d3["fork_prevalence_H5"]["overall"],
        "d3/n_fork_H5_not_H1": d3["n_fork_H5_not_H1"],
        "d3/alpha_H5_median": d3["alpha_H5"].get("median"),
        "d3/alpha_H5_frac_in_0.25_0.75": d3["alpha_H5"].get("frac_in_0.25_0.75"),
        "d4/odds_ratio": d4["odds_ratio"],
        "d4/or_ci95_lo": d4["ci95"][0],
        "d4/or_ci95_hi": d4["ci95"][1],
        "n_samples": d["config"]["n_samples"],
    }
    for t, v in l2["headline"]["delta_F5_vs_F2"].items():
        if v is not None:
            scalars[f"l2/delta_F5_vs_F2/{t}"] = v
    for t, v in l2["headline"]["delta_F6_vs_F4"].items():
        if v is not None:
            scalars[f"l2/delta_F6_vs_F4/{t}"] = v
    wandb.log(scalars)

    # D0 table: overall + strata means per H + floor
    cols = ["series", "H1", "H2", "H3", "H4", "H5"]
    rows = [["overall_mean", *ent_overall], ["floor_p5", *d0["low_entropy_floor_p5"]]]
    for s, cv in d0["per_stratum"].items():
        rows.append([f"stratum_{s}", *cv["mean"]])
    wandb.log({"tables/d0_entropy_vs_H": wandb.Table(columns=cols, data=rows)})

    wandb.log({"tables/d1_r_per_quintile": wandb.Table(
        columns=["quintile", "median_r", "mean_r"],
        data=[[q + 1, d1["median_r_per_quintile"][q], d1["mean_r_per_quintile"][q]] for q in range(5)],
    )})

    pp = d2.get("per_patch_ratio_per_quintile") or [None] * 5
    wandb.log({"tables/d2_shrinkage": wandb.Table(
        columns=["quintile", "pooled_norm_ratio", "per_patch_norm_ratio"],
        data=[[q + 1, d2["norm_ratio_per_quintile"][q], pp[q]] for q in range(5)],
    )})

    fs = list(l2["config"]["feature_sets"].keys())
    l2cols = ["target", "metric", *fs, "delta_F5_vs_F2", "delta_F6_vs_F4", "n_eval"]
    l2rows = []
    for t, row in l2["results"].items():
        l2rows.append([t, l2["metric_per_target"][t], *[row.get(f) for f in fs],
                       row.get("delta_F5_vs_F2"), row.get("delta_F6_vs_F4"), row.get("n_eval")])
    wandb.log({"tables/l2_incremental": wandb.Table(columns=l2cols, data=l2rows)})

    figs = sorted((RESULTS / "figs").glob("*.png"))
    if figs:
        wandb.log({f"figs/{p.stem}": wandb.Image(str(p)) for p in figs})

    print(f"WANDB_URL={run.url}")
    run.finish()


if __name__ == "__main__":
    import multiprocessing as mp
    import os

    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
    sys.stdout.flush()
    os._exit(0)
