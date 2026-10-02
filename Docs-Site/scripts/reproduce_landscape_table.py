"""Rebuild the v5 "DSM accuracy by landscape" table from the run's metrics.json.

Run from Docs-Site/ (standard library only):
    python3 scripts/reproduce_landscape_table.py
    python3 scripts/reproduce_landscape_table.py path/to/metrics.json

The file is the one the v5 final H100 run wrote on Modal. A copy is served at
public/evidence/v5-final/metrics.json. Every number comes from the epoch the
run kept as best.pt (epoch 7). Nothing is re-run.

Pooling: each validation set reports n, RMSE, MAE and Pearson r per landscape.
  RMSE  = sqrt( sum(n_i * rmse_i^2) / sum(n_i) )   exact: the same as RMSE over all pixels
  MAE   = sum(n_i * mae_i) / sum(n_i)              exact
  r     = sum(n_i * r_i) / sum(n_i)                approximate: metrics.json has no
                                                   per-set means or variances, so the
                                                   exact pooled r cannot be rebuilt
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SETS = {"val": "MVS3DM", "val_neon": "NEON", "val_gamus": "GAMUS", "val_us3d": "US3D"}
LANDSCAPES = ["urban", "sparse", "forested"]


def pool(blocks):
    n = sum(b["n"] for b in blocks)
    return {
        "n": n,
        "rmse": math.sqrt(sum(b["n"] * b["rmse_m"] ** 2 for b in blocks) / n),
        "mae": sum(b["n"] * b["mae_m"] for b in blocks) / n,
        "r": sum(b["n"] * b["pearson_r"] for b in blocks) / n,
    }


def best_epoch(metrics):
    return min(metrics["history"], key=lambda h: h["select_score"])


def landscape_table(metrics):
    ep = best_epoch(metrics)
    per_set = []          # every input row, so the pooling can be checked by hand
    for key, name in SETS.items():
        v = ep[key]
        for land in LANDSCAPES + ["overall"]:
            b = v["global"] if land == "overall" else v["per_landscape"].get(land)
            if b:
                per_set.append({"set": name, "landscape": land, "n": b["n"],
                                "tiles": b.get("tiles"), "rmse": b["rmse_m"],
                                "mae": b["mae_m"], "r": b["pearson_r"]})
    rows = []
    for land in LANDSCAPES + ["overall"]:
        blocks = [ep[k]["global"] if land == "overall" else ep[k]["per_landscape"][land]
                  for k in SETS if land == "overall" or land in ep[k]["per_landscape"]]
        g = ep["val_gamus"]["global"] if land == "overall" else ep["val_gamus"]["per_landscape"][land]
        rows.append({"landscape": land,
                     "sets": [SETS[k] for k in SETS
                              if land == "overall" or land in ep[k]["per_landscape"]],
                     **pool(blocks), "gamus_rmse": g["rmse_m"]})
    return {"epoch": ep["epoch"], "select_score": ep["select_score"],
            "rows": rows, "per_set": per_set}


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "public/evidence/v5-final/metrics.json")
    t = landscape_table(json.loads(path.read_text()))
    print(f"best.pt = epoch {t['epoch']}  (select score {t['select_score']:.3f} m)\n")
    print(f"{'landscape':<10}{'pixels':>13}{'RMSE':>8}{'MAE':>8}{'r':>8}{'GAMUS':>8}  sets")
    for r in t["rows"]:
        print(f"{r['landscape']:<10}{r['n']:>13,}{r['rmse']:>8.2f}{r['mae']:>8.2f}"
              f"{r['r']:>8.2f}{r['gamus_rmse']:>8.2f}  {', '.join(r['sets'])}")
