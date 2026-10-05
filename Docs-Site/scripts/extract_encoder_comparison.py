"""Export the completed DAV2/V1 experiment and its DINOv3 comparator.

Run from Docs-Site: python3 scripts/extract_encoder_comparison.py
The published extract preserves metric precision and records source hashes.
"""
import hashlib
import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
SOURCES = {
    "dav2": "Model_Traning/DAV2_V1/outputs/dav2-v1/metrics.json",
    "dinov3": "Model_Traning/V4_modal/Output/metrics.json",
    "report": "Model_Traning/v3/outputs/v3_vs_v4_vs_dav2.md",
}
CONFIG_KEYS = (
    "encoder_model_id", "tile_size", "decoder_dim", "datasets",
    "sampler_weights", "epochs", "encoder_unfreeze_blocks", "llrd",
    "stratum_balance_beta", "stratum_weight_clip", "tta_scales",
    "test_sources", "test_tiles", "test_sliding_tiles",
)
METRIC_KEYS = (
    "global", "balanced_rmse_m", "per_landscape", "tall_gt15m", "flat_lt1m",
)


def extract():
    data = {"sources": {}, "runs": {}}
    for name, relative in SOURCES.items():
        raw = (ROOT / relative).read_bytes()
        data["sources"][name] = {
            "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if name == "report":
            # India results were recorded in the shared evaluation report,
            # not in either training run's test_gamus_test_* blocks.
            labels = {
                "('india_labeled/val', 'plain', 'dav2')": "dav2",
                "('india_labeled/val', 'plain', 'v4')": "dinov3",
            }
            data["india_plain"] = {}
            for line in raw.decode().splitlines():
                cells = [cell.strip() for cell in line.split("|")]
                if len(cells) > 2 and cells[1] in labels:
                    data["india_plain"][labels[cells[1]]] = {
                        "rmse_m": float(cells[2]), "n": int(cells[-2]),
                    }
            assert set(data["india_plain"]) == set(labels.values())
            continue
        run = json.loads(raw)
        data["runs"][name] = {
            "config": {key: run["config"][key] for key in CONFIG_KEYS},
            "history_epochs": len(run["history"]),
            "validation_tta": run["final_tta"]["global"],
            "test": {
                mode: {key: run[f"test_gamus_test_{mode}"][key] for key in METRIC_KEYS}
                for mode in ("plain", "tta", "sliding_tta")
            },
        }
    return data


if __name__ == "__main__":
    out = DOCS / "public/evidence/encoder-comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(extract(), indent=2) + "\n")
    print(f"wrote {out}")
