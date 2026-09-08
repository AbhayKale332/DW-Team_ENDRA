"""DepthWizard v2 — configuration.

One dataclass, every knob, CLI-overridable (same pattern as
`FineTunning/v1/kaggle_phase0.py::parse_config`).

v2 vs v1:
  * streamed data (bounded-LRU HF cache) instead of a pre-downloaded local subset
  * two training stages: SynRS3D pretrain -> GAMUS+GeoNRW fine-tune
  * 3 heads (metric / adaptive-bins / semantics) + a per-pixel gated fusion
  * GSD canonicalisation + jitter
  * late partial encoder unfreeze
  * 8x TTA at eval
  * results are zipped and shared over a cloudflared quick tunnel at the end
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

# 7-class GAMUS land-cover order (index -> name). v1's `class_names`.
CLASS_NAMES: tuple[str, ...] = (
    "ground", "vegetation", "building", "water", "road", "bridge", "other",
)

# Per-height-stratum edges (metres) for the balanced / stratified RMSE table.
HEIGHT_STRATA_M: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 1e9)


@dataclass
class Config:
    # ----- datasets / streaming --------------------------------------
    datasets: str = "gamus,geonrw"          # fine-tune mix; comma list of {gamus,geonrw,synrs3d}
    pretrain_dataset: str = "synrs3d"        # "" -> skip stage P
    data_source: str = "hf"                 # "hf" -> stream; "local" -> read <root>/<name>
    cache_dir: str = "/tmp/dw_cache"
    cache_max_gib: float = 50.0             # bounded-LRU cache ceiling
    local_root: str = "/kaggle/input"       # used only when data_source == "local"

    gamus_repo: str = "earthflow/GAMUS"
    geonrw_repo: str = "torchgeo/geonrw"
    synrs3d_repo: str = "JTRNEO/SynRS3D"

    # subset sizes (0 -> use the whole split). Smoke run shrinks these.
    train_tiles: int = 0
    val_tiles: int = 600
    pretrain_tiles: int = 20000
    sampler_weights: str = "gamus:3,geonrw:1"   # relative sampling frequency in stage F

    # ----- tiling / GSD --------------------------------------------
    tile_size: int = 512                    # model input (multiple of patch 16)
    canonical_gsd_m: float = 0.5            # eval + fusion working GSD
    gsd_jitter_min_m: float = 0.25
    gsd_jitter_max_m: float = 2.0
    gsd_jitter_p: float = 0.8              # prob. a train tile gets jittered
    max_valid_height_m: float = 400.0
    gamus_gsd_m: float = 0.33
    geonrw_gsd_m: float = 1.0

    # ----- model ---------------------------------------------------
    encoder_model_id: str = "facebook/dinov3-vitl16-pretrain-sat493m"
    encoder_feature_indices: tuple = (6, 12, 18, 24)
    encoder_half: bool = True
    freeze_encoder: bool = True
    unfreeze_last_n: int = 4               # blocks to unfreeze in the late phase (0 -> never)
    unfreeze_at_frac: float = 0.7          # of stage-F epochs
    unfreeze_lr_mult: float = 0.05
    decoder_dim: int = 256
    n_bins: int = 128                     # Head B adaptive bins
    bin_min_m: float = 0.0
    bin_max_m: float = 120.0

    # ----- training ----------------------------------------------
    pretrain_epochs: int = 6
    finetune_epochs: int = 30
    pretrain_minutes: float = 120.0
    finetune_minutes: float = 300.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    grad_accum: int = 1
    grad_clip: float = 1.0
    num_workers: int = 16
    prefetch_factor: int = 4
    amp: bool = True
    amp_dtype: str = "bf16"               # bf16 on Blackwell; "fp16" fallback
    channels_last: bool = True
    grad_checkpoint: bool = False         # 96 GB card -> off
    compile_model: bool = False
    seed: int = 42

    # ----- loss weights -----------------------------------------
    w_l1: float = 1.0
    w_grad: float = 0.5
    w_silog: float = 1.0
    silog_lambda: float = 0.85
    silog_shift: float = 1.0
    w_head_a: float = 1.0
    w_head_b: float = 1.0
    w_fused: float = 1.0
    w_bin_ce: float = 0.5
    w_seg: float = 0.3

    # ----- eval / io -------------------------------------------
    eval_every: int = 2
    tta: bool = True
    tta_scales: tuple = (1.0, 1.3)
    per_class_metrics: bool = True
    class_names: tuple = CLASS_NAMES
    output_dir: str = "/kaggle/working/outputs/v2"
    n_qualitative: int = 8

    # ----- delivery -------------------------------------------
    make_zip: bool = True
    share_cloudflared: bool = True
    keep_alive_minutes: float = 180.0

    # ----- misc ----------------------------------------------
    hf_token: str = ""
    smoke: bool = False                   # tiny end-to-end run (Kaggle CPU/T4 sanity)
    skip_pretrain: bool = False
    init_from: str = ""                   # checkpoint to warm-start stage F from

    def apply_smoke(self) -> Config:
        if not self.smoke:
            return self
        self.train_tiles = 12
        self.val_tiles = 6
        self.pretrain_tiles = 12
        self.pretrain_epochs = 1
        self.finetune_epochs = 1
        self.pretrain_minutes = 10.0
        self.finetune_minutes = 15.0
        self.batch_size = 2
        self.num_workers = 2
        self.eval_every = 1
        self.n_qualitative = 2
        self.share_cloudflared = False
        return self

    def validate(self) -> Config:
        assert self.tile_size % 16 == 0, "tile_size must be a multiple of 16"
        assert self.amp_dtype in ("bf16", "fp16")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        if self.skip_pretrain:
            self.pretrain_dataset = ""
        return self

    def dataset_list(self) -> list[str]:
        return [d.strip() for d in self.datasets.split(",") if d.strip()]

    def sampler_weight_map(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in self.sampler_weights.split(","):
            if ":" in part:
                k, v = part.split(":")
                out[k.strip()] = float(v)
        return out


def parse_config(argv: list[str] | None = None) -> Config:
    """Mirror of v1's parser: every non-tuple field becomes a --flag."""
    cfg = Config()
    p = argparse.ArgumentParser(description="DepthWizard v2")
    for f in fields(cfg):
        cur = getattr(cfg, f.name)
        if isinstance(cur, tuple):
            continue
        t = type(cur)
        if t is bool:
            p.add_argument(f"--{f.name}", type=lambda x: str(x).lower() in ("1", "true", "yes"))
        else:
            p.add_argument(f"--{f.name}", type=t)
    args, _ = p.parse_known_args(argv)
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg.apply_smoke().validate()


def safe_config_dict(cfg: Config) -> dict:
    from dataclasses import asdict

    d = asdict(cfg)
    d.pop("hf_token", None)
    return d
