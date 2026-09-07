"""DepthWizard — Phase 0 spike, single-file Kaggle runner (2x T4).

Goal (from `.agents/Depth_Wizard_Plan.md`, Phase 0):
    "GAMUS loader; DINOv3-SAT frozen encoder + DPT head + Head A only;
     ~2-3 h on Kaggle -> first GAMUS val RMSE."

What this script does, end to end, with no other project files:
    1. pip-installs the few deps Kaggle is missing (transformers, huggingface_hub, h5py)
    2. downloads a deterministic subset of earthflow/GAMUS from the Hub
    3. builds  DINOv3-SAT (frozen) -> DPT decoder -> single metric-nDSM head
    4. trains with AMP across both T4s (nn.DataParallel), wall-clock capped
    5. evaluates on the GAMUS val subset -> RMSE / MAE / Pearson r / delta1,
       plus a per-land-cover-class breakdown (the rubric's "stability across
       urban / sparse / hilly / forested")
    6. writes  outputs/v1/metrics.json, a checkpoint, and one sample export for
       the bare Three.js viewer (viewer/phase0_viewer.html)

------------------------------------------------------------------------------
KAGGLE SETUP
    - Notebook settings: Accelerator = "GPU T4 x2", Internet = ON
    - The DINOv3 SAT checkpoint is GATED. Accept the license at
        https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m
      then add your token as a Kaggle Secret named  HF_TOKEN
      (Add-ons -> Secrets).  A lighter gated fallback is
        facebook/dinov3-vitb16-pretrain-lvd1689m
    - Then, in a cell:
        !git clone https://github.com/<you>/SIH.git
        %cd SIH/FineTunning/v1
        !python kaggle_phase0.py
      or just run this file's cells after copy-pasting it into the notebook.

Override any CONFIG field from the CLI, e.g.:
    !python kaggle_phase0.py --train_subset 400 --val_subset 150 --epochs 6 \
        --encoder_model_id facebook/dinov3-vitb16-pretrain-lvd1689m
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path


# ===========================================================================
# 0. dependencies  (Kaggle already ships torch + CUDA + numpy + pillow + mpl)
# ===========================================================================
def _pip_install() -> None:
    pkgs = ["transformers>=4.56.0", "huggingface_hub>=0.34.0", "h5py>=3.10.0", "safetensors>=0.4.3"]
    print("[setup] pip install:", " ".join(pkgs))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *pkgs],
        check=True,
    )


if os.environ.get("DEPTHWIZ_SKIP_INSTALL") != "1":
    try:
        import transformers  # noqa: F401
        import huggingface_hub  # noqa: F401
        import h5py  # noqa: F401
    except Exception:
        _pip_install()

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


# ===========================================================================
# 1. config
# ===========================================================================
@dataclass
class Config:
    # -- data ------------------------------------------------------------
    hf_dataset_repo: str = "earthflow/GAMUS"
    data_source: str = "hf"            # "hf" -> download subset; "local" -> read data_root as-is
    data_root: str = "/kaggle/working/data/gamus"
    train_subset: int = 1200          # 0 -> use every tile in the split
    val_subset: int = 300
    tile_native_px: int = 1024
    train_size: int = 512             # model input; multiple of patch size (16)
    gamus_gsd_m: float = 0.33
    max_valid_height_m: float = 400.0
    num_workers: int = 2
    dl_workers: int = 16             # parallel HF Hub download threads

    # -- model ---------------------------------------------------------
    encoder_model_id: str = "facebook/dinov3-vitl16-pretrain-sat493m"
    encoder_feature_indices: tuple = (6, 12, 18, 24)   # hidden_states idx (0=embeddings)
    freeze_encoder: bool = True
    decoder_dim: int = 256

    # -- training ----------------------------------------------------
    epochs: int = 12
    max_train_minutes: float = 150.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 8              # total across all GPUs (DataParallel splits it)
    grad_accum: int = 1
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 42

    # -- loss  (SiLog + L1 + multi-scale gradient) -----------------
    w_l1: float = 1.0
    w_grad: float = 0.5
    w_silog: float = 1.0
    silog_lambda: float = 0.85
    silog_shift: float = 1.0

    # -- eval / io -------------------------------------------------
    eval_every: int = 1
    per_class_metrics: bool = True
    class_names: tuple = ("ground", "vegetation", "building", "water", "road", "bridge", "other")
    output_dir: str = "/kaggle/working/outputs/v1"
    save_checkpoint: bool = True
    export_viewer_sample: bool = True
    hf_token: str = ""

    def validate(self) -> "Config":
        assert self.train_size % 16 == 0, "train_size must be a multiple of 16"
        assert self.train_size <= self.tile_native_px
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_root).mkdir(parents=True, exist_ok=True)
        return self


def parse_config() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="DepthWizard Phase 0 (Kaggle 2x T4)")
    for f in fields(cfg):
        if f.type in ("tuple",) or isinstance(getattr(cfg, f.name), tuple):
            continue  # keep tuple knobs code-only
        t = type(getattr(cfg, f.name))
        if t is bool:
            p.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("1", "true", "yes"))
        else:
            p.add_argument(f"--{f.name}", type=t)
    args, _ = p.parse_known_args()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg.validate()


# ===========================================================================
# 2. HF token (Kaggle Secret -> env fallback)
# ===========================================================================
def resolve_hf_token(cfg: Config) -> str:
    if cfg.hf_token:
        return cfg.hf_token
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return ""


# ===========================================================================
# 3. GAMUS data
#    earthflow/GAMUS layout (verified against the HF repo):
#      images/<split>/<STEM>_RGB.h5   key "image"  uint8   (1024,1024,3)
#      heights/<split>/<STEM>_AGL.h5  key "image"  float32 (1024,1024)  metres AGL == nDSM
#      classes/<split>/<STEM>_CLS.h5  key "image"  float32 (1024,1024)  class id {0..6}
# ===========================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SUB = {"rgb": ("images", "RGB"), "agl": ("heights", "AGL"), "cls": ("classes", "CLS")}


def _read_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        key = "image" if "image" in f else list(f.keys())[0]
        return np.asarray(f[key][()])


def ensure_hf_subset(cfg: Config, split: str, n: int) -> list[str]:
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(cfg.data_root)
    api = HfApi(token=cfg.hf_token or None)
    repo_files = api.list_repo_files(cfg.hf_dataset_repo, repo_type="dataset")
    prefix = f"images/{split}/"
    stems_all = sorted(
        f[len(prefix): -len("_RGB.h5")]
        for f in repo_files
        if f.startswith(prefix) and f.endswith("_RGB.h5")
    )
    if not stems_all:
        raise RuntimeError(f"no tiles under {prefix} in {cfg.hf_dataset_repo}")
    rng = random.Random(cfg.seed)
    rng.shuffle(stems_all)
    stems = sorted(stems_all if not n else stems_all[:n])
    print(f"[data] {split}: fetching {len(stems)}/{len(stems_all)} tiles -> {root}")

    rels = [
        f"{sub}/{split}/{stem}_{suffix}.h5"
        for stem in stems
        for sub, suffix in _SUB.values()
        if not (root / f"{sub}/{split}/{stem}_{suffix}.h5").exists()
    ]

    def _fetch(rel: str) -> None:
        hf_hub_download(
            cfg.hf_dataset_repo, rel, repo_type="dataset",
            local_dir=str(root), token=cfg.hf_token or None,
        )

    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.dl_workers)) as ex:
        for _ in ex.map(_fetch, rels):
            done += 1
            if done % 300 == 0 or done == len(rels):
                print(f"[data]   {split} {done}/{len(rels)} files  ({time.time() - t0:.0f}s)")
    return stems


def list_local_stems(cfg: Config, split: str) -> list[str]:
    d = Path(cfg.data_root) / "images" / split
    return sorted(p.name[: -len("_RGB.h5")] for p in d.glob("*_RGB.h5")) if d.is_dir() else []


class GamusDataset(Dataset):
    def __init__(self, cfg: Config, split: str, stems: list[str], train: bool):
        self.cfg, self.split, self.stems, self.train = cfg, split, stems, train
        self.root = Path(cfg.data_root)
        self.s = cfg.train_size

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        rgb = _read_h5(self.root / "images" / self.split / f"{stem}_RGB.h5")
        agl = _read_h5(self.root / "heights" / self.split / f"{stem}_AGL.h5").astype(np.float32)
        cls_path = self.root / "classes" / self.split / f"{stem}_CLS.h5"
        cls = _read_h5(cls_path).astype(np.int64) if cls_path.exists() else np.zeros(agl.shape, np.int64)

        h, w = agl.shape
        s = self.s
        rng = random.Random(self.cfg.seed * 1_000_003 + idx * 2 + int(self.train))
        if self.train:
            top = rng.randint(0, max(0, h - s))
            left = rng.randint(0, max(0, w - s))
        else:  # deterministic centre crop -> stable val GSD
            top, left = max(0, (h - s) // 2), max(0, (w - s) // 2)
        sl = (slice(top, top + s), slice(left, left + s))
        rgb, agl, cls = rgb[sl], agl[sl], cls[sl]

        if self.train:
            k = rng.randint(0, 3)
            rgb, agl, cls = (np.rot90(a, k).copy() for a in (rgb, agl, cls))
            if rng.random() < 0.5:
                rgb, agl, cls = (np.fliplr(a).copy() for a in (rgb, agl, cls))

        valid = np.isfinite(agl) & (agl >= 0.0) & (agl <= self.cfg.max_valid_height_m)
        agl = np.where(valid, agl, 0.0).astype(np.float32)
        rgb_n = ((rgb.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(np.transpose(rgb_n, (2, 0, 1)).copy()).float(),
            "rgb_u8": torch.from_numpy(np.ascontiguousarray(rgb)),
            "target": torch.from_numpy(agl).unsqueeze(0),
            "valid": torch.from_numpy(valid).unsqueeze(0),
            "cls": torch.from_numpy(np.ascontiguousarray(cls)).long(),
            "stem": stem,
        }


def build_loaders(cfg: Config):
    if cfg.data_source == "hf":
        tr = ensure_hf_subset(cfg, "train", cfg.train_subset)
        va = ensure_hf_subset(cfg, "val", cfg.val_subset)
    else:
        tr, va = list_local_stems(cfg, "train"), list_local_stems(cfg, "val")
        if cfg.train_subset:
            tr = tr[: cfg.train_subset]
        if cfg.val_subset:
            va = va[: cfg.val_subset]
    if not tr or not va:
        raise RuntimeError("no GAMUS tiles found — check data_source / data_root / internet")
    if len(tr) < cfg.batch_size:
        raise RuntimeError(f"train subset ({len(tr)}) < batch_size ({cfg.batch_size})")

    dl_tr = DataLoader(
        GamusDataset(cfg, "train", tr, True), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0,
    )
    dl_va = DataLoader(
        GamusDataset(cfg, "val", va, False), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )
    return dl_tr, dl_va


# ===========================================================================
# 4. model : DINOv3-SAT (frozen) -> DPT decoder -> metric nDSM head
# ===========================================================================
class DINOv3Encoder(nn.Module):
    """Wraps a HF DINOv3 ViT and returns 4 spatial feature maps (B, C, H/16, W/16)."""

    def __init__(self, cfg: Config):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            cfg.encoder_model_id, token=cfg.hf_token or None, output_hidden_states=True
        )
        self.patch = int(getattr(self.model.config, "patch_size", 16))
        self.hidden = int(self.model.config.hidden_size)
        self.n_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0))
        n_layers = int(getattr(self.model.config, "num_hidden_layers", 24))
        idx = tuple(cfg.encoder_feature_indices)
        if max(idx) > n_layers:  # e.g. ViT-B fallback has only 12 blocks
            idx = tuple(round(n_layers * q) for q in (0.25, 0.5, 0.75, 1.0))
            print(f"[model] configured taps exceed {n_layers} blocks -> using {idx}")
        self.idx = idx
        self.frozen = cfg.freeze_encoder
        if self.frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()
        print(
            f"[model] encoder={cfg.encoder_model_id} hidden={self.hidden} "
            f"patch={self.patch} prefix_tokens={self.n_prefix} taps={self.idx} frozen={self.frozen}"
        )

    def train(self, mode: bool = True):  # keep a frozen encoder in eval mode
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def _tokens_to_map(self, tok: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        patches = tok[:, -(hp * wp):, :]  # drop CLS + register/prefix tokens
        return patches.transpose(1, 2).reshape(tok.shape[0], self.hidden, hp, wp)

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        hp = pixel_values.shape[-2] // self.patch
        wp = pixel_values.shape[-1] // self.patch
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            hs = self.model(pixel_values=pixel_values).hidden_states
        return [self._tokens_to_map(hs[i], hp, wp) for i in self.idx]


class ResidualConvUnit(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        y = self.conv2(F.relu(self.conv1(F.relu(x))))
        return x + y


class FeatureFusionBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.rcu1 = ResidualConvUnit(c)
        self.rcu2 = ResidualConvUnit(c)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out(x)


class DPTDecoder(nn.Module):
    """4x ViT feature maps (all at 1/16) -> reassemble to {1/4,1/8,1/16,1/32} -> RefineNet fusion."""

    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        proj = [96, 192, 384, 768]
        self.proj = nn.ModuleList(nn.Conv2d(in_ch, p, 1) for p in proj)
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(proj[0], proj[0], 4, stride=4),                 # 1/16 -> 1/4
            nn.ConvTranspose2d(proj[1], proj[1], 2, stride=2),                 # 1/16 -> 1/8
            nn.Identity(),                                                     # 1/16
            nn.Conv2d(proj[3], proj[3], 3, stride=2, padding=1),              # 1/16 -> 1/32
        ])
        self.to_dim = nn.ModuleList(nn.Conv2d(p, dim, 3, padding=1, bias=False) for p in proj)
        self.fuse = nn.ModuleList(FeatureFusionBlock(dim) for _ in range(4))
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(dim // 2, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, feats: list[torch.Tensor], out_hw) -> torch.Tensor:
        f = [self.resample[i](self.proj[i](feats[i])) for i in range(4)]
        f = [self.to_dim[i](f[i]) for i in range(4)]
        x = self.fuse[3](f[3])
        x = self.fuse[2](x, f[2])
        x = self.fuse[1](x, f[1])
        x = self.fuse[0](x, f[0])
        x = self.head(x)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


class DepthWizardNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = DINOv3Encoder(cfg)
        self.decoder = DPTDecoder(self.encoder.hidden, cfg.decoder_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(image)
        raw = self.decoder(feats, image.shape[-2:])
        return F.relu(raw)  # nDSM is non-negative


# ===========================================================================
# 5. losses
# ===========================================================================
def gradient_loss(pred, target, valid, scales=4):
    total = 0.0
    p, t, v = pred, target, valid.float()
    for _ in range(scales):
        for d in (1, 2):  # x, y
            dp = (p.diff(dim=-d)).abs()
            dt = (t.diff(dim=-d)).abs()
            m = (v.diff(dim=-d) == 0) & (v.narrow(-d, 0, v.shape[-d] - 1) > 0)
            if m.any():
                total = total + F.l1_loss(dp[m], dt[m])
        p = F.avg_pool2d(p, 2)
        t = F.avg_pool2d(t, 2)
        v = F.avg_pool2d(v, 2)
    return total / scales


def silog_loss(pred, target, valid, lam, shift):
    m = valid.bool()
    if not m.any():
        return pred.sum() * 0.0
    g = torch.log(pred[m].clamp_min(0) + shift) - torch.log(target[m] + shift)
    return torch.sqrt((g ** 2).mean() - lam * (g.mean() ** 2) + 1e-7)


def compute_loss(pred, target, valid, cfg: Config):
    m = valid.bool()
    l1 = F.l1_loss(pred[m], target[m]) if m.any() else pred.sum() * 0.0
    grad = gradient_loss(pred, target, valid)
    sil = silog_loss(pred, target, valid, cfg.silog_lambda, cfg.silog_shift)
    loss = cfg.w_l1 * l1 + cfg.w_grad * grad + cfg.w_silog * sil
    return loss, {"l1": float(l1), "grad": float(grad), "silog": float(sil)}


# ===========================================================================
# 6. streaming metrics  (global + per class)
# ===========================================================================
class MetricAccum:
    def __init__(self):
        self.n = 0
        self.se = self.ae = 0.0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.d1 = 0  # delta < 1.25

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.double()
        target = target.double()
        n = pred.numel()
        if n == 0:
            return
        self.n += n
        diff = pred - target
        self.se += float((diff ** 2).sum())
        self.ae += float(diff.abs().sum())
        self.sx += float(pred.sum())
        self.sy += float(target.sum())
        self.sxx += float((pred ** 2).sum())
        self.syy += float((target ** 2).sum())
        self.sxy += float((pred * target).sum())
        r = torch.maximum(pred + 1.0, target + 1.0) / torch.minimum(pred + 1.0, target + 1.0)
        self.d1 += int((r < 1.25).sum())

    def result(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        rmse = (self.se / self.n) ** 0.5
        mae = self.ae / self.n
        cov = self.sxy / self.n - (self.sx / self.n) * (self.sy / self.n)
        vx = self.sxx / self.n - (self.sx / self.n) ** 2
        vy = self.syy / self.n - (self.sy / self.n) ** 2
        pearson = cov / ((vx * vy) ** 0.5 + 1e-12)
        return {
            "n": self.n, "rmse_m": rmse, "mae_m": mae,
            "pearson_r": pearson, "delta1": self.d1 / self.n,
        }


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device) -> dict:
    model.eval()
    gm = MetricAccum()
    cm = {i: MetricAccum() for i in range(len(cfg.class_names))} if cfg.per_class_metrics else {}
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        val = batch["valid"].to(device, non_blocking=True).bool()
        with torch.autocast("cuda", enabled=cfg.amp):
            pred = model(img).float()
        p, t = pred[val], tgt[val]
        gm.update(p, t)
        if cm:
            cls = batch["cls"].to(device, non_blocking=True).unsqueeze(1)[val]
            for ci, acc in cm.items():
                sel = cls == ci
                if sel.any():
                    acc.update(p[sel], t[sel])
    out = {"global": gm.result()}
    if cm:
        out["per_class"] = {
            cfg.class_names[i]: acc.result() for i, acc in cm.items() if acc.n > 0
        }
    return out


# ===========================================================================
# 7. train
# ===========================================================================
def safe_config_dict(cfg: Config) -> dict:
    d = asdict(cfg)
    d.pop("hf_token", None)  # never persist the token to disk
    return d


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def export_viewer_sample(model, loader, cfg: Config, device):
    """rgb.png + height16.png + meta.json for viewer/phase0_viewer.html."""
    from PIL import Image

    model.eval()
    batch = next(iter(loader))
    with torch.no_grad(), torch.autocast("cuda", enabled=cfg.amp):
        pred = model(batch["image"][:1].to(device)).float()[0, 0].cpu().numpy()
    gt = batch["target"][0, 0].numpy()
    rgb = batch["rgb_u8"][0].numpy().astype(np.uint8)   # (H, W, 3)

    out = Path(cfg.output_dir) / "viewer_sample"
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out / "rgb.png")
    hi, lo = float(pred.max()), float(pred.min())
    span = max(hi - lo, 1e-6)
    Image.fromarray(((pred - lo) / span * 65535).astype(np.uint16)).save(out / "height16.png")
    np.save(out / "pred_ndsm_m.npy", pred.astype(np.float32))
    np.save(out / "gt_ndsm_m.npy", gt.astype(np.float32))
    json.dump(
        {
            "stem": batch["stem"][0], "height_min_m": lo, "height_max_m": hi,
            "gsd_m": cfg.gamus_gsd_m, "size_px": cfg.train_size,
            "encode": "height_m = height_min_m + (png16 / 65535) * (height_max_m - height_min_m)",
        },
        open(out / "meta.json", "w"), indent=2,
    )
    print(f"[viewer] wrote sample -> {out}")


def main() -> None:
    cfg = parse_config()
    cfg.hf_token = resolve_hf_token(cfg)
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpu else "cpu")
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={n_gpu}")
    for i in range(n_gpu):
        print(f"[env]   GPU{i}: {torch.cuda.get_device_name(i)}")
    if not cfg.hf_token:
        print("[warn] no HF token found — gated DINOv3 download will fail. "
              "Add a Kaggle Secret named HF_TOKEN (see the docstring).")

    dl_tr, dl_va = build_loaders(cfg)
    print(f"[data] train batches={len(dl_tr)}  val batches={len(dl_va)}")

    model = DepthWizardNet(cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params")

    if n_gpu > 1:
        model = nn.DataParallel(model)
        print(f"[model] nn.DataParallel over {n_gpu} GPUs, "
              f"per-GPU batch ~{cfg.batch_size // n_gpu}")

    opt = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(dl_tr) // cfg.grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.learning_rate, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)      # torch >= 2.3
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    history = []
    best_rmse = float("inf")
    t_start = time.time()
    stop = False

    for epoch in range(1, cfg.epochs + 1):
        if stop:
            break
        model.train()
        run_loss = 0.0
        parts_sum = {"l1": 0.0, "grad": 0.0, "silog": 0.0}
        parts = {"l1": 0.0, "grad": 0.0, "silog": 0.0}
        step = -1
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(dl_tr):
            img = batch["image"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            val = batch["valid"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=cfg.amp):
                pred = model(img)
                loss, parts = compute_loss(pred.float(), tgt, val, cfg)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()

            run_loss += float(loss) * cfg.grad_accum
            for k in parts_sum:
                parts_sum[k] += parts[k]

            if step % 20 == 0:
                el = (time.time() - t_start) / 60
                print(f"  e{epoch} s{step}/{len(dl_tr)} loss={float(loss)*cfg.grad_accum:.3f} "
                      f"(l1={parts['l1']:.2f} grad={parts['grad']:.2f} silog={parts['silog']:.3f}) "
                      f"lr={sched.get_last_lr()[0]:.2e} {el:.0f}min")

            if (time.time() - t_start) / 60 > cfg.max_train_minutes:
                print(f"[train] hit wall-clock cap ({cfg.max_train_minutes} min) — stopping")
                stop = True
                break

        nb = max(1, step + 1)
        rec = {"epoch": epoch, "train_loss": run_loss / nb,
               **{f"train_{k}": v / nb for k, v in parts_sum.items()}}

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs or stop:
            metrics = evaluate(model, dl_va, cfg, device)
            g = metrics["global"]
            rec["val"] = metrics
            print(f"[eval] epoch {epoch}  RMSE={g['rmse_m']:.3f}m  MAE={g['mae_m']:.3f}m  "
                  f"r={g['pearson_r']:.3f}  d1={g['delta1']:.3f}")
            if cfg.per_class_metrics and "per_class" in metrics:
                for name, cm in metrics["per_class"].items():
                    print(f"[eval]   {name:11s} RMSE={cm['rmse_m']:.3f}m  MAE={cm['mae_m']:.3f}m  n={cm['n']}")
            if g["rmse_m"] < best_rmse and cfg.save_checkpoint:
                best_rmse = g["rmse_m"]
                sd = (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
                torch.save({"epoch": epoch, "config": safe_config_dict(cfg), "metrics": metrics,
                            "model": {k: v for k, v in sd.items() if "encoder.model" not in k}},
                           Path(cfg.output_dir) / "best.pt")
                print(f"[ckpt] new best RMSE {best_rmse:.3f}m -> best.pt (decoder+head only)")

        history.append(rec)
        json.dump(
            {"config": safe_config_dict(cfg), "history": history,
             "best_val_rmse_m": best_rmse, "elapsed_min": (time.time() - t_start) / 60},
            open(Path(cfg.output_dir) / "metrics.json", "w"), indent=2, default=str,
        )

    if cfg.export_viewer_sample:
        try:
            export_viewer_sample(model, dl_va, cfg, device)
        except Exception as e:  # never fail the run over a screenshot asset
            print(f"[viewer] export skipped: {e}")

    print("\n" + "=" * 64)
    print(f"DONE — best GAMUS val RMSE: {best_rmse:.3f} m   "
          f"({(time.time() - t_start) / 60:.0f} min, {n_gpu}x T4)")
    print(f"artifacts: {cfg.output_dir}/metrics.json, best.pt, viewer_sample/")
    print("=" * 64)


if __name__ == "__main__":
    main()
