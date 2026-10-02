"""Score a frozen checkpoint (v3, v4 or v5) on a held-out packed store — one protocol.

    # v5 or v4 weights, scored by this tree
    python eval_test.py --ckpt best.pt --data_root /scratch/dwdata \
        --source gamus --split test --tiles 0 --sliding_tiles 400 \
        --tta_scales 1.0,1.25 --max_valid_height_m 150 --out outputs/v5_test

    # v3 weights, scored by v3's own model code (see "Why --model_code")
    python eval_test.py --model_code ../v3 --ckpt best_2.715.pt ... --out outputs/v3_test

    # qualitative strips + shadow IoU on 12 tiles, next to the metrics
    python eval_test.py --ckpt best.pt ... --qualitative 12

    # the cross-version table (+ strips when every run dumped the same tiles)
    python eval_test.py --compare outputs/v3_test,outputs/v4_test,outputs/v5_test \
        --labels v3,v4,v5 --md outputs/v3_v4_v5.md

Ported from `v3/eval_test.py`; same protocol, so the numbers sit in one table
with v4's `test_gamus_test_*`: every tile centre-crop plain + TTA, then sliding
window + TTA over a seeded sample of `--sliding_tiles` (seed 42).

**Why `--model_code`.**  v4 changed Head B numerically (RMS-normalised pooled
feature, fp32 width softmax, floored widths) without renaming a parameter, so a
v3 checkpoint *loads* into v4/v5 code and predicts through different bin
centres: wrong numbers, no error.  `--model_code ../v3` puts that tree first on
`sys.path`, so `config`, `models`, `dwdata` and `eval` are v3's own; only the
two v5-only helpers this script needs (`viz/shadow.py`, the sharpness metrics)
are loaded from this tree by file path.  v4 checkpoints load into v5 code
directly: v5 only *adds* modules (the detail branch), all behind
`detail_branch`, and the architecture is restored from the checkpoint's own
config so v4 weights build a v4-shaped network.

**What v5 adds on top of v3's script.**
  * `fit_keys` (transformers 4.x `layer.N` <-> 5.x `model.layer.N`) — kept.
  * The architecture is restored from `--run_config`, else from the config the
    checkpoint embeds, so no flag is needed for a v4 vs v5 checkpoint.
  * `edge_rmse_m` / `grad_ratio` for *every* version: v5's evaluator computes
    them itself; for v3 the plain pass is tapped and the same maths applied.
  * `--qualitative N`: RGB / GT / prediction per tile dumped to `qual/*.npz`,
    and shadow IoU — the sun is fitted on the *GT* heights against the image's
    own shadows (GAMUS tiles carry no sun metadata), then the prediction's cast
    shadows are scored under that sun.  No labels needed at deploy time, but the
    GT sun makes it a fair per-tile comparison between versions.
  * σ calibration, when the model returns Head B's spread `b_std` (v4/v5): the
    plain pass is tapped and scored per tile with sparsification curves, AUSE
    and AURG (`eval/sparsification.py`, Poggi et al. CVPR 2020), under
    `<tag>_plain.uncertainty`.  Checks that the σ the web app shows tracks error.
  * `--compare`: `v3_v4_v5.md` from any mix of these `test_metrics.json` files
    and training `metrics.json` files (which carry the same `test_*` keys).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent


def _model_code(argv: list[str]) -> Path:
    for i, x in enumerate(argv):
        if x == "--model_code" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if x.startswith("--model_code="):
            return Path(x.split("=", 1)[1]).resolve()
    return _HERE


_CODE = _model_code(sys.argv[1:])
sys.path.insert(0, str(_CODE))


def _load_here(name: str, rel: str):
    """A module from THIS tree by file path, whatever `sys.path` resolves to."""
    spec = importlib.util.spec_from_file_location(name, _HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shadow = _load_here("_dw_v5_shadow", "viz/shadow.py")
sparse = _load_here("_dw_v5_sparsification", "eval/sparsification.py")

# Config keys that describe the *run*, not the network.  Restoring these from a
# training config would point the eval back at the machine that trained it.
_SKIP = {"output_dir", "resume", "data_root", "datasets", "make_zip", "smoke",
         "num_workers", "prefetch_factor", "batch_size", "epochs", "max_minutes",
         "hf_token", "test_sources", "sampler_weights"}

# DINOv3ViTModel's blocks sit at `layer.N` in transformers 4.56/4.57 and at
# `model.layer.N` in 5.x.  Same tensors, one prefix level apart.
_KEY_ALIASES = (("encoder.model.model.layer.", "encoder.model.layer."),
                ("encoder.model.layer.", "encoder.model.model.layer."))


def fit_keys(sd: dict, want) -> tuple[dict, int]:
    """Rename checkpoint keys across `_KEY_ALIASES`, in either direction, only
    onto a name the model has and the checkpoint does not."""
    want = set(want)
    out, n = {}, 0
    for k, v in sd.items():
        if k not in want:
            for old, new in _KEY_ALIASES:
                nk = new + k[len(old):]
                if k.startswith(old) and nk in want and nk not in sd:
                    k, n = nk, n + 1
                    break
        out[k] = v
    return out, n


# ---------------------------------------------------------------------
# sharpness metrics, version-agnostic (same maths as v5 Evaluator.add_spatial)
# ---------------------------------------------------------------------
class Sharpness:
    def __init__(self, step_m: float = 2.0, band_px: int = 2):
        self.step_m, self.band_px = step_m, band_px
        self.se = 0.0
        self.n = 0
        self.gp = self.gt = 0.0

    def add(self, pred, target, valid):
        with torch.no_grad():
            p = pred.float().reshape(-1, 1, *pred.shape[-2:])
            t = target.float().reshape(-1, 1, *target.shape[-2:]).to(p.device)
            v = valid.bool().reshape(-1, 1, *valid.shape[-2:]).to(p.device)
            big = torch.where(v, t, torch.full_like(t, -1e6))
            small = torch.where(v, t, torch.full_like(t, 1e6))
            rng = F.max_pool2d(big, 3, 1, 1) + F.max_pool2d(-small, 3, 1, 1)
            edge = (rng > self.step_m) & (rng < 5e5)
            k = 2 * self.band_px + 1
            sel = (F.max_pool2d(edge.float(), k, 1, self.band_px) > 0) & v
            if sel.any():
                d = (p[sel] - t[sel]).double()
                self.se += float((d * d).sum())
                self.n += int(sel.sum())
            for dim in (2, 3):
                vv = v.narrow(dim, 0, v.shape[dim] - 1) & v.narrow(dim, 1, v.shape[dim] - 1)
                self.gp += float(p.diff(dim=dim).abs()[vv].sum())
                self.gt += float(t.diff(dim=dim).abs()[vv].sum())

    def result(self) -> dict:
        out = {}
        if self.n:
            out["edge_rmse_m"] = (self.se / self.n) ** 0.5
        if self.gt > 0:
            out["grad_ratio"] = self.gp / self.gt
        return out


class _Tap:
    """Wraps a loader + model so the plain pass also feeds `Sharpness` (when the
    evaluator does not compute it itself) and, when the model returns Head B's
    spread `b_std`, the sparsification meter (AUSE / AURG)."""

    def __init__(self, loader, max_h: float, sharp: bool = True):
        self.loader, self.max_h, self.cur = loader, max_h, None
        # v5's evaluate() reads the source from loader.dataset (no_urban): keep it visible
        self.dataset = getattr(loader, "dataset", None)
        self.sharp = Sharpness() if sharp else None
        self.sparse = sparse.SparsificationMeter()

    def __iter__(self):
        for b in self.loader:
            self.cur = b
            yield b
        self.cur = None

    def __len__(self):
        return len(self.loader)

    def wrap(self, model):
        return _Tapped(model, self)


class _Tapped(torch.nn.Module):
    def __init__(self, model, tap: _Tap):
        super().__init__()
        self.m, self.tap = model, tap

    def forward(self, x):
        out = self.m(x)
        b = self.tap.cur
        if b is not None and tuple(x.shape[-2:]) == tuple(b["target"].shape[-2:]) \
                and x.shape[0] == b["target"].shape[0]:
            t = b["target"]
            v = b["valid"].bool() & (t <= self.tap.max_h)
            if self.tap.sharp is not None:
                self.tap.sharp.add(out["fused"].detach(), t, v)
            if out.get("b_std") is not None:
                self.tap.sparse.add_batch(out["fused"], t, out["b_std"], v)
            self.tap.cur = None
        return out


class Replicated(torch.nn.Module):
    """One frozen copy per device; every batch split across them (eval only).

    nn.DataParallel would re-broadcast ~1.2 GB of ViT-L weights from cuda:0 on
    every forward; here each card gets its copy once.  Autocast state is
    thread-local, so the workers re-enter grad mode and autocast explicitly.
    """

    KEYS = ("fused", "seg", "b_std")

    def __init__(self, model, devices):
        super().__init__()
        self.devices = list(devices)
        self.replicas = torch.nn.ModuleList(
            [model] + [copy.deepcopy(model).to(d) for d in self.devices[1:]])
        self._pool = ThreadPoolExecutor(len(self.devices))

    def forward(self, x):
        chunks = x.chunk(len(self.devices))
        dev = self.devices[0].type
        grad = torch.is_grad_enabled()
        amp, amp_dt = torch.is_autocast_enabled(dev), torch.get_autocast_dtype(dev)

        def run(i):
            d = self.devices[i]
            on = torch.cuda.device(d) if d.type == "cuda" else contextlib.nullcontext()
            with on, torch.set_grad_enabled(grad), \
                    torch.autocast(d.type, dtype=amp_dt, enabled=amp):
                out = self.replicas[i](chunks[i].to(d))
            return {k: out[k] for k in self.KEYS if k in out}

        outs = list(self._pool.map(run, range(len(chunks))))
        return {k: torch.cat([o[k].to(self.devices[0]) for o in outs]) for k in outs[0]}


# ---------------------------------------------------------------------
# argument handling / architecture restore
# ---------------------------------------------------------------------
def _own_args(argv: list[str]):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model_code", default=str(_HERE),
                   help="code tree whose model/config score the checkpoint (../v3 for v3)")
    p.add_argument("--ckpt", default="")
    p.add_argument("--run_config", default="",
                   help="config.json of the run that produced --ckpt; default: the "
                        "config embedded in the checkpoint")
    p.add_argument("--source", default="gamus")
    p.add_argument("--split", default="test")
    p.add_argument("--tiles", type=int, default=0, help="0 = the whole store")
    p.add_argument("--sliding_tiles", type=int, default=400, help="0 = skip")
    p.add_argument("--out", default="outputs/v5_test")
    p.add_argument("--skip_plain", action="store_true")
    p.add_argument("--skip_tta", action="store_true")
    p.add_argument("--tta_scales", default="",
                   help="comma list, e.g. 1.0,1.25; empty keeps the run config's")
    p.add_argument("--qualitative", type=int, default=0,
                   help="dump N tiles (seeded) + shadow IoU; 0 = off")
    p.add_argument("--qual_stems", default="", help="comma list of stems to dump instead")
    p.add_argument("--qual_seed", type=int, default=42)
    p.add_argument("--compare", default="", help="comma list of result dirs / json files")
    p.add_argument("--labels", default="", help="comma list, one per --compare entry")
    p.add_argument("--md", default="outputs/v3_v4_v5.md")
    return p.parse_known_args(argv)


def _overlay_config(cfg, d: dict, explicit: set) -> int:
    names = set(vars(cfg)) - _SKIP - explicit
    n = 0
    for k, v in d.items():
        if k in names:
            cur = getattr(cfg, k)
            setattr(cfg, k, tuple(v) if isinstance(cur, tuple) and isinstance(v, list) else v)
            n += 1
    return n


def _restore_arch(cfg, run_config: str, ck: dict, argv: list[str]) -> str:
    """Overlay the training run's config onto `cfg`, CLI flags winning."""
    explicit = {x[2:].split("=")[0] for x in argv if x.startswith("--")}
    if run_config:
        _overlay_config(cfg, json.loads(Path(run_config).read_text()), explicit)
        return run_config
    emb = ck.get("config") if isinstance(ck, dict) else None
    if isinstance(emb, dict) and emb:
        _overlay_config(cfg, emb, explicit)
        return "checkpoint"
    return "defaults"


def _model_class():
    import models.heads as heads

    for name in ("DepthWizardNet", "DepthWizardNetV3"):
        if hasattr(heads, name):
            return getattr(heads, name)
    raise SystemExit(f"no DepthWizardNet in {heads.__file__}")


def load_checkpoint(cfg, ckpt: str, run_config: str, argv: list[str], device):
    """Build the network the checkpoint was trained as and load it — or refuse."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    src = _restore_arch(cfg, run_config, ck, argv)
    print(f"[cfg] architecture from {src}"
          + (f"  detail_branch={cfg.detail_branch}" if hasattr(cfg, "detail_branch") else ""))
    if isinstance(ck, dict) and "preproc" in ck:
        pp = ck["preproc"]
        for k in ("encoder_model_id", "tile_size", "canonical_gsd_m"):
            if k in pp and hasattr(cfg, k):
                setattr(cfg, k, pp[k])
    model = _model_class()(cfg).to(device)
    sd, renamed = fit_keys(ck.get("model", ck), model.state_dict().keys())
    if renamed:
        print(f"[ckpt] {renamed} encoder keys renamed across the transformers "
              f"DINOv3 layout change (4.x layer.N <-> 5.x model.layer.N)")
    try:
        miss, unexp = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise SystemExit(f"checkpoint does not fit this config — wrong "
                         f"--run_config / --model_code?\n{e}") from e
    enc_ok = not ck.get("encoder_included", True)
    hard = [m for m in miss if not (enc_ok and m.startswith("encoder.model."))]
    print(f"[ckpt] {ckpt}: {len(sd)} tensors, missing {len(miss)} "
          f"({len(miss) - len(hard)} encoder tensors from the hub), unexpected {len(unexp)}"
          + (f"  epoch={ck['epoch']}" if isinstance(ck, dict) and "epoch" in ck else ""))
    if hard:
        raise SystemExit(f"checkpoint does not fit this config — missing {len(hard)} "
                         f"parameters, first few: {hard[:8]}")
    return model.eval(), ck


# ---------------------------------------------------------------------
# qualitative: dumps + shadow IoU
# ---------------------------------------------------------------------
def shadow_iou_tile(rgb_u8: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                    valid: np.ndarray, gsd_m: float) -> dict:
    """Sun fitted on the GT against the image shadows; both maps scored under it."""
    img = shadow.detect_image_shadows(rgb_u8, valid)
    if img[valid].mean() < 0.005:
        return {"informative": False}
    sun = shadow.fit_sun(np.where(valid, gt, 0.0), gsd_m, img, valid)
    az, el = sun["azimuth_deg"], sun["elevation_deg"]
    iou_p = shadow.iou(shadow.cast_shadows(np.where(valid, pred, 0.0), gsd_m, az, el),
                       img, valid)
    return {"informative": True, "azimuth_deg": az, "elevation_deg": el,
            "iou_gt": sun["shadow_iou"], "iou_pred": iou_p}


def qualitative(model, ds, cfg, device, out_dir: Path, n: int, stems: str,
                seed: int) -> dict:
    qd = out_dir / "qual"
    qd.mkdir(parents=True, exist_ok=True)
    want = [s for s in stems.split(",") if s]
    all_stems = list(ds.store.stems)
    if want:
        idx = [all_stems.index(s) for s in want if s in all_stems]
    else:
        idx = sorted(np.random.default_rng(seed).permutation(len(ds))[:n].tolist())
    rows, ious = [], []
    for i in idx:
        b = ds[int(i)]
        x = b["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            p = model(x)["fused"].float()[0, 0].cpu().numpy()
        gt = b["target"][0].numpy()
        v = b["valid"][0].numpy().astype(bool)
        rgb = b["rgb_u8"].numpy()
        g = float(b["gsd_m"])
        np.savez_compressed(qd / f"{b['stem']}.npz", rgb=rgb, gt=gt.astype(np.float16),
                            valid=v, pred=p.astype(np.float16), gsd_m=g)
        s = shadow_iou_tile(rgb, gt, p, v, g)
        rows.append({"stem": b["stem"], **s})
        if s.get("informative"):
            ious.append((s["iou_gt"], s["iou_pred"]))
    res = {"tiles": rows, "n_informative": len(ious)}
    if ious:
        res["shadow_iou_pred_mean"] = float(np.mean([p for _, p in ious]))
        res["shadow_iou_gt_mean"] = float(np.mean([g for g, _ in ious]))
    return res


# ---------------------------------------------------------------------
# compare mode
# ---------------------------------------------------------------------
def _load_result(p: str) -> tuple[dict, Path]:
    q = Path(p)
    f = q / "test_metrics.json" if q.is_dir() else q
    if not f.is_file() and q.is_dir():
        f = q / "metrics.json"
    return json.loads(f.read_text()), (q if q.is_dir() else q.parent)


def _g(d: dict, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def compare_table(results: list[tuple[str, dict]], tag: str | None = None) -> str:
    """Markdown: headline, strata, landscape, class rows for each version."""
    if tag is None:
        keys = [k for _, r in results for k in r if k.startswith("test_") and
                k.endswith("_plain")]
        tag = keys[0][: -len("_plain")] if keys else "test_gamus_test"
    L = [f"# Cross-version comparison — `{tag}`", "",
         "Same protocol for every column: all tiles centre-crop plain + TTA, sliding "
         "window + TTA over a seeded sample of N tiles (eval_test.py).", "",
         "| | " + " | ".join(n for n, _ in results) + " |",
         "|---|" + "---|" * len(results)]

    def row(label, fn, nd=3):
        L.append(f"| {label} | " + " | ".join(_fmt(fn(r), nd) for _, r in results) + " |")

    for suf in ("plain", "tta", "sliding_tta"):
        row(f"RMSE {suf}", lambda r, s=suf: _g(r, f"{tag}_{s}", "global", "rmse_m"))
    row("MAE plain", lambda r: _g(r, f"{tag}_plain", "global", "mae_m"))
    row("r plain", lambda r: _g(r, f"{tag}_plain", "global", "pearson_r"))
    row("balanced RMSE plain", lambda r: _g(r, f"{tag}_plain", "balanced_rmse_m"))
    row("flat (<1 m) bias", lambda r: _g(r, f"{tag}_plain", "flat_lt1m", "bias_m"))
    row("tall (>15 m) bias", lambda r: _g(r, f"{tag}_plain", "tall_gt15m", "bias_m"))
    row("edge RMSE", lambda r: _g(r, f"{tag}_plain", "edge_rmse_m")
        or _g(r, "sharpness", "edge_rmse_m"))
    row("grad ratio", lambda r: _g(r, f"{tag}_plain", "grad_ratio")
        or _g(r, "sharpness", "grad_ratio"), 2)
    row("shadow IoU (qual)", lambda r: _g(r, "qualitative", "shadow_iou_pred_mean"))
    row("sigma AUSE (RMSE)", lambda r: _g(r, f"{tag}_plain", "uncertainty", "ause_rmse_m"))
    row("sigma AURG (RMSE)", lambda r: _g(r, f"{tag}_plain", "uncertainty", "aurg_rmse_m"))

    strata = []
    for _, r in results:
        for k in (_g(r, f"{tag}_plain", "per_stratum") or {}):
            if k not in strata:
                strata.append(k)
    if strata:
        L += ["", "**Per height stratum (plain) — RMSE / bias**", "",
              "| stratum | " + " | ".join(n for n, _ in results) + " |",
              "|---|" + "---|" * len(results)]
        for s in strata:
            cells = []
            for _, r in results:
                d = _g(r, f"{tag}_plain", "per_stratum", s) or {}
                cells.append("-" if not d.get("n") else
                             f"{d['rmse_m']:.3f} / {d['bias_m']:+.2f}")
            L.append(f"| {s} | " + " | ".join(cells) + " |")
    for section, key in (("landscape", "per_landscape"), ("class", "per_class")):
        names = []
        for _, r in results:
            for k in (_g(r, f"{tag}_plain", key) or {}):
                if k not in names:
                    names.append(k)
        if not names:
            continue
        L += ["", f"**Per {section} (plain) — RMSE / bias**", "",
              f"| {section} | " + " | ".join(n for n, _ in results) + " |",
              "|---|" + "---|" * len(results)]
        for s in names:
            cells = []
            for _, r in results:
                d = _g(r, f"{tag}_plain", key, s) or {}
                cells.append("-" if not d.get("n") else
                             f"{d['rmse_m']:.3f} / {d['bias_m']:+.2f}")
            L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


def _colour(h: np.ndarray, vmax: float) -> np.ndarray:
    import matplotlib

    cm = matplotlib.colormaps["turbo"]
    x = np.clip(np.nan_to_num(h.astype(np.float32)) / max(vmax, 1e-3), 0, 1)
    return (cm(x)[..., :3] * 255).astype(np.uint8)


def agreement_rgb(img: np.ndarray, cast: np.ndarray) -> np.ndarray:
    """White both, red image-only (height too low), blue height-only (too high)."""
    out = np.zeros(img.shape + (3,), np.uint8)
    out[img & cast] = (255, 255, 255)
    out[img & ~cast] = (230, 40, 40)
    out[~img & cast] = (40, 90, 230)
    return out


def write_strips(dirs: list[tuple[str, Path]], out_dir: Path) -> list[str]:
    """RGB | GT | each version | shadow agreement of the last, per common stem."""
    from PIL import Image, ImageDraw

    sets = [set(p.stem for p in (d / "qual").glob("*.npz")) for _, d in dirs]
    common = sorted(set.intersection(*sets)) if sets and all(sets) else []
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem in common:
        zs = [np.load(d / "qual" / f"{stem}.npz") for _, d in dirs]
        rgb, gt, v = zs[0]["rgb"], zs[0]["gt"].astype(np.float32), zs[0]["valid"]
        vmax = float(np.percentile(gt[v], 99)) if v.any() else 10.0
        panels = [rgb, _colour(np.where(v, gt, 0), vmax)]
        panels += [_colour(z["pred"].astype(np.float32), vmax) for z in zs]
        g = float(zs[-1]["gsd_m"])
        img = shadow.detect_image_shadows(rgb, v)
        s = shadow_iou_tile(rgb, gt, zs[-1]["pred"].astype(np.float32), v, g)
        if s.get("informative"):
            cast = shadow.cast_shadows(zs[-1]["pred"].astype(np.float32), g,
                                       s["azimuth_deg"], s["elevation_deg"])
            panels.append(agreement_rgb(img, cast))
        strip = Image.fromarray(np.concatenate(panels, 1))
        dr = ImageDraw.Draw(strip)
        w = rgb.shape[1]
        for j, lab in enumerate(["RGB", "GT"] + [n for n, _ in dirs]
                                + (["shadows"] if s.get("informative") else [])):
            dr.text((j * w + 6, 6), lab, fill=(255, 255, 255))
        f = out_dir / f"strip_{stem}.png"
        strip.save(f)
        written.append(str(f))
    return written


def compare_main(a) -> None:
    ents = [e for e in a.compare.split(",") if e]
    labels = [x for x in a.labels.split(",") if x] or [Path(e).name for e in ents]
    if len(labels) != len(ents):
        raise SystemExit("--labels needs one name per --compare entry")
    res, dirs = [], []
    for lab, e in zip(labels, ents):
        r, d = _load_result(e)
        res.append((lab, r))
        dirs.append((lab, d))
    md = compare_table(res)
    strips = write_strips(dirs, Path(a.md).parent / "strips")
    if strips:
        md += "\n## Qualitative strips\n\nRGB | GT | " + " | ".join(labels) + \
              " | shadow agreement (white both, red image only, blue height only)\n\n"
        md += "\n".join(f"![]({Path(s).relative_to(Path(a.md).parent)})" for s in strips) + "\n"
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.md).write_text(md)
    print(md)
    print(f"[out] {a.md}")


# ---------------------------------------------------------------------
def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    a, rest = _own_args(argv)
    if a.compare:
        compare_main(a)
        return
    if not a.ckpt:
        raise SystemExit("--ckpt is required (or --compare)")

    from torch.utils.data import DataLoader

    from config import parse_config, safe_config_dict
    from dwdata.dataset import FullTileDataset, TileDataset
    from dwdata.packed import PackedStore, store_exists
    from dwdata.preprocess import PreprocSpec
    from eval.metrics import Evaluator, evaluate, format_line
    from eval.sliding import sliding_eval
    from train import resolve_hf_token

    cfg = parse_config(rest)
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpu else "cpu")
    print(f"[env] code={_CODE} torch={torch.__version__} gpus={n_gpu}"
          + (f" {torch.cuda.get_device_properties(0).name}" if n_gpu else ""))
    cfg.hf_token = resolve_hf_token(cfg)
    model, ck = load_checkpoint(cfg, a.ckpt, a.run_config, rest, device)
    cfg.gpu_augment = False        # the eval path normalises on the host
    cfg.per_landscape_metrics = True
    if a.tta_scales:
        cfg.tta_scales = tuple(float(x) for x in a.tta_scales.split(",") if x.strip())
    print(f"[cfg] tta_scales={tuple(cfg.tta_scales)} "
          f"max_valid_height_m={cfg.max_valid_height_m}")

    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = PreprocSpec.from_dict(ck["preproc"]) if isinstance(ck, dict) and "preproc" in ck \
        else PreprocSpec.from_config(cfg)
    del ck
    d = Path(cfg.data_root) / a.source / a.split
    if not store_exists(d):
        raise SystemExit(f"no packed store at {d} — attach depthwizard-gamus and run `bash run_kaggle.sh link`, "
                         f"or `prepare_data.py --datasets gamus --gamus_test 0`")
    store = PackedStore(d)
    n = min(a.tiles, len(store)) if a.tiles else len(store)
    print(f"[data] {a.source}/{a.split}: {len(store)} tiles @ {store.tile_px}px "
          f"/ {store.gsd_m} m — scoring {n}")
    if spec.radiometric_stretch:
        store.prime_stretch_bounds(spec.stretch_lo_pct, spec.stretch_hi_pct,
                                   workers=max(4, cfg.num_workers or 8))
    ds = TileDataset(cfg, store, spec, a.source, train=False, length=n)
    dl = DataLoader(
        ds, batch_size=max(1, int(cfg.batch_size * max(1, cfg.eval_batch_mult))),
        shuffle=False, num_workers=cfg.num_workers, pin_memory=n_gpu > 0,
        persistent_workers=False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None)
    full = FullTileDataset(cfg, store, spec, a.source, length=n)

    if cfg.channels_last and n_gpu:
        model = model.to(memory_format=torch.channels_last)
    if n_gpu > 1:
        model = Replicated(model, [torch.device(f"cuda:{i}") for i in range(n_gpu)])
        print(f"[env] one model copy per GPU — every batch split {n_gpu} ways")

    res: dict = {"checkpoint": str(a.ckpt), "model_code": str(_CODE), "store": str(d),
                 "tiles_scored": n, "store_tiles": len(store),
                 "config": safe_config_dict(cfg), "preproc": spec.to_dict()}
    tag = f"test_{a.source}_{a.split}"
    native_sharp = hasattr(Evaluator, "add_spatial")

    for suffix, on, tta in (("plain", not a.skip_plain, False),
                            ("tta", cfg.tta and not a.skip_tta, True)):
        if not on:
            continue
        t0 = time.time()
        if suffix == "plain":
            # tapped: sharpness for evaluators without it, and sigma calibration
            # (sparsification / AUSE) whenever the model returns b_std
            tap = _Tap(dl, cfg.max_valid_height_m, sharp=not native_sharp)
            res[f"{tag}_plain"] = evaluate(tap.wrap(model), tap, cfg, device, use_tta=False)
            if tap.sharp is not None:
                res[f"{tag}_plain"].update(tap.sharp.result())
                res["sharpness"] = tap.sharp.result()
            unc = tap.sparse.result()
            if unc:
                res[f"{tag}_plain"]["uncertainty"] = unc
        else:
            res[f"{tag}_{suffix}"] = evaluate(model, dl, cfg, device, use_tta=tta)
        print(f"[test] {tag}_{suffix:<6} {format_line(res[f'{tag}_{suffix}'])}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        if suffix == "plain" and res[f"{tag}_plain"].get("uncertainty"):
            print(f"[test] {tag}_sigma  {sparse.format_line(res[f'{tag}_plain']['uncertainty'])}",
                  flush=True)
        (out_dir / "test_metrics.json").write_text(json.dumps(res, indent=2))

    if a.sliding_tiles:
        t0 = time.time()
        # A seeded sample, not the sorted-stem prefix (one city).  Picked here,
        # not in sliding_eval, because `--model_code ../v3` runs v3's own copy of
        # that; seed 42 = v5's val_sample_seed, so train.py draws the same tiles.
        k = min(a.sliding_tiles, n)
        idx = np.sort(np.random.default_rng(42).permutation(n)[:k]).tolist()
        res[f"{tag}_sliding_tta"] = sliding_eval(
            model, torch.utils.data.Subset(full, idx), cfg, spec, device,
            tta=cfg.tta, max_tiles=k)
        print(f"[test] {tag}_sliding_tta {format_line(res[f'{tag}_sliding_tta'])}"
              f"  ({time.time() - t0:.0f}s)")

    if a.qualitative or a.qual_stems:
        t0 = time.time()
        base = model.replicas[0] if hasattr(model, "replicas") else model
        res["qualitative"] = qualitative(base, ds, cfg, device, out_dir, a.qualitative,
                                         a.qual_stems, a.qual_seed)
        q = res["qualitative"]
        print(f"[qual] {len(q['tiles'])} tiles -> {out_dir / 'qual'}; shadow IoU "
              f"pred {_fmt(q.get('shadow_iou_pred_mean'))} vs GT "
              f"{_fmt(q.get('shadow_iou_gt_mean'))} on {q['n_informative']} tiles "
              f"({time.time() - t0:.0f}s)")

    (out_dir / "test_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"[out] {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
