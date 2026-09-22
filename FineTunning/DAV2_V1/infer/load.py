"""Rebuild a network + its preprocessing contract from a checkpoint.

Lifted out of v4's `infer/predict.py`, which this variant does not ship: that
module's job is the georeferenced CLI and it imports `geo.calibrate`, while the
only thing the training/eval path needs from it is this loader.

The point of the function is that the encoder id, tile size, patch and
canonical GSD come from the **checkpoint**, not from whatever `config.py`
currently says.  If a checkpoint can be loaded, its inputs can be reproduced
exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import parse_config
from dwdata.preprocess import PreprocSpec


def load_model(ckpt: str, device, hf_token: str = ""):
    """Rebuild the network and its preprocessing spec from a checkpoint."""
    from models.heads import DepthWizardNet

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    spec = PreprocSpec.from_dict(ck["preproc"]) if "preproc" in ck else PreprocSpec()
    if "preproc" not in ck:
        print("[infer] checkpoint has no preproc block — using DAV2/V1 defaults; "
              "metric scale may be wrong")
    if spec.version != PreprocSpec.version:
        print(f"[infer] !! checkpoint preproc version {spec.version!r} != "
              f"{PreprocSpec.version!r}; this checkpoint was trained under a "
              f"different input contract")

    cfg = parse_config([])
    for k, v in (ck.get("config") or {}).items():
        if hasattr(cfg, k) and not isinstance(getattr(cfg, k), tuple):
            setattr(cfg, k, v)
    cfg.hf_token = hf_token or cfg.hf_token
    cfg.encoder_model_id = spec.encoder_model_id
    cfg.tile_size = spec.tile_size
    cfg.encoder_patch = spec.patch
    cfg.canonical_gsd_m = spec.canonical_gsd_m
    cfg.grad_checkpoint_encoder = False

    model = DepthWizardNet(cfg).to(device).eval()
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    enc_missing = [m for m in miss if m.startswith("encoder.model.")]
    other_missing = [m for m in miss if not m.startswith("encoder.model.")]
    print(f"[infer] {ckpt}: epoch={ck.get('epoch')} "
          f"encoder_in_ckpt={ck.get('encoder_included', False)}")
    if other_missing:
        print(f"[infer] WARNING: {len(other_missing)} decoder/head tensors missing "
              f"from the checkpoint, e.g. {other_missing[:3]}")
    if enc_missing and not ck.get("encoder_included", False):
        print(f"[infer] encoder+neck loaded from the hub ({len(enc_missing)} "
              "tensors) — expected when the encoder stayed frozen")
    if unexp:
        print(f"[infer] {len(unexp)} unexpected tensors ignored")
    return model, spec, cfg
