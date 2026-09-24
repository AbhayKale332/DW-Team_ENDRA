"""v5 = v4-modal with the five deltas that cost it 0.76 m reverted to v3.

Why these five and not others: `metrics.json` decomposes the v3->v4 regression
by height stratum, and 85.5 % of it lands in 0-2 m alone (see the table in
README, or re-derive with tools/compare_runs.py). Every knob below is one that
moves the loss mass off that stratum. The knobs that do NOT touch it -- the
extra datasets, the ONNX stage, the landscape metrics -- are kept, because they
are the parts of v4 that earned their place.

Both prior runs used the same encoder, preproc, decoder_dim, n_bins and
crops_per_epoch, so none of those appear here.

Usage, from 02_train.ipynb:

    from v5_flags import FLAGS as V5
    modal_app.argv(**V5)

Read with modal_app.FLAGS beside it: this dict is a full replacement, not an
overlay.
"""

# --- the reverts ----------------------------------------------------------
# Each entry is (v3 value, v4 value) and the reason it is back at v3.
REVERTS = {
    # 0.5 -> 0.7 raised the balancer's tall:flat weight ratio from 2.37x to
    # 3.35x and dropped the 0-2 m weight from 0.675 to 0.558. v4 bought
    # -0.17 MSE on 10-20 m and -0.04 on 20 m+ for +3.90 on 0-2 m. Bad trade.
    "stratum_balance_beta": "0.5",
    # 5 -> 8 was inert: at these stratum frequencies the clip never binds in
    # either config. Reverted for provenance, not for effect -- do not expect
    # this one to move the number.
    "stratum_weight_clip": "5.0",
    # v3 had no soft bin targets. The Gaussian label at sigma=1.5 bins carries
    # ~1.8 nats of irreducible CE, and v4's bin CE ended at 2.53 vs v3's 0.19.
    # 0 restores hard targets. If head B collapses again (CE < 0.15, which is
    # what losses.py:13 records v2 doing), reintroduce at 0.5, not 1.5.
    "bin_soft_sigma": "0.0",
    # The entropy floor pushes the marginal bin distribution toward uniform,
    # which is directly opposed to a ground plane that wants one sharp mode at
    # zero. v3 had no such term and its ground was 1.9x better.
    "w_bin_entropy": "0.0",
    # 16 -> 0 (0 = all of it, encoder.py:83). v3 trained 303.1M encoder params;
    # v4 trained 221.6M and froze the patch embedding plus blocks 1-8, which is
    # where a satellite-pretrained ViT keeps its texture->structure mapping.
    # This is the one revert that costs VRAM: v3 peaked at 80345 MiB with
    # batch_size 16 + grad_accum 2. Hence the batch change below.
    "encoder_unfreeze_blocks": "0",
    # v3's effective batch was 16x2 = 32 at a full-encoder memory profile that
    # already peaked the card. 24x1 = 24 will not fit once the encoder is fully
    # unfrozen, so take v3's shape exactly.
    "batch_size": "16", "grad_accum": "2",
    # 0.9 -> 0.8: with all 24 blocks training, the lower blocks need the deeper
    # decay v3 used, or block 1 sees 6e-5 * 0.9^23 = 5.6e-6 instead of 3.5e-7.
    "llrd": "0.80",
}

# --- kept from v4 ---------------------------------------------------------
# These are v4 additions that the stratum decomposition exonerates. Listed
# explicitly so a future reader knows they were considered and kept.
KEPT = {
    # dfc23 and india_labeled are the only non-GAMUS/SynRS3D supervision there
    # is, and dfc23 is the only genuinely held-out domain in the whole project
    # (4.97 m, never in the sampler at eval time). Keep both.
    "datasets": "gamus,synrs3d_g1,synrs3d_g05,dfc23_g050,india_labeled",
    "sampler_weights": "gamus:3,dfc23:2,synrs3d_g05:1,synrs3d_g1:0.5,india_labeled:1",
    "max_valid_height_m": "150",
    "w_seg": "0.2",
    "test_sources": "gamus:test", "test_tiles": "0", "test_sliding_tiles": "400",
}

# --- fixes ----------------------------------------------------------------
FIXES = {
    # v4's val was `first 400 of 859, prefix not sample` and came out 87.5 %
    # urban against a 57.6 % urban test set. best.pt was selected on that.
    # v5 loaders.py draws a seeded random 400 instead (`sample_indices`), and
    # the log line reads "random 400 of 859, seed 42".
    "val_sample_seed": "42",
    #
    # opset 17 conversion failed (`axes_input_to_attribute.h:55`) and the
    # exporter silently kept 18, while depthwizard.onnx.json recorded 17.
    # Ask for what we actually get.
    "onnx_opset": "18",
}

# --- unchanged infrastructure --------------------------------------------
INFRA = {
    "amp_dtype": "bf16",
    "grad_checkpoint_encoder": "false", "grad_checkpoint_decoder": "false",
    "eval_batch_mult": "2", "num_workers": "12", "prefetch_factor": "6",
    "compile_model": "false",
    "epochs": "26", "eval_every": "1",
    # v3 reached its floor in 26 epochs / 118 min. v4 spent 30 epochs / 119 min
    # and was flat from epoch 22 (3.780 -> 3.755 over the last 8). Do not buy
    # more epochs until the curve stops flattening.
    "max_minutes": "240", "session_minutes": "0",
    "save_full_state": "true", "full_state_every": "5",
    "make_zip": "false",
}

# --- the v5 additions (plan Steps 1-3 and 5) --------------------------------
V5 = {
    # Its own results directory: v4m's best.pt / metrics.json stay untouched.
    "output_dir": "/results/v5",
    # Step 2: full-resolution RGB stem + convex 2x upsampling.  Zero-initialised,
    # so epoch 0 computes what v4 did; `detail_branch false` is the ablation.
    "detail_branch": "true",
    "detail_dim": "64",
    # Step 1: DFC23 / India nDSMs are ~2 m stereo on 0.5 m pixels.  Scored only
    # after 4x4 average pooling, and kept out of the normal / flatness / bin
    # terms -- the fix for v4's blobs.  `coarse_pool` and the vegetation pair
    # below are Step 0 outputs: run tools/audit_labels.py and pass its
    # audit.json through `with_audit()` rather than editing these by hand.
    "coarse_label_sources": "dfc23,india_labeled",
    "coarse_pool": "4",
    # The label resolution in metres; the pool is sized per sample from it
    # (models/losses.py coarse_pool_px).  4 px was only right at 0.5 m.
    "coarse_label_m": "2.0",
    "w_coarse": "1.0",
    "coarse_mask_veg": "false",          # provisional until the audit says trees sit at ~0 m
    "coarse_veg_exg": "0.05",
    # Step 1: the Cartosat look.  MERGED = 0.6 m luminance + 1.6 m colour
    # (1.6/0.6 = 2.67, bracketed); 10 % grayscale for PAN-only uploads.
    "aug_pansharp_p": "0.5",
    "aug_pansharp_lo": "2.0",
    "aug_pansharp_hi": "3.3",
    "aug_gray_p": "0.1",
    # Head B's bins must cover the labels: max_valid_height_m is 150 above, and
    # with the default 120 every 120-150 m pixel landed in the top bin.
    "bin_max_m": "150",
    # Keep the decoder's AdamW state at the unfreeze and ramp the encoder LR in
    # over one epoch (config.py `unfreeze_warmup_epochs`).
    "unfreeze_warmup_epochs": "1.0",
}

FLAGS = {**INFRA, **KEPT, **REVERTS, **FIXES, **V5}

# The Step 0 outputs, and only those, may be overridden from an audit.
AUDIT_KEYS = ("coarse_pool", "coarse_label_m", "coarse_mask_veg", "coarse_veg_exg",
              "aug_pansharp_lo", "aug_pansharp_hi")


def with_audit(path: str | None = None, flags: dict | None = None) -> dict:
    """FLAGS with the Step 0 values from `tools/audit_labels.py`'s audit.json.

    Any other key in the audit is ignored: the audit measures labels and
    radiometry, it does not get to change the schedule or the datasets.
    """
    import json as _json
    import os as _os
    from pathlib import Path as _P

    out = dict(FLAGS if flags is None else flags)
    path = path or _os.environ.get("DW_AUDIT", "")
    if not path:
        return out
    rep = _json.loads(_P(path).read_text())
    sug = rep.get("flags", rep)
    for k in AUDIT_KEYS:
        if k in sug:
            out[k] = str(sug[k])
    return out

# Deliberately absent, and why:
#   w_consistency / unlabeled_source / teacher_ema / consistency_*
#     v4 shipped these in config.json and none of them ran: `india_unlabeled`
#     was not in --datasets, loaders.py:89 got no store, and the warning at
#     loaders.py:92 is gated on the source being in dataset_list() so it
#     printed nothing. No `con=` term appears in any of v4's 1665 log lines.
#     Leave them at their config.py defaults and do NOT put them in a shipped
#     config until the store exists -- a config that claims a branch that
#     never ran is worse than no config.

if __name__ == "__main__":
    import json
    import sys
    from dataclasses import fields
    from pathlib import Path

    # Every key must be a real v5 config field: parse_config ends in
    # parse_known_args, so a typo is silently dropped and the run uses the
    # default you thought you had overridden.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v5"))
    from config import Config, parse_config

    names = {f.name for f in fields(Config)}
    bad = sorted(set(FLAGS) - names)
    if bad:
        raise SystemExit(f"not v5 config fields: {bad}")
    f = with_audit(sys.argv[1] if len(sys.argv) > 1 else None)
    cfg = parse_config([x for k, v in f.items() for x in (f"--{k}", v)]
                       + ["--data_root", "/scratch/dwdata"])
    print(json.dumps(f, indent=2))
    print(f"[ok] {len(f)} flags parse; detail_branch={cfg.detail_branch} "
          f"coarse={cfg.coarse_label_sources!r}@{cfg.coarse_label_m or cfg.coarse_pool}"
          f"{' m' if cfg.coarse_label_m else ' px'} bin_max={cfg.bin_max_m} "
          f"val_sample_seed={cfg.val_sample_seed}")
