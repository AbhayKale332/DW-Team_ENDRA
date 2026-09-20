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
    # A prefix is not a sample; loaders.py:244 says so in the log line it
    # prints and then does it anyway.
    #   >> THIS FLAG DOES NOT EXIST YET. See README "val split" -- loaders.py
    #   >> needs a seeded permutation before this run is worth launching.
    # "val_sample_seed": "42",
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

FLAGS = {**INFRA, **KEPT, **REVERTS, **FIXES}

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
    import json, sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    print(json.dumps(FLAGS, indent=2))
