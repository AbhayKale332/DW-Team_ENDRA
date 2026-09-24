DAV2_V1 — recover from run-1 (phase 1) issues and make phase 2 resumable
Context
Run 1 (FineTunning/DAV2_V1/Logs/da-v2-testing.log, Kaggle 2×T4) finished cleanly. All 24/24 epochs ran in 385 min of training (426 min in total). Best centre-crop val RMSE was 3.562 m at e24, and sliding+TTA gave 3.432 m. AMP was healthy: 0–3 skipped steps per epoch, and the loss scale stayed between 1e3 and 4e3. The logs show these problems. The first four would sabotage phase 2 if we simply re-ran with --resume:

Resuming as-is trains nothing. last_full.pt is saved at epoch 24, so start_epoch = 25 > --epochs 24 and the loop is skipped. The run goes straight to finals and exports.
Raising --epochs gives an LR spike instead of a clean continuation. The cosine is fully annealed: lr was 3.00e-06, which is the lr_scale floor. Progress is max(epoch/epochs, elapsed/max_minutes), so with --epochs 40 progress jumps to 0.6 at e25. The LR then jumps from 3e-6 to ~1.2e-4 (about 40×) in one step with no warmup, on already-converged weights.
best.pt is not carried into the new session. A fresh Kaggle session starts with an empty /kaggle/working/outputs/dav2-v1, but best = 3.562 m is restored from the checkpoint. If phase 2 never beats 3.562 on GAMUS val, no best.pt is written. The final eval then runs on live non-EMA weights and the ONNX export is skipped.
The [sched] line was wrong. It predicted "cosine reaches 50 %, anneal does NOT complete, resume to finish it". It only looked at the clock (480/960), but the epoch fraction was what actually finished the schedule. That line is what made a resume look necessary.
The held-out test was never scored. test_sources="" although gamus/test was linked, so no test_gamus_test_* number exists. The notebook tells us to quote exactly that number.
india_labeled (train+val, packed, with seg) was linked but unused. It was not in --datasets, so it was neither trained on nor reported as a secondary val set.
The mean-teacher was silently off. w_consistency=1.0, but there is no india_unlabeled store: no [mt] line and no con= column. The log never says so, because the warning only fires when that store is named in --datasets.
The ONNX export printed a traceback. Opset 17 was requested, but conversion fails ("No Adapter To Version 17 for Resize"). The exporter fell back to 18 and verified fine (max diff 0.0000 m), so the default should just be 18.
These are known limits and are out of scope:

GSD jitter never goes above 0.44–0.79 m because of the source tile extents; covering the coarse end needs DFC23 or another coarse source packed.
DFC23 is attached raw and unpacked.
nproc: 1 in the check cell is only a reporting artefact: GNU nproc honours the exported OMP_NUM_THREADS=1.
Where india_labeled comes from: it is already a Kaggle dataset, abhaydkale232/depthwizard-india-labeled. It was attached to run 1 and link mounted it (log lines 116–117 and 125–126), so phase 2 needs no new download. It was packed by prepare_data.py --datasets india_labeled --india_dir <dir> (prepare_india_labeled, dwdata/india.py). Per dwdata/india.py, the intended source is the DFC2023 Track 2 New Delhi tiles, downloaded by hand from IEEE DataPort. Each split is a single shard, so the store is small; weight 2 will oversample it heavily. The [data] india_labeled/train: N tiles line in the phase-2 log will give the real tile count.

Decisions (from you): phase 2 adds india_labeled (sampler weight 2) and trains 16 more epochs (e25–e40). best.pt is still selected on GAMUS val, so results stay comparable with v1–v4. India val is reported every epoch.

Design: numbered training phases
Add a phase id to the schedule, so each phase is its own warmup and cosine. Re-running the same phase after a crash then continues that phase's cosine rather than restarting it.

config.py
New fields, in the multi-session block:
phase: int = 1
phase_lr_mult: float = 0.3: peak LR of a phase > 1, relative to the run's saved base_lrs. Run 1's peaks were 3e-4 for the decoder and 6e-5 for the encoder top.
phase_warmup_frac: float = 0.05
In phase > 1, --max_minutes is that phase's budget, measured from the phase start. It is no longer the all-sessions total. Document this next to session_minutes.
validate(): assert phase >= 1 and 0 < phase_lr_mult <= 1.
onnx_opset: int = 18.
train.py
Pure helper next to lr_scale(), so the schedule can be unit-tested: phase_progress(epoch, step, n_steps, phase_start, epochs, phase_elapsed_min, max_minutes) -> float. The per-step LR becomes b * lr_mult * lr_scale(progress, warm). Replaces lines 632–638. Phase 1 uses lr_mult=1 and warm=cfg.warmup_frac, so its behaviour is identical to today.
Resume block (around line 525). Read ck.get("phase"); old checkpoints default to {id: 1, start_epoch: 1, elapsed_min: ck["elapsed_min"], lr_mult: 1.0, warmup_frac: cfg.warmup_frac}.
cfg.phase > saved.id: start a new phase at start_epoch, with phase elapsed = 0, lr_mult=cfg.phase_lr_mult and warm=cfg.phase_warmup_frac. Print [phase] 2 starts at epoch 25 of 40: peak lr … warmup … from floor.
cfg.phase == saved.id: continue the saved phase with its own start epoch, elapsed time, lr_mult and warmup. This makes a crashed phase 2 resumable by re-running the same command.
cfg.phase < saved.id: raise an error with a clear message.
Spike guard. Same phase, start_epoch <= epochs, and the saved schedule already annealed: detect it from the checkpoint's opt.param_groups[0]["lr"] / base_lrs[0] <= 1.5/final_div, which works on the run-1 checkpoint. In that case, auto-bump to saved.id + 1 and print a loud [phase] notice rather than spiking the LR.
Nothing left to train. When start_epoch > cfg.epochs, print [resume] checkpoint already finished epoch 24 of --epochs 24 — nothing to train; running final eval/exports only. To continue: raise --epochs and pass --phase 2.
best.pt carry-over, rank 0, right after the resume load. If out_dir/best.pt is missing and Path(cfg.resume).parent/"best.pt" exists (and is a different file), use shutil.copy2 and print [resume] carried best.pt (3.562 m, e24) from ….
Wall-clock stop (line 815) and the [sched] print (lines 560–573) use the phase clock instead of t0. t0 stays the all-sessions total for elapsed_min and metrics.json.
Reword the sched line to "clock alone would reach X %; the epoch fraction ends it sooner if all N epochs fit". It should print the anneal-incomplete warning only when the projected epoch time cannot finish the phase, not merely because max_minutes > session_minutes.
_save_full(): add a "phase": {id, start_epoch, elapsed_min, lr_mult, warmup_frac} block. Also save rec["lr_start"] and rec["lr_end"] (decoder group) per epoch in history, so metrics.json and the tests can see the schedule.
dwdata/loaders.py
build_unlabeled_loader: when w_consistency > 0, unlabeled_source is set and the store is missing, always print [data] w_consistency=… but no <source> store — mean-teacher OFF, even if the source is not in --datasets. One-line change at line 90.
run_kaggle.sh
New phase2 subcommand. It uses the same perf flags as train, and "$@" still overrides anything.
RESUME="${DW_RESUME:-}". If that is empty, find last_full.pt under $KIN with find -L. Exit with the list of candidates if it finds 0 or more than 1.
torchrun … train.py --resume "$RESUME" --phase 2 --phase_lr_mult 0.3 --phase_warmup_frac 0.05
--datasets gamus,synrs3d_g05,synrs3d_g1,india_labeled. gamus stays first, so it remains the primary val; India val is picked up automatically by build_aux_val_loaders.
--epochs 40 --max_minutes 480 --session_minutes 480 --test_sources gamus:test --save_full_state true --make_zip false
Fix the comment block in train: a resume needs a new --phase once the schedule has finished, and point it to phase2. Add phase2 to the usage line.
Budget comment: 16 × (~16.6 + ~1 min India eval) ≈ 280 min training. Add ~40 min of val finals and ~85 min for gamus/test (2861 tiles plain + TTA, plus 400 sliding). With setup, the total is about 7 h, well inside Kaggle's 12 h.
dav2_train.ipynb
Cell 3 comment: say that india_labeled must be attached (it already is).
Cell 5: split into run-1 (train) and phase-2 (!bash run_kaggle.sh phase2) instructions. Phase 2 also needs the run-1 notebook output attached as an input.
The "If the session dies" markdown: re-run the same phase2 command with DW_RESUME pointing at the phase-2 last_full.pt. The phase id makes this a continuation.
Tests: tests/test_full_state_resume.py
Reuse that file's existing store / tmp_path / monkeypatch pattern (2-epoch stub runs).

phase_progress / LR maths, pure. A phase-2 start gives lr ≈ base·mult·floor at the first step and reaches the peak at the end of warmup. Phase-1 values are unchanged from the current formula.
Resume a finished 2-epoch run with --phase 2 --epochs 4. history[2]["lr_start"] is about the floor, not the spike, and ck2["phase"] == {id: 2, start_epoch: 3, …}.
Re-run phase 2 from a phase-2 checkpoint with --phase 2. The phase start stays 3, so it continues rather than restarting.
Extend an annealed run without --phase: it auto-bumps (capsys sees [phase]) and there is no LR spike.
Resume into a fresh output_dir where no epoch beats best. best.pt exists there and has the old epoch.
start_epoch > epochs: the "nothing to train" line is printed and history is untouched. This extends the existing test_exports_only… pattern.
test_onnx.py: check whether any assertion pins opset 17 and update it to 18.
Verification
Locally: cd FineTunning/DAV2_V1 && python -m pytest -q. The full suite must stay green, including test_full_state_resume.py and test_onnx.py.
bash -n run_kaggle.sh, plus a dry parse check that phase2 builds the expected command.
Commit and push. The notebook git clones the repo, so Kaggle only sees pushed code.
On Kaggle, attach the run-1 notebook output and run cells 1–4, then !bash run_kaggle.sh phase2. In run.log, check for:
[resume] … full-state resume and [resume] carried best.pt (3.562 m …)
[phase] 2 starts at epoch 25 of 40, with e25 s0 lr ≈ 9e-07 rising to ≈ 9e-05 by about e25 s800, then falling. No jump to 1e-4 at s0.
[data] india_labeled/train: … and [data] val+ : … india_labeled — reported, not selected on, then eval eNN [india_labeled] … every epoch
[data] test: 2861 tiles from gamus/test, with [test] test_gamus_test_sliding_tta … and quote this: at the end
[onnx] with no opset-17 traceback