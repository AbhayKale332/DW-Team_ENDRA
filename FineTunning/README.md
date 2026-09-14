# FineTunning

Fine-tuning experiments. Code is written locally, runs on **Kaggle** GPUs.

> **v1 = DepthWizard Phase 0 spike.** DINOv3-SAT (frozen) → DPT decoder →
> metric-nDSM head, trained on a GAMUS subset for a first val RMSE.
> Everything is in one runnable file: **`v1/kaggle_phase0.py`**.
> See **[`v1/README_PHASE0.md`](v1/README_PHASE0.md)**.

## Layout

```
FineTunning/
├── pyproject.toml               # local dev env only (uv)
├── v1/                          # experiment version 1 = DepthWizard Phase 0
│   ├── kaggle_phase0.py         # the whole spike: GAMUS -> model -> train -> eval
│   ├── eval_imele_on_gamus.py   # IMELE baseline row, same val split + metrics
│   ├── README_PHASE0.md         # how to run it
│   ├── main.py / train.py / config.py   # shims -> kaggle_phase0.main()
│   ├── requirements-kaggle.txt
│   └── utils/install_dependency.py
├── v2/ ...                      # copy v1/ to start a new version
├── v4/                          # the deliverable, tuned for 1x H100 (run_lightning.sh)
└── V4_Kaggle/                   # v4 + DDP + full-state resume, for GPU T4 x2
    ├── run_kaggle.sh            # check | prepare | link | smoke | train | finalize
    └── kaggle_train.ipynb       # the four notebook cells
```

**On Kaggle's free GPU T4 x2, use `V4_Kaggle/`, not `v4/`.** It is a full copy of
`v4/` with three changes — DistributedDataParallel across the two cards,
full-state multi-session resume (`last_full.pt`, so a dead 12 h session is
recoverable), and a flag profile that fits 16 GB of Turing. Same data layout as
`v4/`; only the paths differ. `v4/` is unchanged and remains the H100 runbook.
See [`V4_Kaggle/README.md` §5](V4_Kaggle/README.md).

## Run on Kaggle

New notebook, **Accelerator = "GPU T4 x2"**, Internet on. DINOv3-SAT is gated —
accept its license and add an `HF_TOKEN` Kaggle Secret first
(see [`v1/README_PHASE0.md`](v1/README_PHASE0.md)). Then, in a cell:

```python
!git clone https://github.com/<you>/SIH.git
%cd SIH/FineTunning/v1
!python kaggle_phase0.py          # or: !python main.py
```

- `kaggle_phase0.py` self-installs the few missing deps (transformers,
  huggingface_hub, h5py); torch is left alone — Kaggle ships it.
- `main.py` is equivalent: **Step 1** installs `requirements-kaggle.txt`, **Step 2**
  runs `train.run()` → `kaggle_phase0.main()`. Flags `--skip-install`,
  `--install-only`; any other `--flag` is forwarded to `kaggle_phase0`.

## Local dev

```bash
cd FineTunning
uv sync          # creates .venv (no heavy deps by default)
```

## New version

```bash
cp -r v1 v2      # then edit v2/kaggle_phase0.py::Config
```
