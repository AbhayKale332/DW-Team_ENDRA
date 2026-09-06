# FineTunning

Fine-tuning experiments. Code is written locally, runs on **Kaggle** GPUs.

## Layout

```
FineTunning/
├── pyproject.toml          # local dev env only (uv)
├── v1/                     # experiment version 1
│   ├── main.py             # entrypoint: Step 1 install deps -> Step 2 train
│   ├── config.py           # all knobs for this version
│   ├── train.py            # the fine-tuning run
│   ├── requirements-kaggle.txt   # deps installed into the Kaggle runtime
│   └── utils/install_dependency.py
├── v2/ ...                 # copy v1/ to start a new version
```

## Run on Kaggle

New notebook, GPU on, Internet on. In a cell:

```python
!git clone https://github.com/<you>/SIH.git
%cd SIH/FineTunning/v1
!python main.py
```

- `main.py` **Step 1** pip-installs `requirements-kaggle.txt` into the Kaggle
  environment (`sys.executable -m pip install`). torch is left alone — Kaggle
  ships it.
- `main.py` **Step 2** imports `train.py` and runs it.

Flags: `python main.py --skip-install` (deps already installed this session),
`python main.py --install-only` (just Step 1).

Set `HF_TOKEN` as a Kaggle Secret if the base model is gated.

## Local dev

```bash
cd FineTunning
uv sync          # creates .venv (no heavy deps by default)
```

## New version

```bash
cp -r v1 v2      # then edit v2/config.py
```
