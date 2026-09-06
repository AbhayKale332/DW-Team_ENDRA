"""Step 2+: the actual fine-tuning run.

Imports of heavy libs (torch/transformers/...) happen INSIDE run() on purpose,
so this module is importable before Step 1 has installed anything.
"""

from __future__ import annotations

from config import CONFIG


def run() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = CONFIG
    print(f"[train] version        : {cfg.version}")
    print(f"[train] cuda available : {torch.cuda.is_available()}")
    print(f"[train] base model     : {cfg.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, token=cfg.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        token=cfg.hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    dataset = load_dataset(cfg.dataset_name, split="train")
    print(f"[train] dataset rows   : {len(dataset)}")

    # TODO: build SFTTrainer / Trainer here and call trainer.train()
    # TODO: model.save_pretrained(cfg.output_dir); tokenizer.save_pretrained(cfg.output_dir)
    print("[train] TODO: wire up the trainer")
