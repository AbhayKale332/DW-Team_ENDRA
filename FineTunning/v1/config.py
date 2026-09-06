"""Central config for this fine-tuning version (v1).

Copy the whole `vN/` folder to start a new version and tweak here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_output_dir() -> str:
    # Kaggle only persists things written under /kaggle/working
    if Path("/kaggle/working").exists():
        return "/kaggle/working/outputs/v1"
    return str(Path(__file__).resolve().parent / "outputs")


@dataclass
class Config:
    version: str = "v1"

    # --- model / data ---
    base_model: str = "meta-llama/Llama-3.2-1B"
    dataset_name: str = "yahma/alpaca-cleaned"
    max_seq_len: int = 1024

    # --- training ---
    epochs: int = 1
    learning_rate: float = 2e-4
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    seed: int = 42

    # --- LoRA ---
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # --- io ---
    output_dir: str = field(default_factory=_default_output_dir)
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))


CONFIG = Config()
