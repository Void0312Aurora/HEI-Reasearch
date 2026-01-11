"""
L6 Gate Verification (manual script; not collected by pytest by default).

Purpose:
  - Evaluate a Stage6 (A5-T1) adapter checkpoint: vision→core alignment to language prototypes.
  - Report clean and noisy nearest-prototype accuracy in q-space.

Run (recommended):
  /home/void0312/miniconda3/envs/PINNs/bin/python HEI/tests/verify_L6_gate.py

Env knobs:
  HEI_L6_GATE_CKPT          Path to Stage6 checkpoint (default: checkpoints/curriculum/stage6_final.pt)
  HEI_STAGE5_CKPT           Override Stage5 ckpt path (otherwise use metadata inside Stage6 adapter)
  HEI_STAGE6_TEXT_FILE      Text file for Stage5 loader (default: HEI/tests/dummy_corpus.txt)
  HEI_DEVICE                cpu|cuda (default: from adapter config or auto)

  Gate params:
    HEI_L6_GATE_PROFILE      base|robust
    HEI_L6_GATE_ACC          Clean acc threshold
    HEI_L6_GATE_NOISY_ACC    Noisy acc threshold
    HEI_STAGE6_NOISE_STD     Vision noise std for eval
    HEI_STAGE6_EVAL_REPEATS  Noisy eval repeats per latent state
"""

from __future__ import annotations

import os
import sys
from dataclasses import fields
from typing import Any, Dict


def _torch_is_real() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    import torch

    return hasattr(torch, "Tensor") and hasattr(torch, "nn") and hasattr(torch, "__version__")


def _filter_stage6_config_dict(d: Dict[str, Any], cfg_cls) -> Dict[str, Any]:
    allowed = {f.name for f in fields(cfg_cls)}
    return {k: v for k, v in d.items() if k in allowed}


def main() -> int:
    if not _torch_is_real():
        print("PyTorch is not available in this Python. Run this script in the PINNs env.")
        return 2

    import torch

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(repo_root, "HEI"))

    from curriculum.stage6_plural_interfaces import Stage6Config, Stage6Trainer

    ckpt = os.getenv("HEI_L6_GATE_CKPT", "checkpoints/curriculum/stage6_final.pt")
    if not os.path.exists(ckpt):
        print(f"Stage6 checkpoint not found: {ckpt}")
        return 2

    payload = torch.load(ckpt, map_location="cpu")

    cfg = Stage6Config()
    if isinstance(payload, dict) and isinstance(payload.get("config", None), dict):
        cfg_dict = _filter_stage6_config_dict(payload["config"], Stage6Config)
        try:
            cfg = Stage6Config(**cfg_dict)
        except TypeError:
            # Fallback to defaults if the stored config is stale.
            cfg = Stage6Config()

    # Resolve Stage5 ckpt path.
    stage5_override = os.getenv("HEI_STAGE5_CKPT", "")
    if stage5_override:
        cfg.stage5_ckpt = stage5_override
    elif isinstance(payload, dict) and isinstance(payload.get("stage5_ckpt", None), str):
        cfg.stage5_ckpt = payload["stage5_ckpt"]

    # Stage5 loader requires a text file, but Stage6 eval does not depend on its content.
    cfg.stage5_text_file = os.getenv("HEI_STAGE6_TEXT_FILE", getattr(cfg, "stage5_text_file", "HEI/tests/dummy_corpus.txt"))

    # Device override.
    cfg.device = os.getenv("HEI_DEVICE", getattr(cfg, "device", "cpu"))

    # Gate overrides.
    if os.getenv("HEI_L6_GATE_PROFILE") is not None:
        cfg.gate_profile = str(os.getenv("HEI_L6_GATE_PROFILE", cfg.gate_profile)).lower()
    if os.getenv("HEI_L6_GATE_ACC") is not None:
        cfg.gate_acc_clean = float(os.getenv("HEI_L6_GATE_ACC", cfg.gate_acc_clean))
    if os.getenv("HEI_L6_GATE_NOISY_ACC") is not None:
        cfg.gate_acc_noisy = float(os.getenv("HEI_L6_GATE_NOISY_ACC", cfg.gate_acc_noisy))
    if os.getenv("HEI_STAGE6_NOISE_STD") is not None:
        cfg.noise_std = float(os.getenv("HEI_STAGE6_NOISE_STD", cfg.noise_std))
    if os.getenv("HEI_STAGE6_EVAL_REPEATS") is not None:
        cfg.eval_repeats = int(os.getenv("HEI_STAGE6_EVAL_REPEATS", cfg.eval_repeats))

    trainer = Stage6Trainer(cfg)
    trainer.load_checkpoint(ckpt)

    gate = trainer.evaluate_L6_gate()
    print("=== L6 Gate ===")
    print("Profile:", gate["profile"])
    print("Clean:", gate["clean"])
    print("Noisy:", gate["noisy"])
    print("Thresholds:", gate["thresholds"])
    print("Passed:", gate["passed"], gate["passed_components"])

    return 0 if bool(gate["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

