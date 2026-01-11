"""
L5 Gate Verification (manual script; not collected by pytest by default).

Purpose:
  - Evaluate Stage5 byte-level language checkpoint on a small slice of the corpus.
  - Report teacher-forced metrics and robustness to offline steps between tokens.
  - Report a simple open-loop collapse diagnostic (short-cycle + diversity proxy).

Run (recommended):
  /home/void0312/miniconda3/envs/PINNs/bin/python HEI/tests/verify_L5_gate.py

Env knobs:
  HEI_L5_GATE_CKPT            Path to checkpoint (default: checkpoints/curriculum/stage5_best.pt)
  HEI_L5_GATE_TEXT_FILE       Text file path (default: HEI/data/CLUE/CLUECorpusSmall.txt if exists)
  HEI_L5_GATE_MAX_LINES       Pretokenize first N non-empty lines (default: 2000)
  HEI_L5_GATE_EVAL_LINES      Eval split size (default: 200)
  HEI_L5_GATE_SEQ_LEN         Sequence length (default: 32)
  HEI_L5_GATE_BATCH           Eval batch size (default: 128)
  HEI_L5_GATE_EVAL_BATCHES    Number of eval batches (default: 2)
  HEI_L5_GATE_OFFLINE_STEPS   Offline steps between tokens (default: 2)

  Open-loop diagnostic:
    HEI_L5_GATE_OL_BATCH       Batch size (default: 64)
    HEI_L5_GATE_OL_LEN         Generation length (default: 96)
    HEI_L5_GATE_OL_TAIL        Tail length for period detection (default: 48)
    HEI_L5_GATE_OL_MAXP        Max short period to detect (default: 8)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any


def _torch_is_real() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    import torch

    return hasattr(torch, "Tensor") and hasattr(torch, "nn") and hasattr(torch, "__version__")


def _detect_period(seq_tail, *, max_period: int) -> "object":
    import torch

    tail = int(seq_tail.shape[1])
    periods = torch.zeros(seq_tail.shape[0], dtype=torch.long, device=seq_tail.device)
    unresolved = torch.ones(seq_tail.shape[0], dtype=torch.bool, device=seq_tail.device)
    for p in range(1, int(max_period) + 1):
        if p >= tail:
            break
        ok = (seq_tail[:, p:] == seq_tail[:, :-p]).all(dim=1)
        hit = unresolved & ok
        if hit.any():
            periods[hit] = p
            unresolved = unresolved & (~hit)
        if not unresolved.any():
            break
    return periods


def _open_loop_diag(trainer, *, seed: int, batch_size: int, length: int, tail: int, max_period: int) -> Dict[str, Any]:
    import torch

    trainer.eval()
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    b = trainer.eval_data.sample(batch_size=int(batch_size), generator=g)
    tokens = b["tokens"].to(device=trainer.config.device, non_blocking=True)
    B = int(tokens.shape[0])

    dq = int(trainer.config.dim_q)
    D = 2 * dq + 1
    state_flat = torch.zeros((B, D), device=trainer.config.device, dtype=torch.float32)
    prev_w = None

    entity = trainer.entity
    port = trainer.port
    evo = int(trainer.config.evolution_steps)

    # Start from BOS (first token in the pretokenized row).
    tok = tokens[:, 0]
    preds = []
    for _ in range(int(length)):
        u = port.encode(tok)
        chart_w = None
        for _ in range(evo):
            out = entity.forward_tensor(
                state_flat=state_flat,
                u_dict={"language": u},
                dt=float(trainer.config.dt),
                prev_chart_weights=prev_w,
                prediction_error=None,
                detach_next_prev_weights=True,
                compute_action=False,
                router_context=None,
                skip_free_energy=True,
            )
            state_flat = out["next_state_flat"]
            prev_w = out.get("next_prev_chart_weights", None)
            chart_w = out.get("chart_weights", None)

        x = state_flat if bool(getattr(port, "decode_state", False)) else state_flat[:, :dq]
        logits = port.decode(x, chart_weights=chart_w)
        tok = logits.argmax(dim=-1)
        preds.append(tok)

    seq = torch.stack(preds, dim=1)  # [B,L]
    tail = min(int(tail), int(seq.shape[1]))
    seq_tail = seq[:, -tail:]
    periods = _detect_period(seq_tail, max_period=int(max_period))
    locked = periods > 0

    # Diversity proxy: unique token count in tail.
    uniq = torch.tensor([int(torch.unique(seq_tail[i]).numel()) for i in range(B)], device=seq.device, dtype=torch.float32)
    repeat_rate = (seq_tail[:, 1:] == seq_tail[:, :-1]).to(dtype=torch.float32).mean()

    return {
        "lock_rate": float(locked.to(dtype=torch.float32).mean().item()),
        "mean_period": float(periods[locked].to(dtype=torch.float32).mean().item()) if bool(locked.any()) else 0.0,
        "uniq_tail_mean": float(uniq.mean().item()),
        "repeat_rate": float(repeat_rate.item()),
    }


def main() -> int:
    if not _torch_is_real():
        print("PyTorch is not available in this Python. Run this script in the PINNs env.")
        return 2

    import torch

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(repo_root, "HEI"))

    from curriculum.stage5_language import Stage5Config, Stage5Trainer

    ckpt = os.getenv("HEI_L5_GATE_CKPT", "checkpoints/curriculum/stage5_best.pt")
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        return 2

    default_text = "HEI/data/CLUE/CLUECorpusSmall.txt"
    text_file = os.getenv("HEI_L5_GATE_TEXT_FILE", default_text if os.path.exists(default_text) else "HEI/tests/dummy_corpus.txt")
    if not os.path.exists(text_file):
        print(f"Text file not found: {text_file}")
        return 2

    seq_len = int(os.getenv("HEI_L5_GATE_SEQ_LEN", "32"))
    max_lines = int(os.getenv("HEI_L5_GATE_MAX_LINES", "2000"))
    eval_lines = int(os.getenv("HEI_L5_GATE_EVAL_LINES", "200"))
    batch = int(os.getenv("HEI_L5_GATE_BATCH", "128"))
    eval_batches = int(os.getenv("HEI_L5_GATE_EVAL_BATCHES", "2"))
    offline_steps = int(os.getenv("HEI_L5_GATE_OFFLINE_STEPS", "2"))
    dt = float(os.getenv("HEI_L5_GATE_DT", "0.2"))
    evo = int(os.getenv("HEI_L5_GATE_EVO", "2"))

    config = Stage5Config(
        device=str(os.getenv("HEI_DEVICE", "cpu")),
        text_file=text_file,
        sequence_len=seq_len,
        dt=dt,
        evolution_steps=evo,
        max_lines=max_lines,
        eval_lines=eval_lines,
        eval_batches=eval_batches,
        eval_batch_size=batch,
    )

    trainer = Stage5Trainer(config)
    trainer.load_state_dict(torch.load(ckpt, map_location=config.device), strict=False)

    gate = trainer.evaluate_L5_gate(seed=0, offline_steps_between=offline_steps)
    print("=== L5 Gate ===")
    print("Profile:", gate.get("profile", "legacy"))
    print("TeacherForced:", gate["TeacherForced"])
    if gate["OfflineBetween"] is not None:
        print(f"OfflineBetween({offline_steps}):", gate["OfflineBetween"])
    if gate.get("ratios", None) is not None:
        print("OfflineRatios:", gate["ratios"])
    print("Thresholds:", gate["thresholds"])
    print("Passed:", gate["passed"], gate["passed_components"])

    # Open-loop collapse diagnostic (not part of the gate yet; reported for debugging).
    ol = _open_loop_diag(
        trainer,
        seed=0,
        batch_size=int(os.getenv("HEI_L5_GATE_OL_BATCH", "64")),
        length=int(os.getenv("HEI_L5_GATE_OL_LEN", "96")),
        tail=int(os.getenv("HEI_L5_GATE_OL_TAIL", "48")),
        max_period=int(os.getenv("HEI_L5_GATE_OL_MAXP", "8")),
    )
    print("OpenLoopDiag:", ol)

    return 0 if bool(gate["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
