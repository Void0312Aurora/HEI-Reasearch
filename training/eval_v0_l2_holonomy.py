"""
v0 L2 Holonomy Evaluation (cross-chart consistency proxy).

Theory hooks (理论基础-7 / L2):
- Connection should provide non-trivial but controlled parallel transport.
- Holonomy around small loops is a proxy for curvature/frustration; if it explodes
  with scaling, prefer fixing L2 before adding more charts/skills.

Run:
  conda run -n PINNs python HEI/training/eval_v0_l2_holonomy.py --ckpt HEI/checkpoints/v0_lang/last.pt --device cuda
"""

import argparse
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

# Ensure HEI is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from EXP.diag.holonomy import compute_holonomy
from training.checkpoint_io import load_trainer_from_checkpoint


def _stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.array(xs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


@torch.no_grad()
def eval_holonomy(
    *,
    trainer,
    device: str,
    num_loops: int,
    delta: float,
    q_std: float,
    seed: int,
    v_norm: float,
) -> Dict[str, object]:
    conn = trainer.entity.connection
    dim_q = int(trainer.entity.dim_q)
    rng = random.Random(seed)

    ratios: List[float] = []
    errors: List[float] = []

    for _ in range(num_loops):
        q0 = torch.randn(dim_q, device=device) * float(q_std)
        if dim_q >= 2:
            a, b = rng.sample(range(dim_q), 2)
        else:
            a, b = 0, 0

        dx = torch.zeros(dim_q, device=device)
        dy = torch.zeros(dim_q, device=device)
        dx[a] = float(delta)
        dy[b] = float(delta)

        loop = [q0, q0 + dx, q0 + dx + dy, q0 + dy]
        v0 = torch.randn(dim_q, device=device)
        vn = v0.norm().clamp(min=1e-9)
        v0 = v0 * (float(v_norm) / float(vn.item()))

        m = compute_holonomy(conn, loop, v0)
        ratios.append(float(m["holonomy_ratio"]))
        errors.append(float(m["holonomy_error"]))

    return {
        "dim_q": dim_q,
        "connection_mode": getattr(conn, "mode", "unknown"),
        "connection_rank": int(getattr(conn, "rank", 0) or 0),
        "epsilon": float(conn.epsilon.detach().cpu().item()) if hasattr(conn, "epsilon") else 0.0,
        "ratio_stats": _stats(ratios),
        "error_stats": _stats(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path (e.g., .../last.pt)")
    parser.add_argument("--tokenizer_path", default="", help="Optional tokenizer.json path (defaults to ckpt dir).")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--num_loops", type=int, default=64)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--q_std", type=float, default=0.5)
    parser.add_argument("--v_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    tokenizer_path = args.tokenizer_path or None
    trainer, tokenizer, train_cfg, meta = load_trainer_from_checkpoint(
        args.ckpt,
        device=device,
        tokenizer_path=tokenizer_path if tokenizer_path else None,
        strict=False,
    )

    m = eval_holonomy(
        trainer=trainer,
        device=device,
        num_loops=int(args.num_loops),
        delta=float(args.delta),
        q_std=float(args.q_std),
        seed=int(args.seed),
        v_norm=float(args.v_norm),
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(
        "Holonomy:"
        f" dim_q={m['dim_q']}"
        f" mode={m['connection_mode']}"
        f" rank={m['connection_rank']}"
        f" epsilon={m['epsilon']:.4f}"
    )
    rs = m["ratio_stats"]
    es = m["error_stats"]
    print(f"Holonomy ratio: mean={rs['mean']:.6f} p50={rs['p50']:.6f} p90={rs['p90']:.6f} p99={rs['p99']:.6f} max={rs['max']:.6f}")
    print(f"Holonomy error: mean={es['mean']:.6f} p50={es['p50']:.6f} p90={es['p90']:.6f} p99={es['p99']:.6f} max={es['max']:.6f}")


if __name__ == "__main__":
    main()
