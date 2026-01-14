#!/usr/bin/env python3
"""
Scan helper for L7v2-v0: tune `cost_weight` to avoid the "always delta=2" maximum-speed degenerate policy,
while keeping the v0.2-min ecology (forward-only pointer + drift).

This script intentionally reuses `HEI/training/run_l7v2_v0.py` as a subprocess:
- Phase 1: quick sweep on E0 only (fast) to pick a cost_weight that reduces delta_mean.
- Phase 2 (optional): confirm with full E0+E1+E2 at the selected cost_weight.

Example:
  /home/void0312/miniconda3/envs/PINNs/bin/python HEI/training/scripts/scan_l7v2_v0_cost.py \\
    --device cuda --save_root checkpoints/l7v2_v0p2_costscan \\
    --seed 0 --dim_q 64 --num_charts 4 --batch_size 64 \\
    --target_delta 1.5 --trials 8 --cost_weight_start 0.001 --cost_weight_max 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Match: "[L7v2-v0] Saved run to: <path>"
RUN_RE = re.compile(r"^\[L7v2-v0\] Saved run to: (.+)$")


@dataclass
class TrialResult:
    cost_weight: float
    run_dir: str
    elapsed_s: float
    summary: Dict[str, Any]

    @property
    def e0_passive_F(self) -> float:
        return float(self.summary.get("E0", {}).get("passive", {}).get("F_final", float("nan")))

    @property
    def e0_active_F(self) -> float:
        return float(self.summary.get("E0", {}).get("active", {}).get("F_final", float("nan")))

    @property
    def e0_active_delta(self) -> float:
        return float(self.summary.get("E0", {}).get("active", {}).get("delta_mean_final", float("nan")))


def _run_once(cmd: List[str], *, timeout_s: int) -> Tuple[str, str, int, float]:
    t0 = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s)
    return proc.stdout, proc.stderr, int(proc.returncode), float(time.time() - t0)


def _parse_run_dir(stdout: str) -> Optional[str]:
    for line in (stdout or "").splitlines():
        m = RUN_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    return None


def _load_summary(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "summary.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _isfinite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _pick_best(trials: List[TrialResult], *, target_delta: float, e0_tol: float) -> Optional[TrialResult]:
    best: Optional[TrialResult] = None
    best_key: Tuple[float, float] = (float("inf"), float("inf"))
    for tr in trials:
        d = tr.e0_active_delta
        Fp = tr.e0_passive_F
        Fa = tr.e0_active_F
        if not (_isfinite(d) and _isfinite(Fp) and _isfinite(Fa)):
            continue
        if Fa > Fp * (1.0 + float(e0_tol)):
            continue
        if d > float(target_delta):
            continue
        key = (abs(d - float(target_delta)), float(tr.cost_weight))
        if key < best_key:
            best_key = key
            best = tr
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="checkpoints/l7v2_v0_costscan")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dim_q", type=int, default=64)
    ap.add_argument("--num_charts", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--text_a", type=str, default="HEI/data/CLUE/CLUECorpusSmall_head5000.txt")
    ap.add_argument("--text_b", type=str, default="HEI/data/cilin/new_cilin.txt")
    ap.add_argument("--max_chars", type=int, default=200000)
    ap.add_argument("--vocab_size", type=int, default=8000)

    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta_kl", type=float, default=0.01)
    ap.add_argument("--gamma_pred", type=float, default=1.0)
    ap.add_argument("--noise_std", type=float, default=0.05)

    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--step_size", type=float, default=1.0)
    # NOTE: including delta=0 makes the sweep vulnerable to "stall/dark-room" policies where the
    # controller chooses delta=0 regardless of cost_weight; use forward-only by default.
    ap.add_argument("--candidates", type=str, default="1,2")

    ap.add_argument("--pointer_drift", type=float, default=0.3)
    ap.add_argument("--pointer_bounds", type=str, default="clamp", choices=["clamp", "reflect"])
    ap.add_argument("--anti_lock_weight", type=float, default=0.0)
    ap.add_argument("--anti_lock_sigma", type=float, default=0.5)
    ap.add_argument("--revisit_weight", type=float, default=0.0)
    ap.add_argument("--revisit_window", type=int, default=0)
    ap.add_argument("--lookahead_steps", type=int, default=1)
    ap.add_argument("--lookahead_discount", type=float, default=0.9)
    ap.add_argument("--pref_repeat_weight", type=float, default=0.0)
    ap.add_argument("--cover_weight", type=float, default=0.0)
    ap.add_argument("--cover_sigma", type=float, default=2.0)
    ap.add_argument("--cover_window", type=int, default=0)
    ap.add_argument("--cover_warmup_frac", type=float, default=0.3)
    ap.add_argument("--epi_weight", type=float, default=0.0)
    ap.add_argument("--epi_normalize", type=int, default=1)

    ap.add_argument("--cost_power", type=float, default=2.0)
    ap.add_argument("--cost_offset", type=float, default=0.0)

    ap.add_argument("--e0_steps", type=int, default=400)
    ap.add_argument("--episodes_per_class", type=int, default=20)
    ap.add_argument("--online_steps", type=int, default=80)
    ap.add_argument("--offline_steps", type=int, default=20)
    ap.add_argument("--replay_mode", type=str, default="random")
    ap.add_argument("--log_every", type=int, default=50)

    # Scan controls
    ap.add_argument("--target_delta", type=float, default=1.5)
    ap.add_argument("--e0_tol", type=float, default=0.0, help="Allow active.F <= passive.F*(1+tol) in phase1 gating.")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--cost_weight_start", type=float, default=1e-3)
    ap.add_argument("--cost_weight_max", type=float, default=5e-2)
    ap.add_argument("--timeout_s", type=int, default=3600)
    ap.add_argument("--confirm", type=int, default=1, help="Run a final full E0+E1+E2 confirm at the selected cost_weight.")
    args = ap.parse_args()

    save_root = str(args.save_root)
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.jsonl")

    base = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "..", "run_l7v2_v0.py"),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--dim_q",
        str(int(args.dim_q)),
        "--num_charts",
        str(int(args.num_charts)),
        "--batch_size",
        str(int(args.batch_size)),
        "--text_a",
        str(args.text_a),
        "--text_b",
        str(args.text_b),
        "--max_chars",
        str(int(args.max_chars)),
        "--vocab_size",
        str(int(args.vocab_size)),
        "--dt",
        str(float(args.dt)),
        "--lr",
        str(float(args.lr)),
        "--beta_kl",
        str(float(args.beta_kl)),
        "--gamma_pred",
        str(float(args.gamma_pred)),
        "--noise_std",
        str(float(args.noise_std)),
        "--sigma",
        str(float(args.sigma)),
        "--step_size",
        str(float(args.step_size)),
        "--candidates",
        str(args.candidates),
        "--pointer_drift",
        str(float(args.pointer_drift)),
        "--pointer_bounds",
        str(args.pointer_bounds),
        "--anti_lock_weight",
        str(float(args.anti_lock_weight)),
        "--anti_lock_sigma",
        str(float(args.anti_lock_sigma)),
        "--revisit_weight",
        str(float(args.revisit_weight)),
        "--revisit_window",
        str(int(args.revisit_window)),
        "--lookahead_steps",
        str(int(args.lookahead_steps)),
        "--lookahead_discount",
        str(float(args.lookahead_discount)),
        "--pref_repeat_weight",
        str(float(args.pref_repeat_weight)),
        "--cover_weight",
        str(float(args.cover_weight)),
        "--cover_sigma",
        str(float(args.cover_sigma)),
        "--cover_window",
        str(int(args.cover_window)),
        "--cover_warmup_frac",
        str(float(args.cover_warmup_frac)),
        "--epi_weight",
        str(float(args.epi_weight)),
        "--epi_normalize",
        str(int(args.epi_normalize)),
        "--cost_power",
        str(float(args.cost_power)),
        "--cost_offset",
        str(float(args.cost_offset)),
        "--e0_steps",
        str(int(args.e0_steps)),
        "--episodes_per_class",
        str(int(args.episodes_per_class)),
        "--online_steps",
        str(int(args.online_steps)),
        "--offline_steps",
        str(int(args.offline_steps)),
        "--replay_mode",
        str(args.replay_mode),
        "--log_every",
        str(int(args.log_every)),
    ]

    trials: List[TrialResult] = []
    cw = float(args.cost_weight_start)
    for idx in range(int(args.trials)):
        if cw > float(args.cost_weight_max) + 1e-12:
            break
        trial_root = os.path.join(save_root, f"trial_costw_{cw:.6g}")
        cmd = base + ["--save_dir", trial_root, "--cost_weight", f"{cw:.12g}", "--run_e0", "1", "--run_e1e2", "0"]
        print(f"[scan] ({idx+1}/{int(args.trials)}) cost_weight={cw:.6g}")
        stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
        run_dir = _parse_run_dir(stdout)
        rec: Dict[str, Any] = {
            "phase": "sweep_e0",
            "idx": int(idx),
            "cost_weight": float(cw),
            "elapsed_s": float(elapsed),
            "returncode": int(code),
        }
        if run_dir is None or code != 0:
            rec["stdout_tail"] = (stdout or "")[-2000:]
            rec["stderr_tail"] = (stderr or "")[-2000:]
            _write_jsonl(results_path, rec)
            cw *= 2.0
            continue

        summary = _load_summary(run_dir)
        tr = TrialResult(cost_weight=float(cw), run_dir=str(run_dir), elapsed_s=float(elapsed), summary=summary)
        trials.append(tr)

        rec.update(
            {
                "run_dir": str(run_dir),
                "E0_passive_F": tr.e0_passive_F,
                "E0_active_F": tr.e0_active_F,
                "E0_active_delta": tr.e0_active_delta,
            }
        )
        _write_jsonl(results_path, rec)
        cw *= 2.0

    best = _pick_best(trials, target_delta=float(args.target_delta), e0_tol=float(args.e0_tol))
    if best is None:
        print("[scan] No acceptable cost_weight found in sweep. Try increasing --cost_weight_max or relaxing --target_delta.")
        return

    print(
        "[scan] Selected cost_weight="
        f"{best.cost_weight:.6g} (active_delta={best.e0_active_delta:.3f} active_F={best.e0_active_F:.4f} passive_F={best.e0_passive_F:.4f})"
    )

    if int(args.confirm) != 1:
        return

    confirm_root = os.path.join(save_root, f"confirm_costw_{best.cost_weight:.6g}")
    cmd = base + [
        "--save_dir",
        confirm_root,
        "--cost_weight",
        f"{best.cost_weight:.12g}",
        "--run_e0",
        "1",
        "--run_e1e2",
        "1",
    ]
    print(f"[scan] confirm: cost_weight={best.cost_weight:.6g}")
    stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
    run_dir = _parse_run_dir(stdout)
    rec = {
        "phase": "confirm_full",
        "cost_weight": float(best.cost_weight),
        "elapsed_s": float(elapsed),
        "returncode": int(code),
    }
    if run_dir is None or code != 0:
        rec["stdout_tail"] = (stdout or "")[-2000:]
        rec["stderr_tail"] = (stderr or "")[-2000:]
        _write_jsonl(results_path, rec)
        print("[scan] confirm failed, see results.jsonl")
        return

    summary = _load_summary(run_dir)
    rec["run_dir"] = str(run_dir)
    rec["summary"] = summary
    _write_jsonl(results_path, rec)
    print(f"[scan] confirm saved: {run_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
