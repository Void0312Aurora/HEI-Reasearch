#!/usr/bin/env python3
"""
Scan E3+ efficiency tier vs speak mix sharpness (k).

Goal (per temp-11 "controllability"):
  - Vary --e3_speak_mix_sharpness and record how:
      - speak_mix_prob_mean
      - resolved_any_rate / first_hit_median / dwell_hit_mean
    respond, under fixed budget (key_len=1, online_steps=80, val_align=0).

Outputs:
  - results.jsonl (one line per run)
  - scan_summary.json (aggregated by k)
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


RUN_RE = re.compile(r"^\[L7v2-v0\] Saved run to: (.+)$")


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


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


def _run_once(cmd: List[str], *, timeout_s: int) -> Tuple[str, str, int, float]:
    t0 = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s)
    return proc.stdout, proc.stderr, int(proc.returncode), float(time.time() - t0)


def _write_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@dataclass
class Flat:
    seed: int
    k: float
    run_dir: str
    elapsed_s: float
    e3: Dict[str, Any]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="checkpoints/l7v2_v0_e3p_speak_sharpness")
    ap.add_argument("--timeout_s", type=int, default=7200)
    ap.add_argument("--seeds", type=str, default="2")
    ap.add_argument("--ks", type=str, default="1,2,4,8,12")

    # Fixed efficiency tier config
    ap.add_argument("--dim_q", type=int, default=64)
    ap.add_argument("--num_charts", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--pointer_bounds", type=str, default="wrap", choices=["clamp", "reflect", "wrap"])
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--cost_weight", type=float, default=0.001)
    ap.add_argument("--cost_power", type=float, default=1.0)
    ap.add_argument("--anti_lock_weight", type=float, default=0.1)
    ap.add_argument("--anti_lock_sigma", type=float, default=0.5)
    ap.add_argument("--cover_weight", type=float, default=0.1)
    ap.add_argument("--cover_window", type=int, default=64)
    ap.add_argument("--cover_sigma", type=float, default=2.0)
    ap.add_argument("--cover_warmup_frac", type=float, default=0.3)

    ap.add_argument("--e3_seq_len", type=int, default=4096)
    ap.add_argument("--e3_value_len", type=int, default=32)
    ap.add_argument("--e3_key_len", type=int, default=1)
    ap.add_argument("--e3_clue_margin", type=int, default=64)
    ap.add_argument("--e3_pairs", type=int, default=8)
    ap.add_argument("--online_steps", type=int, default=80)
    ap.add_argument("--offline_steps", type=int, default=20)
    ap.add_argument("--pointer_drift", type=float, default=0.0)
    ap.add_argument("--candidates", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--query_steps", type=int, default=8)
    ap.add_argument("--query_key", type=str, default="k1", choices=["k1", "k2", "random"])
    ap.add_argument("--fixate_steps", type=int, default=8)
    ap.add_argument("--fixate_mad_mult", type=float, default=6.0)

    # Speak config (fixed except k)
    ap.add_argument("--speak_input", type=str, default="y", choices=["q", "y"])
    ap.add_argument("--speak_loss_weight", type=float, default=0.1)
    ap.add_argument("--speak_pos_weight", type=float, default=0.0)
    ap.add_argument("--speak_pos_weight_max", type=float, default=50.0)

    args = ap.parse_args()

    save_root = str(args.save_root)
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.jsonl")
    out_path = os.path.join(save_root, "scan_summary.json")

    seeds = _parse_csv_ints(str(args.seeds))
    ks = _parse_csv_floats(str(args.ks))
    if not ks:
        raise ValueError("empty --ks")

    runner = os.path.join(os.path.dirname(__file__), "..", "run_l7v2_v0.py")

    base = [
        sys.executable,
        runner,
        "--device",
        str(args.device),
        "--dim_q",
        str(int(args.dim_q)),
        "--num_charts",
        str(int(args.num_charts)),
        "--batch_size",
        str(int(args.batch_size)),
        "--dt",
        str(float(args.dt)),
        "--sigma",
        str(float(args.sigma)),
        "--step_size",
        str(float(args.step_size)),
        "--pointer_bounds",
        str(args.pointer_bounds),
        "--noise_std",
        str(float(args.noise_std)),
        "--cost_weight",
        str(float(args.cost_weight)),
        "--cost_power",
        str(float(args.cost_power)),
        "--anti_lock_weight",
        str(float(args.anti_lock_weight)),
        "--anti_lock_sigma",
        str(float(args.anti_lock_sigma)),
        "--cover_weight",
        str(float(args.cover_weight)),
        "--cover_window",
        str(int(args.cover_window)),
        "--cover_sigma",
        str(float(args.cover_sigma)),
        "--cover_warmup_frac",
        str(float(args.cover_warmup_frac)),
        "--online_steps",
        str(int(args.online_steps)),
        "--offline_steps",
        str(int(args.offline_steps)),
        "--pointer_drift",
        str(float(args.pointer_drift)),
        f"--candidates={str(args.candidates)}",
        "--run_e0",
        "0",
        "--run_e1e2",
        "0",
        "--run_e3",
        "1",
        "--log_every",
        "0",
        "--e3_task",
        "kv_swap",
        "--e3_pairs",
        str(int(args.e3_pairs)),
        "--e3_seq_len",
        str(int(args.e3_seq_len)),
        "--e3_clue_len",
        str(int(args.e3_value_len)),
        "--e3_key_len",
        str(int(args.e3_key_len)),
        "--e3_clue_margin",
        str(int(args.e3_clue_margin)),
        "--e3_policy",
        "active_fixate",
        "--e3_fixate_steps",
        str(int(args.fixate_steps)),
        "--e3_fixate_mad_mult",
        str(float(args.fixate_mad_mult)),
        "--e3_fixate_mode",
        "freeze",
        "--e3_query_steps",
        str(int(args.query_steps)),
        "--e3_query_key",
        str(args.query_key),
        "--e3_val_align",
        "0",
        "--e3_speak",
        "1",
        "--e3_speak_vocab",
        "2",
        "--e3_speak_input",
        str(args.speak_input),
        "--e3_speak_loss_weight",
        str(float(args.speak_loss_weight)),
        "--e3_speak_use",
        "2",
        "--e3_speak_pos_weight",
        str(float(args.speak_pos_weight)),
        "--e3_speak_pos_weight_max",
        str(float(args.speak_pos_weight_max)),
    ]

    flats: List[Flat] = []
    for seed in seeds:
        for k in ks:
            save_dir = os.path.join(save_root, f"seed{int(seed)}_k{str(k).replace('.','p')}")
            cmd = base + [
                "--save_dir",
                str(save_dir),
                "--seed",
                str(int(seed)),
                "--e3_speak_mix_sharpness",
                str(float(k)),
            ]
            stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
            run_dir = _parse_run_dir(stdout)
            rec: Dict[str, Any] = {
                "seed": int(seed),
                "k": float(k),
                "returncode": int(code),
                "elapsed_s": float(elapsed),
                "run_dir": run_dir,
                "stderr_tail": "\n".join((stderr or "").splitlines()[-40:]),
            }
            if code == 0 and run_dir:
                summary = _load_summary(run_dir)
                e3 = dict(summary.get("E3") or {})
                rec["e3"] = e3
                flats.append(Flat(seed=int(seed), k=float(k), run_dir=run_dir, elapsed_s=float(elapsed), e3=e3))
            _write_jsonl(results_path, rec)

    # Aggregate by k (mean over seeds).
    agg: Dict[str, Dict[str, float]] = {}
    for k in ks:
        rows = [f for f in flats if abs(float(f.k) - float(k)) < 1e-12]
        if not rows:
            continue
        def _mean(field: str) -> float:
            vals = []
            for r in rows:
                v = r.e3.get(field, float("nan"))
                try:
                    v = float(v)
                except Exception:
                    v = float("nan")
                if math.isfinite(v):
                    vals.append(v)
            return float(sum(vals) / max(1, len(vals))) if vals else float("nan")

        agg[str(k)] = {
            "resolved_any_rate_mean": _mean("resolved_any_rate"),
            "first_hit_median_mean": _mean("first_hit_median"),
            "dwell_hit_mean_mean": _mean("dwell_hit_mean"),
            "speak_mix_prob_mean_mean": _mean("speak_mix_prob_mean"),
            "speak_tpr_mean": _mean("speak_tpr"),
            "speak_precision_mean": _mean("speak_precision"),
        }

    out = {
        "config": vars(args),
        "rows": [f.__dict__ for f in flats],
        "agg_by_k": agg,
        "results_jsonl": results_path,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[scan] wrote: {out_path}")


if __name__ == "__main__":
    main()

