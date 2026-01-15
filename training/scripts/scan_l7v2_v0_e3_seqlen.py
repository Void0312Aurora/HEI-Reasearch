#!/usr/bin/env python3
"""
E3 scan helper for L7v2-v0: sweep seq_len at fixed clue_len.

Motivation (per temp-18 / temp-16):
  - With fixed online_steps and small step candidates, evidence exposure is geometry-limited.
  - This sweep answers: for the current search capability, what is the max seq_len
    before hit_rate/probe_auc collapses?

Outputs:
  - results.jsonl (one line per run)
  - grid_summary.json (aggregated per seq_len)
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


def _isfinite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _write_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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


@dataclass
class FlatResult:
    seed: int
    seq_len: int
    run_dir: str
    elapsed_s: float
    summary: Dict[str, Any]

    def e3(self) -> Dict[str, Any]:
        return dict(self.summary.get("E3", {}) or {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="checkpoints/l7v2_v0_e3_seqlen")
    ap.add_argument("--timeout_s", type=int, default=7200)

    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--seq_lens", type=str, default="512,1024,2048,4096")
    ap.add_argument("--clue_len", type=int, default=32)
    ap.add_argument("--clue_margin", type=int, default=64)
    ap.add_argument(
        "--policy",
        type=str,
        default="active",
        choices=["active", "active_fixate", "passive", "random", "frozen"],
    )
    ap.add_argument("--fixate_steps", type=int, default=8)
    ap.add_argument("--fixate_mad_mult", type=float, default=6.0)

    # Core config (defaults match temp-13 style)
    ap.add_argument("--dim_q", type=int, default=64)
    ap.add_argument("--num_charts", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--online_steps", type=int, default=80)
    ap.add_argument("--offline_steps", type=int, default=20)
    ap.add_argument("--e3_pairs", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--candidates", type=str, default="1,2")
    ap.add_argument("--pointer_drift", type=float, default=0.3)
    ap.add_argument("--pointer_bounds", type=str, default="clamp", choices=["clamp", "reflect", "wrap"])
    ap.add_argument("--lookahead_steps", type=int, default=1)
    ap.add_argument("--lookahead_discount", type=float, default=0.9)
    ap.add_argument("--cost_weight", type=float, default=0.001)
    ap.add_argument("--cost_power", type=float, default=2.0)
    ap.add_argument("--anti_lock_weight", type=float, default=0.1)
    ap.add_argument("--anti_lock_sigma", type=float, default=0.5)
    ap.add_argument("--cover_weight", type=float, default=0.1)
    ap.add_argument("--cover_sigma", type=float, default=2.0)
    ap.add_argument("--cover_window", type=int, default=64)
    ap.add_argument("--cover_warmup_frac", type=float, default=0.3)
    ap.add_argument("--noise_std", type=float, default=0.05)

    ap.add_argument("--text_a", type=str, default="HEI/data/CLUE/CLUECorpusSmall_head5000.txt")
    ap.add_argument("--text_b", type=str, default="HEI/data/cilin/new_cilin.txt")
    ap.add_argument("--max_chars", type=int, default=200000)
    ap.add_argument("--vocab_size", type=int, default=8000)

    ap.add_argument("--log_every", type=int, default=80)
    args = ap.parse_args()

    save_root = str(args.save_root)
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.jsonl")
    grid_path = os.path.join(save_root, "grid_summary.json")

    seeds = _parse_csv_ints(str(args.seeds))
    seq_lens = _parse_csv_ints(str(args.seq_lens))

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
        "--sigma",
        str(float(args.sigma)),
        "--step_size",
        str(float(args.step_size)),
        f"--candidates={str(args.candidates)}",
        "--pointer_drift",
        str(float(args.pointer_drift)),
        "--pointer_bounds",
        str(args.pointer_bounds),
        "--lookahead_steps",
        str(int(args.lookahead_steps)),
        "--lookahead_discount",
        str(float(args.lookahead_discount)),
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
        "--cover_sigma",
        str(float(args.cover_sigma)),
        "--cover_window",
        str(int(args.cover_window)),
        "--cover_warmup_frac",
        str(float(args.cover_warmup_frac)),
        "--noise_std",
        str(float(args.noise_std)),
        "--online_steps",
        str(int(args.online_steps)),
        "--offline_steps",
        str(int(args.offline_steps)),
        "--run_e0",
        "0",
        "--run_e1e2",
        "0",
        "--run_e3",
        "1",
        "--e3_pairs",
        str(int(args.e3_pairs)),
        "--e3_clue_len",
        str(int(args.clue_len)),
        "--e3_clue_margin",
        str(int(args.clue_margin)),
        "--e3_policy",
        str(args.policy),
        "--e3_fixate_steps",
        str(int(args.fixate_steps)),
        "--e3_fixate_mad_mult",
        str(float(args.fixate_mad_mult)),
        "--log_every",
        str(int(args.log_every)),
    ]

    flat: List[FlatResult] = []
    total = len(seeds) * len(seq_lens)
    idx = 0
    for seed in seeds:
        for seq_len in seq_lens:
            idx += 1
            trial_root = os.path.join(save_root, f"seed{seed}", f"len{seq_len}")
            cmd = base + [
                "--save_dir",
                trial_root,
                "--seed",
                str(int(seed)),
                "--e3_seq_len",
                str(int(seq_len)),
            ]
            print(f"[scan] ({idx}/{total}) seed={seed} seq_len={seq_len}")
            stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
            run_dir = _parse_run_dir(stdout)

            rec: Dict[str, Any] = {
                "seed": int(seed),
                "seq_len": int(seq_len),
                "returncode": int(code),
                "elapsed_s": float(elapsed),
                "run_dir": run_dir,
                "stdout_tail": "\n".join(stdout.splitlines()[-20:]),
                "stderr_tail": "\n".join(stderr.splitlines()[-20:]),
            }
            if code != 0 or not run_dir:
                _write_jsonl(results_path, rec)
                continue

            summary = _load_summary(run_dir)
            fr = FlatResult(seed=int(seed), seq_len=int(seq_len), run_dir=str(run_dir), elapsed_s=float(elapsed), summary=summary)
            flat.append(fr)
            rec["E3"] = fr.e3()
            _write_jsonl(results_path, rec)

    # Aggregate per seq_len
    by_len: Dict[int, List[FlatResult]] = {}
    for fr in flat:
        by_len.setdefault(int(fr.seq_len), []).append(fr)

    def mean(vals: List[float]) -> float:
        xs = [float(v) for v in vals if _isfinite(v)]
        return float(sum(xs) / max(1, len(xs))) if xs else float("nan")

    grid: Dict[str, Any] = {}
    for seq_len, rs in sorted(by_len.items(), key=lambda kv: kv[0]):
        vals: Dict[str, List[float]] = {
            "hit_rate": [],
            "probe_auc": [],
            "probe_auc_shuffled": [],
            "dwell_mean": [],
            "dwell_hit_mean": [],
            "span_mean": [],
            "span_hit_mean": [],
        }
        for fr in rs:
            e3 = fr.e3()
            vals["hit_rate"].append(float(e3.get("hit_rate", float("nan"))))
            vals["probe_auc"].append(float(e3.get("probe_auc", float("nan"))))
            vals["probe_auc_shuffled"].append(float(e3.get("probe_auc_shuffled", float("nan"))))
            vals["dwell_mean"].append(float(e3.get("dwell_mean", float("nan"))))
            vals["dwell_hit_mean"].append(float(e3.get("dwell_hit_mean", float("nan"))))
            vals["span_mean"].append(float(e3.get("span_mean", float("nan"))))
            vals["span_hit_mean"].append(float(e3.get("span_hit_mean", float("nan"))))
        grid[str(int(seq_len))] = {k: mean(v) for k, v in vals.items()}

    out = {
        "config": {
            "device": str(args.device),
            "policy": str(args.policy),
            "seeds": seeds,
            "seq_lens": seq_lens,
            "clue_len": int(args.clue_len),
            "clue_margin": int(args.clue_margin),
            "online_steps": int(args.online_steps),
            "offline_steps": int(args.offline_steps),
            "pairs": int(args.e3_pairs),
            "batch_size": int(args.batch_size),
        },
        "grid": grid,
        "n_success": len(flat),
    }
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[scan] Done. Wrote: {results_path}")
    print(f"[scan] Grid summary: {grid_path}")


if __name__ == "__main__":
    main()
