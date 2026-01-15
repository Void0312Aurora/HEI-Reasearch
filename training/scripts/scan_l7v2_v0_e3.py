#!/usr/bin/env python3
"""
E3 scan helper for L7v2-v0 (minimal reading closed-loop).

Purpose (per temp-14):
  - Run a small, falsifiable matrix over:
      - seed
      - e3_policy (active/passive/frozen, optionally random)
  - Record E3 metrics:
      - hit_rate / first_hit_median / dwell_mean / span_mean
      - probe_auc + shuffled control
      - non-boundary cycle hits
  - Optionally evaluate a minimal E3 gate.

Outputs:
  - results.jsonl (one line per run)
  - group_summary.json (aggregated per-policy statistics)
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


def _parse_csv_str(s: str) -> List[str]:
    out: List[str] = []
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
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
    policy: str
    run_dir: str
    elapsed_s: float
    summary: Dict[str, Any]

    def e3(self) -> Dict[str, Any]:
        return dict(self.summary.get("E3", {}) or {})


def _gate_e3(e3: Dict[str, Any], *, online_steps: int, args: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    checks: Dict[str, bool] = {}
    hit_rate = float(e3.get("hit_rate", float("nan")))
    first_hit_median = float(e3.get("first_hit_median", float("nan")))
    probe_auc = float(e3.get("probe_auc", float("nan")))
    probe_auc_shuf = float(e3.get("probe_auc_shuffled", float("nan")))
    cycles = dict(e3.get("ptr_cycles", {}) or {})
    non_boundary_hits = int(cycles.get("ptr_cycle_hits_non_boundary", 0) or 0)

    checks["hit_rate"] = _isfinite(hit_rate) and (hit_rate >= float(args.gate_hit_rate))
    thr_first = float(args.gate_first_hit_frac) * float(max(1, int(online_steps)))
    checks["first_hit_median"] = _isfinite(first_hit_median) and (first_hit_median <= thr_first)
    checks["probe_auc"] = _isfinite(probe_auc) and (probe_auc >= float(args.gate_probe_auc))
    checks["probe_auc_shuffled"] = _isfinite(probe_auc_shuf) and (
        float(args.gate_shuf_low) <= probe_auc_shuf <= float(args.gate_shuf_high)
    )
    if int(args.gate_non_boundary_zero) == 1:
        checks["non_boundary_cycles"] = int(non_boundary_hits) == 0
    else:
        checks["non_boundary_cycles"] = True

    passed = all(checks.values())
    return passed, checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="checkpoints/l7v2_v0_e3_matrix")
    ap.add_argument("--timeout_s", type=int, default=7200)

    # Matrix
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--policies", type=str, default="active,passive,frozen")

    # v0 baseline knobs (reuse run_l7v2_v0 defaults unless overridden)
    ap.add_argument("--dim_q", type=int, default=64)
    ap.add_argument("--num_charts", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--beta_kl", type=float, default=0.01)
    ap.add_argument("--gamma_pred", type=float, default=1.0)
    ap.add_argument("--noise_std", type=float, default=0.05)

    ap.add_argument("--text_a", type=str, default="HEI/data/CLUE/CLUECorpusSmall_head5000.txt")
    ap.add_argument("--text_b", type=str, default="HEI/data/cilin/new_cilin.txt")
    ap.add_argument("--max_chars", type=int, default=200000)
    ap.add_argument("--vocab_size", type=int, default=8000)

    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--candidates", type=str, default="1,2")
    ap.add_argument("--pointer_drift", type=float, default=0.3)
    ap.add_argument("--pointer_bounds", type=str, default="clamp", choices=["clamp", "reflect", "wrap"])
    ap.add_argument("--lookahead_steps", type=int, default=1)
    ap.add_argument("--lookahead_discount", type=float, default=0.9)
    ap.add_argument("--cost_weight", type=float, default=0.001)
    ap.add_argument("--anti_lock_weight", type=float, default=0.1)
    ap.add_argument("--anti_lock_sigma", type=float, default=0.5)
    ap.add_argument("--cover_weight", type=float, default=0.1)
    ap.add_argument("--cover_sigma", type=float, default=2.0)
    ap.add_argument("--cover_window", type=int, default=64)
    ap.add_argument("--cover_warmup_frac", type=float, default=0.3)

    # E3 config
    ap.add_argument("--online_steps", type=int, default=80)
    ap.add_argument("--offline_steps", type=int, default=20)
    ap.add_argument("--e3_pairs", type=int, default=8)
    ap.add_argument("--e3_seq_len", type=int, default=4096)
    ap.add_argument("--e3_clue_len", type=int, default=32)
    ap.add_argument("--e3_clue_margin", type=int, default=64)
    ap.add_argument("--e3_boundary_window", type=int, default=0)
    ap.add_argument("--fixate_steps", type=int, default=8)
    ap.add_argument("--fixate_mad_mult", type=float, default=6.0)
    ap.add_argument("--log_every", type=int, default=50)

    # Minimal gate (optional; defaults match temp-14 suggestion)
    ap.add_argument("--gate", type=int, default=1)
    ap.add_argument("--gate_hit_rate", type=float, default=0.6)
    ap.add_argument("--gate_first_hit_frac", type=float, default=0.7)
    ap.add_argument("--gate_probe_auc", type=float, default=0.8)
    ap.add_argument("--gate_shuf_low", type=float, default=0.45)
    ap.add_argument("--gate_shuf_high", type=float, default=0.55)
    ap.add_argument("--gate_non_boundary_zero", type=int, default=1)

    args = ap.parse_args()

    save_root = str(args.save_root)
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.jsonl")
    group_path = os.path.join(save_root, "group_summary.json")

    seeds = _parse_csv_ints(str(args.seeds))
    policies = [p.strip().lower() for p in _parse_csv_str(str(args.policies))]
    for p in policies:
        if p not in ("active", "active_fixate", "passive", "random", "frozen"):
            raise ValueError(f"unknown policy: {p}")

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
        "--lr",
        str(float(args.lr)),
        "--weight_decay",
        str(float(args.weight_decay)),
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
        "--e3_seq_len",
        str(int(args.e3_seq_len)),
        "--e3_clue_len",
        str(int(args.e3_clue_len)),
        "--e3_clue_margin",
        str(int(args.e3_clue_margin)),
        "--e3_boundary_window",
        str(int(args.e3_boundary_window)),
        "--e3_fixate_steps",
        str(int(args.fixate_steps)),
        "--e3_fixate_mad_mult",
        str(float(args.fixate_mad_mult)),
        "--log_every",
        str(int(args.log_every)),
    ]

    flat: List[FlatResult] = []
    total = len(seeds) * len(policies)
    idx = 0
    for seed in seeds:
        for pol in policies:
            idx += 1
            trial_root = os.path.join(save_root, f"seed{seed}", f"policy_{pol}")
            cmd = (
                base
                + [
                    "--save_dir",
                    trial_root,
                    "--seed",
                    str(int(seed)),
                    "--e3_policy",
                    str(pol),
                ]
            )

            print(f"[scan] ({idx}/{total}) seed={seed} policy={pol}")
            stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
            run_dir = _parse_run_dir(stdout)

            rec: Dict[str, Any] = {
                "seed": int(seed),
                "policy": str(pol),
                "cmd": cmd,
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
            fr = FlatResult(seed=int(seed), policy=str(pol), run_dir=str(run_dir), elapsed_s=float(elapsed), summary=summary)
            flat.append(fr)

            e3 = fr.e3()
            if int(args.gate) == 1:
                passed, checks = _gate_e3(e3, online_steps=int(args.online_steps), args=args)
                rec["gate_passed"] = bool(passed)
                rec["gate_checks"] = checks
            rec["E3"] = e3
            _write_jsonl(results_path, rec)

    # Aggregate per policy
    groups: Dict[str, Dict[str, Any]] = {}
    for fr in flat:
        e3 = fr.e3()
        pol = str(fr.policy)
        g = groups.setdefault(pol, {"runs": 0, "seeds": [], "means": {}, "passes": 0})
        g["runs"] += 1
        g["seeds"].append(int(fr.seed))
        if int(args.gate) == 1:
            passed, _ = _gate_e3(e3, online_steps=int(args.online_steps), args=args)
            g["passes"] += int(bool(passed))

    def _mean(vals: List[float]) -> float:
        xs = [float(x) for x in vals if _isfinite(x)]
        return float(sum(xs) / max(1, len(xs))) if xs else float("nan")

    for pol, g in groups.items():
        vals: Dict[str, List[float]] = {
            "hit_rate": [],
            "first_hit_median": [],
            "dwell_mean": [],
            "span_mean": [],
            "probe_auc": [],
            "probe_auc_shuffled": [],
            "delta_abs_mean": [],
            "near_boundary_rate": [],
            "non_boundary_hits": [],
        }
        for fr in flat:
            if fr.policy != pol:
                continue
            e3 = fr.e3()
            vals["hit_rate"].append(float(e3.get("hit_rate", float("nan"))))
            vals["first_hit_median"].append(float(e3.get("first_hit_median", float("nan"))))
            vals["dwell_mean"].append(float(e3.get("dwell_mean", float("nan"))))
            vals["span_mean"].append(float(e3.get("span_mean", float("nan"))))
            vals["probe_auc"].append(float(e3.get("probe_auc", float("nan"))))
            vals["probe_auc_shuffled"].append(float(e3.get("probe_auc_shuffled", float("nan"))))
            vals["delta_abs_mean"].append(float(e3.get("delta_abs_mean", float("nan"))))
            vals["near_boundary_rate"].append(float(e3.get("near_boundary_rate", float("nan"))))
            cyc = dict(e3.get("ptr_cycles", {}) or {})
            vals["non_boundary_hits"].append(float(int(cyc.get("ptr_cycle_hits_non_boundary", 0) or 0)))

        g["means"] = {k: _mean(v) for k, v in vals.items()}

    out = {
        "config": {
            "device": str(args.device),
            "seeds": seeds,
            "policies": policies,
            "online_steps": int(args.online_steps),
            "offline_steps": int(args.offline_steps),
            "e3_pairs": int(args.e3_pairs),
            "e3_seq_len": int(args.e3_seq_len),
            "e3_clue_len": int(args.e3_clue_len),
            "e3_clue_margin": int(args.e3_clue_margin),
            "gate": int(args.gate),
        },
        "groups": groups,
        "n_success": len(flat),
    }
    with open(group_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[scan] Done. Wrote: {results_path}")
    print(f"[scan] Group summary: {group_path}")


if __name__ == "__main__":
    main()
