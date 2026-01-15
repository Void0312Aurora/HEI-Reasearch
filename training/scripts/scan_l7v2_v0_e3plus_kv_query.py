#!/usr/bin/env python3
"""
E3+ scan helper for L7v2-v0: kv_swap + query/readout (key→value) across seq_len × candidates.

Motivation (per temp-20 and 1·15/temp-01):
  - With query/readout enabled, "binding is learnable when hit"; the bottleneck becomes resolved_rate.
  - We want a falsifiable grid:
      - seq_len ∈ {1024, 2048, 4096}
      - candidates ∈ { {1,2}, {1,2,4,8,16,32}, {32} }
    to see whether search capability slows the resolved_rate collapse with longer sequences.

Outputs:
  - results.jsonl (one line per run)
  - grid_summary.json (aggregated per (seq_len,candidate_set))
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


def _parse_candidate_sets(s: str) -> List[str]:
    sets: List[str] = []
    for chunk in str(s or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",") if p.strip()]
        if not parts:
            continue
        sets.append(",".join(parts))
    return sets


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
    candidate_set: str
    run_dir: str
    elapsed_s: float
    summary: Dict[str, Any]

    def e3(self) -> Dict[str, Any]:
        return dict(self.summary.get("E3", {}) or {})


def _gate_e3plus(e3: Dict[str, Any], *, args: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    checks: Dict[str, bool] = {}
    resolved_any_rate = float(e3.get("resolved_any_rate", float("nan")))
    query_acc_hit = float(e3.get("query_acc_hit", float("nan")))

    checks["resolved_any_rate"] = _isfinite(resolved_any_rate) and (
        resolved_any_rate >= float(args.gate_resolved_any_rate)
    )
    checks["query_acc_hit"] = _isfinite(query_acc_hit) and (query_acc_hit >= float(args.gate_query_acc_hit))
    passed = all(checks.values())
    return passed, checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="checkpoints/l7v2_v0_e3plus_kv_query")
    ap.add_argument("--timeout_s", type=int, default=7200)

    # Matrix
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--seq_lens", type=str, default="1024,2048,4096")
    ap.add_argument("--candidate_sets", type=str, default="1,2;1,2,4,8,16,32;32")

    # Core config
    ap.add_argument("--dim_q", type=int, default=64)
    ap.add_argument("--num_charts", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--pointer_drift", type=float, default=0.3)
    ap.add_argument("--pointer_bounds", type=str, default="wrap", choices=["clamp", "reflect", "wrap"])
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

    # E3+ kv_swap config
    ap.add_argument("--online_steps", type=int, default=80)
    ap.add_argument("--offline_steps", type=int, default=20)
    ap.add_argument("--e3_pairs", type=int, default=8)
    ap.add_argument("--key_len", type=int, default=1)
    ap.add_argument("--value_len", type=int, default=32)
    ap.add_argument("--clue_margin", type=int, default=64)
    ap.add_argument(
        "--policy",
        type=str,
        default="active_fixate",
        choices=["active", "active_fixate", "passive", "random", "frozen"],
    )
    ap.add_argument("--fixate_steps", type=int, default=8)
    ap.add_argument("--fixate_mad_mult", type=float, default=6.0)
    ap.add_argument("--boundary_window", type=int, default=0)

    # Query/readout
    ap.add_argument("--query_steps", type=int, default=8)
    ap.add_argument("--query_key", type=str, default="k1", choices=["k1", "k2", "random"])
    ap.add_argument(
        "--val_align",
        type=int,
        default=0,
        help="If 1, enable value-hit triggered one-step alignment to the corresponding key (v0p11 efficiency).",
    )

    ap.add_argument("--log_every", type=int, default=50)

    # Gate (optional; for regression)
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--gate_resolved_any_rate", type=float, default=0.18)
    ap.add_argument("--gate_query_acc_hit", type=float, default=0.95)

    args = ap.parse_args()

    save_root = str(args.save_root)
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.jsonl")
    grid_path = os.path.join(save_root, "grid_summary.json")

    seeds = _parse_csv_ints(str(args.seeds))
    seq_lens = _parse_csv_ints(str(args.seq_lens))
    cand_sets = _parse_candidate_sets(str(args.candidate_sets))
    if not cand_sets:
        raise ValueError("empty --candidate_sets")

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
        "--e3_task",
        "kv_swap",
        "--e3_pairs",
        str(int(args.e3_pairs)),
        "--e3_key_len",
        str(int(args.key_len)),
        "--e3_clue_len",
        str(int(args.value_len)),
        "--e3_clue_margin",
        str(int(args.clue_margin)),
        "--e3_boundary_window",
        str(int(args.boundary_window)),
        "--e3_policy",
        str(args.policy),
        "--e3_fixate_steps",
        str(int(args.fixate_steps)),
        "--e3_fixate_mad_mult",
        str(float(args.fixate_mad_mult)),
        "--e3_query_steps",
        str(int(args.query_steps)),
        "--e3_query_key",
        str(args.query_key),
        "--e3_val_align",
        str(int(args.val_align)),
        "--log_every",
        str(int(args.log_every)),
    ]

    flat: List[FlatResult] = []
    total = len(seeds) * len(seq_lens) * len(cand_sets)
    idx = 0
    for seed in seeds:
        for seq_len in seq_lens:
            for cand in cand_sets:
                idx += 1
                trial_root = os.path.join(
                    save_root,
                    f"seed{seed}",
                    f"len{seq_len}",
                    f"cand_{cand.replace(',', '-')}",
                )
                cmd = base + [
                    "--save_dir",
                    trial_root,
                    "--seed",
                    str(int(seed)),
                    "--e3_seq_len",
                    str(int(seq_len)),
                    f"--candidates={cand}",
                ]
                print(f"[scan] ({idx}/{total}) seed={seed} seq_len={seq_len} candidates={cand}")
                stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
                run_dir = _parse_run_dir(stdout)

                rec: Dict[str, Any] = {
                    "seed": int(seed),
                    "seq_len": int(seq_len),
                    "candidate_set": str(cand),
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
                fr = FlatResult(
                    seed=int(seed),
                    seq_len=int(seq_len),
                    candidate_set=str(cand),
                    run_dir=str(run_dir),
                    elapsed_s=float(elapsed),
                    summary=summary,
                )
                flat.append(fr)
                e3 = fr.e3()
                if int(args.gate) == 1:
                    passed, checks = _gate_e3plus(e3, args=args)
                    rec["gate_passed"] = bool(passed)
                    rec["gate_checks"] = checks
                rec["E3"] = e3
                _write_jsonl(results_path, rec)

    # Aggregate per (seq_len,candidate_set)
    groups: Dict[str, Dict[str, Any]] = {}
    for fr in flat:
        key = f"len{fr.seq_len}/cand:{fr.candidate_set}"
        g = groups.setdefault(key, {"runs": 0, "seeds": [], "means": {}, "passes": 0})
        g["runs"] += 1
        g["seeds"].append(int(fr.seed))
        if int(args.gate) == 1:
            passed, _ = _gate_e3plus(fr.e3(), args=args)
            g["passes"] += int(bool(passed))

    def _mean(vals: List[float]) -> float:
        xs = [float(x) for x in vals if _isfinite(x)]
        return float(sum(xs) / max(1, len(xs))) if xs else float("nan")

    for key, g in groups.items():
        vals: Dict[str, List[float]] = {
            "resolved_any_rate": [],
            "resolved_rate": [],
            "query_acc": [],
            "query_acc_hit": [],
            "probe_auc_hit": [],
            "span_hit_mean": [],
            "delta_abs_mean": [],
        }
        for fr in flat:
            k = f"len{fr.seq_len}/cand:{fr.candidate_set}"
            if k != key:
                continue
            e3 = fr.e3()
            vals["resolved_any_rate"].append(float(e3.get("resolved_any_rate", float("nan"))))
            vals["resolved_rate"].append(float(e3.get("resolved_rate", float("nan"))))
            vals["query_acc"].append(float(e3.get("query_acc", float("nan"))))
            vals["query_acc_hit"].append(float(e3.get("query_acc_hit", float("nan"))))
            vals["probe_auc_hit"].append(float(e3.get("probe_auc_hit", float("nan"))))
            vals["span_hit_mean"].append(float(e3.get("span_hit_mean", float("nan"))))
            vals["delta_abs_mean"].append(float(e3.get("delta_abs_mean", float("nan"))))
        g["means"] = {k: _mean(v) for k, v in vals.items()}

    out = {
        "config": {
            "device": str(args.device),
            "seeds": seeds,
            "seq_lens": seq_lens,
            "candidate_sets": cand_sets,
            "policy": str(args.policy),
            "online_steps": int(args.online_steps),
            "offline_steps": int(args.offline_steps),
            "e3_pairs": int(args.e3_pairs),
            "key_len": int(args.key_len),
            "value_len": int(args.value_len),
            "clue_margin": int(args.clue_margin),
            "query_steps": int(args.query_steps),
            "query_key": str(args.query_key),
            "val_align": int(args.val_align),
            "gate": int(args.gate),
        },
        "groups": groups,
        "n_success": len(flat),
    }
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[scan] Done. Wrote: {results_path}")
    print(f"[scan] Grid summary: {grid_path}")


if __name__ == "__main__":
    main()
