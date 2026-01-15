#!/usr/bin/env python3
"""
E3-v0 scan helper for L7v2-v0: compare candidate step sets (search capability).

Motivation (per temp-18/19/20 and 1·15/temp-01):
  - E3-v0 on long sequences is geometry-limited; the key is search/coverage, not "write chain".
  - We want a falsifiable comparison between:
      - stride-only (e.g. candidates={32})
      - multi-scale (e.g. candidates={1,2,4,8,16,32})
  - Gate should focus on: hit_rate, probe_auc_hit, dwell_hit_mean (cycles are diagnostic).

Outputs:
  - results.jsonl (one line per run)
  - group_summary.json (aggregated per candidate_set)
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
    """
    Parse semicolon-separated candidate sets, each as a CSV list.
    Example: "32;1,2,4,8,16,32"
    """
    sets: List[str] = []
    for chunk in str(s or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # normalize "1, 2, 4" -> "1,2,4"
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
    candidate_set: str
    run_dir: str
    elapsed_s: float
    summary: Dict[str, Any]

    def e3(self) -> Dict[str, Any]:
        return dict(self.summary.get("E3", {}) or {})


def _gate_e3_v0(e3: Dict[str, Any], *, online_steps: int, args: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    checks: Dict[str, bool] = {}
    hit_rate = float(e3.get("hit_rate", float("nan")))
    first_hit_median = float(e3.get("first_hit_median", float("nan")))
    dwell_hit_mean = float(e3.get("dwell_hit_mean", float("nan")))
    probe_auc_hit = float(e3.get("probe_auc_hit", float("nan")))
    probe_auc_shuf = float(e3.get("probe_auc_shuffled", float("nan")))

    checks["hit_rate"] = _isfinite(hit_rate) and (hit_rate >= float(args.gate_hit_rate))
    thr_first = float(args.gate_first_hit_frac) * float(max(1, int(online_steps)))
    checks["first_hit_median"] = _isfinite(first_hit_median) and (first_hit_median <= thr_first)
    checks["dwell_hit_mean"] = _isfinite(dwell_hit_mean) and (dwell_hit_mean >= float(args.gate_dwell_hit_mean))
    checks["probe_auc_hit"] = _isfinite(probe_auc_hit) and (probe_auc_hit >= float(args.gate_probe_auc_hit))
    checks["probe_auc_shuffled"] = _isfinite(probe_auc_shuf) and (
        float(args.gate_shuf_low) <= probe_auc_shuf <= float(args.gate_shuf_high)
    )

    passed = all(checks.values())
    return passed, checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="checkpoints/l7v2_v0_e3_candidates")
    ap.add_argument("--timeout_s", type=int, default=7200)

    # Matrix
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--candidate_sets", type=str, default="32;1,2,4,8,16,32")

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

    # E3 config (Q1 defaults: 4096×32, active_fixate, wrap)
    ap.add_argument("--online_steps", type=int, default=80)
    ap.add_argument("--offline_steps", type=int, default=20)
    ap.add_argument("--e3_pairs", type=int, default=8)
    ap.add_argument("--e3_seq_len", type=int, default=4096)
    ap.add_argument("--e3_clue_len", type=int, default=32)
    ap.add_argument("--e3_clue_margin", type=int, default=64)
    ap.add_argument("--e3_boundary_window", type=int, default=0)
    ap.add_argument(
        "--e3_policy",
        type=str,
        default="active_fixate",
        choices=["active", "active_fixate", "passive", "random", "frozen"],
    )
    ap.add_argument("--fixate_steps", type=int, default=8)
    ap.add_argument("--fixate_mad_mult", type=float, default=6.0)
    ap.add_argument("--log_every", type=int, default=50)

    # Gate (optional; aligned with 1·15/temp-01)
    ap.add_argument("--gate", type=int, default=1)
    ap.add_argument("--gate_hit_rate", type=float, default=0.6)
    ap.add_argument("--gate_first_hit_frac", type=float, default=0.7)
    ap.add_argument("--gate_dwell_hit_mean", type=float, default=6.0)
    ap.add_argument("--gate_probe_auc_hit", type=float, default=0.95)
    ap.add_argument("--gate_shuf_low", type=float, default=0.45)
    ap.add_argument("--gate_shuf_high", type=float, default=0.55)

    args = ap.parse_args()

    save_root = str(args.save_root)
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.jsonl")
    group_path = os.path.join(save_root, "group_summary.json")

    seeds = _parse_csv_ints(str(args.seeds))
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
        "marker_patch",
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
        "--e3_policy",
        str(args.e3_policy),
        "--e3_fixate_steps",
        str(int(args.fixate_steps)),
        "--e3_fixate_mad_mult",
        str(float(args.fixate_mad_mult)),
        "--log_every",
        str(int(args.log_every)),
    ]

    flat: List[FlatResult] = []
    total = len(seeds) * len(cand_sets)
    idx = 0
    for seed in seeds:
        for cand in cand_sets:
            idx += 1
            trial_root = os.path.join(save_root, f"seed{seed}", f"cand_{cand.replace(',', '-')}")
            cmd = base + [
                "--save_dir",
                trial_root,
                "--seed",
                str(int(seed)),
                f"--candidates={cand}",
            ]

            print(f"[scan] ({idx}/{total}) seed={seed} candidates={cand}")
            stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
            run_dir = _parse_run_dir(stdout)

            rec: Dict[str, Any] = {
                "seed": int(seed),
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
                candidate_set=str(cand),
                run_dir=str(run_dir),
                elapsed_s=float(elapsed),
                summary=summary,
            )
            flat.append(fr)
            e3 = fr.e3()
            if int(args.gate) == 1:
                passed, checks = _gate_e3_v0(e3, online_steps=int(args.online_steps), args=args)
                rec["gate_passed"] = bool(passed)
                rec["gate_checks"] = checks
            rec["E3"] = e3
            _write_jsonl(results_path, rec)

    # Aggregate per candidate set
    groups: Dict[str, Dict[str, Any]] = {}
    for fr in flat:
        e3 = fr.e3()
        cand = str(fr.candidate_set)
        g = groups.setdefault(cand, {"runs": 0, "seeds": [], "means": {}, "passes": 0})
        g["runs"] += 1
        g["seeds"].append(int(fr.seed))
        if int(args.gate) == 1:
            passed, _ = _gate_e3_v0(e3, online_steps=int(args.online_steps), args=args)
            g["passes"] += int(bool(passed))

    def _mean(vals: List[float]) -> float:
        xs = [float(x) for x in vals if _isfinite(x)]
        return float(sum(xs) / max(1, len(xs))) if xs else float("nan")

    for cand, g in groups.items():
        vals: Dict[str, List[float]] = {
            "hit_rate": [],
            "first_hit_median": [],
            "dwell_hit_mean": [],
            "span_hit_mean": [],
            "probe_auc_hit": [],
            "probe_auc": [],
            "probe_auc_shuffled": [],
            "delta_abs_mean": [],
            "near_boundary_rate": [],
        }
        for fr in flat:
            if fr.candidate_set != cand:
                continue
            e3 = fr.e3()
            vals["hit_rate"].append(float(e3.get("hit_rate", float("nan"))))
            vals["first_hit_median"].append(float(e3.get("first_hit_median", float("nan"))))
            vals["dwell_hit_mean"].append(float(e3.get("dwell_hit_mean", float("nan"))))
            vals["span_hit_mean"].append(float(e3.get("span_hit_mean", float("nan"))))
            vals["probe_auc_hit"].append(float(e3.get("probe_auc_hit", float("nan"))))
            vals["probe_auc"].append(float(e3.get("probe_auc", float("nan"))))
            vals["probe_auc_shuffled"].append(float(e3.get("probe_auc_shuffled", float("nan"))))
            vals["delta_abs_mean"].append(float(e3.get("delta_abs_mean", float("nan"))))
            vals["near_boundary_rate"].append(float(e3.get("near_boundary_rate", float("nan"))))
        g["means"] = {k: _mean(v) for k, v in vals.items()}

    out = {
        "config": {
            "device": str(args.device),
            "seeds": seeds,
            "candidate_sets": cand_sets,
            "e3_policy": str(args.e3_policy),
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
