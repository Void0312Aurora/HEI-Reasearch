#!/usr/bin/env python3
"""
E3p-L1: "speak is not a brittle heuristic" ablation/transfer gate for L7v2-v0.

This script focuses on the E3+ efficiency tier (kv_swap + query, key_len=1, steps=80),
and optionally runs:
  - baseline (val_align=0, speak=0)
  - speak-mix (val_align=0, speak=1, speak_use=2)

Motivation (per temp-11):
  - Rule out false positives: speak should remain effective when candidate multiscale is removed.
  - Check minimal transfer: query_key changes (k1/k2/random) should not break the gate.

Outputs:
  - gate_summary.json (full per-run metrics, checks, and deltas vs baseline).
Exit code:
  - 0 if all enabled speak runs pass the thresholds (and runner succeeded)
  - 1 otherwise
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


def _parse_csv_strs(s: str) -> List[str]:
    out: List[str] = []
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
    return out


def _parse_candidate_sets(s: str) -> List[str]:
    """
    Semicolon-separated candidate sets; each set is a CSV list.
    Example: "1,2,4,8,16,32;32"
    """
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


def _tail(text: str, n: int = 80) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[-n:])


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


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def _gate_e3plus(e3: Dict[str, Any], *, args: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    resolved_any = float(e3.get("resolved_any_rate", float("nan")))
    query_acc_hit = float(e3.get("query_acc_hit", float("nan")))
    checks = {
        "resolved_any_rate": _finite(resolved_any) and resolved_any >= float(args.gate_resolved_any_rate),
        "query_acc_hit": _finite(query_acc_hit) and query_acc_hit >= float(args.gate_query_acc_hit),
    }
    return all(checks.values()), checks


@dataclass
class GateRun:
    variant: str
    seed: int
    query_key: str
    candidate_set: str
    run_dir: Optional[str]
    returncode: int
    elapsed_s: float
    passed: bool
    checks: Dict[str, bool]
    e3: Dict[str, Any]
    stdout_tail: str
    stderr_tail: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="checkpoints/l7v2_v0_gate_e3p_l1")
    ap.add_argument("--timeout_s", type=int, default=7200)

    ap.add_argument("--seeds", type=str, default="2")
    ap.add_argument("--query_keys", type=str, default="k1,k2,random")
    ap.add_argument("--candidate_sets", type=str, default="1,2,4,8,16,32;32")
    ap.add_argument("--run_baseline", type=int, default=1)
    ap.add_argument("--run_speak", type=int, default=1)

    # Core model knobs (match E3+ efficiency tier defaults).
    ap.add_argument("--dim_q", type=int, default=64)
    ap.add_argument("--num_charts", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--pointer_bounds", type=str, default="wrap", choices=["clamp", "reflect", "wrap"])
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--cost_weight", type=float, default=0.001)
    ap.add_argument("--anti_lock_weight", type=float, default=0.1)
    ap.add_argument("--anti_lock_sigma", type=float, default=0.5)
    ap.add_argument("--cover_weight", type=float, default=0.1)
    ap.add_argument("--cover_window", type=int, default=64)
    ap.add_argument("--cover_sigma", type=float, default=2.0)
    ap.add_argument("--cover_warmup_frac", type=float, default=0.3)

    # E3+ kv_swap config.
    ap.add_argument("--e3_seq_len", type=int, default=4096)
    ap.add_argument("--e3_value_len", type=int, default=32)
    ap.add_argument("--e3_key_len", type=int, default=1)
    ap.add_argument("--e3_clue_margin", type=int, default=64)
    ap.add_argument("--e3_pairs", type=int, default=8)
    ap.add_argument("--online_steps", type=int, default=80)
    ap.add_argument("--offline_steps", type=int, default=20)
    ap.add_argument("--pointer_drift", type=float, default=0.0)
    ap.add_argument("--cost_power", type=float, default=1.0)
    ap.add_argument("--fixate_steps", type=int, default=8)
    ap.add_argument("--fixate_mad_mult", type=float, default=6.0)

    ap.add_argument("--query_steps", type=int, default=8)

    # Speak config (variant=speak)
    ap.add_argument("--speak_vocab", type=int, default=2)
    ap.add_argument("--speak_input", type=str, default="y", choices=["q", "y"])
    ap.add_argument("--speak_loss_weight", type=float, default=0.1)
    ap.add_argument("--speak_use", type=int, default=2)
    ap.add_argument("--speak_back_threshold", type=float, default=0.9)
    ap.add_argument("--speak_pos_weight", type=float, default=0.0)
    ap.add_argument("--speak_pos_weight_max", type=float, default=50.0)
    ap.add_argument("--speak_mix_sharpness", type=float, default=8.0)

    # Gate thresholds (aligned with E3+ efficiency tier).
    ap.add_argument("--gate_resolved_any_rate", type=float, default=0.18)
    ap.add_argument("--gate_query_acc_hit", type=float, default=0.95)

    args = ap.parse_args()

    os.makedirs(str(args.save_root), exist_ok=True)
    out_path = os.path.join(str(args.save_root), "gate_summary.json")

    seeds = _parse_csv_ints(str(args.seeds))
    query_keys = _parse_csv_strs(str(args.query_keys))
    cand_sets = _parse_candidate_sets(str(args.candidate_sets))
    if not query_keys:
        raise ValueError("empty --query_keys")
    if not cand_sets:
        raise ValueError("empty --candidate_sets")

    runner = os.path.join(os.path.dirname(__file__), "..", "run_l7v2_v0.py")

    def _common(seed: int, *, query_key: str, cand_set: str, save_dir: str) -> List[str]:
        return [
            sys.executable,
            runner,
            "--device",
            str(args.device),
            "--save_dir",
            str(save_dir),
            "--seed",
            str(int(seed)),
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
            f"--candidates={str(cand_set)}",
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
            str(query_key),
        ]

    runs: List[GateRun] = []
    any_fail = False
    by_key: Dict[str, Dict[str, GateRun]] = {}

    for seed in seeds:
        for query_key in query_keys:
            for cand_set in cand_sets:
                key = f"seed{int(seed)}|q={query_key}|cand={cand_set}"
                by_key.setdefault(key, {})

                if int(args.run_baseline) != 0:
                    cmd = _common(
                        int(seed),
                        query_key=str(query_key),
                        cand_set=str(cand_set),
                        save_dir=os.path.join(str(args.save_root), "baseline"),
                    ) + [
                        "--e3_val_align",
                        "0",
                        "--e3_speak",
                        "0",
                    ]
                    stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
                    run_dir = _parse_run_dir(stdout)
                    e3 = {}
                    checks: Dict[str, bool] = {}
                    passed = False
                    if code == 0 and run_dir:
                        e3 = dict((_load_summary(run_dir).get("E3") or {}) if run_dir else {})
                        passed, checks = _gate_e3plus(e3, args=args)
                    else:
                        checks = {"runner_ok": False}
                    runs.append(
                        GateRun(
                            variant="baseline",
                            seed=int(seed),
                            query_key=str(query_key),
                            candidate_set=str(cand_set),
                            run_dir=run_dir,
                            returncode=int(code),
                            elapsed_s=float(elapsed),
                            passed=bool(passed),
                            checks=checks,
                            e3=e3,
                            stdout_tail=_tail(stdout),
                            stderr_tail=_tail(stderr),
                        )
                    )
                    by_key[key]["baseline"] = runs[-1]

                if int(args.run_speak) != 0:
                    cmd = _common(
                        int(seed),
                        query_key=str(query_key),
                        cand_set=str(cand_set),
                        save_dir=os.path.join(str(args.save_root), "speak"),
                    ) + [
                        "--e3_val_align",
                        "0",
                        "--e3_speak",
                        "1",
                        "--e3_speak_vocab",
                        str(int(args.speak_vocab)),
                        "--e3_speak_input",
                        str(args.speak_input),
                        "--e3_speak_loss_weight",
                        str(float(args.speak_loss_weight)),
                        "--e3_speak_use",
                        str(int(args.speak_use)),
                        "--e3_speak_back_threshold",
                        str(float(args.speak_back_threshold)),
                        "--e3_speak_pos_weight",
                        str(float(args.speak_pos_weight)),
                        "--e3_speak_pos_weight_max",
                        str(float(args.speak_pos_weight_max)),
                        "--e3_speak_mix_sharpness",
                        str(float(args.speak_mix_sharpness)),
                    ]
                    stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
                    run_dir = _parse_run_dir(stdout)
                    e3 = {}
                    checks = {}
                    passed = False
                    if code == 0 and run_dir:
                        e3 = dict((_load_summary(run_dir).get("E3") or {}) if run_dir else {})
                        passed, checks = _gate_e3plus(e3, args=args)
                    else:
                        checks = {"runner_ok": False}
                    any_fail = any_fail or (not passed)
                    runs.append(
                        GateRun(
                            variant="speak",
                            seed=int(seed),
                            query_key=str(query_key),
                            candidate_set=str(cand_set),
                            run_dir=run_dir,
                            returncode=int(code),
                            elapsed_s=float(elapsed),
                            passed=bool(passed),
                            checks=checks,
                            e3=e3,
                            stdout_tail=_tail(stdout),
                            stderr_tail=_tail(stderr),
                        )
                    )
                    by_key[key]["speak"] = runs[-1]

    # Deltas vs baseline (resolved_any_rate, query_acc_hit).
    deltas: Dict[str, Dict[str, float]] = {}
    for key, variants in by_key.items():
        base = variants.get("baseline")
        spk = variants.get("speak")
        if base is None or spk is None:
            continue
        b = base.e3
        s = spk.e3
        try:
            b_res = float(b.get("resolved_any_rate", float("nan")))
            s_res = float(s.get("resolved_any_rate", float("nan")))
            b_q = float(b.get("query_acc_hit", float("nan")))
            s_q = float(s.get("query_acc_hit", float("nan")))
        except Exception:
            continue
        deltas[key] = {
            "resolved_any_rate": (s_res - b_res) if (_finite(b_res) and _finite(s_res)) else float("nan"),
            "query_acc_hit": (s_q - b_q) if (_finite(b_q) and _finite(s_q)) else float("nan"),
        }

    out = {
        "config": vars(args),
        "runs": [r.__dict__ for r in runs],
        "deltas_vs_baseline": deltas,
        "passed": (not any_fail) if int(args.run_speak) != 0 else True,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[gate] wrote: {out_path}")
    print(f"[gate] passed={out['passed']}")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
