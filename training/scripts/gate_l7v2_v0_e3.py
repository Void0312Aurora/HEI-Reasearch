#!/usr/bin/env python3
"""
Unified E3 gate runner for L7v2-v0.

This script runs a small, reproducible suite and emits a single PASS/FAIL exit code:
  - E3-v0 (marker_patch): proves "hit -> offline write -> separable" on 4096×32.
  - E3+  (kv_swap + query): proves "hit -> can answer" and measures resolved_any_rate.
  - E3+  efficiency tier (kv_swap + query): key_len=1 at steps=80 must still pass.
  - E3-text (kv_text + text query): proves "natural-language surface doc + text query -> answer + citation"
    and checks cite_hit_rate for resolved queries.

Defaults align with the v0p10 baseline/curriculum recorded under:
  HEI/docs/temp/2026/1/A/1·15/01/temp-04.md

Outputs:
  - gate_summary.json (full per-run metrics and checks)
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


def _tail(text: str, n: int = 60) -> str:
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
    return v == v and abs(v) != float("inf")


def _gate_e3_v0(e3: Dict[str, Any], *, online_steps: int, args: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    hit_rate = float(e3.get("hit_rate", float("nan")))
    first_hit_median = float(e3.get("first_hit_median", float("nan")))
    dwell_hit_mean = float(e3.get("dwell_hit_mean", float("nan")))
    probe_auc_hit = float(e3.get("probe_auc_hit", float("nan")))
    probe_auc_shuf = float(e3.get("probe_auc_shuffled", float("nan")))

    thr_first = float(args.e3v0_gate_first_hit_frac) * float(max(1, int(online_steps)))
    checks = {
        "hit_rate": _finite(hit_rate) and hit_rate >= float(args.e3v0_gate_hit_rate),
        "first_hit_median": _finite(first_hit_median) and first_hit_median <= thr_first,
        "dwell_hit_mean": _finite(dwell_hit_mean) and dwell_hit_mean >= float(args.e3v0_gate_dwell_hit_mean),
        "probe_auc_hit": _finite(probe_auc_hit) and probe_auc_hit >= float(args.e3v0_gate_probe_auc_hit),
        "probe_auc_shuffled": _finite(probe_auc_shuf)
        and float(args.e3v0_gate_shuf_low) <= probe_auc_shuf <= float(args.e3v0_gate_shuf_high),
    }
    return all(checks.values()), checks


def _gate_e3_plus(e3: Dict[str, Any], *, args: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    resolved_any = float(e3.get("resolved_any_rate", float("nan")))
    query_acc_hit = float(e3.get("query_acc_hit", float("nan")))
    checks = {
        "resolved_any_rate": _finite(resolved_any) and resolved_any >= float(args.e3p_gate_resolved_any_rate),
        "query_acc_hit": _finite(query_acc_hit) and query_acc_hit >= float(args.e3p_gate_query_acc_hit),
    }
    return all(checks.values()), checks


def _gate_e3_text(e3: Dict[str, Any], *, args: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    resolved_any = float(e3.get("resolved_any_rate", float("nan")))
    resolved = float(e3.get("resolved_rate", float("nan")))
    query_acc_hit = float(e3.get("query_acc_hit", float("nan")))
    cite_hit_rate = float(e3.get("cite_hit_rate", float("nan")))
    checks = {
        "resolved_any_rate": _finite(resolved_any) and resolved_any >= float(args.e3text_gate_resolved_any_rate),
        "resolved_rate": _finite(resolved) and resolved >= float(args.e3text_gate_resolved_rate),
        "query_acc_hit": _finite(query_acc_hit) and query_acc_hit >= float(args.e3text_gate_query_acc_hit),
        "cite_hit_rate": _finite(cite_hit_rate) and cite_hit_rate >= float(args.e3text_gate_cite_hit_rate),
    }
    return all(checks.values()), checks


def _gate_e3_text_both(e3: Dict[str, Any], *, args: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    resolved = float(e3.get("resolved_rate", float("nan")))
    resolved_strict = float(e3.get("resolved_strict_rate", float("nan")))
    query_acc_hit = float(e3.get("query_acc_hit", float("nan")))
    combine_acc_hit = float(e3.get("combine_acc_hit", float("nan")))
    cite_hit_rate = float(e3.get("cite_hit_rate", float("nan")))
    checks = {
        "resolved_rate": _finite(resolved) and resolved >= float(args.e3text_both_gate_resolved_rate),
        "query_acc_hit": _finite(query_acc_hit) and query_acc_hit >= float(args.e3text_both_gate_query_acc_hit),
        "cite_hit_rate": _finite(cite_hit_rate) and cite_hit_rate >= float(args.e3text_both_gate_cite_hit_rate),
    }
    comb_thr = float(args.e3text_both_gate_combine_acc_hit)
    if comb_thr > 0.0:
        checks["combine_acc_hit"] = _finite(combine_acc_hit) and combine_acc_hit >= comb_thr
    strict_thr = float(args.e3text_both_gate_resolved_strict_rate)
    if strict_thr > 0.0:
        checks["resolved_strict_rate"] = _finite(resolved_strict) and resolved_strict >= strict_thr
    return all(checks.values()), checks


@dataclass
class GateRun:
    suite: str
    seed: int
    run_dir: Optional[str]
    returncode: int
    elapsed_s: float
    checks: Dict[str, bool]
    passed: bool
    e3: Dict[str, Any]
    stdout_tail: str
    stderr_tail: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="checkpoints/l7v2_v0_gate_e3")
    ap.add_argument("--timeout_s", type=int, default=7200)
    ap.add_argument("--seeds", type=str, default="0,1,2")

    # Common model sizes
    ap.add_argument("--dim_q", type=int, default=64)
    ap.add_argument("--num_charts", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.1)

    # Shared environment knobs
    ap.add_argument("--pointer_bounds", type=str, default="wrap", choices=["clamp", "reflect", "wrap"])
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--cost_weight", type=float, default=0.001)
    ap.add_argument("--anti_lock_weight", type=float, default=0.1)
    ap.add_argument("--anti_lock_sigma", type=float, default=0.5)
    ap.add_argument("--cover_weight", type=float, default=0.1)
    ap.add_argument("--cover_window", type=int, default=64)
    ap.add_argument("--cover_sigma", type=float, default=2.0)
    ap.add_argument("--cover_warmup_frac", type=float, default=0.3)

    # E3-v0 baseline config
    ap.add_argument("--e3v0_seq_len", type=int, default=4096)
    ap.add_argument("--e3v0_clue_len", type=int, default=32)
    ap.add_argument("--e3v0_clue_margin", type=int, default=64)
    ap.add_argument("--e3v0_pairs", type=int, default=8)
    ap.add_argument("--e3v0_online_steps", type=int, default=80)
    ap.add_argument("--e3v0_offline_steps", type=int, default=20)
    ap.add_argument("--e3v0_candidates", type=str, default="32")
    ap.add_argument("--e3v0_pointer_drift", type=float, default=0.3)
    ap.add_argument("--e3v0_cost_power", type=float, default=2.0)

    # E3+ curriculum config
    ap.add_argument("--e3p_seq_len", type=int, default=4096)
    ap.add_argument("--e3p_value_len", type=int, default=32)
    ap.add_argument("--e3p_clue_margin", type=int, default=64)
    ap.add_argument("--e3p_pairs", type=int, default=8)
    ap.add_argument("--e3p_candidates", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--e3p_pointer_drift", type=float, default=0.0)
    ap.add_argument("--e3p_cost_power", type=float, default=1.0)
    ap.add_argument("--e3p_query_steps", type=int, default=8)
    ap.add_argument("--e3p_query_key", type=str, default="k1", choices=["k1", "k2", "random"])

    ap.add_argument("--e3p_key_len_a", type=int, default=2)
    ap.add_argument("--e3p_online_steps_a", type=int, default=80)
    ap.add_argument("--e3p_key_len_b", type=int, default=1)
    ap.add_argument("--e3p_online_steps_b", type=int, default=120)
    ap.add_argument("--e3p_online_steps_c", type=int, default=80, help="E3+ efficiency tier online_steps.")
    ap.add_argument("--e3p_val_align_c", type=int, default=1, help="E3+ efficiency tier: enable --e3_val_align.")
    ap.add_argument(
        "--e3p_eff_speak",
        type=int,
        default=0,
        help="E3+ efficiency tier: enable de-privileged speak control (forces --e3_val_align=0).",
    )
    ap.add_argument("--e3p_eff_speak_vocab", type=int, default=2, help="E3 speak vocab size (>=2).")
    ap.add_argument(
        "--e3p_eff_speak_input",
        type=str,
        default="y",
        choices=["q", "y"],
        help="E3 speak input (q=internal state, y=current sensory read).",
    )
    ap.add_argument(
        "--e3p_eff_speak_loss_weight",
        type=float,
        default=0.1,
        help="E3 speak supervised loss weight (0 disables).",
    )
    ap.add_argument(
        "--e3p_eff_speak_use",
        type=int,
        default=2,
        help="E3 speak control (0=off, 1=threshold, 2=mix into lookahead).",
    )
    ap.add_argument(
        "--e3p_eff_speak_back_threshold",
        type=float,
        default=0.9,
        help="E3 speak BACK threshold when --e3_speak_use=1.",
    )
    ap.add_argument(
        "--e3p_eff_speak_pos_weight",
        type=float,
        default=0.0,
        help="E3 speak pos_weight for imbalance (0=auto).",
    )
    ap.add_argument(
        "--e3p_eff_speak_pos_weight_max",
        type=float,
        default=50.0,
        help="E3 speak max pos_weight (auto or clip manual).",
    )
    ap.add_argument(
        "--e3p_eff_speak_mix_sharpness",
        type=float,
        default=6.0,
        help="E3 speak mix-control sharpness k for p_mix = sigmoid(k * logit_diff).",
    )
    ap.add_argument("--e3p_offline_steps", type=int, default=20)

    # Gate thresholds (aligned with temp-03/temp-04)
    ap.add_argument("--e3v0_gate_hit_rate", type=float, default=0.6)
    ap.add_argument("--e3v0_gate_first_hit_frac", type=float, default=0.7)
    ap.add_argument("--e3v0_gate_dwell_hit_mean", type=float, default=6.0)
    ap.add_argument("--e3v0_gate_probe_auc_hit", type=float, default=0.95)
    ap.add_argument("--e3v0_gate_shuf_low", type=float, default=0.45)
    ap.add_argument("--e3v0_gate_shuf_high", type=float, default=0.55)

    ap.add_argument("--e3p_gate_resolved_any_rate", type=float, default=0.18)
    ap.add_argument("--e3p_gate_query_acc_hit", type=float, default=0.95)

    # E3-text config (kv_text + text query + citation)
    ap.add_argument("--e3text_seq_len", type=int, default=4096)
    ap.add_argument("--e3text_value_len", type=int, default=32)
    ap.add_argument("--e3text_key_len", type=int, default=1)
    ap.add_argument("--e3text_clue_margin", type=int, default=64)
    ap.add_argument("--e3text_pairs", type=int, default=8)
    ap.add_argument("--e3text_online_steps", type=int, default=80)
    ap.add_argument("--e3text_offline_steps", type=int, default=20)
    ap.add_argument("--e3text_candidates", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--e3text_pointer_drift", type=float, default=0.0)
    ap.add_argument("--e3text_cost_power", type=float, default=1.0)
    ap.add_argument("--e3text_query_steps", type=int, default=8)
    ap.add_argument("--e3text_query_key", type=str, default="k1", choices=["k1", "k2", "random"])
    ap.add_argument("--e3text_kv_text_gap", type=int, default=3)

    ap.add_argument("--e3text_gate_resolved_any_rate", type=float, default=0.13)
    ap.add_argument("--e3text_gate_resolved_rate", type=float, default=0.06)
    ap.add_argument("--e3text_gate_query_acc_hit", type=float, default=0.95)
    ap.add_argument("--e3text_gate_cite_hit_rate", type=float, default=0.95)

    # E3-text both-mode (dual evidence + combine via joint correctness)
    ap.add_argument("--run_e3text_both", type=int, default=0, help="If 1, also run E3-text query_mode=both suite.")
    ap.add_argument("--e3text_both_window", type=int, default=128, help="K2 is placed within +/-window of K1.")
    ap.add_argument("--e3text_both_query_steps", type=int, default=32, help="Query/readout steps per key in both-mode.")
    ap.add_argument(
        "--e3text_both_key_seek",
        type=int,
        default=1,
        help="If 1, enable E3 kv_text value→key micro-read alignment (raises resolved_strict_rate).",
    )
    ap.add_argument(
        "--e3text_both_key_seek_steps",
        type=int,
        default=40,
        help="E3 kv_text key-seek: steps to keep key-seek active after entering a value span (per sample).",
    )
    ap.add_argument(
        "--e3text_both_eval_samples",
        type=int,
        default=1024,
        help="Approx eval samples for BOTH suite (n = pairs * 2 * batch_size). 0 => use --e3text_pairs.",
    )
    ap.add_argument("--e3text_both_comb_mode", type=str, default="add_mod", choices=["concat", "add_mod"])
    ap.add_argument("--e3text_both_comb_mod", type=int, default=256, help="COMB-v1 modulus M (and value pool size).")
    ap.add_argument("--e3text_both_gate_resolved_rate", type=float, default=0.05)
    ap.add_argument(
        "--e3text_both_gate_resolved_strict_rate",
        type=float,
        default=0.0,
        help="If >0, also require resolved_strict_rate >= thr (tighten after window sweep).",
    )
    ap.add_argument("--e3text_both_gate_query_acc_hit", type=float, default=0.95)
    ap.add_argument("--e3text_both_gate_combine_acc_hit", type=float, default=0.95)
    ap.add_argument("--e3text_both_gate_cite_hit_rate", type=float, default=0.95)

    ap.add_argument("--log_every", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(str(args.save_root), exist_ok=True)
    out_path = os.path.join(str(args.save_root), "gate_summary.json")

    seeds = _parse_csv_ints(str(args.seeds))
    runner = os.path.join(os.path.dirname(__file__), "..", "run_l7v2_v0.py")

    def _common_base() -> List[str]:
        return [
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
            "--run_e0",
            "0",
            "--run_e1e2",
            "0",
            "--run_e3",
            "1",
            "--log_every",
            str(int(args.log_every)),
        ]

    runs: List[GateRun] = []
    any_fail = False

    for seed in seeds:
        # --- E3-v0 (marker_patch) ---
        cmd_v0 = _common_base() + [
            "--save_dir",
            os.path.join(str(args.save_root), "e3v0_marker_patch"),
            "--seed",
            str(int(seed)),
            "--online_steps",
            str(int(args.e3v0_online_steps)),
            "--offline_steps",
            str(int(args.e3v0_offline_steps)),
            "--pointer_drift",
            str(float(args.e3v0_pointer_drift)),
            "--cost_power",
            str(float(args.e3v0_cost_power)),
            f"--candidates={str(args.e3v0_candidates)}",
            "--e3_task",
            "marker_patch",
            "--e3_pairs",
            str(int(args.e3v0_pairs)),
            "--e3_seq_len",
            str(int(args.e3v0_seq_len)),
            "--e3_clue_len",
            str(int(args.e3v0_clue_len)),
            "--e3_clue_margin",
            str(int(args.e3v0_clue_margin)),
            "--e3_policy",
            "active_fixate",
            "--e3_fixate_steps",
            "8",
            "--e3_fixate_mad_mult",
            "6.0",
        ]
        stdout, stderr, code, elapsed = _run_once(cmd_v0, timeout_s=int(args.timeout_s))
        run_dir = _parse_run_dir(stdout)
        e3 = {}
        checks: Dict[str, bool] = {}
        passed = False
        if code == 0 and run_dir:
            e3 = dict((_load_summary(run_dir).get("E3") or {}) if run_dir else {})
            passed, checks = _gate_e3_v0(e3, online_steps=int(args.e3v0_online_steps), args=args)
        else:
            checks = {"runner_ok": False}
        any_fail = any_fail or (not passed)
        runs.append(
            GateRun(
                suite="E3-v0",
                seed=int(seed),
                run_dir=run_dir,
                returncode=int(code),
                elapsed_s=float(elapsed),
                checks=checks,
                passed=bool(passed),
                e3=e3,
                stdout_tail=_tail(stdout),
                stderr_tail=_tail(stderr),
            )
        )

        # --- E3+ Level A (key_len=2) ---
        cmd_k2 = _common_base() + [
            "--save_dir",
            os.path.join(str(args.save_root), "e3p_kv_keylen2"),
            "--seed",
            str(int(seed)),
            "--online_steps",
            str(int(args.e3p_online_steps_a)),
            "--offline_steps",
            str(int(args.e3p_offline_steps)),
            "--pointer_drift",
            str(float(args.e3p_pointer_drift)),
            "--cost_power",
            str(float(args.e3p_cost_power)),
            f"--candidates={str(args.e3p_candidates)}",
            "--e3_task",
            "kv_swap",
            "--e3_pairs",
            str(int(args.e3p_pairs)),
            "--e3_seq_len",
            str(int(args.e3p_seq_len)),
            "--e3_clue_len",
            str(int(args.e3p_value_len)),
            "--e3_key_len",
            str(int(args.e3p_key_len_a)),
            "--e3_clue_margin",
            str(int(args.e3p_clue_margin)),
            "--e3_policy",
            "active_fixate",
            "--e3_fixate_steps",
            "8",
            "--e3_fixate_mad_mult",
            "6.0",
            "--e3_fixate_mode",
            "freeze",
            "--e3_query_steps",
            str(int(args.e3p_query_steps)),
            "--e3_query_key",
            str(args.e3p_query_key),
        ]
        stdout, stderr, code, elapsed = _run_once(cmd_k2, timeout_s=int(args.timeout_s))
        run_dir = _parse_run_dir(stdout)
        e3 = {}
        checks = {}
        passed = False
        if code == 0 and run_dir:
            e3 = dict((_load_summary(run_dir).get("E3") or {}) if run_dir else {})
            passed, checks = _gate_e3_plus(e3, args=args)
        else:
            checks = {"runner_ok": False}
        any_fail = any_fail or (not passed)
        runs.append(
            GateRun(
                suite="E3+ key_len=2",
                seed=int(seed),
                run_dir=run_dir,
                returncode=int(code),
                elapsed_s=float(elapsed),
                checks=checks,
                passed=bool(passed),
                e3=e3,
                stdout_tail=_tail(stdout),
                stderr_tail=_tail(stderr),
            )
        )

        # --- E3+ Level B (key_len=1) ---
        cmd_k1 = _common_base() + [
            "--save_dir",
            os.path.join(str(args.save_root), "e3p_kv_keylen1"),
            "--seed",
            str(int(seed)),
            "--online_steps",
            str(int(args.e3p_online_steps_b)),
            "--offline_steps",
            str(int(args.e3p_offline_steps)),
            "--pointer_drift",
            str(float(args.e3p_pointer_drift)),
            "--cost_power",
            str(float(args.e3p_cost_power)),
            f"--candidates={str(args.e3p_candidates)}",
            "--e3_task",
            "kv_swap",
            "--e3_pairs",
            str(int(args.e3p_pairs)),
            "--e3_seq_len",
            str(int(args.e3p_seq_len)),
            "--e3_clue_len",
            str(int(args.e3p_value_len)),
            "--e3_key_len",
            str(int(args.e3p_key_len_b)),
            "--e3_clue_margin",
            str(int(args.e3p_clue_margin)),
            "--e3_policy",
            "active_fixate",
            "--e3_fixate_steps",
            "8",
            "--e3_fixate_mad_mult",
            "6.0",
            "--e3_fixate_mode",
            "freeze",
            "--e3_query_steps",
            str(int(args.e3p_query_steps)),
            "--e3_query_key",
            str(args.e3p_query_key),
        ]
        stdout, stderr, code, elapsed = _run_once(cmd_k1, timeout_s=int(args.timeout_s))
        run_dir = _parse_run_dir(stdout)
        e3 = {}
        checks = {}
        passed = False
        if code == 0 and run_dir:
            e3 = dict((_load_summary(run_dir).get("E3") or {}) if run_dir else {})
            passed, checks = _gate_e3_plus(e3, args=args)
        else:
            checks = {"runner_ok": False}
        any_fail = any_fail or (not passed)
        runs.append(
            GateRun(
                suite="E3+ key_len=1",
                seed=int(seed),
                run_dir=run_dir,
                returncode=int(code),
                elapsed_s=float(elapsed),
                checks=checks,
                passed=bool(passed),
                e3=e3,
                stdout_tail=_tail(stdout),
                stderr_tail=_tail(stderr),
            )
        )

        # --- E3+ Level C (efficiency: key_len=1, steps=80) ---
        cmd_eff = _common_base() + [
            "--save_dir",
            os.path.join(str(args.save_root), "e3p_kv_keylen1_eff"),
            "--seed",
            str(int(seed)),
            "--online_steps",
            str(int(args.e3p_online_steps_c)),
            "--offline_steps",
            str(int(args.e3p_offline_steps)),
            "--pointer_drift",
            str(float(args.e3p_pointer_drift)),
            "--cost_power",
            str(float(args.e3p_cost_power)),
            f"--candidates={str(args.e3p_candidates)}",
            "--e3_task",
            "kv_swap",
            "--e3_pairs",
            str(int(args.e3p_pairs)),
            "--e3_seq_len",
            str(int(args.e3p_seq_len)),
            "--e3_clue_len",
            str(int(args.e3p_value_len)),
            "--e3_key_len",
            str(int(args.e3p_key_len_b)),
            "--e3_clue_margin",
            str(int(args.e3p_clue_margin)),
            "--e3_policy",
            "active_fixate",
            "--e3_fixate_steps",
            "8",
            "--e3_fixate_mad_mult",
            "6.0",
            "--e3_fixate_mode",
            "freeze",
            "--e3_query_steps",
            str(int(args.e3p_query_steps)),
            "--e3_query_key",
            str(args.e3p_query_key),
        ]
        if int(args.e3p_eff_speak) != 0:
            cmd_eff += [
                "--e3_val_align",
                "0",
                "--e3_speak",
                "1",
                "--e3_speak_vocab",
                str(int(args.e3p_eff_speak_vocab)),
                "--e3_speak_input",
                str(args.e3p_eff_speak_input),
                "--e3_speak_loss_weight",
                str(float(args.e3p_eff_speak_loss_weight)),
                "--e3_speak_use",
                str(int(args.e3p_eff_speak_use)),
                "--e3_speak_back_threshold",
                str(float(args.e3p_eff_speak_back_threshold)),
                "--e3_speak_pos_weight",
                str(float(args.e3p_eff_speak_pos_weight)),
                "--e3_speak_pos_weight_max",
                str(float(args.e3p_eff_speak_pos_weight_max)),
                "--e3_speak_mix_sharpness",
                str(float(args.e3p_eff_speak_mix_sharpness)),
            ]
        else:
            cmd_eff += [
                "--e3_val_align",
                str(int(args.e3p_val_align_c)),
            ]
        stdout, stderr, code, elapsed = _run_once(cmd_eff, timeout_s=int(args.timeout_s))
        run_dir = _parse_run_dir(stdout)
        e3 = {}
        checks = {}
        passed = False
        if code == 0 and run_dir:
            e3 = dict((_load_summary(run_dir).get("E3") or {}) if run_dir else {})
            passed, checks = _gate_e3_plus(e3, args=args)
        else:
            checks = {"runner_ok": False}
        any_fail = any_fail or (not passed)
        runs.append(
            GateRun(
                suite="E3+ key_len=1 (efficiency)",
                seed=int(seed),
                run_dir=run_dir,
                returncode=int(code),
                elapsed_s=float(elapsed),
                checks=checks,
                passed=bool(passed),
                e3=e3,
                stdout_tail=_tail(stdout),
                stderr_tail=_tail(stderr),
            )
        )

        # --- E3-text (kv_text + text query + citation) ---
        cmd_text = _common_base() + [
            "--save_dir",
            os.path.join(str(args.save_root), "e3text_kv_text"),
            "--seed",
            str(int(seed)),
            "--online_steps",
            str(int(args.e3text_online_steps)),
            "--offline_steps",
            str(int(args.e3text_offline_steps)),
            "--pointer_drift",
            str(float(args.e3text_pointer_drift)),
            "--cost_power",
            str(float(args.e3text_cost_power)),
            f"--candidates={str(args.e3text_candidates)}",
            "--e3_task",
            "kv_text",
            "--e3_pairs",
            str(int(args.e3text_pairs)),
            "--e3_seq_len",
            str(int(args.e3text_seq_len)),
            "--e3_clue_len",
            str(int(args.e3text_value_len)),
            "--e3_key_len",
            str(int(args.e3text_key_len)),
            "--e3_clue_margin",
            str(int(args.e3text_clue_margin)),
            "--e3_kv_text_gap",
            str(int(args.e3text_kv_text_gap)),
            "--e3_policy",
            "active_fixate",
            "--e3_fixate_steps",
            "8",
            "--e3_fixate_mad_mult",
            "6.0",
            "--e3_fixate_mode",
            "freeze",
            "--e3_query_steps",
            str(int(args.e3text_query_steps)),
            "--e3_query_key",
            str(args.e3text_query_key),
            "--e3_query_input",
            "text",
            "--e3_two_phase",
            "1",
            "--e3_two_phase_speak_thresh",
            "0.6",
            "--e3_two_phase_trigger",
            "back",
            "--e3_two_phase_back_delta",
            "4.0",
            "--e3_val_align",
            "0",
            "--e3_speak",
            "1",
            "--e3_speak_vocab",
            "3",
            "--e3_speak_input",
            "y",
            "--e3_speak_loss_weight",
            "0.1",
            "--e3_speak_use",
            "2",
            "--e3_speak_mix_sharpness",
            "6.0",
        ]
        stdout, stderr, code, elapsed = _run_once(cmd_text, timeout_s=int(args.timeout_s))
        run_dir = _parse_run_dir(stdout)
        e3 = {}
        checks = {}
        passed = False
        if code == 0 and run_dir:
            e3 = dict((_load_summary(run_dir).get("E3") or {}) if run_dir else {})
            passed, checks = _gate_e3_text(e3, args=args)
        else:
            checks = {"runner_ok": False}
        any_fail = any_fail or (not passed)
        runs.append(
            GateRun(
                suite="E3-text (kv_text)",
                seed=int(seed),
                run_dir=run_dir,
                returncode=int(code),
                elapsed_s=float(elapsed),
                checks=checks,
                passed=bool(passed),
                e3=e3,
                stdout_tail=_tail(stdout),
                stderr_tail=_tail(stderr),
            )
        )

        if int(args.run_e3text_both) != 0:
            both_pairs = int(args.e3text_pairs)
            n_req = int(max(0, int(args.e3text_both_eval_samples)))
            if n_req > 0:
                both_pairs = int(max(1, int(math.ceil(float(n_req) / float(max(1, 2 * int(args.batch_size)))))))

            # --- E3-text BOTH (kv_text + text query + 2 citations) ---
            cmd_both = _common_base() + [
                "--save_dir",
                os.path.join(str(args.save_root), "e3text_both_kv_text"),
                "--seed",
                str(int(seed)),
                "--online_steps",
                str(int(args.e3text_online_steps)),
                "--offline_steps",
                str(int(args.e3text_offline_steps)),
                "--pointer_drift",
                str(float(args.e3text_pointer_drift)),
                "--cost_power",
                str(float(args.e3text_cost_power)),
                f"--candidates={str(args.e3text_candidates)}",
                "--e3_task",
                "kv_text",
                "--e3_pairs",
                str(int(both_pairs)),
                "--e3_seq_len",
                str(int(args.e3text_seq_len)),
                "--e3_clue_len",
                str(int(args.e3text_value_len)),
                "--e3_key_len",
                str(int(args.e3text_key_len)),
                "--e3_clue_margin",
                str(int(args.e3text_clue_margin)),
                "--e3_kv_text_gap",
                str(int(args.e3text_kv_text_gap)),
                "--e3_policy",
                "active_fixate",
                "--e3_fixate_steps",
                "8",
                "--e3_fixate_mad_mult",
                "6.0",
                "--e3_fixate_mode",
                "freeze",
                "--e3_query_steps",
                str(int(args.e3text_both_query_steps)),
                "--e3_query_mode",
                "both",
                "--e3_both_window",
                str(int(args.e3text_both_window)),
                "--e3_key_seek",
                str(int(args.e3text_both_key_seek)),
                "--e3_key_seek_steps",
                str(int(args.e3text_both_key_seek_steps)),
                "--e3_query_input",
                "text",
                "--e3_comb_mode",
                str(args.e3text_both_comb_mode),
                "--e3_comb_mod",
                str(int(args.e3text_both_comb_mod)),
                "--e3_two_phase",
                "0",
                "--e3_two_phase_trigger",
                "auto",
                "--e3_val_align",
                "0",
                "--e3_speak",
                "1",
                "--e3_speak_vocab",
                "3",
                "--e3_speak_input",
                "y",
                "--e3_speak_loss_weight",
                "0.1",
                "--e3_speak_use",
                "2",
                "--e3_speak_mix_sharpness",
                "6.0",
            ]
            stdout, stderr, code, elapsed = _run_once(cmd_both, timeout_s=int(args.timeout_s))
            run_dir = _parse_run_dir(stdout)
            e3 = {}
            checks = {}
            passed = False
            if code == 0 and run_dir:
                e3 = dict((_load_summary(run_dir).get("E3") or {}) if run_dir else {})
                passed, checks = _gate_e3_text_both(e3, args=args)
            else:
                checks = {"runner_ok": False}
            any_fail = any_fail or (not passed)
            runs.append(
                GateRun(
                    suite="E3-text BOTH (kv_text)",
                    seed=int(seed),
                    run_dir=run_dir,
                    returncode=int(code),
                    elapsed_s=float(elapsed),
                    checks=checks,
                    passed=bool(passed),
                    e3=e3,
                    stdout_tail=_tail(stdout),
                    stderr_tail=_tail(stderr),
                )
            )

    out = {
        "config": vars(args),
        "runs": [r.__dict__ for r in runs],
        "passed": (not any_fail),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[gate] wrote: {out_path}")
    print(f"[gate] passed={not any_fail}")
    sys.exit(0 if not any_fail else 1)


if __name__ == "__main__":
    main()
