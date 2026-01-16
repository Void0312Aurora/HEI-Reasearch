#!/usr/bin/env python3
"""
L7v2-v0 E3-text (kv_text) BOTH scan: widen geometry (both_window) and measure degradation.

This script runs ONLY the E3 suite via `HEI/training/run_l7v2_v0.py` as a subprocess and records:
  - resolved_rate (both-mode; value-hit curriculum)
  - resolved_strict_rate (diagnostic; key+value hit)
  - query_acc_hit, cite_hit_rate

Default settings follow the Iter-3 "both 可发生版" gate:
  - query_mode=both, both_window=1
  - query_steps=32
  - two_phase=0 (avoid harming both coverage)

Example:
  /home/void0312/miniconda3/envs/PINNs/bin/python HEI/training/scripts/scan_l7v2_v0_e3text_both_window.py \\
    --device cuda --save_root checkpoints/l7v2_e3text_both_window_scan \\
    --seeds 0,1,2 --windows 1,64,128,256,512 --confirm 1
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


def _parse_csv_ints_allow_spaces(s: str) -> List[int]:
    s = str(s or "").replace(" ", "")
    return _parse_csv_ints(s)


def _isfinite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _write_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _clear_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def _find_latest_run_dir(save_root: str, *, window: int, seed: int) -> Optional[str]:
    base = os.path.join(str(save_root), f"w{int(window)}")
    if not os.path.isdir(base):
        return None
    cand: List[str] = []
    suf = f"seed{int(seed)}"
    try:
        for name in os.listdir(base):
            if suf not in name:
                continue
            run_dir = os.path.join(base, name)
            if not os.path.isdir(run_dir):
                continue
            if not os.path.isfile(os.path.join(run_dir, "summary.json")):
                continue
            cand.append(run_dir)
    except Exception:
        return None
    if not cand:
        return None
    cand.sort()
    return cand[-1]

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
class ScanResult:
    seed: int
    window: int
    run_dir: Optional[str]
    returncode: int
    elapsed_s: float
    e3: Dict[str, Any]
    gate_passed: bool
    gate_checks: Dict[str, bool]
    stdout_tail: str
    stderr_tail: str


def _tail(s: str, n: int = 2000) -> str:
    s = s or ""
    return s if len(s) <= n else s[-n:]


def _gate(e3: Dict[str, Any], *, args: argparse.Namespace) -> Tuple[bool, Dict[str, bool]]:
    resolved = float(e3.get("resolved_rate", float("nan")))
    strict = float(e3.get("resolved_strict_rate", float("nan")))
    query_acc_hit = float(e3.get("query_acc_hit", float("nan")))
    cite_hit_rate = float(e3.get("cite_hit_rate", float("nan")))

    checks: Dict[str, bool] = {
        "resolved_rate": _isfinite(resolved) and resolved >= float(args.gate_resolved_rate),
        "query_acc_hit": _isfinite(query_acc_hit) and query_acc_hit >= float(args.gate_query_acc_hit),
        "cite_hit_rate": _isfinite(cite_hit_rate) and cite_hit_rate >= float(args.gate_cite_hit_rate),
    }
    if args.gate_resolved_strict_rate is not None:
        checks["resolved_strict_rate"] = _isfinite(strict) and strict >= float(args.gate_resolved_strict_rate)
    return all(checks.values()), checks


def _agg(xs: List[float]) -> Dict[str, float]:
    xs = [float(x) for x in xs if _isfinite(x)]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = sum(xs) / float(len(xs))
    var = sum((x - mean) ** 2 for x in xs) / float(max(1, len(xs) - 1))
    return {"mean": float(mean), "std": float(math.sqrt(var)), "min": float(min(xs)), "max": float(max(xs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="checkpoints/l7v2_e3text_both_window_scan")
    ap.add_argument("--timeout_s", type=int, default=7200)

    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--windows", type=str, default="1,64,128,256,512")

    # Shared knobs (keep aligned with gate defaults)
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
    ap.add_argument("--pointer_drift", type=float, default=0.0)
    ap.add_argument("--candidates", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--replay_mode", type=str, default="random", choices=["none", "random", "recent", "prioritized"])

    # E3-text BOTH config
    ap.add_argument("--e3_pairs", type=int, default=8)
    ap.add_argument("--e3_seq_len", type=int, default=4096)
    ap.add_argument("--e3_value_len", type=int, default=32)
    ap.add_argument("--e3_key_len", type=int, default=1)
    ap.add_argument("--e3_clue_margin", type=int, default=64)
    ap.add_argument("--e3_kv_text_gap", type=int, default=3)
    ap.add_argument("--online_steps", type=int, default=80)
    ap.add_argument("--offline_steps", type=int, default=20)
    ap.add_argument("--query_steps", type=int, default=32)
    ap.add_argument(
        "--two_phase_trigger",
        type=str,
        default="auto",
        choices=["auto", "back", "hold"],
        help="E3 speak trigger semantics (even when two_phase=0 it affects speak target via hold-on-fixate).",
    )

    # Gate thresholds for scanning summary (defaults match Iter-3 both gate)
    ap.add_argument("--gate_resolved_rate", type=float, default=0.05)
    ap.add_argument("--gate_query_acc_hit", type=float, default=0.95)
    ap.add_argument("--gate_cite_hit_rate", type=float, default=0.95)
    ap.add_argument(
        "--gate_resolved_strict_rate",
        type=float,
        default=None,
        help="Optional: require resolved_strict_rate >= thr (enables strict tightening scan).",
    )

    ap.add_argument("--confirm", type=int, default=0)
    ap.add_argument(
        "--reuse_existing",
        type=int,
        default=0,
        help="If 1, do not run new jobs; reuse existing run dirs under save_root/w{window}/... and only rebuild results/summary.",
    )
    ap.add_argument("--overwrite", type=int, default=1, help="If 1, overwrite results.jsonl instead of appending.")
    args = ap.parse_args()

    seeds = _parse_csv_ints_allow_spaces(str(args.seeds))
    windows = _parse_csv_ints_allow_spaces(str(args.windows))
    windows = [int(w) for w in windows if int(w) >= 0]
    if not windows:
        raise ValueError("empty --windows")

    save_root = str(args.save_root)
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.jsonl")
    group_path = os.path.join(save_root, "group_summary.json")

    if int(args.confirm) != 1:
        print("[scan] dry-run (add --confirm 1 to execute)")
        print(f"[scan] save_root={save_root}")
        print(f"[scan] seeds={seeds}")
        print(f"[scan] windows={windows}")
        return

    if int(args.overwrite) != 0:
        _clear_file(results_path)

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
        "--pointer_drift",
        str(float(args.pointer_drift)),
        f"--candidates={str(args.candidates)}",
        "--replay_mode",
        str(args.replay_mode),
        "--run_e0",
        "0",
        "--run_e1e2",
        "0",
        "--run_e3",
        "1",
        "--online_steps",
        str(int(args.online_steps)),
        "--offline_steps",
        str(int(args.offline_steps)),
        "--e3_task",
        "kv_text",
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
        "--e3_kv_text_gap",
        str(int(args.e3_kv_text_gap)),
        "--e3_policy",
        "active_fixate",
        "--e3_fixate_steps",
        "8",
        "--e3_fixate_mad_mult",
        "6.0",
        "--e3_fixate_mode",
        "freeze",
        "--e3_query_steps",
        str(int(args.query_steps)),
        "--e3_query_mode",
        "both",
        "--e3_query_input",
        "text",
        "--e3_two_phase",
        "0",
        "--e3_two_phase_trigger",
        str(args.two_phase_trigger),
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

    results: List[ScanResult] = []
    for seed in seeds:
        for w in windows:
            if int(args.reuse_existing) != 0:
                run_dir = _find_latest_run_dir(save_root, window=int(w), seed=int(seed))
                code = 0 if run_dir else 1
                elapsed = float("nan")
                stdout = ""
                stderr = ""
            else:
                cmd = list(base) + [
                    "--save_dir",
                    os.path.join(save_root, f"w{int(w)}"),
                    "--seed",
                    str(int(seed)),
                    "--e3_both_window",
                    str(int(w)),
                ]
                stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
                run_dir = _parse_run_dir(stdout)

            e3: Dict[str, Any] = {}
            gate_passed = False
            gate_checks: Dict[str, bool] = {}
            if int(code) == 0 and run_dir:
                summary = _load_summary(run_dir)
                e3 = dict(summary.get("E3", {}) or {})
                gate_passed, gate_checks = _gate(e3, args=args)
            else:
                gate_checks = {"runner_ok": False}

            rec = ScanResult(
                seed=int(seed),
                window=int(w),
                run_dir=run_dir,
                returncode=int(code),
                elapsed_s=float(elapsed),
                e3=e3,
                gate_passed=bool(gate_passed),
                gate_checks=gate_checks,
                stdout_tail=_tail(stdout),
                stderr_tail=_tail(stderr),
            )
            results.append(rec)
            _write_jsonl(
                results_path,
                {
                    "seed": int(seed),
                    "window": int(w),
                    "run_dir": run_dir,
                    "returncode": int(code),
                    "elapsed_s": float(elapsed),
                    "gate_passed": bool(gate_passed),
                    "gate_checks": gate_checks,
                    "E3": e3,
                },
            )

    by_w: Dict[int, List[ScanResult]] = {}
    for r in results:
        by_w.setdefault(int(r.window), []).append(r)

    rows: List[Dict[str, Any]] = []
    for w in sorted(by_w.keys()):
        rs = by_w[w]
        rows.append(
            {
                "window": int(w),
                "n": int(len(rs)),
                "pass_rate": float(sum(1 for x in rs if x.gate_passed) / float(max(1, len(rs)))),
                "resolved_rate": _agg([float(x.e3.get("resolved_rate", float("nan"))) for x in rs]),
                "resolved_strict_rate": _agg([float(x.e3.get("resolved_strict_rate", float("nan"))) for x in rs]),
                "query_acc_hit": _agg([float(x.e3.get("query_acc_hit", float("nan"))) for x in rs]),
                "cite_hit_rate": _agg([float(x.e3.get("cite_hit_rate", float("nan"))) for x in rs]),
                "resolved_any_rate": _agg([float(x.e3.get("resolved_any_rate", float("nan"))) for x in rs]),
            }
        )

    out = {
        "seeds": seeds,
        "windows": windows,
        "gate": {
            "resolved_rate": float(args.gate_resolved_rate),
            "query_acc_hit": float(args.gate_query_acc_hit),
            "cite_hit_rate": float(args.gate_cite_hit_rate),
            "resolved_strict_rate": (None if args.gate_resolved_strict_rate is None else float(args.gate_resolved_strict_rate)),
        },
        "rows": rows,
    }
    with open(group_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[scan] wrote: {results_path}")
    print(f"[scan] wrote: {group_path}")


if __name__ == "__main__":
    main()
