#!/usr/bin/env python3
"""
Scan helper for L7v2-v0 v0.4 epistemic proxies.

Purpose (per temp-10):
  - Evaluate small, falsifiable grids over:
      - finite-difference epsilon (epi_obs_eps)
      - jac_E denominator floor (epi_pred_floor)
      - seed
      - (optionally) epi_weight sign (+/-)
  - Run E0-only by default (fast) and record:
      - epi_proxy_final (active)
      - delta_mean_final / V_term_final / pred_err_final (active)
  - Aggregate sign-controllability per (seed, eps, pred_floor) when both +/- weights are run.

This script reuses `HEI/training/run_l7v2_v0.py` as a subprocess and writes:
  - results.jsonl (one line per run)
  - group_summary.json (per-group sign controllability)
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
from typing import Any, Dict, Iterable, List, Optional, Tuple


RUN_RE = re.compile(r"^\[L7v2-v0\] Saved run to: (.+)$")


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


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
    epi_weight: float
    epi_obs_eps: float
    epi_pred_floor: float
    run_dir: str
    elapsed_s: float
    summary: Dict[str, Any]

    def e0_active(self) -> Dict[str, Any]:
        return dict(self.summary.get("E0", {}).get("active", {}) or {})

    def key_group(self) -> Tuple[int, float, float]:
        return (int(self.seed), float(self.epi_obs_eps), float(self.epi_pred_floor))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_root", default="checkpoints/l7v2_v0p4_jace_scan")
    ap.add_argument("--timeout_s", type=int, default=3600)

    # Grid
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epi_weights", type=str, default="-0.02,0.02")
    ap.add_argument("--eps_values", type=str, default="0.1,1.0")
    ap.add_argument("--pred_floors", type=str, default="0,0.001")

    # v0.3c baseline knobs (override if needed)
    ap.add_argument("--dim_q", type=int, default=64)
    ap.add_argument("--num_charts", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--e0_steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
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
    ap.add_argument("--pointer_bounds", type=str, default="clamp", choices=["clamp", "reflect"])
    ap.add_argument("--lookahead_steps", type=int, default=1)
    ap.add_argument("--lookahead_discount", type=float, default=0.9)
    ap.add_argument("--cover_weight", type=float, default=0.02)
    ap.add_argument("--cover_sigma", type=float, default=2.0)
    ap.add_argument("--cover_window", type=int, default=32)
    ap.add_argument("--cover_warmup_frac", type=float, default=0.3)
    ap.add_argument("--cost_weight", type=float, default=0.0028)
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--epi_mode", type=str, default="jac_E")
    ap.add_argument("--run_full", type=int, default=0, help="If 1, run E0+E1+E2 for each config (slow).")

    args = ap.parse_args()

    save_root = str(args.save_root)
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.jsonl")
    group_path = os.path.join(save_root, "group_summary.json")

    seeds = _parse_csv_ints(str(args.seeds))
    epi_weights = _parse_csv_floats(str(args.epi_weights))
    eps_values = _parse_csv_floats(str(args.eps_values))
    pred_floors = _parse_csv_floats(str(args.pred_floors))

    base = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "..", "run_l7v2_v0.py"),
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
        "--lookahead_steps",
        str(int(args.lookahead_steps)),
        "--lookahead_discount",
        str(float(args.lookahead_discount)),
        "--cover_weight",
        str(float(args.cover_weight)),
        "--cover_sigma",
        str(float(args.cover_sigma)),
        "--cover_window",
        str(int(args.cover_window)),
        "--cover_warmup_frac",
        str(float(args.cover_warmup_frac)),
        "--cost_weight",
        str(float(args.cost_weight)),
        "--e0_steps",
        str(int(args.e0_steps)),
        "--log_every",
        str(int(args.log_every)),
        "--epi_mode",
        str(args.epi_mode),
    ]

    run_e1e2 = int(args.run_full) == 1

    flat: List[FlatResult] = []
    total = len(seeds) * len(eps_values) * len(pred_floors) * len(epi_weights)
    idx = 0
    for seed in seeds:
        for eps in eps_values:
            for floor in pred_floors:
                for w in epi_weights:
                    idx += 1
                    trial_root = os.path.join(
                        save_root,
                        f"seed{seed}",
                        f"eps{eps:.6g}",
                        f"floor{floor:.6g}",
                        f"w{w:+.6g}",
                    )
                    cmd = (
                        base
                        + [
                            "--save_dir",
                            trial_root,
                            "--seed",
                            str(int(seed)),
                            "--epi_obs_eps",
                            f"{float(eps):.12g}",
                            "--epi_pred_floor",
                            f"{float(floor):.12g}",
                            "--epi_weight",
                            f"{float(w):.12g}",
                            "--run_e0",
                            "1",
                            "--run_e1e2",
                            "1" if run_e1e2 else "0",
                        ]
                    )
                    print(
                        f"[scan] ({idx}/{total}) seed={seed} eps={eps:.6g} pred_floor={floor:.6g} w={w:+.6g} full={int(run_e1e2)}"
                    )
                    stdout, stderr, code, elapsed = _run_once(cmd, timeout_s=int(args.timeout_s))
                    run_dir = _parse_run_dir(stdout)
                    rec: Dict[str, Any] = {
                        "seed": int(seed),
                        "epi_weight": float(w),
                        "epi_obs_eps": float(eps),
                        "epi_pred_floor": float(floor),
                        "elapsed_s": float(elapsed),
                        "returncode": int(code),
                        "run_dir": str(run_dir) if run_dir else None,
                    }
                    if run_dir is None or code != 0:
                        rec["stdout_tail"] = (stdout or "")[-2000:]
                        rec["stderr_tail"] = (stderr or "")[-2000:]
                        _write_jsonl(results_path, rec)
                        continue

                    summary = _load_summary(run_dir)
                    fr = FlatResult(
                        seed=int(seed),
                        epi_weight=float(w),
                        epi_obs_eps=float(eps),
                        epi_pred_floor=float(floor),
                        run_dir=str(run_dir),
                        elapsed_s=float(elapsed),
                        summary=summary,
                    )
                    flat.append(fr)
                    e0a = fr.e0_active()
                    rec.update(
                        {
                            "e0_active_epi_proxy": float(e0a.get("epi_proxy_final", float("nan"))),
                            "e0_active_delta": float(e0a.get("delta_mean_final", float("nan"))),
                            "e0_active_V": float(e0a.get("V_term_final", float("nan"))),
                            "e0_active_pred": float(e0a.get("pred_err_final", float("nan"))),
                        }
                    )
                    _write_jsonl(results_path, rec)

    # Aggregate sign controllability when both +/- weights exist per group.
    group: Dict[str, Any] = {
        "meta": {
            "seeds": seeds,
            "epi_weights": epi_weights,
            "eps_values": eps_values,
            "pred_floors": pred_floors,
            "run_full": bool(run_e1e2),
        },
        "groups": [],
    }

    by_group: Dict[Tuple[int, float, float], List[FlatResult]] = {}
    for fr in flat:
        by_group.setdefault(fr.key_group(), []).append(fr)

    for (seed, eps, floor), runs in sorted(by_group.items(), key=lambda x: x[0]):
        runs = sorted(runs, key=lambda r: float(r.epi_weight))
        # Find min/max weights (for typical +/- grids).
        w_min = min(r.epi_weight for r in runs)
        w_max = max(r.epi_weight for r in runs)
        r_min = next((r for r in runs if r.epi_weight == w_min), None)
        r_max = next((r for r in runs if r.epi_weight == w_max), None)
        proxy_min = float(r_min.e0_active().get("epi_proxy_final", float("nan"))) if r_min else float("nan")
        proxy_max = float(r_max.e0_active().get("epi_proxy_final", float("nan"))) if r_max else float("nan")
        ok = _isfinite(proxy_min) and _isfinite(proxy_max) and (proxy_max > proxy_min)
        group["groups"].append(
            {
                "seed": int(seed),
                "epi_obs_eps": float(eps),
                "epi_pred_floor": float(floor),
                "w_min": float(w_min),
                "w_max": float(w_max),
                "epi_proxy_min": proxy_min,
                "epi_proxy_max": proxy_max,
                "delta_proxy": (proxy_max - proxy_min) if (_isfinite(proxy_min) and _isfinite(proxy_max)) else float("nan"),
                "sign_controllable": bool(ok),
            }
        )

    with open(group_path, "w", encoding="utf-8") as f:
        json.dump(group, f, ensure_ascii=False, indent=2)
    print(f"[scan] wrote: {results_path}")
    print(f"[scan] wrote: {group_path}")


if __name__ == "__main__":
    main()

