#!/usr/bin/env python
"""
Stage-7 hyperparameter scan helper (train + robust/base gate).

Motivation:
- Stage7-Robust currently fails mainly on open-loop collapse metrics (cycle/repetition).
- Iterating manually is slow; this script runs short "finetune bursts" from a base
  checkpoint and ranks configs by gate metrics.

Usage (example):
  /home/void0312/miniconda3/envs/PINNs/bin/python HEI/training/scripts/scan_stage7_robust.py \
    --python /home/void0312/miniconda3/envs/PINNs/bin/python \
    --base_ckpt checkpoints/curriculum/stage7_active/last.pt \
    --save_root checkpoints/curriculum/stage7_scan \
    --delta_steps 200 \
    --grid_json '{"lr":[3e-4,1e-4],"unlikelihood_weight":[0.2,0.5,1.0],"unlikelihood_window":[1,4,8],"q_clip_norm":[100,50]}' \
    --profile robust
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from hashlib import sha1
from random import Random
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json_or_inline(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    if os.path.exists(text):
        with open(text, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(text)


def _reservoir_sample_lines(path: str, *, n: int, seed: int) -> List[str]:
    n = int(n)
    if n <= 0:
        return []
    rng = Random(int(seed))
    sample: List[str] = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            seen += 1
            if len(sample) < n:
                sample.append(ln)
                continue
            j = rng.randrange(seen)
            if j < n:
                sample[j] = ln
    return sample


def _prepare_shared_eval_texts(
    *,
    src_path: str,
    save_root: str,
    eval_samples: int,
    seed: int,
) -> str:
    src_path = os.path.abspath(src_path)
    save_root = os.path.abspath(save_root)
    os.makedirs(save_root, exist_ok=True)
    try:
        st = os.stat(src_path)
        stamp = f"{int(st.st_size)}:{int(st.st_mtime)}"
    except Exception:
        stamp = "nostat"
    key = f"{src_path}|{stamp}|{int(eval_samples)}|{int(seed)}"
    h = sha1(key.encode("utf-8")).hexdigest()[:10]
    base = os.path.basename(src_path).replace(os.sep, "_")
    out_path = os.path.join(save_root, f"eval_texts_{base}_n{int(eval_samples)}_seed{int(seed)}_{h}.txt")
    if os.path.exists(out_path):
        return out_path
    lines = _reservoir_sample_lines(src_path, n=int(eval_samples), seed=int(seed))
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            ln = str(ln).strip()
            if ln:
                f.write(ln + "\n")
    return out_path


def _format_short(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        av = abs(v)
        if av >= 1e3 or av < 1e-3:
            return f"{v:.2e}".replace("+", "")
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    s = str(v)
    s = s.replace(os.sep, "_")
    s = re.sub(r"[^A-Za-z0-9._=-]+", "_", s)
    return s[:64] if len(s) > 64 else s


def _stable_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha1(blob).hexdigest()[:10]


def _read_ckpt_meta(python_exe: str, ckpt_path: str) -> Dict[str, Any]:
    code = (
        "import json, torch\n"
        "ckpt=torch.load(r'''%s''', map_location='cpu')\n"
        "meta={'step': int(ckpt.get('step',0) or 0), 'total_tokens': int(ckpt.get('total_tokens',0) or 0)}\n"
        "opt=ckpt.get('optimizer_state',{})\n"
        "try:\n"
        "  meta['optimizer_lr']=float(opt.get('param_groups',[{}])[0].get('lr',0.0) or 0.0)\n"
        "except Exception:\n"
        "  meta['optimizer_lr']=0.0\n"
        "cfg=ckpt.get('train_cfg',{})\n"
        "meta['train_cfg']=cfg if isinstance(cfg,dict) else {}\n"
        "print(json.dumps(meta, ensure_ascii=False))\n"
        % ckpt_path.replace("\\", "\\\\")
    )
    out = subprocess.check_output([python_exe, "-c", code], text=True)
    return json.loads(out.strip() or "{}")


@dataclass
class GateResult:
    passed: bool
    tf_ppl_ratio: float
    tf_top1_acc: float
    open_rep: float
    open_cycle_rate: float
    open_dominant_cycle: float
    open_unique_ratio: float
    raw: str

    def sort_key(self) -> Tuple[int, float, float, float, float]:
        return (
            0 if self.passed else 1,
            self.open_cycle_rate,
            self.open_dominant_cycle,
            self.open_rep,
            self.tf_ppl_ratio,
        )


_RE_TF = re.compile(r"TeacherForced:.*?PPL_ratio=(?P<ppl_ratio>[0-9.]+).*?top1_acc=(?P<acc>[0-9.]+)")
_RE_OL = re.compile(
    r"OpenLoop:.*?rep(?P<ng>[0-9]+)=(?P<rep>[0-9.]+).*?cycle_rate=(?P<cycle>[0-9.]+).*?"
    r"dominant_cycle=(?P<dom>[0-9.]+).*?unique_ratio=(?P<uniq>[0-9.]+)"
)
_RE_PASSED = re.compile(r"Passed:\s*(?P<passed>True|False)")


def _parse_gate_output(text: str) -> GateResult:
    tf_m = _RE_TF.search(text)
    ol_m = _RE_OL.search(text)
    p_m = _RE_PASSED.search(text)
    if not (tf_m and ol_m and p_m):
        raise ValueError("Failed to parse eval_stage7_gate output.")
    return GateResult(
        passed=(p_m.group("passed") == "True"),
        tf_ppl_ratio=float(tf_m.group("ppl_ratio")),
        tf_top1_acc=float(tf_m.group("acc")),
        open_rep=float(ol_m.group("rep")),
        open_cycle_rate=float(ol_m.group("cycle")),
        open_dominant_cycle=float(ol_m.group("dom")),
        open_unique_ratio=float(ol_m.group("uniq")),
        raw=text,
    )


def _product_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = sorted(grid.keys())
    values = []
    for k in keys:
        v = grid[k]
        if not isinstance(v, list) or not v:
            raise ValueError(f"grid_json['{k}'] must be a non-empty list.")
        values.append(v)
    for combo in itertools.product(*values):
        yield {k: v for k, v in zip(keys, combo)}


def _maybe_int_bool(v: Any) -> Any:
    if isinstance(v, bool):
        return 1 if v else 0
    return v


def _build_train_cmd(
    *,
    python_exe: str,
    base_ckpt: str,
    save_dir: str,
    target_steps: int,
    device: str,
    wiki_path: str,
    clue_path: str,
    max_samples_wiki: int,
    max_samples_clue: int,
    batch_size: int,
    max_seq_len: int,
    tf32: int,
    amp: int,
    amp_dtype: str,
    fused_adamw: int,
    cuda_graph: int,
    pretokenize_samples: int,
    data_on_device: int,
    resume_optimizer: int,
    resume_strict: int,
    base_cfg_overrides: Dict[str, Any],
    trial_overrides: Dict[str, Any],
    extra_train_args: List[str],
) -> List[str]:
    cmd = [
        python_exe,
        os.path.join(os.path.dirname(__file__), "..", "train_active_sampling.py"),
        "--resume_ckpt",
        base_ckpt,
        "--resume_optimizer",
        str(int(resume_optimizer)),
        "--resume_strict",
        str(int(resume_strict)),
        "--device",
        device,
        "--wiki_path",
        wiki_path,
        "--clue_path",
        clue_path,
        "--max_samples_wiki",
        str(int(max_samples_wiki)),
        "--max_samples_clue",
        str(int(max_samples_clue)),
        "--batch_size",
        str(int(batch_size)),
        "--max_seq_len",
        str(int(max_seq_len)),
        "--tf32",
        str(int(tf32)),
        "--amp",
        str(int(amp)),
        "--amp_dtype",
        str(amp_dtype),
        "--fused_adamw",
        str(int(fused_adamw)),
        "--cuda_graph",
        str(int(cuda_graph)),
        "--steps",
        str(int(target_steps)),
        "--save_dir",
        save_dir,
        "--save_every",
        "0",
        "--sample_every",
        "0",
        "--pretokenize_samples",
        str(int(pretokenize_samples)),
        "--data_on_device",
        str(int(data_on_device)),
        # Avoid sampler loss updates during scan to reduce sync/variance.
        "--sampler_loss_update_every",
        "0",
    ]

    merged = {k: _maybe_int_bool(v) for k, v in dict(base_cfg_overrides).items()}
    merged.update({k: _maybe_int_bool(v) for k, v in trial_overrides.items()})

    # Pass only known CLI flags (keep explicit list to avoid silently doing nothing).
    allow = {
        "lr",
        "dt",
        "num_evolution_steps",
        "beta_kl",
        "gamma_pred",
        "num_offline_steps",
        "offline_dt",
        "offline_replay_mode",
        "offline_weight",
        "offline_loss_mode",
        "offline_margin",
        "offline_detach_init",
        "reset_each_batch",
        "port_trainable",
        "port_init_std",
        "scheduled_sampling_prob",
        "scheduled_sampling_mode",
        "scheduled_sampling_top_k",
        "scheduled_sampling_temperature",
        "unlikelihood_weight",
        "unlikelihood_window",
        "router_balance_weight",
        "router_entropy_weight",
        "experience_push_n",
        "port_coupling_top_k",
        "port_coupling_impl",
        "transport_threshold",
        "connection_rank",
        "q_clip_norm",
        "p_clip_norm",
        "s_clip_abs",
        "sanitize_nonfinite",
        "detach_every",
    }
    for k, v in merged.items():
        if k not in allow:
            continue
        cmd.extend([f"--{k}", str(v)])

    if extra_train_args:
        cmd.extend(extra_train_args)
    return cmd


def _build_gate_cmd(
    *,
    python_exe: str,
    ckpt_path: str,
    device: str,
    profile: str,
    eval_text_file: str,
    wiki_path: str,
    clue_path: str,
    max_samples_wiki: int,
    max_samples_clue: int,
    eval_samples: int,
    seed: int,
    batch_size: int,
    max_seq_len: int,
    num_prompts: int,
    prompt_chars: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    ngram_n: int,
    cycle_max_period: int,
    cycle_min_repeats: int,
    cycle_tail_window: int,
    margin_window: int,
    eval_cache: int,
) -> List[str]:
    cmd = [
        python_exe,
        os.path.join(os.path.dirname(__file__), "..", "eval_stage7_gate.py"),
        "--ckpt",
        ckpt_path,
        "--device",
        device,
        "--profile",
        profile,
        "--eval_text_file",
        eval_text_file,
        "--wiki_path",
        wiki_path,
        "--clue_path",
        clue_path,
        "--max_samples_wiki",
        str(int(max_samples_wiki)),
        "--max_samples_clue",
        str(int(max_samples_clue)),
        "--eval_samples",
        str(int(eval_samples)),
        "--seed",
        str(int(seed)),
        "--batch_size",
        str(int(batch_size)),
        "--max_seq_len",
        str(int(max_seq_len)),
        "--num_prompts",
        str(int(num_prompts)),
        "--prompt_chars",
        str(int(prompt_chars)),
        "--max_new_tokens",
        str(int(max_new_tokens)),
        "--temperature",
        str(float(temperature)),
        "--top_k",
        str(int(top_k)),
        "--ngram_n",
        str(int(ngram_n)),
        "--cycle_max_period",
        str(int(cycle_max_period)),
        "--cycle_min_repeats",
        str(int(cycle_min_repeats)),
        "--cycle_tail_window",
        str(int(cycle_tail_window)),
        "--margin_window",
        str(int(margin_window)),
        "--eval_cache",
        str(int(eval_cache)),
    ]
    return cmd


def _run(cmd: List[str], *, cwd: str, log_path: str, timeout_s: Optional[int]) -> Tuple[int, str, float]:
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        out = p.stdout or ""
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + "\n[scan] TIMEOUT\n"
        p = subprocess.CompletedProcess(cmd, returncode=124, stdout=out, stderr=None)
    dt = time.time() - t0
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(out)
    return int(p.returncode), out, float(dt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="Python executable to run train/eval (PINNs recommended).")
    ap.add_argument("--base_ckpt", required=True, help="Base checkpoint path (e.g. .../stage7_active/last.pt).")
    ap.add_argument("--save_root", default="checkpoints/curriculum/stage7_scan")
    ap.add_argument("--delta_steps", type=int, default=200, help="How many extra steps to finetune per trial.")
    ap.add_argument("--target_steps", type=int, default=0, help="Absolute target step number (overrides --delta_steps).")
    ap.add_argument("--device", default="cuda")

    # Data (train)
    ap.add_argument("--wiki_path", default="HEI/data/wiki/wikipedia-zh-20250901.json")
    ap.add_argument("--clue_path", default="HEI/data/CLUE/CLUECorpusSmall.txt")
    ap.add_argument("--max_samples_wiki", type=int, default=50000)
    ap.add_argument("--max_samples_clue", type=int, default=50000)

    # Runtime/perf
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--tf32", type=int, default=1)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--fused_adamw", type=int, default=1)
    ap.add_argument("--cuda_graph", type=int, default=1)
    ap.add_argument("--pretokenize_samples", type=int, default=1)
    ap.add_argument("--data_on_device", type=int, default=0)
    ap.add_argument("--resume_optimizer", type=int, default=0)
    ap.add_argument("--resume_strict", type=int, default=1)
    ap.add_argument("--extra_train_args", default="", help="Extra args appended to train_active_sampling.py")

    # Gate/eval
    ap.add_argument("--profile", default="robust", choices=["base", "robust"])
    ap.add_argument("--eval_text_file", default="HEI/data/CLUE/CLUECorpusSmall.txt")
    ap.add_argument("--eval_samples", type=int, default=512)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument(
        "--shared_eval_texts",
        type=int,
        default=1,
        help="Create a small, shared eval-texts file under save_root to avoid re-scanning huge corpora per trial (1/0).",
    )
    ap.add_argument("--gate_batch_size", type=int, default=64)
    ap.add_argument("--num_prompts", type=int, default=200)
    ap.add_argument("--prompt_chars", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0, help="0=auto (base:20, robust:1)")
    ap.add_argument("--ngram_n", type=int, default=3)
    ap.add_argument("--cycle_max_period", type=int, default=8)
    ap.add_argument("--cycle_min_repeats", type=int, default=3)
    ap.add_argument("--cycle_tail_window", type=int, default=32)
    ap.add_argument("--margin_window", type=int, default=10)
    ap.add_argument("--eval_cache", type=int, default=1)

    # Search space
    ap.add_argument("--grid_json", default="", help="JSON dict mapping param->list values (inline JSON or path).")
    ap.add_argument("--max_trials", type=int, default=0, help="If >0, stop after this many trials.")
    ap.add_argument("--skip_done", type=int, default=1, help="Skip trials that already have gate.json (1/0).")
    ap.add_argument("--timeout_train_s", type=int, default=0, help="Optional per-trial train timeout (0=disable).")
    ap.add_argument("--timeout_gate_s", type=int, default=0, help="Optional per-trial gate timeout (0=disable).")
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    extra_train_args = shlex.split(args.extra_train_args) if args.extra_train_args else []
    base_meta = _read_ckpt_meta(args.python, args.base_ckpt)
    base_step = int(base_meta.get("step", 0) or 0)
    base_cfg = dict(base_meta.get("train_cfg", {}) or {})

    # Baseline overrides: ensure we don't accidentally change important fields due to CLI defaults.
    base_cfg_overrides: Dict[str, Any] = {}
    for k in [
        "dt",
        "num_evolution_steps",
        "beta_kl",
        "gamma_pred",
        "num_offline_steps",
        "offline_dt",
        "offline_replay_mode",
        "offline_weight",
        "offline_loss_mode",
        "offline_margin",
        "offline_detach_init",
        "reset_each_batch",
        "port_trainable",
        "port_init_std",
        "scheduled_sampling_prob",
        "scheduled_sampling_mode",
        "scheduled_sampling_top_k",
        "scheduled_sampling_temperature",
        "unlikelihood_weight",
        "unlikelihood_window",
        "router_balance_weight",
        "router_entropy_weight",
        "experience_push_n",
        "port_coupling_top_k",
        "port_coupling_impl",
        "transport_threshold",
        "connection_rank",
        "q_clip_norm",
        "p_clip_norm",
        "s_clip_abs",
        "sanitize_nonfinite",
        "detach_every",
    ]:
        if k in base_cfg:
            base_cfg_overrides[k] = base_cfg[k]

    # Default grid: start by scanning a few "safe" knobs.
    grid = _load_json_or_inline(args.grid_json) if args.grid_json else {}
    if not grid:
        grid = {
            "lr": [3e-4, 1e-4],
            "unlikelihood_weight": [0.2, 0.5, 1.0],
            "unlikelihood_window": [1, 4, 8],
            "q_clip_norm": [100.0, 50.0],
        }

    trials = list(_product_grid(grid))
    if args.max_trials and len(trials) > int(args.max_trials):
        trials = trials[: int(args.max_trials)]

    if args.target_steps and int(args.target_steps) > 0:
        target_steps = int(args.target_steps)
    else:
        target_steps = base_step + int(args.delta_steps)
    if target_steps <= base_step:
        raise SystemExit(f"target_steps={target_steps} must be > base_step={base_step}")

    save_root = os.path.abspath(args.save_root)
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.jsonl")

    best: Optional[Tuple[GateResult, str]] = None

    print(f"[scan] base_ckpt={args.base_ckpt} step={base_step} -> target_steps={target_steps}")
    print(f"[scan] trials={len(trials)} save_root={save_root}")

    eval_text_file = args.eval_text_file
    if bool(args.shared_eval_texts) and args.eval_text_file and int(args.eval_samples) > 0:
        try:
            eval_text_file = _prepare_shared_eval_texts(
                src_path=args.eval_text_file,
                save_root=save_root,
                eval_samples=int(args.eval_samples),
                seed=int(args.eval_seed),
            )
            print(f"[scan] shared_eval_texts={os.path.relpath(eval_text_file)}")
        except Exception as exc:
            print(f"[scan] shared_eval_texts failed, fallback to --eval_text_file: {exc}")
            eval_text_file = args.eval_text_file

    for idx, trial in enumerate(trials, start=1):
        trial = dict(trial)
        trial_id = _stable_hash(trial)
        trial_dir = os.path.join(save_root, f"t{idx:03d}_{trial_id}")
        os.makedirs(trial_dir, exist_ok=True)

        gate_json_path = os.path.join(trial_dir, "gate.json")
        if bool(args.skip_done) and os.path.exists(gate_json_path):
            try:
                with open(gate_json_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                gr = GateResult(
                    passed=bool(cached.get("passed", False)),
                    tf_ppl_ratio=float(cached.get("tf_ppl_ratio", 1e9)),
                    tf_top1_acc=float(cached.get("tf_top1_acc", 0.0)),
                    open_rep=float(cached.get("open_rep", 1e9)),
                    open_cycle_rate=float(cached.get("open_cycle_rate", 1e9)),
                    open_dominant_cycle=float(cached.get("open_dominant_cycle", 1e9)),
                    open_unique_ratio=float(cached.get("open_unique_ratio", 0.0)),
                    raw=str(cached.get("raw", "")),
                )
                if best is None or gr.sort_key() < best[0].sort_key():
                    best = (gr, trial_dir)
            except Exception:
                pass
            continue

        train_cmd = _build_train_cmd(
            python_exe=args.python,
            base_ckpt=args.base_ckpt,
            save_dir=trial_dir,
            target_steps=target_steps,
            device=args.device,
            wiki_path=args.wiki_path,
            clue_path=args.clue_path,
            max_samples_wiki=args.max_samples_wiki,
            max_samples_clue=args.max_samples_clue,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            tf32=args.tf32,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            fused_adamw=args.fused_adamw,
            cuda_graph=args.cuda_graph,
            pretokenize_samples=args.pretokenize_samples,
            data_on_device=args.data_on_device,
            resume_optimizer=args.resume_optimizer,
            resume_strict=args.resume_strict,
            base_cfg_overrides=base_cfg_overrides,
            trial_overrides=trial,
            extra_train_args=extra_train_args,
        )

        top_k = int(args.top_k)
        if top_k <= 0:
            top_k = 20 if str(args.profile).lower() == "base" else 1

        last_ckpt = os.path.join(trial_dir, "last.pt")
        gate_cmd = _build_gate_cmd(
            python_exe=args.python,
            ckpt_path=last_ckpt,
            device=args.device,
            profile=args.profile,
            eval_text_file=eval_text_file,
            wiki_path=args.wiki_path,
            clue_path=args.clue_path,
            max_samples_wiki=args.max_samples_wiki,
            max_samples_clue=args.max_samples_clue,
            eval_samples=args.eval_samples,
            seed=args.eval_seed,
            batch_size=args.gate_batch_size,
            max_seq_len=args.max_seq_len,
            num_prompts=args.num_prompts,
            prompt_chars=args.prompt_chars,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            ngram_n=args.ngram_n,
            cycle_max_period=args.cycle_max_period,
            cycle_min_repeats=args.cycle_min_repeats,
            cycle_tail_window=args.cycle_tail_window,
            margin_window=args.margin_window,
            eval_cache=args.eval_cache,
        )

        record = {
            "trial_dir": os.path.relpath(trial_dir),
            "trial": trial,
            "base_ckpt": os.path.relpath(os.path.abspath(args.base_ckpt)),
            "base_step": base_step,
            "target_steps": target_steps,
            "train_cmd": train_cmd,
            "gate_cmd": gate_cmd,
        }
        with open(os.path.join(trial_dir, "trial.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        if bool(args.dry_run):
            print(f"[dry] {trial_dir}")
            print("  train:", " ".join(shlex.quote(x) for x in train_cmd))
            print("  gate: ", " ".join(shlex.quote(x) for x in gate_cmd))
            continue

        print(f"[scan] ({idx}/{len(trials)}) {os.path.basename(trial_dir)} {trial}")

        train_rc, train_out, train_dt = _run(
            train_cmd,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
            log_path=os.path.join(trial_dir, "train.log"),
            timeout_s=(int(args.timeout_train_s) if int(args.timeout_train_s) > 0 else None),
        )
        record["train_rc"] = train_rc
        record["train_time_s"] = train_dt
        record["train_tail"] = train_out.strip().splitlines()[-5:]

        gate_rc = 1
        gate_dt = 0.0
        gate_out = ""
        gate_result: Optional[GateResult] = None
        if train_rc == 0 and os.path.exists(last_ckpt):
            gate_rc, gate_out, gate_dt = _run(
                gate_cmd,
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
                log_path=os.path.join(trial_dir, "gate.log"),
                timeout_s=(int(args.timeout_gate_s) if int(args.timeout_gate_s) > 0 else None),
            )
            record["gate_rc"] = gate_rc
            record["gate_time_s"] = gate_dt
            try:
                gate_result = _parse_gate_output(gate_out)
                record["gate"] = {
                    "passed": gate_result.passed,
                    "tf_ppl_ratio": gate_result.tf_ppl_ratio,
                    "tf_top1_acc": gate_result.tf_top1_acc,
                    "open_rep": gate_result.open_rep,
                    "open_cycle_rate": gate_result.open_cycle_rate,
                    "open_dominant_cycle": gate_result.open_dominant_cycle,
                    "open_unique_ratio": gate_result.open_unique_ratio,
                    "raw": gate_result.raw,
                }
                with open(gate_json_path, "w", encoding="utf-8") as f:
                    json.dump(record["gate"], f, ensure_ascii=False, indent=2)
            except Exception as exc:
                record["gate_parse_error"] = str(exc)
        else:
            record["gate_skipped"] = True

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if gate_result is not None:
            if best is None or gate_result.sort_key() < best[0].sort_key():
                best = (gate_result, trial_dir)
            print(
                f"[scan] gate: passed={gate_result.passed} "
                f"cycle={gate_result.open_cycle_rate:.3f} dom={gate_result.open_dominant_cycle:.3f} "
                f"rep={gate_result.open_rep:.3f} pplR={gate_result.tf_ppl_ratio:.3f} "
                f"({gate_dt:.1f}s)"
            )
            if gate_result.passed:
                print(f"[scan] FOUND PASS: {trial_dir}")
                break

    if best is not None:
        gr, tdir = best
        best_path = os.path.join(save_root, "best.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trial_dir": os.path.relpath(tdir),
                    "passed": gr.passed,
                    "tf_ppl_ratio": gr.tf_ppl_ratio,
                    "tf_top1_acc": gr.tf_top1_acc,
                    "open_rep": gr.open_rep,
                    "open_cycle_rate": gr.open_cycle_rate,
                    "open_dominant_cycle": gr.open_dominant_cycle,
                    "open_unique_ratio": gr.open_unique_ratio,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[scan] best: {os.path.relpath(tdir)} passed={gr.passed} cycle={gr.open_cycle_rate:.3f} rep={gr.open_rep:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
