"""
Stage 7 gate (v0): teacher-forced + Protocol-5 open-loop diagnostics with pass/fail.

This gate is intentionally not "human-level language". It checks:
- Teacher-forced improves vs uniform baseline (PPL ratio).
- Open-loop does not collapse into short cycles / excessive repetition.

Example:
  conda run -n PINNs python HEI/training/eval_stage7_gate.py \
    --ckpt checkpoints/curriculum/stage7_active/last.pt --device cuda \
    --wiki_path HEI/data/wiki/wikipedia-zh-20250901.json --clue_path HEI/data/CLUE/CLUECorpusSmall.txt \
    --profile base
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from typing import Dict, List

import torch

# Ensure HEI is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.active_sampling import ActiveSamplingConfig, CorpusPool
from training.checkpoint_io import load_trainer_from_checkpoint
from training.eval_protocol5_closed_loop import (
    collect_eval_texts,
    collect_eval_texts_from_pool,
    detect_tail_cycle,
    extract_prompts,
    generate_open_loop,
    ngram_repetition_rate,
    teacher_forced_diagnostics,
)


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _profile_defaults(profile: str) -> Dict[str, float]:
    profile = str(profile or "base").lower()
    if profile == "robust":
        return {
            "gate_ppl_ratio": 0.60,
            "gate_cycle_rate": 0.40,
            "gate_dominant_cycle": 0.10,
            "gate_rep": 0.50,
            "gate_unique_ratio": 0.60,
        }
    # base
    return {
        "gate_ppl_ratio": 0.85,
        "gate_cycle_rate": 0.60,
        "gate_dominant_cycle": 0.20,
        "gate_rep": 0.60,
        "gate_unique_ratio": 0.40,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path (e.g., .../last.pt)")
    parser.add_argument("--tokenizer_path", default="", help="Optional tokenizer.json path (defaults to ckpt dir).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", default="base", choices=["base", "robust"])

    # Data
    parser.add_argument("--eval_text_file", default="", help="Optional newline-separated evaluation texts.")
    parser.add_argument("--wiki_path", default="HEI/data/wiki/wikipedia-zh-20250901.json")
    parser.add_argument("--clue_path", default="HEI/data/CLUE/CLUECorpusSmall.txt")
    parser.add_argument("--max_samples_wiki", type=int, default=200000)
    parser.add_argument("--max_samples_clue", type=int, default=200000)
    parser.add_argument("--eval_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)

    # Teacher-forced batching
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=128)

    # Open-loop generation
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--prompt_chars", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0, help="0=auto (base:20, robust:1)")

    # Cycle / repetition detection
    parser.add_argument("--cycle_max_period", type=int, default=8)
    parser.add_argument("--cycle_min_repeats", type=int, default=3)
    parser.add_argument("--cycle_tail_window", type=int, default=32)
    parser.add_argument("--ngram_n", type=int, default=3)
    parser.add_argument("--margin_window", type=int, default=10)

    # Gate thresholds (override per profile)
    parser.add_argument("--gate_ppl_ratio", type=float, default=-1.0, help="Teacher-forced PPL / vocab must be <= this.")
    parser.add_argument("--gate_cycle_rate", type=float, default=-1.0, help="Open-loop cycle_rate must be <= this.")
    parser.add_argument("--gate_dominant_cycle", type=float, default=-1.0, help="Open-loop dominant_cycle must be <= this.")
    parser.add_argument("--gate_rep", type=float, default=-1.0, help="Open-loop ngram repetition rate must be <= this.")
    parser.add_argument("--gate_unique_ratio", type=float, default=-1.0, help="Open-loop unique_outputs/prompts must be >= this.")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    seed = int(args.seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    tokenizer_path = args.tokenizer_path or None
    trainer, tokenizer, train_cfg, meta = load_trainer_from_checkpoint(
        args.ckpt,
        device=device,
        tokenizer_path=tokenizer_path if tokenizer_path else None,
        override_cfg={"batch_size": int(args.batch_size), "max_seq_len": int(args.max_seq_len), "experience_push_n": 0},
        strict=False,
    )

    # Collect evaluation texts
    if args.eval_text_file:
        texts = collect_eval_texts(eval_text_file=args.eval_text_file, eval_samples=args.eval_samples, seed=args.seed)
    else:
        cfg = ActiveSamplingConfig(
            wiki_path=args.wiki_path,
            clue_path=args.clue_path,
            max_samples_wiki=args.max_samples_wiki,
            max_samples_clue=args.max_samples_clue,
            source_weights={"wiki": 0.5, "clue": 0.5, "hf": 0.0},
        )
        pool = CorpusPool(cfg)
        pool.load()
        texts = collect_eval_texts_from_pool(pool=pool, eval_samples=args.eval_samples, seed=args.seed)

    if not texts:
        raise RuntimeError("No texts found. Provide --eval_text_file or ensure wiki/clue paths are valid.")

    prompts = extract_prompts(texts, num_prompts=args.num_prompts, prompt_chars=args.prompt_chars)
    if not prompts:
        raise RuntimeError("No prompts extracted. Try larger --eval_samples or smaller --prompt_chars.")

    # --- Teacher-forced diagnostics ---
    tf = teacher_forced_diagnostics(
        trainer=trainer,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    vocab = float(len(tokenizer))
    ppl_ratio = float(tf["tf_PPL"]) / max(vocab, 1.0)

    # --- Open-loop diagnostics ---
    cycle_counter: Counter = Counter()
    unique_outputs: set = set()
    cycle_hits = 0
    rep_rates: List[float] = []
    amp_ratios: List[float] = []

    top_k = int(args.top_k)
    if top_k <= 0:
        top_k = 20 if str(args.profile).lower() == "base" else 1

    w = max(int(args.margin_window), 1)
    for prompt in prompts:
        tokens, margins = generate_open_loop(
            trainer=trainer,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )
        out_text = tokenizer.decode(tokens)
        unique_outputs.add(out_text)

        ids = tokenizer.encode(prompt)
        prefix_ids = ids[:-1] if len(ids) >= 2 else ids
        prefix_len = max(len(prefix_ids), 1)
        cont = tokens[prefix_len:] if len(tokens) > prefix_len else []
        rep_rates.append(ngram_repetition_rate(cont, args.ngram_n))

        cinfo = detect_tail_cycle(
            cont,
            max_period=args.cycle_max_period,
            min_repeats=args.cycle_min_repeats,
            tail_window=args.cycle_tail_window,
        )
        if cinfo is not None:
            cycle_hits += 1
            cycle_counter[(cinfo.period, cinfo.pattern)] += 1

        early = margins[:w]
        late = margins[-w:] if len(margins) >= w else margins
        early_m = _mean([float(x) for x in early])
        late_m = _mean([float(x) for x in late])
        if early_m > 0:
            amp_ratios.append(late_m / early_m)

    cycle_rate = cycle_hits / max(len(prompts), 1)
    dominant_cycle = 0.0
    if cycle_counter:
        dominant_cycle = cycle_counter.most_common(1)[0][1] / max(len(prompts), 1)

    rep = _mean(rep_rates)
    unique_ratio = len(unique_outputs) / max(len(prompts), 1)
    margin_amp = _mean(amp_ratios)

    defaults = _profile_defaults(args.profile)
    gate_ppl_ratio = defaults["gate_ppl_ratio"] if float(args.gate_ppl_ratio) < 0 else float(args.gate_ppl_ratio)
    gate_cycle_rate = defaults["gate_cycle_rate"] if float(args.gate_cycle_rate) < 0 else float(args.gate_cycle_rate)
    gate_dominant_cycle = (
        defaults["gate_dominant_cycle"] if float(args.gate_dominant_cycle) < 0 else float(args.gate_dominant_cycle)
    )
    gate_rep = defaults["gate_rep"] if float(args.gate_rep) < 0 else float(args.gate_rep)
    gate_unique_ratio = defaults["gate_unique_ratio"] if float(args.gate_unique_ratio) < 0 else float(args.gate_unique_ratio)

    checks = {
        "TeacherForced:PPL_ratio": (ppl_ratio <= gate_ppl_ratio),
        "OpenLoop:cycle_rate": (cycle_rate <= gate_cycle_rate),
        "OpenLoop:dominant_cycle": (dominant_cycle <= gate_dominant_cycle),
        f"OpenLoop:rep{int(args.ngram_n)}": (rep <= gate_rep),
        "OpenLoop:unique_ratio": (unique_ratio >= gate_unique_ratio),
    }
    passed = all(checks.values())

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(
        f"Config: port_arch={getattr(getattr(trainer, 'config', trainer), 'port_arch', 'unknown')} "
        f"sequence_mode={getattr(getattr(trainer, 'config', trainer), 'sequence_mode', 'unknown')} "
        f"vocab={len(tokenizer)} dim_q={trainer.entity.dim_q} charts={int(getattr(getattr(trainer,'config',trainer),'num_charts',0) or 0)}"
    )
    print(
        "TeacherForced:"
        f" E_pred={tf['tf_E_pred']:.4f}"
        f" PPL={tf['tf_PPL']:.2f}"
        f" PPL_ratio={ppl_ratio:.4f}"
        f" top1_acc={tf['tf_top1_acc']:.4f}"
        f" margin={tf['tf_margin']:.4f}"
    )
    print(
        "OpenLoop:"
        f" prompts={len(prompts)}"
        f" rep{int(args.ngram_n)}={rep:.4f}"
        f" cycle_rate={cycle_rate:.4f}"
        f" dominant_cycle={dominant_cycle:.4f}"
        f" unique_outputs={len(unique_outputs)}/{len(prompts)}"
        f" unique_ratio={unique_ratio:.4f}"
        f" margin_amp={margin_amp:.4f}"
    )
    print(
        "Thresholds:"
        f" ppl_ratio<={gate_ppl_ratio:.4f}"
        f" cycle_rate<={gate_cycle_rate:.4f}"
        f" dominant_cycle<={gate_dominant_cycle:.4f}"
        f" rep<={gate_rep:.4f}"
        f" unique_ratio>={gate_unique_ratio:.4f}"
    )
    print(f"Passed: {passed} {checks}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
