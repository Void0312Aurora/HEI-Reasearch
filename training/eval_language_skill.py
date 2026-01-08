"""
Evaluate SoulEntity language skill checkpoints.

Supports:
- Local corpora (wiki/clue)
- Optional HuggingFace streaming corpora (requires `pip install datasets`)

Example:
python HEI/training/eval_language_skill.py --ckpt HEI/checkpoints/lang_active/last.pt --device cuda --eval_samples 20000
"""

import argparse
import os
import random
import sys
from typing import Dict, List, Optional

import torch

# Ensure HEI is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.active_sampling import ActiveSamplingConfig, CorpusPool
from training.checkpoint_io import load_trainer_from_checkpoint


def build_batch(texts: List[str], tokenizer, max_len: int) -> Dict[str, torch.Tensor]:
    input_ids = []
    attention_masks = []

    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) > max_len:
            ids = ids[:max_len]

        mask = [1] * len(ids)
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [tokenizer.pad_id] * pad_len
            mask = mask + [0] * pad_len

        input_ids.append(torch.tensor(ids, dtype=torch.long))
        attention_masks.append(torch.tensor(mask, dtype=torch.long))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
    }


def collect_eval_texts(
    *,
    eval_text_file: str,
    eval_samples: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    with open(eval_text_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []
    if eval_samples <= 0 or eval_samples >= len(lines):
        rng.shuffle(lines)
        return lines
    return rng.sample(lines, eval_samples)


def collect_eval_texts_from_pool(
    *,
    pool: CorpusPool,
    eval_samples: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    if not pool.samples:
        return []
    if eval_samples <= 0 or eval_samples >= len(pool.samples):
        indices = list(range(len(pool.samples)))
        rng.shuffle(indices)
        return [pool.samples[i].text for i in indices]
    indices = rng.sample(range(len(pool.samples)), eval_samples)
    return [pool.samples[i].text for i in indices]


@torch.no_grad()
def evaluate(
    *,
    trainer,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int,
    max_seq_len: int,
) -> Dict[str, float]:
    trainer.eval()
    total = 0
    sum_E = 0.0
    sum_F = 0.0
    sum_F_on = 0.0
    sum_F_off = 0.0
    sum_dF_off = 0.0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch = build_batch(batch_texts, tokenizer, max_seq_len)
        batch = {k: v.to(device) for k, v in batch.items()}

        out = trainer.train_step(batch)
        bs = int(batch["input_ids"].shape[0])
        total += bs
        sum_E += float(out["prediction_error"].item()) * bs
        sum_F += float(out["free_energy"].item()) * bs
        sum_F_on += float(out["free_energy_online"].item()) * bs
        sum_F_off += float(out.get("free_energy_offline_end", out.get("free_energy_offline", 0.0)).item()) * bs
        sum_dF_off += float(out.get("delta_free_energy_offline", torch.tensor(0.0)).item()) * bs

    avg_E = sum_E / max(total, 1)
    return {
        "avg_prediction_error": avg_E,
        "avg_perplexity": float(torch.exp(torch.tensor(avg_E)).item()),
        "avg_free_energy": sum_F / max(total, 1),
        "avg_free_energy_online": sum_F_on / max(total, 1),
        "avg_free_energy_offline_end": sum_F_off / max(total, 1),
        "avg_delta_free_energy_offline": sum_dF_off / max(total, 1),
        "num_samples": float(total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path (e.g., .../last.pt)")
    parser.add_argument("--tokenizer_path", default="", help="Optional tokenizer.json path (defaults to ckpt dir).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)

    # Eval data (choose one)
    parser.add_argument("--eval_text_file", default="", help="Optional newline-separated evaluation texts.")
    parser.add_argument("--wiki_path", default="HEI/data/wiki/wikipedia-zh-20250901.json")
    parser.add_argument("--clue_path", default="HEI/data/CLUE/CLUECorpusSmall.txt")
    parser.add_argument("--max_samples_wiki", type=int, default=200000)
    parser.add_argument("--max_samples_clue", type=int, default=200000)

    # Optional HF streaming data
    parser.add_argument("--hf_dataset", default="", help="Optional HuggingFace dataset name (requires `pip install datasets`).")
    parser.add_argument("--hf_name", default="", help="Optional HuggingFace dataset config name.")
    parser.add_argument("--hf_split", default="validation", help="HuggingFace split (streaming).")
    parser.add_argument("--hf_text_field", default="text", help="Text field key for HuggingFace dataset.")
    parser.add_argument("--max_samples_hf", type=int, default=0, help="How many HF samples to stream into eval pool.")

    # Sampling / display
    parser.add_argument("--prompt", action="append", default=[], help="Prompt for qualitative generation (repeatable).")
    parser.add_argument("--gen_len", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    tokenizer_path = args.tokenizer_path or None
    trainer, tokenizer, train_cfg, meta = load_trainer_from_checkpoint(
        args.ckpt,
        device=device,
        tokenizer_path=tokenizer_path if tokenizer_path else None,
        override_cfg={"batch_size": args.batch_size, "max_seq_len": args.max_seq_len},
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
            hf_dataset=args.hf_dataset or None,
            hf_name=args.hf_name or None,
            hf_split=args.hf_split,
            hf_text_field=args.hf_text_field,
            max_samples_hf=args.max_samples_hf,
            source_weights={"wiki": 0.5, "clue": 0.5, "hf": 0.0},
        )
        pool = CorpusPool(cfg)
        pool.load()
        texts = collect_eval_texts_from_pool(pool=pool, eval_samples=args.eval_samples, seed=args.seed)

    if not texts:
        raise RuntimeError("No evaluation texts found. Provide --eval_text_file or ensure corpora paths are valid.")

    metrics = evaluate(
        trainer=trainer,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(
        "Eval:"
        f" samples={int(metrics['num_samples'])}"
        f" E_pred={metrics['avg_prediction_error']:.4f}"
        f" PPL={metrics['avg_perplexity']:.2f}"
        f" F={metrics['avg_free_energy']:.4f}"
        f" F_on={metrics['avg_free_energy_online']:.4f}"
        f" F_off_end={metrics['avg_free_energy_offline_end']:.4f}"
        f" dF_off={metrics['avg_delta_free_energy_offline']:.4f}"
    )

    prompts = args.prompt or ["如何", "人工智能", "数学是"]
    for p in prompts:
        try:
            text = trainer.generate(p, max_len=args.gen_len, temperature=args.temperature, top_k=args.top_k)
            print(f"[Sample] {p} -> {text}")
        except Exception as exc:
            print(f"[Sample] generation failed for prompt={p!r}: {exc}")


if __name__ == "__main__":
    main()

