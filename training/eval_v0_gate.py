"""
v0 Gate Evaluation (theory-aligned quick checks).

Goal:
- Confirm a checkpoint is loadable and produces finite outputs.
- Provide a compact numeric "health" report aligned with v0 gate intent
  (not "can talk like a human" yet).

Run:
  conda run -n PINNs python HEI/training/eval_v0_gate.py --ckpt HEI/checkpoints/v0_lang/last.pt --device cuda
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


def collect_eval_samples(
    *,
    wiki_path: str,
    clue_path: str,
    max_samples_wiki: int,
    max_samples_clue: int,
    hf_dataset: Optional[str],
    hf_name: Optional[str],
    hf_split: str,
    hf_text_field: str,
    max_samples_hf: int,
    eval_samples: int,
    seed: int,
) -> List[str]:
    cfg = ActiveSamplingConfig(
        wiki_path=wiki_path,
        clue_path=clue_path,
        max_samples_wiki=max_samples_wiki,
        max_samples_clue=max_samples_clue,
        hf_dataset=hf_dataset,
        hf_name=hf_name,
        hf_split=hf_split,
        hf_text_field=hf_text_field,
        max_samples_hf=max_samples_hf,
        source_weights={"wiki": 0.5, "clue": 0.5, "hf": 0.0},
    )
    pool = CorpusPool(cfg)
    pool.load()

    if not pool.samples:
        return []

    rng = random.Random(seed)
    indices = list(range(len(pool.samples)))
    rng.shuffle(indices)
    indices = indices[: max(eval_samples, 1)]
    return [pool.samples[i].text for i in indices]


@torch.no_grad()
def quick_metrics(
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
    sum_F_off_end = 0.0
    sum_dF_off = 0.0
    nonfinite_batches = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch = build_batch(batch_texts, tokenizer, max_seq_len)
        batch = {k: v.to(device) for k, v in batch.items()}

        out = trainer.train_step(batch)

        scalars = [
            out["prediction_error"],
            out["free_energy"],
            out["free_energy_online"],
            out.get("free_energy_offline_end", out.get("free_energy_offline", torch.tensor(0.0, device=device))),
            out.get("delta_free_energy_offline", torch.tensor(0.0, device=device)),
        ]
        if not all(torch.isfinite(x).all().item() for x in scalars):
            nonfinite_batches += 1
            continue

        bs = int(batch["input_ids"].shape[0])
        total += bs
        sum_E += float(out["prediction_error"].item()) * bs
        sum_F += float(out["free_energy"].item()) * bs
        sum_F_on += float(out["free_energy_online"].item()) * bs
        sum_F_off_end += float(out.get("free_energy_offline_end", out.get("free_energy_offline", 0.0)).item()) * bs
        sum_dF_off += float(out.get("delta_free_energy_offline", torch.tensor(0.0)).item()) * bs

    total = max(total, 1)
    avg_E = sum_E / total
    return {
        "avg_E_pred": avg_E,
        "avg_PPL": float(torch.exp(torch.tensor(avg_E)).item()),
        "avg_F": sum_F / total,
        "avg_F_on": sum_F_on / total,
        "avg_F_off_end": sum_F_off_end / total,
        "avg_dF_off": sum_dF_off / total,
        "nonfinite_batches": float(nonfinite_batches),
        "num_samples": float(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path (e.g., .../last.pt)")
    parser.add_argument("--tokenizer_path", default="", help="Optional tokenizer.json path (defaults to ckpt dir).")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--wiki_path", default="HEI/data/wiki/wikipedia-zh-20250901.json")
    parser.add_argument("--clue_path", default="HEI/data/CLUE/CLUECorpusSmall.txt")
    parser.add_argument("--max_samples_wiki", type=int, default=200000)
    parser.add_argument("--max_samples_clue", type=int, default=200000)

    parser.add_argument("--hf_dataset", default="", help="Optional HuggingFace dataset name (streaming).")
    parser.add_argument("--hf_name", default="", help="Optional HuggingFace dataset config name.")
    parser.add_argument("--hf_split", default="validation")
    parser.add_argument("--hf_text_field", default="text")
    parser.add_argument("--max_samples_hf", type=int, default=0)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    tokenizer_path = args.tokenizer_path or None
    trainer, tokenizer, train_cfg, meta = load_trainer_from_checkpoint(
        args.ckpt,
        device=device,
        tokenizer_path=tokenizer_path if tokenizer_path else None,
        override_cfg={
            "batch_size": int(args.batch_size),
            "max_seq_len": int(args.max_seq_len),
            "experience_push_n": 0,  # keep evaluation side-effect minimal
        },
        strict=False,
    )

    texts = collect_eval_samples(
        wiki_path=args.wiki_path,
        clue_path=args.clue_path,
        max_samples_wiki=args.max_samples_wiki,
        max_samples_clue=args.max_samples_clue,
        hf_dataset=args.hf_dataset or None,
        hf_name=args.hf_name or None,
        hf_split=args.hf_split,
        hf_text_field=args.hf_text_field,
        max_samples_hf=args.max_samples_hf,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )
    if not texts:
        raise RuntimeError("No evaluation texts found (check corpus paths / hf args).")

    m = quick_metrics(
        trainer=trainer,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    port_arch = getattr(getattr(trainer, "config", trainer), "port_arch", getattr(trainer, "port_arch", "unknown"))
    sequence_mode = getattr(getattr(trainer, "config", trainer), "sequence_mode", getattr(trainer, "sequence_mode", "unknown"))
    num_charts = int(getattr(getattr(trainer, "config", trainer), "num_charts", getattr(trainer.entity, "num_charts", 0)) or 0)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Config: port_arch={port_arch} sequence_mode={sequence_mode} vocab={len(tokenizer)} dim_q={trainer.entity.dim_q} charts={num_charts}")
    print(
        "Gate:"
        f" samples={int(m['num_samples'])}"
        f" E_pred={m['avg_E_pred']:.4f}"
        f" PPL={m['avg_PPL']:.2f}"
        f" F={m['avg_F']:.4f}"
        f" F_on={m['avg_F_on']:.4f}"
        f" F_off_end={m['avg_F_off_end']:.4f}"
        f" dF_off={m['avg_dF_off']:.4f}"
        f" nonfinite_batches={int(m['nonfinite_batches'])}"
    )

    try:
        diag = trainer.get_entity_diagnostics()
        if isinstance(diag, dict):
            qn = diag.get("q_norm", None)
            pn = diag.get("p_norm", None)
            sn = diag.get("s_value", None)
            zn = diag.get("z_norm", None)
            exp = diag.get("experience_size", None)
            print(f"Diag: q_norm={qn} p_norm={pn} s_value={sn} z_norm={zn} experience_size={exp}")
    except Exception as exc:
        print(f"Diag: unavailable ({exc})")


if __name__ == "__main__":
    main()

