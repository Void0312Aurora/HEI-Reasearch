"""
v0 Skill/Atlas Evaluation.

Theory hooks (理论基础-7):
- Skills ≈ atlas charts; evaluate whether the router collapses or distributes usage.
- This is NOT a language-quality benchmark; it's a structural diagnostic.

Outputs:
- Coverage distribution + entropy (effective chart count)
- Switching rate (argmax chart changes)
- Sparsity proxies (mean max weight, mean token entropy)
- Optional MI(source;chart) to check if charts correlate with environments/sources

Run:
  conda run -n PINNs python HEI/training/eval_v0_skill_atlas.py --ckpt HEI/checkpoints/v0_lang/last.pt --device cuda
"""

import argparse
import math
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

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


def sample_pool(
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
    hf_weight: float,
    eval_samples: int,
    seed: int,
) -> List[Tuple[str, str]]:
    base_weights = {"wiki": 0.5, "clue": 0.5}
    if hf_dataset and max_samples_hf > 0 and hf_weight > 0:
        hf_w = max(0.0, min(float(hf_weight), 1.0))
        scale = max(0.0, 1.0 - hf_w)
        base_weights = {"wiki": base_weights["wiki"] * scale, "clue": base_weights["clue"] * scale, "hf": hf_w}

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
        source_weights=base_weights,
    )
    pool = CorpusPool(cfg)
    pool.load()
    if not pool.samples:
        return []

    rng = random.Random(seed)
    indices = list(range(len(pool.samples)))
    rng.shuffle(indices)
    indices = indices[: max(eval_samples, 1)]
    return [(pool.samples[i].text, pool.samples[i].source) for i in indices]


def _entropy(p: torch.Tensor) -> float:
    p = p.clamp(min=1e-12)
    return float((-(p * torch.log(p))).sum().item())


def _mutual_information_source_chart(counts: Dict[str, Counter]) -> float:
    # counts[source][chart] = token_count
    total = 0
    for src, ctr in counts.items():
        total += sum(ctr.values())
    if total <= 0:
        return 0.0

    # p(source), p(chart)
    p_src: Dict[str, float] = {}
    p_chart: Dict[int, float] = defaultdict(float)
    for src, ctr in counts.items():
        src_total = float(sum(ctr.values()))
        p_src[src] = src_total / total
        for c, n in ctr.items():
            p_chart[int(c)] += float(n) / total

    mi = 0.0
    for src, ctr in counts.items():
        for c, n in ctr.items():
            p_sc = float(n) / total
            denom = (p_src.get(src, 0.0) * p_chart.get(int(c), 0.0))
            if p_sc > 0 and denom > 0:
                mi += p_sc * math.log(p_sc / denom)
    return float(mi)


@torch.no_grad()
def eval_recurrent_router_stats(
    *,
    trainer,
    tokenizer,
    samples: List[Tuple[str, str]],
    device: str,
    batch_size: int,
    max_seq_len: int,
) -> Dict[str, object]:
    trainer.eval()

    port_arch = getattr(getattr(trainer, "config", trainer), "port_arch", getattr(trainer, "port_arch", "unknown"))
    sequence_mode = getattr(getattr(trainer, "config", trainer), "sequence_mode", getattr(trainer, "sequence_mode", "unknown"))
    if port_arch != "minimal" or sequence_mode != "recurrent":
        raise ValueError(f"eval_v0_skill_atlas expects minimal+recurrent; got port_arch={port_arch} sequence_mode={sequence_mode}")

    K = int(getattr(getattr(trainer, "config", trainer), "num_charts", 0) or trainer.entity.atlas.num_charts)
    sum_weights = torch.zeros(K, dtype=torch.double)
    sum_token_entropy = 0.0
    sum_max_w = 0.0
    total_tokens = 0
    total_switches = 0
    src_counts: Dict[str, Counter] = defaultdict(Counter)

    texts = [t for (t, _s) in samples]
    sources = [s for (_t, s) in samples]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_sources = sources[i : i + batch_size]
        batch = build_batch(batch_texts, tokenizer, max_seq_len)
        token_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        bs, seq_len = token_ids.shape

        trainer.reset_entity(bs)
        trainer.entity.enter_online()
        state_flat = trainer.entity.state.flat
        prev_weights = None
        prev_arg: List[Optional[int]] = [None] * bs

        for t in range(seq_len):
            mask_curr = attn[:, t].float().unsqueeze(1)  # [B,1]
            if mask_curr.sum().item() <= 0:
                continue
            tok = token_ids[:, t]
            u_t = trainer.token_embedding(tok) * mask_curr  # [B,Dq]

            last_w = None
            for _ in range(trainer.config.num_evolution_steps):
                out = trainer.entity.forward_tensor(
                    state_flat=state_flat,
                    u_dict={"language": u_t},
                    dt=trainer.config.dt,
                    prev_chart_weights=prev_weights,
                    prediction_error=None,
                    detach_next_prev_weights=True,
                )
                next_state_flat = out["next_state_flat"]
                next_prev = out["next_prev_chart_weights"]
                last_w = out.get("chart_weights", None)

                # Freeze padded samples
                state_flat = mask_curr * next_state_flat + (1.0 - mask_curr) * state_flat
                prev_weights = next_prev if prev_weights is None else (mask_curr * next_prev + (1.0 - mask_curr) * prev_weights)

            if last_w is None:
                continue

            mask1 = attn[:, t].bool()  # [B]
            if not mask1.any():
                continue

            w = last_w[mask1]  # [N,K]
            sum_weights += w.double().sum(dim=0).cpu()
            sum_max_w += float(w.max(dim=1).values.sum().item())
            token_ent = -(w * torch.log(w + 1e-12)).sum(dim=1)
            sum_token_entropy += float(token_ent.sum().item())

            arg_all = torch.argmax(last_w, dim=1).tolist()  # [B]
            for b in range(bs):
                if not bool(mask1[b].item()):
                    continue
                a = int(arg_all[b])
                src_counts[batch_sources[b]][a] += 1
                if prev_arg[b] is not None and prev_arg[b] != a:
                    total_switches += 1
                prev_arg[b] = a

            total_tokens += int(mask1.sum().item())

    denom = max(total_tokens, 1)
    cov = (sum_weights / float(denom)).float()
    cov = cov / cov.sum().clamp(min=1e-12)
    coverage_entropy = _entropy(cov)
    eff = float(math.exp(coverage_entropy))

    mean_token_entropy = float(sum_token_entropy / denom)
    mean_max_w = float(sum_max_w / denom)
    switch_rate = float(total_switches / max(total_tokens - 1, 1))
    mi = _mutual_information_source_chart(src_counts)

    # Top charts by usage
    top = torch.topk(cov, k=min(8, cov.shape[0])).indices.cpu().tolist()
    top_items = [(int(i), float(cov[i].item())) for i in top]

    return {
        "num_tokens": int(total_tokens),
        "num_charts": int(K),
        "coverage_entropy": float(coverage_entropy),
        "effective_charts": float(eff),
        "mean_token_entropy": float(mean_token_entropy),
        "mean_max_weight": float(mean_max_w),
        "switch_rate": float(switch_rate),
        "mi_source_chart": float(mi),
        "top_charts": top_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path (e.g., .../last.pt)")
    parser.add_argument("--tokenizer_path", default="", help="Optional tokenizer.json path (defaults to ckpt dir).")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--eval_samples", type=int, default=256)
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
    parser.add_argument("--hf_weight", type=float, default=0.0)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    tokenizer_path = args.tokenizer_path or None
    trainer, tokenizer, train_cfg, meta = load_trainer_from_checkpoint(
        args.ckpt,
        device=device,
        tokenizer_path=tokenizer_path if tokenizer_path else None,
        override_cfg={"batch_size": int(args.batch_size), "max_seq_len": int(args.max_seq_len), "experience_push_n": 0},
        strict=False,
    )

    samples = sample_pool(
        wiki_path=args.wiki_path,
        clue_path=args.clue_path,
        max_samples_wiki=args.max_samples_wiki,
        max_samples_clue=args.max_samples_clue,
        hf_dataset=args.hf_dataset or None,
        hf_name=args.hf_name or None,
        hf_split=args.hf_split,
        hf_text_field=args.hf_text_field,
        max_samples_hf=args.max_samples_hf,
        hf_weight=args.hf_weight,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )
    if not samples:
        raise RuntimeError("No samples loaded (check paths / hf args).")

    metrics = eval_recurrent_router_stats(
        trainer=trainer,
        tokenizer=tokenizer,
        samples=samples,
        device=device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(
        "Atlas:"
        f" tokens={metrics['num_tokens']}"
        f" charts={metrics['num_charts']}"
        f" H_cov={metrics['coverage_entropy']:.4f}"
        f" N_eff={metrics['effective_charts']:.2f}"
        f" H_tok={metrics['mean_token_entropy']:.4f}"
        f" max_w={metrics['mean_max_weight']:.4f}"
        f" switch_rate={metrics['switch_rate']:.4f}"
        f" MI(source;chart)={metrics['mi_source_chart']:.6f}"
    )
    print(f"Top charts: {metrics['top_charts']}")


if __name__ == "__main__":
    main()
