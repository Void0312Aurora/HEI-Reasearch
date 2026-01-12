"""
Protocol-5 diagnostics: closed-loop (open-loop generation) amplification & attractor lock-in.

理论基础-7/快慢变量/诊断协议.md 协议 5 要点：
- teacher-forced（条件在真实前缀）排名/误差在改善，但 open-loop（把输出喂回输入）仍可能锁死到短周期循环；
- 这更像端口/解码闭环放大偏置，而不应直接归咎于“动力学本体坍塌”。

This script reports:
1) Teacher-forced one-step diagnostics (margin/accuracy) on a held-out corpus.
2) Open-loop diagnostics for many prompts:
   - n-gram repetition (proxy for mode collapse /循环)
   - detected short-period cycles in the tail
   - attractor occupancy (how many prompts fall into the same cycle signature)
   - margin amplification (early vs late top1 logit margin)

Example:
python HEI/training/eval_protocol5_closed_loop.py \
  --ckpt HEI/checkpoints/lang_active/step0017000_tok881684020.pt \
  --device cuda --eval_samples 2000 --num_prompts 200 --prompt_chars 4 \
  --top_k 1 --temperature 1.0 --max_new_tokens 64
"""

import argparse
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


def collect_eval_texts_from_pool(*, pool: CorpusPool, eval_samples: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    if not pool.samples:
        return []
    if eval_samples <= 0 or eval_samples >= len(pool.samples):
        indices = list(range(len(pool.samples)))
        rng.shuffle(indices)
        return [pool.samples[i].text for i in indices]
    indices = rng.sample(range(len(pool.samples)), eval_samples)
    return [pool.samples[i].text for i in indices]


def collect_eval_texts(*, eval_text_file: str, eval_samples: int, seed: int) -> List[str]:
    """
    Stream-sample lines from a potentially huge text file.

    NOTE: The old implementation loaded the entire file into memory, which is
    unusable for multi-GB corpora (e.g. CLUECorpusSmall). Here we use reservoir
    sampling to keep memory O(eval_samples).
    """
    rng = random.Random(seed)
    if eval_samples <= 0:
        # Backward-compatible "return all": only safe for small files.
        out: List[str] = []
        with open(eval_text_file, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    out.append(ln)
        if out:
            rng.shuffle(out)
        return out

    reservoir: List[str] = []
    seen = 0
    with open(eval_text_file, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if seen < eval_samples:
                reservoir.append(ln)
                seen += 1
                continue
            seen += 1
            j = rng.randrange(seen)
            if j < eval_samples:
                reservoir[j] = ln
    return reservoir


def extract_prompts(texts: Sequence[str], *, num_prompts: int, prompt_chars: int) -> List[str]:
    prompts: List[str] = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        p = t[:prompt_chars].strip()
        if not p:
            continue
        prompts.append(p)
        if len(prompts) >= num_prompts:
            break
    return prompts


def _ngrams(tokens: Sequence[int], n: int) -> List[Tuple[int, ...]]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]


def ngram_repetition_rate(tokens: Sequence[int], n: int) -> float:
    grams = _ngrams(tokens, n)
    total = len(grams)
    if total <= 0:
        return 0.0
    uniq = len(set(grams))
    return 1.0 - (uniq / total)


@dataclass(frozen=True)
class CycleInfo:
    period: int
    pattern: Tuple[int, ...]


def detect_tail_cycle(
    tokens: Sequence[int],
    *,
    max_period: int,
    min_repeats: int,
    tail_window: int,
) -> Optional[CycleInfo]:
    if max_period <= 0 or min_repeats <= 1:
        return None
    tail = list(tokens[-tail_window:]) if tail_window > 0 else list(tokens)
    if not tail:
        return None

    for p in range(1, max_period + 1):
        need = p * min_repeats
        if len(tail) < need:
            continue
        pattern = tuple(tail[-p:])
        ok = True
        for k in range(2, min_repeats + 1):
            seg = tuple(tail[-k * p : -(k - 1) * p])
            if seg != pattern:
                ok = False
                break
        if ok:
            return CycleInfo(period=p, pattern=pattern)
    return None


@torch.no_grad()
def teacher_forced_diagnostics(
    *,
    trainer,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int,
    max_seq_len: int,
) -> Dict[str, float]:
    trainer.eval()
    port_arch = getattr(getattr(trainer, "config", trainer), "port_arch", getattr(trainer, "port_arch", "transformer"))

    sum_margin = 0.0
    sum_max_logit = 0.0
    sum_gt_logit = 0.0
    correct = 0.0
    total_tokens = 0.0
    sum_E = 0.0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch = build_batch(batch_texts, tokenizer, max_seq_len)
        batch = {k: v.to(device) for k, v in batch.items()}

        token_ids = batch["input_ids"]
        attn = batch["attention_mask"]
        bs = int(token_ids.shape[0])

        if port_arch == "transformer":
            # --- Online rollout (no offline) ---
            if trainer.entity.state is None or trainer.entity.state.batch_size != bs:
                trainer.reset_entity(bs)
            trainer.entity.enter_online()

            u = trainer.encode_tokens(token_ids, attn)
            for _ in range(trainer.config.num_evolution_steps):
                trainer.entity.step({"language": u}, dt=trainer.config.dt)
            q_final = trainer.entity.state.q

            logits = trainer.decode_state(q_final, token_ids)  # [B,L,V]

            # Prediction error (matches training metric)
            E_pred, _ = trainer.compute_prediction_error(logits, token_ids, attn)
            sum_E += float(E_pred.item()) * bs

            # Shift for next-token stats
            shift_logits = logits[:, :-1, :]
            shift_targets = token_ids[:, 1:]
            shift_mask = attn[:, 1:].bool()

            top2 = torch.topk(shift_logits, k=2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]

            pred = shift_logits.argmax(dim=-1)
            correct_masked = (pred == shift_targets) & shift_mask

            gt_logit = shift_logits.gather(dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
            max_logit = top2[..., 0]

            token_count = float(shift_mask.sum().item())
            if token_count <= 0:
                continue

            sum_margin += float((margin * shift_mask).sum().item())
            sum_max_logit += float((max_logit * shift_mask).sum().item())
            sum_gt_logit += float((gt_logit * shift_mask).sum().item())
            correct += float(correct_masked.sum().item())
            total_tokens += token_count
        elif port_arch == "minimal":
            # --- Recurrent rollout (teacher-forced): vectorize readout across time ---
            trainer.reset_entity(bs)
            trainer.entity.enter_online()

            state_flat = trainer.entity.state.flat
            prev_weights = None
            u_seq = trainer.encode_token_sequence(token_ids)  # [B,L,Dq]

            seq_len = int(token_ids.shape[1])
            if seq_len > 1:
                q_seq = torch.empty((bs, seq_len - 1, trainer.entity.dim_q), device=device, dtype=state_flat.dtype)
            else:
                q_seq = torch.empty((bs, 0, trainer.entity.dim_q), device=device, dtype=state_flat.dtype)

            for t in range(seq_len):
                mask_curr = attn[:, t].float().unsqueeze(1)
                u_t = u_seq[:, t, :] * mask_curr
                for _ in range(trainer.config.num_evolution_steps):
                    out = trainer.entity.forward_tensor(
                        state_flat=state_flat,
                        u_dict={"language": u_t},
                        dt=trainer.config.dt,
                        prev_chart_weights=prev_weights,
                        prediction_error=None,
                        detach_next_prev_weights=True,
                        compute_action=False,
                    )
                    next_state_flat = out["next_state_flat"]
                    next_prev = out["next_prev_chart_weights"]
                    state_flat = mask_curr * next_state_flat + (1.0 - mask_curr) * state_flat
                    prev_weights = next_prev if prev_weights is None else (mask_curr * next_prev + (1.0 - mask_curr) * prev_weights)

                if t < seq_len - 1:
                    q_seq[:, t, :] = state_flat[:, : trainer.entity.dim_q]

            logits_seq = trainer.decode_logits_from_q(q_seq)  # [B,L-1,V]
            targets_seq = token_ids[:, 1:]
            mask_seq = attn[:, 1:].bool()

            flat_logits = logits_seq.reshape(-1, logits_seq.shape[-1])
            flat_targets = targets_seq.reshape(-1)
            flat_loss = torch.nn.functional.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=tokenizer.pad_id,
                reduction="none",
            )
            loss_mat = flat_loss.view(bs, -1)
            mask_f = mask_seq.float()
            per_sample_loss = (loss_mat * mask_f).sum(dim=1)
            per_sample_count = mask_f.sum(dim=1).clamp(min=1.0)
            E_pred_per_sample = per_sample_loss / per_sample_count
            sum_E += float(E_pred_per_sample.mean().item()) * bs

            top2 = torch.topk(logits_seq, k=2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
            pred = logits_seq.argmax(dim=-1)
            correct_masked = (pred == targets_seq) & mask_seq
            gt_logit = logits_seq.gather(dim=-1, index=targets_seq.unsqueeze(-1)).squeeze(-1)
            max_logit = top2[..., 0]

            token_count = float(mask_seq.sum().item())
            if token_count > 0:
                sum_margin += float((margin * mask_f).sum().item())
                sum_max_logit += float((max_logit * mask_f).sum().item())
                sum_gt_logit += float((gt_logit * mask_f).sum().item())
                correct += float(correct_masked.sum().item())
                total_tokens += token_count
        else:
            raise ValueError(f"Unknown port_arch: {port_arch}")

    avg_E = sum_E / max(len(texts), 1)
    return {
        "tf_E_pred": avg_E,
        "tf_PPL": float(torch.exp(torch.tensor(avg_E)).item()),
        "tf_top1_acc": correct / max(total_tokens, 1.0),
        "tf_margin": sum_margin / max(total_tokens, 1.0),
        "tf_max_logit": sum_max_logit / max(total_tokens, 1.0),
        "tf_gt_logit": sum_gt_logit / max(total_tokens, 1.0),
        "tf_tokens": total_tokens,
    }


@torch.no_grad()
def generate_open_loop(
    *,
    trainer,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> Tuple[List[int], List[float]]:
    trainer.eval()
    port_arch = getattr(getattr(trainer, "config", trainer), "port_arch", getattr(trainer, "port_arch", "transformer"))
    trainer.reset_entity(1)

    ids = tokenizer.encode(prompt)
    prefix_ids = ids[:-1] if len(ids) >= 2 else ids
    if not prefix_ids:
        prefix_ids = [tokenizer.bos_id]

    if port_arch == "transformer":
        token_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones_like(token_ids, dtype=torch.long, device=device)

        u = trainer.encode_tokens(token_ids, attn)
        for _ in range(trainer.config.num_evolution_steps):
            trainer.entity.step({"language": u}, dt=trainer.config.dt)

        q = trainer.entity.state.q  # [1,dim_q]

        generated = list(prefix_ids)
        margins: List[float] = []
        for _ in range(max_new_tokens):
            prev = torch.tensor([generated], dtype=torch.long, device=device)
            logits = trainer.language_port.decoder(q, prev)[:, -1, :].squeeze(0)  # [V]

            logits = logits / max(float(temperature), 1e-6)
            top2 = torch.topk(logits, k=2, dim=-1).values
            margins.append(float((top2[0] - top2[1]).item()))

            if top_k > 0:
                v, _ = torch.topk(logits, min(int(top_k), int(logits.shape[-1])))
                logits = logits.clone()
                logits[logits < v[-1]] = float("-inf")

            if top_k == 1:
                next_token = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())

            generated.append(next_token)
            if next_token == tokenizer.eos_id:
                break

        return generated, margins

    if port_arch == "minimal":
        trainer.entity.enter_online()
        state_flat = trainer.entity.state.flat
        prev_weights = None

        # Consume prefix
        for tok in prefix_ids:
            tok_t = torch.tensor([[tok]], device=device, dtype=torch.long)
            u_t = trainer.token_embedding(tok_t).squeeze(1)  # [1,Dq]
            for _ in range(int(trainer.config.num_evolution_steps)):
                out = trainer.entity.forward_tensor(
                    state_flat=state_flat,
                    u_dict={"language": u_t},
                    dt=trainer.config.dt,
                    prev_chart_weights=prev_weights,
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                )
                state_flat = out["next_state_flat"]
                prev_weights = out["next_prev_chart_weights"]

        generated = list(prefix_ids)
        margins: List[float] = []
        for _ in range(max_new_tokens):
            q = state_flat[:, : trainer.entity.dim_q]
            logits = trainer.decode_logits_from_q(q).squeeze(0)  # [V]

            logits = logits / max(float(temperature), 1e-6)
            top2 = torch.topk(logits, k=2, dim=-1).values
            margins.append(float((top2[0] - top2[1]).item()))

            if top_k > 0:
                v, _ = torch.topk(logits, min(int(top_k), int(logits.shape[-1])))
                logits = logits.clone()
                logits[logits < v[-1]] = float("-inf")

            if top_k == 1:
                next_token = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())

            generated.append(next_token)
            if next_token == tokenizer.eos_id:
                break

            tok_t = torch.tensor([[next_token]], device=device, dtype=torch.long)
            u_t = trainer.token_embedding(tok_t).squeeze(1)
            for _ in range(int(trainer.config.num_evolution_steps)):
                out = trainer.entity.forward_tensor(
                    state_flat=state_flat,
                    u_dict={"language": u_t},
                    dt=trainer.config.dt,
                    prev_chart_weights=prev_weights,
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                )
                state_flat = out["next_state_flat"]
                prev_weights = out["next_prev_chart_weights"]

        return generated, margins

    raise ValueError(f"Unknown port_arch: {port_arch}")


@torch.no_grad()
def generate_open_loop_batch(
    *,
    trainer,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Batched open-loop generation for gate evaluation.

    This is critical for Stage7 gate speed: the naïve per-prompt loop issues
    thousands of tiny kernels and keeps GPU utilization low.
    """
    if not prompts:
        return [], []

    trainer.eval()
    port_arch = getattr(getattr(trainer, "config", trainer), "port_arch", getattr(trainer, "port_arch", "transformer"))
    if port_arch != "minimal":
        # Fallback: keep old behavior.
        out_tokens: List[List[int]] = []
        out_margins: List[List[float]] = []
        for p in prompts:
            tks, ms = generate_open_loop(
                trainer=trainer,
                tokenizer=tokenizer,
                prompt=p,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            out_tokens.append(tks)
            out_margins.append(ms)
        return out_tokens, out_margins

    B = int(len(prompts))
    eos_id = int(getattr(tokenizer, "eos_id", getattr(tokenizer, "pad_id", 0)))
    pad_id = int(getattr(tokenizer, "pad_id", 0))

    # Encode prompts.
    prefix_ids_list: List[List[int]] = []
    prefix_lens: List[int] = []
    for p in prompts:
        ids = tokenizer.encode(p)
        prefix_ids = ids[:-1] if len(ids) >= 2 else ids
        if not prefix_ids:
            prefix_ids = [tokenizer.bos_id]
        prefix_ids_list.append(prefix_ids)
        prefix_lens.append(len(prefix_ids))

    prefix_max = max(prefix_lens) if prefix_lens else 1
    prefix_tokens = torch.full((B, prefix_max), pad_id, dtype=torch.long, device=device)
    prefix_mask = torch.zeros((B, prefix_max), dtype=torch.float32, device=device)
    for i, ids in enumerate(prefix_ids_list):
        L = len(ids)
        if L <= 0:
            continue
        prefix_tokens[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        prefix_mask[i, :L] = 1.0

    # Roll-in: consume prefix (teacher tokens).
    trainer.reset_entity(B)
    trainer.entity.enter_online()
    state_flat = trainer.entity.state.flat
    prev_weights = None
    for t in range(prefix_max):
        mask_t = prefix_mask[:, t].unsqueeze(1)
        tok_t = prefix_tokens[:, t]
        u_t = trainer.token_embedding(tok_t.unsqueeze(1)).squeeze(1) * mask_t
        for _ in range(int(trainer.config.num_evolution_steps)):
            out = trainer.entity.forward_tensor(
                state_flat=state_flat,
                u_dict={"language": u_t},
                dt=trainer.config.dt,
                prev_chart_weights=prev_weights,
                prediction_error=None,
                detach_next_prev_weights=True,
                compute_action=False,
            )
            next_state = out["next_state_flat"]
            next_prev = out["next_prev_chart_weights"]
            state_flat = mask_t * next_state + (1.0 - mask_t) * state_flat
            prev_weights = next_prev if prev_weights is None else (mask_t * next_prev + (1.0 - mask_t) * prev_weights)

    gen_tokens = torch.full((B, max_new_tokens), pad_id, dtype=torch.long, device=device)
    gen_margins = torch.zeros((B, max_new_tokens), dtype=torch.float32, device=device)
    alive = torch.ones((B,), dtype=torch.bool, device=device)

    temp = max(float(temperature), 1e-6)
    k = int(top_k)
    for t in range(int(max_new_tokens)):
        q = state_flat[:, : trainer.entity.dim_q]
        logits = trainer.decode_logits_from_q(q)  # [B,V]
        logits = logits / temp

        top2 = torch.topk(logits, k=2, dim=-1).values
        gen_margins[:, t] = (top2[:, 0] - top2[:, 1]).float()

        step_logits = logits
        if k > 0:
            v, _ = torch.topk(step_logits, min(k, int(step_logits.shape[-1])), dim=-1)
            cutoff = v[:, -1].unsqueeze(-1)
            step_logits = step_logits.masked_fill(step_logits < cutoff, float("-inf"))

        if k == 1:
            next_tok = torch.argmax(step_logits, dim=-1)
        else:
            probs = torch.softmax(step_logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs.clamp(min=0.0)
            probs = probs + 1e-8
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            next_tok = torch.multinomial(probs, 1).squeeze(-1)

        alive_before = alive
        next_tok = torch.where(alive_before, next_tok, torch.full_like(next_tok, eos_id))
        gen_tokens[:, t] = next_tok

        update_mask = (alive_before & (next_tok != eos_id)).float().unsqueeze(1)
        alive = alive_before & (next_tok != eos_id)

        # Feed back generated token into dynamics for sequences still alive.
        u_t = trainer.token_embedding(next_tok.unsqueeze(1)).squeeze(1) * update_mask
        for _ in range(int(trainer.config.num_evolution_steps)):
            out = trainer.entity.forward_tensor(
                state_flat=state_flat,
                u_dict={"language": u_t},
                dt=trainer.config.dt,
                prev_chart_weights=prev_weights,
                prediction_error=None,
                detach_next_prev_weights=True,
                compute_action=False,
            )
            next_state = out["next_state_flat"]
            next_prev = out["next_prev_chart_weights"]
            state_flat = update_mask * next_state + (1.0 - update_mask) * state_flat
            prev_weights = next_prev if prev_weights is None else (update_mask * next_prev + (1.0 - update_mask) * prev_weights)

        if not bool(alive.any()):
            break

    gen_tokens_cpu = gen_tokens.detach().cpu()
    gen_margins_cpu = gen_margins.detach().cpu()

    out_tokens: List[List[int]] = []
    out_margins: List[List[float]] = []
    for i in range(B):
        prefix = prefix_ids_list[i]
        row = gen_tokens_cpu[i].tolist()
        mrow = gen_margins_cpu[i].tolist()
        cut = len(row)
        if eos_id in row:
            cut = row.index(eos_id) + 1
        out_tokens.append(prefix + row[:cut])
        out_margins.append(mrow[:cut])
    return out_tokens, out_margins


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path (e.g., .../last.pt)")
    parser.add_argument("--tokenizer_path", default="", help="Optional tokenizer.json path (defaults to ckpt dir).")
    parser.add_argument("--device", default="cuda")

    # Evaluation corpus
    parser.add_argument("--eval_text_file", default="", help="Optional newline-separated evaluation texts.")
    parser.add_argument("--wiki_path", default="HEI/data/wiki/wikipedia-zh-20250901.json")
    parser.add_argument("--clue_path", default="HEI/data/CLUE/CLUECorpusSmall.txt")
    parser.add_argument("--max_samples_wiki", type=int, default=200000)
    parser.add_argument("--max_samples_clue", type=int, default=200000)
    parser.add_argument("--eval_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)

    # Optional HF streaming data
    parser.add_argument("--hf_dataset", default="", help="Optional HuggingFace dataset name (requires `pip install datasets`).")
    parser.add_argument("--hf_name", default="", help="Optional HuggingFace dataset config name.")
    parser.add_argument("--hf_split", default="validation", help="HuggingFace split (streaming).")
    parser.add_argument("--hf_text_field", default="text", help="Text field key for HuggingFace dataset.")
    parser.add_argument("--max_samples_hf", type=int, default=0, help="How many HF samples to stream into eval pool.")
    parser.add_argument("--hf_weight", type=float, default=0.0, help="Source weight assigned to HF samples (renormalizes wiki/clue weights).")

    # Protocol 5 prompt set
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--prompt_chars", type=int, default=4)

    # Teacher-forced batching
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=128)

    # Open-loop generation
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--show_samples", type=int, default=4)

    # Cycle / repetition detection
    parser.add_argument("--cycle_max_period", type=int, default=8)
    parser.add_argument("--cycle_min_repeats", type=int, default=3)
    parser.add_argument("--cycle_tail_window", type=int, default=32)
    parser.add_argument("--ngram_n", type=int, default=3)
    parser.add_argument("--margin_window", type=int, default=10, help="Compute early/late margin means using this window.")
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
        base_weights = {"wiki": 0.5, "clue": 0.5}
        if args.hf_dataset and args.max_samples_hf > 0 and args.hf_weight > 0:
            hf_w = float(args.hf_weight)
            hf_w = max(0.0, min(1.0, hf_w))
            scale = max(0.0, 1.0 - hf_w)
            base_weights = {"wiki": base_weights["wiki"] * scale, "clue": base_weights["clue"] * scale, "hf": hf_w}

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
            source_weights=base_weights,
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

    # --- Open-loop diagnostics ---
    cycle_counter: Counter = Counter()
    unique_outputs: set = set()
    cycle_hits = 0
    rep_rates: List[float] = []
    amp_ratios: List[float] = []

    show = max(int(args.show_samples), 0)
    for i, prompt in enumerate(prompts):
        tokens, margins = generate_open_loop(
            trainer=trainer,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        out_text = tokenizer.decode(tokens)
        unique_outputs.add(out_text)

        # Approx repetition on the continuation (exclude the prefix prompt tokens where possible)
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

        w = max(int(args.margin_window), 1)
        early = margins[:w]
        late = margins[-w:] if len(margins) >= w else margins
        early_m = _mean(early)
        late_m = _mean(late)
        if early_m > 0:
            amp_ratios.append(late_m / early_m)

        if i < show:
            print(f"[Sample] prompt={prompt!r} -> {out_text}")

    cycle_rate = cycle_hits / max(len(prompts), 1)
    dominant_cycle = 0.0
    if cycle_counter:
        dominant_cycle = cycle_counter.most_common(1)[0][1] / max(len(prompts), 1)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(
        "TeacherForced:"
        f" E_pred={tf['tf_E_pred']:.4f}"
        f" PPL={tf['tf_PPL']:.2f}"
        f" top1_acc={tf['tf_top1_acc']:.4f}"
        f" margin={tf['tf_margin']:.4f}"
        f" max_logit={tf['tf_max_logit']:.4f}"
        f" gt_logit={tf['tf_gt_logit']:.4f}"
        f" tokens={int(tf['tf_tokens'])}"
    )
    print(
        "OpenLoop:"
        f" prompts={len(prompts)}"
        f" top_k={args.top_k} temp={args.temperature}"
        f" max_new_tokens={args.max_new_tokens}"
        f" rep{args.ngram_n}={_mean(rep_rates):.4f}"
        f" cycle_rate={cycle_rate:.4f}"
        f" dominant_cycle={dominant_cycle:.4f}"
        f" unique_outputs={len(unique_outputs)}/{len(prompts)}"
        f" margin_amp={_mean(amp_ratios):.4f}"
    )


if __name__ == "__main__":
    main()
