"""
Active Sampling Training for SoulEntity language skills.

Data sources:
- HEI/data/wiki/wikipedia-zh-20250901.json
- HEI/data/CLUE/CLUECorpusSmall.txt

Run:
python HEI/training/train_active_sampling.py --tokenizer byte --port_arch minimal --sequence_mode recurrent --port_trainable 0 --steps 2000 --batch_size 64

Baseline (Transformer port, pooled u):
python HEI/training/train_active_sampling.py --tokenizer char --port_arch transformer --sequence_mode pooled --steps 2000 --batch_size 512
"""

import argparse
import hashlib
import contextlib
import math
import os
import queue
import random
import signal
import sys
import threading
import time
import traceback
from collections import Counter
from typing import Dict, List, Tuple

import torch
from dataclasses import asdict

import torch
from torch.optim import AdamW

# Ensure HEI is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from he_core.language_interface import SimpleTokenizer, ByteTokenizer
from training.active_sampling import ActiveSamplingConfig, CorpusPool, ActiveSampler
from training.soul_language_training import TrainingConfig, SoulLanguageTrainer
from training.checkpoint_io import load_trainer_from_checkpoint, save_checkpoint


def count_parameters(module: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def count_unique_parameters(modules: List[torch.nn.Module]) -> Dict[str, int]:
    seen = set()
    total = 0
    trainable = 0
    for module in modules:
        for p in module.parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            n = int(p.numel())
            total += n
            if p.requires_grad:
                trainable += n
    return {"total": total, "trainable": trainable}


def build_tokenizer(pool: CorpusPool, vocab_size: int, max_samples: int, tokenizer_type: str):
    if tokenizer_type == "byte":
        return ByteTokenizer()
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, mode="char")
    texts = [s.text for s in random.sample(pool.samples, min(max_samples, len(pool.samples)))]
    tokenizer.build_vocab(texts)
    return tokenizer


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


def _clip_grad_norm_no_sync_(params, max_norm: float = 1.0, eps: float = 1e-6) -> None:
    max_norm = float(max_norm)
    if max_norm <= 0.0:
        return
    grads = [p.grad for p in params if getattr(p, "grad", None) is not None]
    if not grads:
        return
    # CPU fallback: native implementation is fine.
    if grads[0].device.type != "cuda":
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        return

    device = grads[0].device
    dtype = grads[0].dtype
    total_sq = torch.zeros((), device=device, dtype=dtype)
    for g in grads:
        total_sq = total_sq + (g * g).sum()
    total_norm = torch.sqrt(total_sq)
    coef = max_norm / (total_norm + float(eps))
    coef = torch.clamp(coef, max=1.0)
    torch._foreach_mul_(grads, coef)


def _pretokenize_samples(
    samples: List[str], tokenizer, seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    pad_id = int(tokenizer.pad_id)
    eos_id = int(getattr(tokenizer, "eos_id", pad_id))
    seq_len = int(seq_len)
    token_buf = torch.full((len(samples), seq_len), pad_id, dtype=torch.long)
    mask_buf = torch.zeros((len(samples), seq_len), dtype=torch.long)
    for i, text in enumerate(samples):
        ids = tokenizer.encode(text, add_special=True)
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            if eos_id not in ids:
                ids[-1] = eos_id
        mask = [1] * len(ids)
        if len(ids) < seq_len:
            pad_len = seq_len - len(ids)
            ids = ids + [pad_id] * pad_len
            mask = mask + [0] * pad_len
        token_buf[i, : seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
        mask_buf[i, :] = torch.tensor(mask[:seq_len], dtype=torch.long)
    return token_buf, mask_buf


def _apply_denoise(
    token_in: torch.Tensor,
    *,
    token_clean: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
    prob: float,
    vocab_size: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    unk_id: int | None = None,
) -> torch.Tensor:
    mode = str(mode or "none").lower()
    prob = float(prob or 0.0)
    if (not math.isfinite(prob)) or prob <= 0.0 or mode == "none":
        return token_in

    if token_in.shape != token_clean.shape or token_in.shape != attention_mask.shape:
        raise ValueError("denoise: token_in, token_clean, attention_mask must share the same shape.")

    device = token_in.device
    mask = attention_mask.to(dtype=torch.bool)

    corr = (torch.rand(token_in.shape, device=device) < prob) & mask
    # Keep BOS position stable (prefix anchor); do not corrupt special tokens.
    if token_in.shape[1] > 0:
        corr[:, 0] = False
    corr = corr & (token_clean != pad_id) & (token_clean != bos_id) & (token_clean != eos_id)
    if unk_id is not None:
        corr = corr & (token_clean != int(unk_id))

    if mode == "replace":
        special_max = max(int(pad_id), int(bos_id), int(eos_id), int(unk_id) if unk_id is not None else 0)
        low = special_max + 1
        if low >= int(vocab_size):
            low = 0
        rand = torch.randint(low, int(vocab_size), token_in.shape, device=device, dtype=torch.long)
        token_in = torch.where(corr, rand, token_in)
        return token_in

    if mode == "repeat":
        prev = token_clean.clone()
        if token_clean.shape[1] > 1:
            prev[:, 1:] = token_clean[:, :-1]
        token_in = torch.where(corr, prev, token_in)
        return token_in

    raise ValueError(f"Unknown denoise_mode: {mode}")


def _maybe_init_entity_from_state_dict(entity: torch.nn.Module, ckpt_path: str) -> None:
    ckpt_path = str(ckpt_path or "").strip()
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        print(f"[Init] entity init checkpoint not found: {ckpt_path}")
        return
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        print(f"[Init] failed to load entity init checkpoint: {ckpt_path} ({exc})")
        return
    if not isinstance(payload, dict):
        print(f"[Init] unsupported entity init payload type: {type(payload)} ({ckpt_path})")
        return

    # Common prefixes for curriculum stage checkpoints.
    prefix_candidates = ["entity.", "seq_model.entity.", "trainer.entity.", "model.entity."]
    best_prefix = ""
    best_state: Dict[str, torch.Tensor] = {}
    for prefix in prefix_candidates:
        mapped = {k[len(prefix) :]: v for k, v in payload.items() if isinstance(k, str) and k.startswith(prefix)}
        if len(mapped) > len(best_state):
            best_state = mapped
            best_prefix = prefix

    state = best_state if len(best_state) >= 10 else {k: v for k, v in payload.items() if isinstance(k, str)}
    used = best_prefix if state is best_state and best_prefix else "<none>"
    incompatible = entity.load_state_dict(state, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    print(
        f"[Init] entity={type(entity).__name__} init_from={ckpt_path} prefix={used} "
        f"loaded={len(state)} missing={len(missing)} unexpected={len(unexpected)}"
    )


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _stage7_gate_profile_defaults(profile: str) -> Dict[str, float]:
    profile = str(profile or "base").lower()
    if profile == "robust":
        return {
            "gate_ppl_ratio": 0.60,
            "gate_cycle_rate": 0.40,
            "gate_dominant_cycle": 0.10,
            "gate_rep": 0.50,
            "gate_unique_ratio": 0.60,
        }
    return {
        "gate_ppl_ratio": 0.85,
        "gate_cycle_rate": 0.60,
        "gate_dominant_cycle": 0.20,
        "gate_rep": 0.60,
        "gate_unique_ratio": 0.40,
    }


def _read_nonempty_lines(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(ln)
    return out


def _eval_cache_path(*, eval_text_file: str, eval_samples: int, seed: int, cache_dir: str) -> str:
    try:
        st = os.stat(eval_text_file)
        stamp = f"{int(st.st_size)}:{int(st.st_mtime)}"
    except Exception:
        stamp = "nostat"
    key = f"{os.path.abspath(eval_text_file)}|{stamp}|{int(eval_samples)}|{int(seed)}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    base = os.path.basename(eval_text_file).replace(os.sep, "_")
    name = f"stage7_eval_texts_{base}_n{int(eval_samples)}_seed{int(seed)}_{h}.txt"
    return os.path.join(cache_dir, name)


@torch.no_grad()
def _run_stage7_gate_eval(
    *,
    trainer,
    tokenizer,
    texts: List[str],
    prompts: List[str],
    device: str,
    profile: str,
    batch_size: int,
    max_seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    cycle_max_period: int,
    cycle_min_repeats: int,
    cycle_tail_window: int,
    ngram_n: int,
    margin_window: int,
    gate_ppl_ratio: float,
    gate_cycle_rate: float,
    gate_dominant_cycle: float,
    gate_rep: float,
    gate_unique_ratio: float,
) -> Dict[str, object]:
    from training.eval_protocol5_closed_loop import (
        detect_tail_cycle,
        generate_open_loop_batch,
        ngram_repetition_rate,
        teacher_forced_diagnostics,
    )

    trainer.eval()

    tf = teacher_forced_diagnostics(
        trainer=trainer,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=int(batch_size),
        max_seq_len=int(max_seq_len),
    )
    vocab = float(len(tokenizer))
    ppl_ratio = float(tf["tf_PPL"]) / max(vocab, 1.0)

    cycle_counter: Counter = Counter()
    unique_outputs: set = set()
    cycle_hits = 0
    rep_rates: List[float] = []
    amp_ratios: List[float] = []

    top_k_eff = int(top_k)
    if top_k_eff <= 0:
        top_k_eff = 20 if str(profile).lower() == "base" else 1

    w = max(int(margin_window), 1)
    all_tokens, all_margins = generate_open_loop_batch(
        trainer=trainer,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=top_k_eff,
    )
    for prompt, tokens, margins in zip(prompts, all_tokens, all_margins):
        out_text = tokenizer.decode(tokens)
        unique_outputs.add(out_text)

        ids = tokenizer.encode(prompt)
        prefix_ids = ids[:-1] if len(ids) >= 2 else ids
        prefix_len = max(len(prefix_ids), 1)
        cont = tokens[prefix_len:] if len(tokens) > prefix_len else []
        rep_rates.append(float(ngram_repetition_rate(cont, int(ngram_n))))

        cinfo = detect_tail_cycle(
            cont,
            max_period=int(cycle_max_period),
            min_repeats=int(cycle_min_repeats),
            tail_window=int(cycle_tail_window),
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

    checks = {
        "TeacherForced:PPL_ratio": (ppl_ratio <= float(gate_ppl_ratio)),
        "OpenLoop:cycle_rate": (cycle_rate <= float(gate_cycle_rate)),
        "OpenLoop:dominant_cycle": (dominant_cycle <= float(gate_dominant_cycle)),
        f"OpenLoop:rep{int(ngram_n)}": (rep <= float(gate_rep)),
        "OpenLoop:unique_ratio": (unique_ratio >= float(gate_unique_ratio)),
    }
    passed = all(bool(v) for v in checks.values())

    return {
        "passed": bool(passed),
        "checks": checks,
        "teacher_forced": tf,
        "ppl_ratio": float(ppl_ratio),
        "open_loop": {
            "rep": float(rep),
            "cycle_rate": float(cycle_rate),
            "dominant_cycle": float(dominant_cycle),
            "unique_ratio": float(unique_ratio),
            "margin_amp": float(margin_amp),
            "prompts": int(len(prompts)),
        },
    }


class BatchPrefetcher:
    def __init__(
        self,
        *,
        sampler,
        tokenizer,
        seq_len: int,
        batch_size: int,
        prefetch_batches: int,
        device: torch.device,
    ):
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.device = device
        self.queue: queue.Queue = queue.Queue(maxsize=max(2, int(prefetch_batches)))
        self.shutdown = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        while not self.shutdown.is_set():
            texts, buckets, tokens, mask = self.sampler.sample_batch(self.batch_size)
            if not texts:
                continue
            if tokens is None or mask is None:
                batch = build_batch(texts, self.tokenizer, self.seq_len)
                tokens = batch["input_ids"]
                mask = batch["attention_mask"]
            if tokens.device.type == "cpu" and not tokens.is_pinned():
                tokens = tokens.pin_memory()
                mask = mask.pin_memory()
            try:
                self.queue.put((tokens, mask, buckets), timeout=1.0)
            except queue.Full:
                continue

    def next_batch(self, timeout: float | None = None):
        return self.queue.get(timeout=timeout)

    def stop(self) -> None:
        self.shutdown.set()
        self.thread.join(timeout=5.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_path", default="HEI/data/wiki/wikipedia-zh-20250901.json")
    parser.add_argument("--clue_path", default="HEI/data/CLUE/CLUECorpusSmall.txt")
    parser.add_argument("--hf_dataset", default="", help="Optional HuggingFace dataset name (requires `pip install datasets`).")
    parser.add_argument("--hf_name", default="", help="Optional HuggingFace dataset config name.")
    parser.add_argument("--hf_split", default="train", help="HuggingFace split (streaming).")
    parser.add_argument("--hf_text_field", default="text", help="Text field key for HuggingFace dataset.")
    parser.add_argument("--max_samples_hf", type=int, default=0, help="How many samples to stream from HuggingFace into the active pool (0 = disabled).")
    parser.add_argument("--hf_weight", type=float, default=0.0, help="Source weight assigned to HuggingFace samples (renormalizes wiki/clue weights).")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dim_q", type=int, default=64)
    parser.add_argument("--dim_z", type=int, default=16)
    parser.add_argument("--dim_embed", type=int, default=256)
    parser.add_argument("--num_charts", type=int, default=4)
    parser.add_argument("--tokenizer", default="byte", choices=["byte", "char"])
    parser.add_argument("--port_arch", default="minimal", choices=["minimal", "transformer"])
    parser.add_argument("--sequence_mode", default="recurrent", choices=["recurrent", "pooled"])
    parser.add_argument("--tie_io_weights", type=int, default=1, help="minimal port only (1/0)")
    parser.add_argument("--port_trainable", type=int, default=0, help="minimal port only (1/0): if 0, keep UTF-8/byte port parameter-free")
    parser.add_argument("--port_init_std", type=float, default=0.02, help="minimal port only: embedding/logit init std (0=disable)")
    parser.add_argument("--scheduled_sampling_prob", type=float, default=0.0, help="recurrent only: probability to feed model token instead of teacher token")
    parser.add_argument("--scheduled_sampling_mode", default="sample", choices=["sample", "argmax"])
    parser.add_argument("--scheduled_sampling_top_k", type=int, default=20)
    parser.add_argument("--scheduled_sampling_temperature", type=float, default=1.0)
    parser.add_argument(
        "--unlikelihood_weight",
        type=float,
        default=0.0,
        help="Protocol-5 hardening: unlikelihood weight penalizing p(x_{t+1}=x_t) unless target repeats (0=disable).",
    )
    parser.add_argument(
        "--unlikelihood_window",
        type=int,
        default=1,
        help="Protocol-5 hardening: penalize repeating any of the last K fed tokens (K=1 matches immediate-repeat).",
    )
    parser.add_argument(
        "--denoise_mode",
        default="none",
        choices=["none", "replace", "repeat"],
        help="B3 denoising: corrupt the fed token stream but predict clean targets (none/replace/repeat).",
    )
    parser.add_argument(
        "--denoise_prob",
        type=float,
        default=0.0,
        help="B3 denoising: corruption probability per non-pad token (0=disable).",
    )
    parser.add_argument("--router_balance_weight", type=float, default=0.0, help="encourage uniform chart usage (skills capacity)")
    parser.add_argument("--router_entropy_weight", type=float, default=0.0, help="penalize router entropy (encourage sparse chart selection)")
    # NOTE: pushing experience involves randomness + bookkeeping that adds overhead and can induce syncs.
    # Keep default off; enable explicitly when training with offline replay.
    parser.add_argument(
        "--experience_push_n",
        type=int,
        default=0,
        help="recurrent only: push N end-states per batch into experience buffer (0=disable)",
    )
    parser.add_argument("--port_coupling_top_k", type=int, default=0, help="L3: only use top-k charts per step in port coupling (0 = dense)")
    parser.add_argument(
        "--port_coupling_impl",
        default="dense",
        choices=["grouped", "dense"],
        help="L3 impl detail: grouped=top-k via per-chart grouping; dense=dense einsum then mask to top-k (usually higher GPU utilization).",
    )
    parser.add_argument("--transport_threshold", type=float, default=0.1, help="L2: apply parallel transport when chart-weight change >= threshold")
    parser.add_argument("--connection_rank", type=int, default=0, help="L2: connection low-rank (0=auto, <0=full, >0=rank)")
    parser.add_argument("--q_clip_norm", type=float, default=100.0, help="stability: clip ||q|| per sample (0=disable)")
    parser.add_argument("--p_clip_norm", type=float, default=100.0, help="stability: clip ||p|| per sample (0=disable)")
    parser.add_argument("--s_clip_abs", type=float, default=100.0, help="stability: clip |s| (0=disable)")
    parser.add_argument("--sanitize_nonfinite", type=int, default=1, help="stability: replace NaN/Inf in state with 0 (1/0)")
    parser.add_argument("--detach_every", type=int, default=0, help="recurrent only: truncate BPTT every N tokens (0=disabled)")
    parser.add_argument("--vocab_size", type=int, default=10000, help="char tokenizer only (ignored for byte)")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.1, help="Online dynamics step size.")
    parser.add_argument("--num_evolution_steps", type=int, default=1, help="Integrator steps per token (online).")
    parser.add_argument("--beta_kl", type=float, default=0.01, help="A3: free-energy KL weight.")
    parser.add_argument("--gamma_pred", type=float, default=1.0, help="A3: prediction-error weight.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_samples_wiki", type=int, default=200000)
    parser.add_argument("--max_samples_clue", type=int, default=200000)
    parser.add_argument("--vocab_build_samples", type=int, default=50000)
    parser.add_argument("--device", default="cuda")
    # Default on for CUDA to match later curriculum stages (higher throughput, stable numerics for this workload).
    parser.add_argument("--tf32", type=int, default=1, help="Enable TF32 matmul on Ampere+ (1/0).")
    parser.add_argument("--amp", type=int, default=0, help="Enable CUDA autocast mixed precision (1/0).")
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"], help="Autocast dtype when --amp=1.")
    parser.add_argument("--fused_adamw", type=int, default=1, help="Use fused AdamW on CUDA if available (1/0).")
    parser.add_argument("--num_offline_steps", type=int, default=1)
    parser.add_argument("--offline_dt", type=float, default=0.1)
    parser.add_argument("--offline_replay_mode", default="prioritized")
    parser.add_argument("--offline_weight", type=float, default=1.0)
    parser.add_argument(
        "--offline_loss_mode",
        default="relu_delta",
        choices=["end", "delta", "relu_delta"],
        help="How offline free-energy participates in training: end=use F_end; delta=use (F_end-F_start); relu_delta=penalize increases only.",
    )
    parser.add_argument(
        "--offline_margin",
        type=float,
        default=0.0,
        help="Margin for relu_delta: loss = relu((F_end-F_start)+margin).",
    )
    parser.add_argument("--offline_detach_init", type=int, default=0)
    parser.add_argument("--reset_each_batch", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=0, help="Stop after processing this many non-pad tokens (0 = disabled).")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sample_every", type=int, default=200)
    parser.add_argument(
        "--sampler_loss_update_every",
        type=int,
        default=0,
        help="Active sampler: update bucket loss EMA every N steps (0 = disable loss updates; counts still update).",
    )
    parser.add_argument(
        "--sampler_loss_update_mode",
        default="batch_mean",
        choices=["per_sample", "batch_mean"],
        help="Active sampler: per_sample uses per-sample prediction error (slower, syncs); batch_mean uses batch mean only (faster).",
    )
    parser.add_argument(
        "--sampler_loss_weight",
        type=float,
        default=1.0,
        help="Active sampler score weight for normalized loss (difficulty).",
    )
    parser.add_argument(
        "--sampler_coverage_weight",
        type=float,
        default=0.5,
        help="Active sampler score weight for coverage deficit (target distribution matching).",
    )
    parser.add_argument(
        "--sampler_progress_weight",
        type=float,
        default=0.0,
        help="Active sampler score weight for learning progress (EMA loss decrease).",
    )
    parser.add_argument(
        "--sampler_temperature",
        type=float,
        default=1.0,
        help="Active sampler temperature (higher=flatter, lower=sharper).",
    )
    parser.add_argument(
        "--sampler_random_ratio",
        type=float,
        default=0.1,
        help="Active sampler random batch fraction in [0,1].",
    )
    parser.add_argument(
        "--sampler_ema_rate",
        type=float,
        default=0.1,
        help="Active sampler EMA rate for bucket loss/progress in (0,1].",
    )
    parser.add_argument("--save_dir", default="", help="If set, save checkpoints (tokenizer/config/last.pt).")
    parser.add_argument("--save_every", type=int, default=0, help="Also keep step checkpoints every N steps (0 = only last.pt).")
    parser.add_argument("--resume_ckpt", default="", help="Resume training from this checkpoint .pt (e.g. .../last.pt).")
    parser.add_argument("--resume_dir", default="", help="Resume training from <dir>/last.pt.")
    parser.add_argument("--resume_tokenizer_path", default="", help="Optional tokenizer.json path for resume.")
    parser.add_argument("--resume_optimizer", type=int, default=1, help="Load optimizer state if present in checkpoint (1/0).")
    parser.add_argument("--resume_strict", type=int, default=1, help="Strict state_dict loading (1/0).")
    parser.add_argument(
        "--init_entity_ckpt",
        default="",
        help="Optional: initialize SoulEntity weights from an external state_dict (e.g. Stage5 entity checkpoint).",
    )
    parser.add_argument("--pretokenize_samples", type=int, default=0, help="Pre-tokenize the corpus sample pool before training (1/0).")
    parser.add_argument("--data_on_device", type=int, default=0, help="Keep the cached token tensors on device when possible.")
    # Performance: CUDA graph capture (reduces Python/kernel-launch overhead, similar to L4/L5).
    # -1 = auto (enable for CUDA+minimal+recurrent+teacher-forced+pretokenized on device)
    parser.add_argument("--cuda_graph", type=int, default=-1, help="Enable CUDA graph capture (1/0; -1=auto).")
    parser.add_argument("--cuda_graph_warmup", type=int, default=3, help="Warmup steps before graph capture.")
    # Stage7 maintenance: periodically run Protocol-5 gate and auto-harden on failure.
    parser.add_argument("--auto_gate", type=int, default=0, help="Enable periodic Stage7 gate checks (1/0).")
    parser.add_argument("--auto_gate_profile", default="robust", choices=["base", "robust"], help="Gate profile.")
    parser.add_argument("--auto_gate_every", type=int, default=500, help="Run gate every N steps (0=disable).")
    parser.add_argument("--auto_gate_at_end", type=int, default=1, help="Also run gate at end (1/0).")
    parser.add_argument(
        "--auto_gate_stop_on_fail",
        type=int,
        default=1,
        help="If auto-harden cannot restore the gate, stop training (1/0).",
    )
    parser.add_argument("--auto_gate_seed", type=int, default=0, help="Seed for fixed eval set.")
    parser.add_argument("--auto_gate_eval_text_file", default="", help="Optional eval texts file (newline-separated).")
    parser.add_argument("--auto_gate_eval_samples", type=int, default=2000, help="How many eval texts to sample.")
    parser.add_argument("--auto_gate_eval_cache", type=int, default=1, help="Cache sampled eval texts in save_dir (1/0).")
    parser.add_argument("--auto_gate_batch_size", type=int, default=64, help="Teacher-forced eval batch size.")
    parser.add_argument("--auto_gate_max_seq_len", type=int, default=-1, help="Teacher-forced eval seq len (-1=use train).")
    parser.add_argument("--auto_gate_num_prompts", type=int, default=200, help="Open-loop prompts.")
    parser.add_argument("--auto_gate_prompt_chars", type=int, default=4, help="Prompt length in chars.")
    parser.add_argument("--auto_gate_max_new_tokens", type=int, default=64, help="Open-loop continuation length.")
    parser.add_argument("--auto_gate_temperature", type=float, default=1.0, help="Open-loop temperature.")
    parser.add_argument("--auto_gate_top_k", type=int, default=0, help="Open-loop top-k (0=auto by profile).")
    parser.add_argument("--auto_gate_cycle_max_period", type=int, default=8)
    parser.add_argument("--auto_gate_cycle_min_repeats", type=int, default=3)
    parser.add_argument("--auto_gate_cycle_tail_window", type=int, default=32)
    parser.add_argument("--auto_gate_ngram_n", type=int, default=3)
    parser.add_argument("--auto_gate_margin_window", type=int, default=10)
    # Threshold overrides (negative => profile default)
    parser.add_argument("--auto_gate_ppl_ratio", type=float, default=-1.0)
    parser.add_argument("--auto_gate_cycle_rate", type=float, default=-1.0)
    parser.add_argument("--auto_gate_dominant_cycle", type=float, default=-1.0)
    parser.add_argument("--auto_gate_rep", type=float, default=-1.0)
    parser.add_argument("--auto_gate_unique_ratio", type=float, default=-1.0)
    # Auto-hardening burst settings (used when auto_gate fails).
    parser.add_argument("--auto_harden_steps", type=int, default=200, help="Hardening burst steps per round.")
    parser.add_argument("--auto_harden_max_rounds", type=int, default=6, help="Max hardening rounds before stop.")
    parser.add_argument(
        "--auto_harden_unlikelihood_weight",
        type=float,
        default=20.0,
        help="Hardening: unlikelihood weight.",
    )
    parser.add_argument("--auto_harden_unlikelihood_window", type=int, default=32, help="Hardening: unlikelihood window.")
    parser.add_argument("--auto_harden_denoise_mode", default="repeat", choices=["none", "replace", "repeat"])
    parser.add_argument("--auto_harden_denoise_prob", type=float, default=0.5)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    if str(device).startswith("cuda"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        if bool(args.tf32):
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    use_amp = bool(args.amp) and str(device).startswith("cuda")
    amp_dtype = torch.bfloat16 if str(args.amp_dtype) == "bf16" else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler) if str(device).startswith("cuda") else None

    # CUDA-graph policy (mirrors L4/L5 assumptions: fixed shapes + recurrent + pretokenized).
    # Allow bf16 autocast (no GradScaler) but keep fp16 GradScaler unsupported in graph mode.
    graph_safe = (
        str(device).startswith("cuda")
        and (not use_scaler)
        and args.port_arch == "minimal"
        and args.sequence_mode == "recurrent"
        and bool(args.pretokenize_samples)
        # Graph capture cannot include RNG-dependent control flow (scheduled sampling).
        and float(getattr(args, "scheduled_sampling_prob", 0.0) or 0.0) <= 0.0
        # Experience push / replay sampling use randomness + dynamic buffers.
        and int(getattr(args, "experience_push_n", 0) or 0) == 0
        and str(getattr(args, "offline_replay_mode", "none") or "none") == "none"
    )
    if int(args.cuda_graph) == -1:
        use_cuda_graph = bool(graph_safe)
    else:
        requested = bool(int(args.cuda_graph))
        if requested and (not graph_safe):
            raise RuntimeError(
                "CUDA graph is not supported with the current settings. "
                "Disable graph-friendly blockers (scheduled sampling / experience push / offline replay) "
                "or set --cuda_graph -1 to auto-disable."
            )
        use_cuda_graph = bool(requested)

    def _make_optimizer(params):
        opt_kwargs = {"lr": args.lr}
        if str(device).startswith("cuda") and bool(args.fused_adamw):
            opt_kwargs["fused"] = True
        # CUDA graphs: prefer capturable + foreach where supported (matches curriculum stages).
        if str(device).startswith("cuda") and bool(use_cuda_graph):
            opt_kwargs["capturable"] = True
            opt_kwargs["foreach"] = True
        try:
            return AdamW(params, **opt_kwargs)
        except Exception:
            opt_kwargs.pop("fused", None)
            opt_kwargs.pop("capturable", None)
            opt_kwargs.pop("foreach", None)
            return AdamW(params, **opt_kwargs)
    stop_requested = False
    interrupt_count = 0

    def _handle_stop_signal(signum, frame):
        nonlocal stop_requested, interrupt_count
        stop_requested = True
        interrupt_count += 1
        # If the user sends multiple interrupts, fall back to immediate KeyboardInterrupt.
        if interrupt_count >= 2:
            raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handle_stop_signal)
        signal.signal(signal.SIGTERM, _handle_stop_signal)
    except Exception:
        # Some environments (or threads) may not allow installing signal handlers.
        pass

    if args.port_arch == "minimal" and args.sequence_mode != "recurrent":
        raise ValueError("port_arch=minimal requires --sequence_mode recurrent")
    if args.port_arch == "transformer" and args.sequence_mode != "pooled":
        raise ValueError("port_arch=transformer requires --sequence_mode pooled")
    if args.port_arch != "minimal":
        args.port_trainable = 1

    # Source weights
    base_weights = {"wiki": 0.6, "clue": 0.4}
    if args.hf_dataset and args.max_samples_hf > 0 and args.hf_weight > 0:
        hf_w = float(args.hf_weight)
        hf_w = max(0.0, min(1.0, hf_w))
        scale = max(0.0, 1.0 - hf_w)
        base_weights = {"wiki": base_weights["wiki"] * scale, "clue": base_weights["clue"] * scale, "hf": hf_w}

    sampler_cfg = ActiveSamplingConfig(
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
        loss_weight=max(float(args.sampler_loss_weight), 0.0) if math.isfinite(float(args.sampler_loss_weight)) else 1.0,
        coverage_weight=max(float(args.sampler_coverage_weight), 0.0) if math.isfinite(float(args.sampler_coverage_weight)) else 0.5,
        progress_weight=max(float(args.sampler_progress_weight), 0.0) if math.isfinite(float(args.sampler_progress_weight)) else 0.0,
        temperature=max(float(args.sampler_temperature), 1e-6) if math.isfinite(float(args.sampler_temperature)) else 1.0,
        random_ratio=min(max(float(args.sampler_random_ratio), 0.0), 1.0) if math.isfinite(float(args.sampler_random_ratio)) else 0.1,
        ema_rate=min(max(float(args.sampler_ema_rate), 1e-6), 1.0) if math.isfinite(float(args.sampler_ema_rate)) else 0.1,
    )

    pool = CorpusPool(sampler_cfg)
    pool.load()

    resume_ckpt = args.resume_ckpt
    if not resume_ckpt and args.resume_dir:
        resume_ckpt = os.path.join(args.resume_dir, "last.pt")

    resume_tokenizer_path = args.resume_tokenizer_path or None

    start_step = 0
    total_tokens = 0
    if resume_ckpt:
        trainer, tokenizer, train_cfg, meta = load_trainer_from_checkpoint(
            resume_ckpt,
            device=device,
            tokenizer_path=resume_tokenizer_path,
            override_cfg={
                "batch_size": args.batch_size,
                "max_seq_len": args.max_seq_len,
                "dt": float(args.dt),
                "num_evolution_steps": int(args.num_evolution_steps),
                "beta_kl": float(args.beta_kl),
                "gamma_pred": float(args.gamma_pred),
                "num_offline_steps": args.num_offline_steps,
                "offline_dt": args.offline_dt,
                "offline_replay_mode": args.offline_replay_mode,
                "offline_weight": args.offline_weight,
                "offline_loss_mode": args.offline_loss_mode,
                "offline_margin": args.offline_margin,
                "offline_detach_init": bool(args.offline_detach_init),
                "reset_each_batch": bool(args.reset_each_batch),
                "port_trainable": bool(args.port_trainable),
                "port_init_std": float(args.port_init_std),
                "scheduled_sampling_prob": float(args.scheduled_sampling_prob),
                "scheduled_sampling_mode": str(args.scheduled_sampling_mode),
                "scheduled_sampling_top_k": int(args.scheduled_sampling_top_k),
                "scheduled_sampling_temperature": float(args.scheduled_sampling_temperature),
                "unlikelihood_weight": float(args.unlikelihood_weight),
                "unlikelihood_window": int(args.unlikelihood_window),
                "router_balance_weight": float(args.router_balance_weight),
                "router_entropy_weight": float(args.router_entropy_weight),
                "experience_push_n": int(args.experience_push_n),
                "port_coupling_top_k": int(args.port_coupling_top_k),
                "port_coupling_impl": str(args.port_coupling_impl),
                "transport_threshold": float(args.transport_threshold),
                "connection_rank": int(args.connection_rank),
                "q_clip_norm": float(args.q_clip_norm),
                "p_clip_norm": float(args.p_clip_norm),
                "s_clip_abs": float(args.s_clip_abs),
                "sanitize_nonfinite": bool(args.sanitize_nonfinite),
                "detach_every": int(args.detach_every),
            },
            strict=bool(args.resume_strict),
        )
        start_step = int(meta.get("step", 0) or 0)
        total_tokens = int(meta.get("total_tokens", 0) or 0)
        train_params = [p for p in trainer.parameters() if p.requires_grad]
        optimizer = _make_optimizer(train_params)
        if args.resume_optimizer and isinstance(meta.get("optimizer_state", None), dict):
            optimizer.load_state_dict(meta["optimizer_state"])
            # Move optimizer state tensors onto the target device.
            for state in optimizer.state.values():
                for k, v in list(state.items()):
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
        print(f"Resumed: ckpt={resume_ckpt} step={start_step} total_tokens={total_tokens} resume_optimizer={bool(args.resume_optimizer)}")
    else:
        tokenizer = build_tokenizer(pool, args.vocab_size, args.vocab_build_samples, args.tokenizer)
        train_cfg = TrainingConfig(
            dim_q=args.dim_q,
            dim_z=args.dim_z,
            dim_embed=args.dim_embed,
            vocab_size=len(tokenizer),
            num_charts=args.num_charts,
            port_arch=args.port_arch,
            sequence_mode=args.sequence_mode,
            tie_io_weights=bool(args.tie_io_weights),
            port_trainable=bool(args.port_trainable),
            port_init_std=float(args.port_init_std),
            scheduled_sampling_prob=float(args.scheduled_sampling_prob),
            scheduled_sampling_mode=str(args.scheduled_sampling_mode),
            scheduled_sampling_top_k=int(args.scheduled_sampling_top_k),
            scheduled_sampling_temperature=float(args.scheduled_sampling_temperature),
            unlikelihood_weight=float(args.unlikelihood_weight),
            unlikelihood_window=int(args.unlikelihood_window),
            router_balance_weight=float(args.router_balance_weight),
            router_entropy_weight=float(args.router_entropy_weight),
            experience_push_n=int(args.experience_push_n),
            port_coupling_top_k=int(args.port_coupling_top_k),
            port_coupling_impl=str(args.port_coupling_impl),
            transport_threshold=float(args.transport_threshold),
            connection_rank=int(args.connection_rank),
            q_clip_norm=float(args.q_clip_norm),
            p_clip_norm=float(args.p_clip_norm),
            s_clip_abs=float(args.s_clip_abs),
            sanitize_nonfinite=bool(args.sanitize_nonfinite),
            detach_every=int(args.detach_every),
            dt=float(args.dt),
            num_evolution_steps=int(args.num_evolution_steps),
            beta_kl=float(args.beta_kl),
            gamma_pred=float(args.gamma_pred),
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            device=device,
            num_offline_steps=args.num_offline_steps,
            offline_dt=args.offline_dt,
            offline_replay_mode=args.offline_replay_mode,
            offline_weight=args.offline_weight,
            offline_loss_mode=args.offline_loss_mode,
            offline_margin=args.offline_margin,
            offline_detach_init=bool(args.offline_detach_init),
            reset_each_batch=bool(args.reset_each_batch),
        )
        trainer = SoulLanguageTrainer(train_cfg, tokenizer).to(device)
        _maybe_init_entity_from_state_dict(trainer.entity, args.init_entity_ckpt)
        train_params = [p for p in trainer.parameters() if p.requires_grad]
        optimizer = _make_optimizer(train_params)

    sampler = ActiveSampler(pool, sampler_cfg)
    if bool(args.pretokenize_samples):
        sample_texts = [sample.text for sample in pool.samples]
        token_cache, mask_cache = _pretokenize_samples(sample_texts, tokenizer, args.max_seq_len)
        if str(device).startswith("cuda") and (not bool(args.data_on_device)):
            # Pinned host cache enables async H2D copies (higher utilization, fewer stalls).
            try:
                token_cache = token_cache.pin_memory()
                mask_cache = mask_cache.pin_memory()
            except Exception:
                pass
        if bool(args.data_on_device):
            token_cache = token_cache.to(device=device, non_blocking=True)
            mask_cache = mask_cache.to(device=device, non_blocking=True)
        sampler.set_token_cache(token_cache, mask_cache)

    params = count_parameters(trainer)
    entity_params = count_unique_parameters([trainer.entity])
    port_modules = []
    if getattr(train_cfg, "port_arch", "transformer") == "transformer" and getattr(trainer, "language_port", None) is not None:
        port_modules = [trainer.language_port]
    elif getattr(train_cfg, "port_arch", "") == "minimal":
        if getattr(trainer, "token_embedding", None) is not None:
            port_modules.append(trainer.token_embedding)
        if getattr(trainer, "output_proj", None) is not None:
            port_modules.append(trainer.output_proj)
    port_params = count_unique_parameters(port_modules) if port_modules else {"total": 0, "trainable": 0}

    print(
        f"Tokenizer: {type(tokenizer).__name__} vocab={len(tokenizer)} "
        f"port_arch={train_cfg.port_arch} sequence_mode={train_cfg.sequence_mode} "
        f"port_trainable={int(getattr(train_cfg, 'port_trainable', True))}"
    )
    print(f"Active sampling pool size: {len(pool.samples)}")
    print(f"Buckets: {len(pool.bucket_to_indices)}")
    print(f"Device: {device}")
    if str(device).startswith("cuda"):
        print(
            f"Precision: tf32={bool(args.tf32)} amp={bool(use_amp)} amp_dtype={args.amp_dtype} fused_adamw={bool(args.fused_adamw)}"
        )
        if bool(use_cuda_graph):
            print(f"CUDA Graph: enabled (warmup={int(args.cuda_graph_warmup)})")
        else:
            print("CUDA Graph: disabled")
    print(f"Params: total={params['total']:,} trainable={params['trainable']:,}")
    print(f"Params breakdown: entity={entity_params['trainable']:,}/{entity_params['total']:,} port={port_params['trainable']:,}/{port_params['total']:,}")
    print(f"Integrator steps / train step: online={train_cfg.num_evolution_steps} offline={train_cfg.num_offline_steps}")
    print(f"Offline loss: mode={train_cfg.offline_loss_mode} weight={train_cfg.offline_weight} margin={train_cfg.offline_margin}")
    if getattr(train_cfg, "sequence_mode", "") == "recurrent":
        print(
            f"Closed-loop: ss_prob={getattr(train_cfg,'scheduled_sampling_prob',0.0)} "
            f"ss_mode={getattr(train_cfg,'scheduled_sampling_mode','sample')} "
            f"ss_top_k={getattr(train_cfg,'scheduled_sampling_top_k',0)} "
            f"ss_temp={getattr(train_cfg,'scheduled_sampling_temperature',1.0)} "
            f"ul_w={getattr(train_cfg,'unlikelihood_weight',0.0)} "
            f"ul_win={int(getattr(train_cfg,'unlikelihood_window',1) or 1)}"
        )
        print(
            f"Atlas: balance_w={getattr(train_cfg,'router_balance_weight',0.0)} "
            f"entropy_w={getattr(train_cfg,'router_entropy_weight',0.0)} "
            f"experience_push_n={getattr(train_cfg,'experience_push_n',0)} "
            f"port_top_k={getattr(train_cfg,'port_coupling_top_k',0)} "
            f"port_impl={getattr(train_cfg,'port_coupling_impl','grouped')}"
        )
    if args.save_dir:
        print(f"Checkpoint dir: {args.save_dir} (keep_every={args.save_every})")

    auto_gate_enabled = bool(int(getattr(args, "auto_gate", 0) or 0)) and int(getattr(args, "auto_gate_every", 0) or 0) > 0
    auto_gate_profile = str(getattr(args, "auto_gate_profile", "robust") or "robust")
    auto_gate_texts: List[str] = []
    auto_gate_prompts: List[str] = []
    auto_gate_thresholds: Dict[str, float] = {}

    quality_denoise_mode = str(getattr(args, "denoise_mode", "none") or "none")
    quality_denoise_prob = float(getattr(args, "denoise_prob", 0.0) or 0.0)
    harden_denoise_mode = str(getattr(args, "auto_harden_denoise_mode", "repeat") or "repeat")
    harden_denoise_prob = float(getattr(args, "auto_harden_denoise_prob", 0.0) or 0.0)
    denoise_mode = quality_denoise_mode
    denoise_prob = quality_denoise_prob

    quality_ul_w = float(getattr(args, "unlikelihood_weight", 0.0) or 0.0)
    quality_ul_win = int(getattr(args, "unlikelihood_window", 1) or 1)
    harden_ul_w = float(getattr(args, "auto_harden_unlikelihood_weight", 0.0) or 0.0)
    harden_ul_win = int(getattr(args, "auto_harden_unlikelihood_window", 1) or 1)

    mode = "quality"
    harden_remaining = 0
    harden_round = 0

    if auto_gate_enabled:
        from training.eval_protocol5_closed_loop import collect_eval_texts, collect_eval_texts_from_pool, extract_prompts

        seed = int(getattr(args, "auto_gate_seed", 0) or 0)
        cache_dir = args.save_dir or (os.path.dirname(os.path.abspath(resume_ckpt)) if resume_ckpt else os.getcwd())
        cache_dir = str(cache_dir or "").strip() or os.getcwd()
        if bool(getattr(args, "auto_gate_eval_cache", 1)) and cache_dir:
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception:
                pass

        eval_text_file = str(getattr(args, "auto_gate_eval_text_file", "") or "").strip()
        eval_samples = int(getattr(args, "auto_gate_eval_samples", 0) or 0)
        if eval_samples <= 0:
            eval_samples = 2000
        if eval_text_file:
            texts: List[str] = []
            cache_path = ""
            if bool(getattr(args, "auto_gate_eval_cache", 1)) and eval_samples > 0 and cache_dir:
                cache_path = _eval_cache_path(
                    eval_text_file=eval_text_file,
                    eval_samples=eval_samples,
                    seed=seed,
                    cache_dir=cache_dir,
                )
                if os.path.exists(cache_path):
                    texts = _read_nonempty_lines(cache_path)
            if not texts:
                texts = collect_eval_texts(eval_text_file=eval_text_file, eval_samples=eval_samples, seed=seed)
                if bool(getattr(args, "auto_gate_eval_cache", 1)) and cache_path:
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            for t in texts:
                                t = str(t).strip()
                                if t:
                                    f.write(t + "\n")
                    except Exception:
                        pass
            auto_gate_texts = texts
        else:
            auto_gate_texts = collect_eval_texts_from_pool(pool=pool, eval_samples=eval_samples, seed=seed)

        if not auto_gate_texts:
            raise RuntimeError("auto_gate: no eval texts found (check --auto_gate_eval_text_file / corpus paths).")

        auto_gate_prompts = extract_prompts(
            auto_gate_texts,
            num_prompts=int(getattr(args, "auto_gate_num_prompts", 200) or 200),
            prompt_chars=int(getattr(args, "auto_gate_prompt_chars", 4) or 4),
        )
        if not auto_gate_prompts:
            raise RuntimeError("auto_gate: no prompts extracted (try smaller --auto_gate_prompt_chars).")

        defaults = _stage7_gate_profile_defaults(auto_gate_profile)
        auto_gate_thresholds = {
            "gate_ppl_ratio": defaults["gate_ppl_ratio"]
            if float(getattr(args, "auto_gate_ppl_ratio", -1.0)) < 0
            else float(getattr(args, "auto_gate_ppl_ratio")),
            "gate_cycle_rate": defaults["gate_cycle_rate"]
            if float(getattr(args, "auto_gate_cycle_rate", -1.0)) < 0
            else float(getattr(args, "auto_gate_cycle_rate")),
            "gate_dominant_cycle": defaults["gate_dominant_cycle"]
            if float(getattr(args, "auto_gate_dominant_cycle", -1.0)) < 0
            else float(getattr(args, "auto_gate_dominant_cycle")),
            "gate_rep": defaults["gate_rep"]
            if float(getattr(args, "auto_gate_rep", -1.0)) < 0
            else float(getattr(args, "auto_gate_rep")),
            "gate_unique_ratio": defaults["gate_unique_ratio"]
            if float(getattr(args, "auto_gate_unique_ratio", -1.0)) < 0
            else float(getattr(args, "auto_gate_unique_ratio")),
        }
        print(
            "[AutoGate] enabled:"
            f" profile={auto_gate_profile}"
            f" every={int(args.auto_gate_every)}"
            f" eval_texts={len(auto_gate_texts)}"
            f" prompts={len(auto_gate_prompts)}"
            f" harden_steps={int(getattr(args,'auto_harden_steps',0) or 0)}"
            f" harden_ul_w={harden_ul_w}"
            f" harden_ul_win={harden_ul_win}"
            f" harden_denoise={harden_denoise_mode}:{harden_denoise_prob}"
        )

        def _auto_gate_eval(step_now: int, *, tag: str) -> Dict[str, object]:
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            eval_max_seq_len = int(getattr(args, "auto_gate_max_seq_len", -1) or -1)
            if eval_max_seq_len <= 0:
                eval_max_seq_len = int(args.max_seq_len)
            out = _run_stage7_gate_eval(
                trainer=trainer,
                tokenizer=tokenizer,
                texts=auto_gate_texts,
                prompts=auto_gate_prompts,
                device=device,
                profile=auto_gate_profile,
                batch_size=int(getattr(args, "auto_gate_batch_size", 64) or 64),
                max_seq_len=int(eval_max_seq_len),
                max_new_tokens=int(getattr(args, "auto_gate_max_new_tokens", 64) or 64),
                temperature=float(getattr(args, "auto_gate_temperature", 1.0) or 1.0),
                top_k=int(getattr(args, "auto_gate_top_k", 0) or 0),
                cycle_max_period=int(getattr(args, "auto_gate_cycle_max_period", 8) or 8),
                cycle_min_repeats=int(getattr(args, "auto_gate_cycle_min_repeats", 3) or 3),
                cycle_tail_window=int(getattr(args, "auto_gate_cycle_tail_window", 32) or 32),
                ngram_n=int(getattr(args, "auto_gate_ngram_n", 3) or 3),
                margin_window=int(getattr(args, "auto_gate_margin_window", 10) or 10),
                gate_ppl_ratio=float(auto_gate_thresholds["gate_ppl_ratio"]),
                gate_cycle_rate=float(auto_gate_thresholds["gate_cycle_rate"]),
                gate_dominant_cycle=float(auto_gate_thresholds["gate_dominant_cycle"]),
                gate_rep=float(auto_gate_thresholds["gate_rep"]),
                gate_unique_ratio=float(auto_gate_thresholds["gate_unique_ratio"]),
            )
            ol = out.get("open_loop", {}) if isinstance(out, dict) else {}
            passed = bool(out.get("passed", False)) if isinstance(out, dict) else False
            print(
                f"[AutoGate:{tag}] step={int(step_now)} passed={passed}"
                f" ppl_ratio={float(out.get('ppl_ratio',0.0)):.4f}"
                f" rep={float(ol.get('rep',0.0)):.4f}"
                f" cycle_rate={float(ol.get('cycle_rate',0.0)):.4f}"
                f" dominant_cycle={float(ol.get('dominant_cycle',0.0)):.4f}"
                f" unique_ratio={float(ol.get('unique_ratio',0.0)):.4f}"
            )
            if not passed and isinstance(out, dict):
                print(f"[AutoGate:{tag}] checks={out.get('checks',{})}")
            trainer.train()
            return out

    window_tokens = 0
    window_time = 0.0
    start_time = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    stop_reason = "steps_exhausted"
    last_step = start_step
    if args.max_tokens and total_tokens >= args.max_tokens:
        stop_reason = "max_tokens_already_reached"
        step = start_step
        last_step = start_step
    elif args.steps <= start_step:
        stop_reason = "target_steps_already_reached"
        step = start_step
        last_step = start_step
    else:
        step = start_step
        try:
            if use_cuda_graph:
                # CUDA graphs can work with bf16 autocast (no GradScaler). Keep fp16 unsupported here
                # to avoid GradScaler/overflow complexities inside capture.
                if use_amp and amp_dtype != torch.bfloat16:
                    raise RuntimeError("CUDA graph mode supports --amp=1 only with --amp_dtype bf16.")
                if use_scaler:
                    raise RuntimeError("CUDA graph mode does not support fp16 GradScaler.")

                B = int(args.batch_size)
                L = int(args.max_seq_len)
                token_buf = torch.empty((B, L), device=device, dtype=torch.long)
                target_buf = torch.empty((B, L), device=device, dtype=torch.long)
                mask_buf = torch.empty((B, L), device=device, dtype=torch.long)

                loss_buf = torch.zeros((), device=device, dtype=torch.float32)
                F_on_buf = torch.zeros((), device=device, dtype=torch.float32)
                F_off_buf = torch.zeros((), device=device, dtype=torch.float32)
                dF_off_buf = torch.zeros((), device=device, dtype=torch.float32)
                off_loss_buf = torch.zeros((), device=device, dtype=torch.float32)
                rb_buf = torch.zeros((), device=device, dtype=torch.float32)
                re_buf = torch.zeros((), device=device, dtype=torch.float32)
                ar_buf = torch.zeros((), device=device, dtype=torch.float32)
                ul_buf = torch.zeros((), device=device, dtype=torch.float32)
                qn_buf = torch.zeros((), device=device, dtype=torch.float32)
                E_buf = torch.zeros((), device=device, dtype=torch.float32)
                E_ps_buf = torch.empty((B,), device=device, dtype=torch.float32)
                ppl_buf = torch.zeros((), device=device, dtype=torch.float32)
                zero_scalar = torch.zeros((), device=device, dtype=torch.float32)

                token_cache = getattr(sampler, "token_cache", None)
                mask_cache = getattr(sampler, "mask_cache", None)
                cache_ready = token_cache is not None and mask_cache is not None
                cache_device = token_cache.device if cache_ready else None
                token_host = None
                mask_host = None
                if cache_ready and cache_device is not None and cache_device.type == "cpu":
                    # Keep a pinned host staging buffer so H2D copies stay async even
                    # though `index_select` allocates regular CPU memory by default.
                    try:
                        token_host = torch.empty((B, L), dtype=token_cache.dtype, device="cpu", pin_memory=True)
                        mask_host = torch.empty((B, L), dtype=mask_cache.dtype, device="cpu", pin_memory=True)
                    except Exception:
                        token_host = torch.empty((B, L), dtype=token_cache.dtype, device="cpu")
                        mask_host = torch.empty((B, L), dtype=mask_cache.dtype, device="cpu")

                def refill() -> List[str]:
                    indices, bucket_keys = sampler.sample_indices(B)
                    if not indices:
                        return []

                    if cache_ready and token_cache is not None and mask_cache is not None and cache_device is not None:
                        if cache_device.type == "cuda":
                            idx = torch.as_tensor(indices, dtype=torch.long, device=cache_device)
                            torch.index_select(token_cache, 0, idx, out=token_buf)
                            torch.index_select(mask_cache, 0, idx, out=mask_buf)
                        else:
                            idx = torch.as_tensor(indices, dtype=torch.long, device="cpu")
                            if token_host is not None and mask_host is not None:
                                torch.index_select(token_cache, 0, idx, out=token_host)
                                torch.index_select(mask_cache, 0, idx, out=mask_host)
                                token_buf.copy_(token_host, non_blocking=True)
                                mask_buf.copy_(mask_host, non_blocking=True)
                            else:
                                token_batch = token_cache.index_select(0, idx)
                                mask_batch = mask_cache.index_select(0, idx)
                                token_buf.copy_(token_batch, non_blocking=True)
                                mask_buf.copy_(mask_batch, non_blocking=True)

                        target_buf.copy_(token_buf)
                        if denoise_prob > 0.0 and denoise_mode != "none":
                            token_buf.copy_(
                                _apply_denoise(
                                    token_buf,
                                    token_clean=target_buf,
                                    attention_mask=mask_buf,
                                    mode=denoise_mode,
                                    prob=denoise_prob,
                                    vocab_size=len(tokenizer),
                                    pad_id=int(tokenizer.pad_id),
                                    bos_id=int(tokenizer.bos_id),
                                    eos_id=int(tokenizer.eos_id),
                                    unk_id=int(getattr(tokenizer, "unk_id", -1)) if hasattr(tokenizer, "unk_id") else None,
                                )
                            )
                        return bucket_keys

                    # Fallback: this is slower; CUDA-graph mode expects pretokenized cache.
                    texts = [sampler.pool.samples[i].text for i in indices]
                    batch2 = build_batch(texts, tokenizer, args.max_seq_len)
                    token_buf.copy_(batch2["input_ids"].to(device=device, non_blocking=True))
                    mask_buf.copy_(batch2["attention_mask"].to(device=device, non_blocking=True))
                    target_buf.copy_(token_buf)
                    if denoise_prob > 0.0 and denoise_mode != "none":
                        token_buf.copy_(
                            _apply_denoise(
                                token_buf,
                                token_clean=target_buf,
                                attention_mask=mask_buf,
                                mode=denoise_mode,
                                prob=denoise_prob,
                                vocab_size=len(tokenizer),
                                pad_id=int(tokenizer.pad_id),
                                bos_id=int(tokenizer.bos_id),
                                eos_id=int(tokenizer.eos_id),
                                unk_id=int(getattr(tokenizer, "unk_id", -1)) if hasattr(tokenizer, "unk_id") else None,
                            )
                        )
                    return bucket_keys

                def fwd_bwd_step() -> None:
                    optimizer.zero_grad(set_to_none=False)
                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
                        if str(device).startswith("cuda")
                        else contextlib.nullcontext()
                    )
                    with autocast_ctx:
                        out = trainer.train_step({"input_ids": token_buf, "target_ids": target_buf, "attention_mask": mask_buf})
                        loss = out["loss"]
                    loss_buf.copy_(loss.detach().float())
                    F_on_buf.copy_(out["free_energy_online"].detach().float())
                    F_off_buf.copy_(out["free_energy_offline"].detach().float())
                    dF_off_buf.copy_(out.get("delta_free_energy_offline", zero_scalar).detach().float())
                    off_loss_buf.copy_(out.get("offline_loss", zero_scalar).detach().float())
                    rb_buf.copy_(out.get("router_balance", zero_scalar).detach().float())
                    re_buf.copy_(out.get("router_entropy", zero_scalar).detach().float())
                    ar_buf.copy_(out.get("atlas_reg", zero_scalar).detach().float())
                    ul_buf.copy_(out.get("unlikelihood_loss", zero_scalar).detach().float())
                    qn_buf.copy_(out.get("q_norm", zero_scalar).detach().float())
                    E_buf.copy_(out["prediction_error"].detach().float())
                    E_ps_buf.copy_(out["prediction_error_per_sample"].detach().float())
                    ppl_buf.copy_(out["perplexity"].detach().float())
                    loss.backward()
                    _clip_grad_norm_no_sync_(train_params, 1.0)

                warmup = max(1, int(args.cuda_graph_warmup))

                # Capture a quality graph (baseline config) and optionally a harden graph.
                train_cfg.unlikelihood_weight = float(quality_ul_w)
                train_cfg.unlikelihood_window = int(quality_ul_win)
                for _ in range(warmup):
                    bucket_keys = refill()
                    fwd_bwd_step()
                    optimizer.step()
                    sampler.update_counts(bucket_keys)

                torch.cuda.synchronize()
                graph_pool = torch.cuda.graphs.graph_pool_handle()
                graph_quality = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_quality, pool=graph_pool):
                    fwd_bwd_step()

                graph_harden = graph_quality
                if auto_gate_enabled and (float(harden_ul_w) != float(quality_ul_w) or int(harden_ul_win) != int(quality_ul_win)):
                    train_cfg.unlikelihood_weight = float(harden_ul_w)
                    train_cfg.unlikelihood_window = int(harden_ul_win)
                    for _ in range(warmup):
                        bucket_keys = refill()
                        fwd_bwd_step()
                        optimizer.step()
                        sampler.update_counts(bucket_keys)
                    torch.cuda.synchronize()
                    graph_harden = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph_harden, pool=graph_pool):
                        fwd_bwd_step()
                    train_cfg.unlikelihood_weight = float(quality_ul_w)
                    train_cfg.unlikelihood_window = int(quality_ul_win)
                    print(f"[AutoGate] Captured harden CUDA graph (ul_w={harden_ul_w} ul_win={harden_ul_win}).")

                window_t0 = time.perf_counter()
                for step in range(start_step + 1, args.steps + 1):
                    if stop_requested:
                        stop_reason = "interrupted"
                        break
                    mode_used = mode
                    if mode_used == "harden":
                        denoise_mode = harden_denoise_mode
                        denoise_prob = harden_denoise_prob
                        active_graph = graph_harden
                    else:
                        denoise_mode = quality_denoise_mode
                        denoise_prob = quality_denoise_prob
                        active_graph = graph_quality

                    bucket_keys = refill()
                    active_graph.replay()
                    optimizer.step()

                    # No-sync approximate token count in CUDA-graph mode.
                    step_tokens = int(token_buf.numel())
                    total_tokens += step_tokens
                    window_tokens += step_tokens
                    sampler.update_counts(bucket_keys)
                    last_step = step

                    update_loss_now = bool(args.sampler_loss_update_every) and step % int(args.sampler_loss_update_every) == 0
                    log_now = step % args.log_every == 0
                    if update_loss_now or log_now:
                        torch.cuda.synchronize()

                    if update_loss_now:
                        if str(args.sampler_loss_update_mode) == "per_sample":
                            loss_cpu = E_ps_buf.detach().cpu()
                            sampler.update_losses(bucket_keys, loss_cpu)
                        else:
                            mean_loss = float(E_buf.item())
                            sampler.update_losses(bucket_keys, mean_loss)

                    if log_now:
                        now = time.perf_counter()
                        tok_s = (window_tokens / max(now - window_t0, 1e-9))
                        elapsed = now - start_time
                        msg = (
                            f"Step {step} ({mode_used}): F={float(loss_buf.item()):.4f} "
                            f"F_on={float(F_on_buf.item()):.4f} F_off={float(F_off_buf.item()):.4f} "
                            f"dF_off={float(dF_off_buf.item()):.4f} L_off={float(off_loss_buf.item()):.4f} "
                            f"R_bal={float(rb_buf.item()):.4f} R_ent={float(re_buf.item()):.4f} R_atlas={float(ar_buf.item()):.4f} "
                            f"UL={float(ul_buf.item()):.4f} q={float(qn_buf.item()):.2f} "
                            f"E_pred={float(E_buf.item()):.4f} PPL={float(ppl_buf.item()):.2f} "
                            f"tok={total_tokens} tok/s={tok_s:.1f} t={elapsed:.1f}s"
                        )
                        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        msg += f" peak_mem={peak_gb:.2f}GB"
                        torch.cuda.reset_peak_memory_stats()
                        print(msg)
                        window_tokens = 0
                        window_t0 = now

                    if args.sample_every and args.sample_every > 0 and step % args.sample_every == 0:
                        try:
                            with torch.no_grad():
                                sample = trainer.generate("数学是", max_len=30)
                            print(f"[Sample] {sample}")
                        except Exception as exc:
                            print(f"[Sample] generation failed: {exc}")

                    if args.save_dir and args.save_every and step % args.save_every == 0:
                        save_checkpoint(
                            args.save_dir,
                            trainer=trainer,
                            tokenizer=tokenizer,
                            train_cfg=train_cfg,
                            step=step,
                            total_tokens=total_tokens,
                            optimizer=optimizer,
                            extra={"sampler_cfg": asdict(sampler_cfg)},
                            keep_every=args.save_every,
                        )

                    if auto_gate_enabled:
                        max_rounds = int(getattr(args, "auto_harden_max_rounds", 1) or 1)
                        harden_steps = int(getattr(args, "auto_harden_steps", 0) or 0)
                        stop_on_fail = bool(int(getattr(args, "auto_gate_stop_on_fail", 1)))
                        try:
                            if mode_used == "quality":
                                run_now = (step % int(args.auto_gate_every) == 0) or (
                                    bool(int(getattr(args, "auto_gate_at_end", 1) or 1)) and step == int(args.steps)
                                )
                                if run_now:
                                    tag = "periodic" if step % int(args.auto_gate_every) == 0 else "end"
                                    gate_out = _auto_gate_eval(step, tag=tag)
                                    if not bool(gate_out.get("passed", False)):
                                        if harden_steps <= 0:
                                            stop_reason = "auto_gate_failed"
                                            break
                                        mode = "harden"
                                        harden_round = 1
                                        harden_remaining = harden_steps
                                        print(f"[AutoGate] FAIL -> enter harden round {harden_round} ({harden_remaining} steps)")
                            else:
                                harden_remaining -= 1
                                if harden_remaining <= 0:
                                    gate_out = _auto_gate_eval(step, tag=f"post_harden_r{harden_round}")
                                    if bool(gate_out.get("passed", False)):
                                        mode = "quality"
                                        harden_round = 0
                                        harden_remaining = 0
                                        print("[AutoGate] PASS -> back to quality")
                                    else:
                                        harden_round += 1
                                        if max_rounds > 0 and harden_round > max_rounds:
                                            if stop_on_fail:
                                                stop_reason = "auto_gate_failed"
                                                break
                                            harden_round = max_rounds
                                            harden_remaining = max(harden_steps, 1)
                                            mode = "harden"
                                            print("[AutoGate] exceeded max rounds; continue harden (stop_on_fail=0)")
                                        else:
                                            harden_remaining = max(harden_steps, 1)
                                            mode = "harden"
                                            print(
                                                f"[AutoGate] still failing -> harden round {harden_round} ({harden_remaining} steps)"
                                            )
                        except Exception as exc:
                            stop_reason = "auto_gate_error"
                            print(f"[AutoGate] error: {exc}")
                            traceback.print_exc()
                            break

                    if args.max_tokens and total_tokens >= args.max_tokens:
                        stop_reason = "max_tokens_reached"
                        break

            else:
                for step in range(start_step + 1, args.steps + 1):
                    if stop_requested:
                        stop_reason = "interrupted"
                        break

                    mode_used = mode
                    if mode_used == "harden":
                        denoise_mode = harden_denoise_mode
                        denoise_prob = harden_denoise_prob
                        train_cfg.unlikelihood_weight = float(harden_ul_w)
                        train_cfg.unlikelihood_window = int(harden_ul_win)
                    else:
                        denoise_mode = quality_denoise_mode
                        denoise_prob = quality_denoise_prob
                        train_cfg.unlikelihood_weight = float(quality_ul_w)
                        train_cfg.unlikelihood_window = int(quality_ul_win)

                    t0 = time.perf_counter()

                    want_texts = not (
                        getattr(sampler, "token_cache", None) is not None and getattr(sampler, "mask_cache", None) is not None
                    )
                    texts, bucket_keys, token_ids_batch, mask_batch = sampler.sample_batch(args.batch_size, with_texts=want_texts)
                    if (not bucket_keys) and (not texts):
                        stop_reason = "empty_pool"
                        break

                    if token_ids_batch is not None and mask_batch is not None:
                        batch = {"input_ids": token_ids_batch, "attention_mask": mask_batch}
                    else:
                        batch = build_batch(texts, tokenizer, args.max_seq_len)

                    # Avoid per-step GPU sync: when cached tensors live on-device, `.item()` forces a synchronize.
                    mask = batch["attention_mask"]
                    if mask.device.type == "cpu":
                        step_tokens = int(mask.sum().item())
                    else:
                        # Approximate: counts pads too, but keeps tok/s stable and avoids sync.
                        step_tokens = int(mask.numel())

                    target_device = torch.device(device)
                    batch = {k: (v if v.device == target_device else v.to(target_device, non_blocking=True)) for k, v in batch.items()}
                    if denoise_prob > 0.0 and denoise_mode != "none":
                        clean = batch["input_ids"]
                        token_in = clean.clone()
                        token_in = _apply_denoise(
                            token_in,
                            token_clean=clean,
                            attention_mask=batch["attention_mask"],
                            mode=denoise_mode,
                            prob=denoise_prob,
                            vocab_size=len(tokenizer),
                            pad_id=int(tokenizer.pad_id),
                            bos_id=int(tokenizer.bos_id),
                            eos_id=int(tokenizer.eos_id),
                            unk_id=int(getattr(tokenizer, "unk_id", -1)) if hasattr(tokenizer, "unk_id") else None,
                        )
                        batch["target_ids"] = clean
                        batch["input_ids"] = token_in

                    try:
                        autocast_ctx = (
                            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
                            if str(device).startswith("cuda")
                            else contextlib.nullcontext()
                        )
                        with autocast_ctx:
                            result = trainer.train_step(batch)
                            loss = result["loss"]
                        if not torch.isfinite(loss).all():
                            stop_reason = "nonfinite_loss"
                            print(f"[Error] Non-finite loss at step {step}; stopping to preserve checkpoint.")
                            break

                        optimizer.zero_grad(set_to_none=True)
                        if use_scaler and scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            _clip_grad_norm_no_sync_(train_params, 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            _clip_grad_norm_no_sync_(train_params, 1.0)
                            optimizer.step()
                    except torch.cuda.OutOfMemoryError:
                        stop_reason = "oom"
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()
                        break
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            stop_reason = "oom"
                            if device.startswith("cuda"):
                                torch.cuda.empty_cache()
                            break
                        stop_reason = "error"
                        print(f"[Error] RuntimeError during training step {step}: {exc}")
                        traceback.print_exc()
                        break
                    except Exception as exc:
                        stop_reason = "error"
                        print(f"[Error] Exception during training step {step}: {exc}")
                        traceback.print_exc()
                        break

                    total_tokens += step_tokens
                    window_tokens += step_tokens

                    # Active sampler bookkeeping:
                    # - Always update counts (pure CPU, no sync)
                    # - Update loss EMA less frequently to avoid GPU→CPU sync/copies per step
                    sampler.update_counts(bucket_keys)
                    if args.sampler_loss_update_every and step % args.sampler_loss_update_every == 0:
                        if args.sampler_loss_update_mode == "batch_mean":
                            mean_loss = float(result["prediction_error"].detach().float().cpu().item())
                            sampler.update_losses(bucket_keys, mean_loss)
                        else:
                            loss_cpu = result["prediction_error_per_sample"].detach().float().cpu()
                            sampler.update_losses(bucket_keys, loss_cpu)
                    last_step = step

                    dt = time.perf_counter() - t0
                    window_time += dt

                    if step % args.log_every == 0:
                        avg_loss = result["free_energy"].item()
                        pred_err = result["prediction_error"].item()
                        F_online = result["free_energy_online"].item()
                        F_offline = result["free_energy_offline"].item()
                        delta_off = result.get("delta_free_energy_offline", torch.tensor(0.0)).item()
                        off_loss = result.get("offline_loss", torch.tensor(0.0)).item()
                        rb = result.get("router_balance", torch.tensor(0.0)).item()
                        re = result.get("router_entropy", torch.tensor(0.0)).item()
                        ar = result.get("atlas_reg", torch.tensor(0.0)).item()
                        ul = result.get("unlikelihood_loss", torch.tensor(0.0)).item()
                        qn = result.get("q_norm", torch.tensor(0.0)).item()
                        tok_s = (window_tokens / max(window_time, 1e-9))
                        elapsed = time.perf_counter() - start_time
                        msg = (
                            f"Step {step} ({mode_used}): F={avg_loss:.4f} "
                            f"F_on={F_online:.4f} F_off={F_offline:.4f} "
                            f"dF_off={delta_off:.4f} L_off={off_loss:.4f} "
                            f"R_bal={rb:.4f} R_ent={re:.4f} R_atlas={ar:.4f} "
                            f"UL={ul:.4f} q={qn:.2f} "
                            f"E_pred={pred_err:.4f} PPL={result['perplexity']:.2f} "
                            f"tok={total_tokens} tok/s={tok_s:.1f} t={elapsed:.1f}s"
                        )
                        if device.startswith("cuda"):
                            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                            msg += f" peak_mem={peak_gb:.2f}GB"
                            torch.cuda.reset_peak_memory_stats()
                        print(msg)
                        window_tokens = 0
                        window_time = 0.0

                    if stop_requested:
                        stop_reason = "interrupted"
                        break

                    if args.sample_every and args.sample_every > 0 and step % args.sample_every == 0:
                        try:
                            with torch.no_grad():
                                sample = trainer.generate("数学是", max_len=30)
                            print(f"[Sample] {sample}")
                        except Exception as exc:
                            print(f"[Sample] generation failed: {exc}")

                    if args.save_dir and args.save_every and step % args.save_every == 0:
                        save_checkpoint(
                            args.save_dir,
                            trainer=trainer,
                            tokenizer=tokenizer,
                            train_cfg=train_cfg,
                            step=step,
                            total_tokens=total_tokens,
                            optimizer=optimizer,
                            extra={"sampler_cfg": asdict(sampler_cfg)},
                            keep_every=args.save_every,
                        )

                    if auto_gate_enabled:
                        max_rounds = int(getattr(args, "auto_harden_max_rounds", 1) or 1)
                        harden_steps = int(getattr(args, "auto_harden_steps", 0) or 0)
                        stop_on_fail = bool(int(getattr(args, "auto_gate_stop_on_fail", 1)))
                        try:
                            if mode_used == "quality":
                                run_now = (step % int(args.auto_gate_every) == 0) or (
                                    bool(int(getattr(args, "auto_gate_at_end", 1) or 1)) and step == int(args.steps)
                                )
                                if run_now:
                                    tag = "periodic" if step % int(args.auto_gate_every) == 0 else "end"
                                    gate_out = _auto_gate_eval(step, tag=tag)
                                    if not bool(gate_out.get("passed", False)):
                                        if harden_steps <= 0:
                                            stop_reason = "auto_gate_failed"
                                            break
                                        mode = "harden"
                                        harden_round = 1
                                        harden_remaining = harden_steps
                                        print(f"[AutoGate] FAIL -> enter harden round {harden_round} ({harden_remaining} steps)")
                            else:
                                harden_remaining -= 1
                                if harden_remaining <= 0:
                                    gate_out = _auto_gate_eval(step, tag=f"post_harden_r{harden_round}")
                                    if bool(gate_out.get("passed", False)):
                                        mode = "quality"
                                        harden_round = 0
                                        harden_remaining = 0
                                        print("[AutoGate] PASS -> back to quality")
                                    else:
                                        harden_round += 1
                                        if max_rounds > 0 and harden_round > max_rounds:
                                            if stop_on_fail:
                                                stop_reason = "auto_gate_failed"
                                                break
                                            harden_round = max_rounds
                                            harden_remaining = max(harden_steps, 1)
                                            mode = "harden"
                                            print("[AutoGate] exceeded max rounds; continue harden (stop_on_fail=0)")
                                        else:
                                            harden_remaining = max(harden_steps, 1)
                                            mode = "harden"
                                            print(
                                                f"[AutoGate] still failing -> harden round {harden_round} ({harden_remaining} steps)"
                                            )
                        except Exception as exc:
                            stop_reason = "auto_gate_error"
                            print(f"[AutoGate] error: {exc}")
                            traceback.print_exc()
                            break

                    if args.max_tokens and total_tokens >= args.max_tokens:
                        stop_reason = "max_tokens_reached"
                        break
        except KeyboardInterrupt:
            stop_reason = "interrupted"

    if args.save_dir:
        save_checkpoint(
            args.save_dir,
            trainer=trainer,
            tokenizer=tokenizer,
            train_cfg=train_cfg,
            step=last_step,
            total_tokens=total_tokens,
            optimizer=optimizer,
            extra={"sampler_cfg": asdict(sampler_cfg)},
            keep_every=args.save_every,
        )

    avg_tok_per_step = total_tokens / max(last_step, 1)
    print(
        f"Stopped: {stop_reason} step={last_step} total_tokens={total_tokens} "
        f"avg_tok/step={avg_tok_per_step:.1f}"
    )


if __name__ == "__main__":
    main()
