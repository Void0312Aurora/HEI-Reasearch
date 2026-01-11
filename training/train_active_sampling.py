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
import contextlib
import os
import queue
import random
import signal
import sys
import threading
import time
import traceback
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

    # CUDA-graph policy (mirrors L4/L5 assumptions: fixed shapes + teacher-forced recurrent + pretokenized on device).
    if str(device).startswith("cuda") and (not use_amp):
        auto = (
            args.port_arch == "minimal"
            and args.sequence_mode == "recurrent"
            and bool(args.pretokenize_samples)
        )
        use_cuda_graph = bool(auto) if int(args.cuda_graph) == -1 else bool(int(args.cuda_graph))
    else:
        use_cuda_graph = bool(int(args.cuda_graph)) if int(args.cuda_graph) >= 0 else False

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
        random_ratio=0.1,
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
            f"ss_temp={getattr(train_cfg,'scheduled_sampling_temperature',1.0)}"
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
                if use_amp:
                    raise RuntimeError("CUDA graph mode currently requires --amp=0.")

                B = int(args.batch_size)
                L = int(args.max_seq_len)
                token_buf = torch.empty((B, L), device=device, dtype=torch.long)
                mask_buf = torch.empty((B, L), device=device, dtype=torch.long)

                loss_buf = torch.zeros((), device=device, dtype=torch.float32)
                F_on_buf = torch.zeros((), device=device, dtype=torch.float32)
                F_off_buf = torch.zeros((), device=device, dtype=torch.float32)
                dF_off_buf = torch.zeros((), device=device, dtype=torch.float32)
                off_loss_buf = torch.zeros((), device=device, dtype=torch.float32)
                rb_buf = torch.zeros((), device=device, dtype=torch.float32)
                re_buf = torch.zeros((), device=device, dtype=torch.float32)
                ar_buf = torch.zeros((), device=device, dtype=torch.float32)
                E_buf = torch.zeros((), device=device, dtype=torch.float32)
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
                        return bucket_keys

                    # Fallback: this is slower; CUDA-graph mode expects pretokenized cache.
                    texts = [sampler.pool.samples[i].text for i in indices]
                    batch2 = build_batch(texts, tokenizer, args.max_seq_len)
                    token_buf.copy_(batch2["input_ids"].to(device=device, non_blocking=True))
                    mask_buf.copy_(batch2["attention_mask"].to(device=device, non_blocking=True))
                    return bucket_keys

                def fwd_bwd_step() -> None:
                    optimizer.zero_grad(set_to_none=False)
                    out = trainer.train_step({"input_ids": token_buf, "attention_mask": mask_buf})
                    loss = out["loss"]
                    loss_buf.copy_(loss.detach().float())
                    F_on_buf.copy_(out["free_energy_online"].detach().float())
                    F_off_buf.copy_(out["free_energy_offline"].detach().float())
                    dF_off_buf.copy_(out.get("delta_free_energy_offline", zero_scalar).detach().float())
                    off_loss_buf.copy_(out.get("offline_loss", zero_scalar).detach().float())
                    rb_buf.copy_(out.get("router_balance", zero_scalar).detach().float())
                    re_buf.copy_(out.get("router_entropy", zero_scalar).detach().float())
                    ar_buf.copy_(out.get("atlas_reg", zero_scalar).detach().float())
                    E_buf.copy_(out["prediction_error"].detach().float())
                    ppl_buf.copy_(out["perplexity"].detach().float())
                    loss.backward()
                    _clip_grad_norm_no_sync_(train_params, 1.0)

                warmup = max(1, int(args.cuda_graph_warmup))
                for _ in range(warmup):
                    bucket_keys = refill()
                    fwd_bwd_step()
                    optimizer.step()
                    sampler.update_counts(bucket_keys)

                torch.cuda.synchronize()
                pool = torch.cuda.graphs.graph_pool_handle()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    fwd_bwd_step()

                for step in range(start_step + 1, args.steps + 1):
                    if stop_requested:
                        stop_reason = "interrupted"
                        break
                    t0 = time.perf_counter()
                    bucket_keys = refill()
                    graph.replay()
                    optimizer.step()

                    # No-sync approximate token count in CUDA-graph mode.
                    step_tokens = int(token_buf.numel())
                    total_tokens += step_tokens
                    window_tokens += step_tokens
                    sampler.update_counts(bucket_keys)
                    last_step = step

                    dt = time.perf_counter() - t0
                    window_time += dt

                    if step % args.log_every == 0:
                        torch.cuda.synchronize()
                        tok_s = (window_tokens / max(window_time, 1e-9))
                        elapsed = time.perf_counter() - start_time
                        msg = (
                            f"Step {step}: F={float(loss_buf.item()):.4f} "
                            f"F_on={float(F_on_buf.item()):.4f} F_off={float(F_off_buf.item()):.4f} "
                            f"dF_off={float(dF_off_buf.item()):.4f} L_off={float(off_loss_buf.item()):.4f} "
                            f"R_bal={float(rb_buf.item()):.4f} R_ent={float(re_buf.item()):.4f} R_atlas={float(ar_buf.item()):.4f} "
                            f"E_pred={float(E_buf.item()):.4f} PPL={float(ppl_buf.item()):.2f} "
                            f"tok={total_tokens} tok/s={tok_s:.1f} t={elapsed:.1f}s"
                        )
                        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        msg += f" peak_mem={peak_gb:.2f}GB"
                        torch.cuda.reset_peak_memory_stats()
                        print(msg)
                        window_tokens = 0
                        window_time = 0.0

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

                    if args.max_tokens and total_tokens >= args.max_tokens:
                        stop_reason = "max_tokens_reached"
                        break

            else:
                for step in range(start_step + 1, args.steps + 1):
                    if stop_requested:
                        stop_reason = "interrupted"
                        break

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
                        tok_s = (window_tokens / max(window_time, 1e-9))
                        elapsed = time.perf_counter() - start_time
                        msg = (
                            f"Step {step}: F={avg_loss:.4f} "
                            f"F_on={F_online:.4f} F_off={F_offline:.4f} "
                            f"dF_off={delta_off:.4f} L_off={off_loss:.4f} "
                            f"R_bal={rb:.4f} R_ent={re:.4f} R_atlas={ar:.4f} "
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
