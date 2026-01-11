"""
Stage 5: Byte-level language skill (simple language modeling).

This module re-establishes the configuration/trainer contracts that Stage 6,
Stage 7, and the gate verification scripts rely on (e.g., `Stage5Config`,
`Stage5Trainer`, `evaluate_L5_gate`). The trainer wraps the reused SequenceModel
from Stage 4 but consumes real text-to-byte data for evaluation.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure HEI is on sys.path so sibling modules can import cleanly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from curriculum.common.base_trainer import BaseCurriculumTrainer
from curriculum.stage4_context import SequenceModel, Stage4Config, SymbolPort
from he_core.language_interface import ByteTokenizer


def _resolve_text_file(primary: str, fallback: str = "HEI/tests/dummy_corpus.txt") -> str:
    if primary and os.path.exists(primary):
        return primary
    if fallback and os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Text file not found: {primary or fallback}")


class TextCorpus:
    """Simple byte-level corpus loader for Stage 5 gate evaluation."""

    def __init__(self, path: str, tokenizer: ByteTokenizer, max_lines: int):
        self.path = path
        self.tokenizer = tokenizer
        self.max_lines = max_lines if max_lines and max_lines > 0 else None
        self.tokens: List[List[int]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Text corpus missing: {self.path}")
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    encoded = self.tokenizer.encode(text, add_special=True)
                except Exception:
                    continue
                if len(encoded) < 3:
                    continue
                self.tokens.append(encoded)
                if self.max_lines and len(self.tokens) >= self.max_lines:
                    break
        if not self.tokens:
            raise RuntimeError(f"No usable lines found in: {self.path}")

    def sample_window(
        self,
        *,
        seq_len: int,
        rng: random.Random,
    ) -> Tuple[Sequence[int], int, int]:
        if not self.tokens:
            raise RuntimeError("Corpus is empty.")
        # Try a few times before falling back to the first sample.
        for _ in range(8):
            tokens = rng.choice(self.tokens)
            if len(tokens) < 3:
                continue
            pos = rng.randint(1, len(tokens) - 2)
            return tokens[:pos], tokens[pos], tokens[pos + 1]
        tokens = self.tokens[0]
        pos = min(len(tokens) - 2, 1)
        return tokens[:pos], tokens[pos], tokens[pos + 1]


class Stage5EvalSampler:
    """Provides pretokenized sequences for the gate open-loop diagnostic."""

    def __init__(self, corpus: TextCorpus, seq_len: int, tokenizer: ByteTokenizer):
        self.corpus = corpus
        self.seq_len = int(seq_len)
        self.pad_id = tokenizer.pad_id

    def sample(self, *, batch_size: int, generator: torch.Generator) -> Dict[str, torch.Tensor]:
        if batch_size <= 0:
            return {"tokens": torch.full((0, self.seq_len), self.pad_id, dtype=torch.long)}
        tokens = torch.full((batch_size, self.seq_len), self.pad_id, dtype=torch.long)
        seeds = torch.randint(0, 2**31 - 1, (batch_size,), generator=generator)
        for i in range(batch_size):
            rng = random.Random(int(seeds[i].item()))
            context, query_tok, target_tok = self.corpus.sample_window(seq_len=self.seq_len, rng=rng)
            seq = list(context[-self.seq_len :]) + [query_tok, target_tok]
            seq = seq[: self.seq_len]
            if not seq:
                continue
            filled = torch.tensor(seq, dtype=torch.long)
            tokens[i, : filled.shape[0]] = filled
        return {"tokens": tokens}


@dataclass
class Stage5Config(Stage4Config):
    """Re-exported configuration object for Stage 5."""

    num_charts: int = 32
    vocab_size: int = 260
    router_context_dim: int = 0
    text_file: str = "HEI/data/CLUE/CLUECorpusSmall.txt"
    max_lines: int = 200000
    eval_lines: int = 2000
    eval_batches: int = 2
    eval_batch_size: int = 128
    sequence_len: int = 32
    a3: float = 0.0
    gate_profile: str = "robust"
    gate_acc: float = 0.3
    gate_ppl: float = 15.0
    gate_offline_ppl_ratio: float = 10.0
    gate_offline_acc_ratio: float = 0.3
    offline_steps_between: int = 2
    seed: int = 0
    extra_args: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.context_window_train = int(self.sequence_len or self.context_window_train)
        if self.sequence_len <= 0:
            self.sequence_len = int(self.context_window_train or 32)
        self.alpha_chart = float(self.a3 or self.alpha_chart)
        self.num_charts = int(getattr(self, "num_charts", 32))
        self.vocab_size = int(getattr(self, "vocab_size", 260))
        if self.port_decode_state is None:
            self.port_decode_state = True

    def __str__(self) -> str:
        return (
            "Stage5Config("
            f"device={self.device}, steps={self.steps}, batch_size={self.batch_size}, "
            f"seq={self.sequence_len}, charts={self.num_charts}, dt={self.dt}, "
            f"evo={self.evolution_steps}, lr={self.lr}, a3={self.a3}, "
            f"cuda_graph={int(bool(self.use_cuda_graph))}"
            ")"
        )


class Stage5Trainer(BaseCurriculumTrainer):
    """Minimal Stage 5 trainer that exposes the language gate diagnostics."""

    def __init__(self, config: Stage5Config):
        super().__init__(config)
        self.config = config
        self.tokenizer = ByteTokenizer()
        self.port = SymbolPort(
            vocab_size=int(config.vocab_size),
            dim_q=int(config.dim_q),
            num_charts=int(config.num_charts),
            decode_state=bool(config.port_decode_state),
        ).to(config.device)
        self.entity.add_interface("language", int(config.dim_q))
        self.seq_model = SequenceModel(
            entity=self.entity,
            port=self.port,
            config=config,
            phase_embed=None,
        )
        self._train_rng = random.Random(int(config.seed or 0))
        self._eval_rng = random.Random(int(config.seed or 0) + 1)
        self._eval_corpus = TextCorpus(
            path=_resolve_text_file(config.text_file),
            tokenizer=self.tokenizer,
            max_lines=max(1, int(config.eval_lines)),
        )
        self.eval_data = Stage5EvalSampler(
            corpus=self._eval_corpus,
            seq_len=self.config.sequence_len,
            tokenizer=self.tokenizer,
        )
        torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def evaluate_L5_gate(
        self,
        *,
        batch_size: Optional[int] = None,
        offline_steps_between: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        batch_size = int(batch_size or self.config.eval_batch_size)
        offline_steps = int(offline_steps_between or self.config.offline_steps_between)
        teacher = self._run_gate_eval(
            batch_size=batch_size,
            offline_steps=0,
            rng=self._gate_rng(seed=seed, offset=0),
        )
        offline = self._run_gate_eval(
            batch_size=batch_size,
            offline_steps=offline_steps,
            rng=self._gate_rng(seed=seed, offset=1),
        )

        ratios = {
            "ppl_ratio": offline["ppl"] / max(teacher["ppl"], 1e-6),
            "acc_ratio": offline["acc"] / max(teacher["acc"], 1e-6),
        }
        thresholds = {
            "gate_acc": float(self.config.gate_acc),
            "gate_ppl": float(self.config.gate_ppl),
            "gate_offline_ppl_ratio": float(self.config.gate_offline_ppl_ratio),
            "gate_offline_acc_ratio": float(self.config.gate_offline_acc_ratio),
            "offline_steps_between": offline_steps,
            "offline_ratio_ref": "teacher_forced",
        }
        passed_tf = (teacher["acc"] >= thresholds["gate_acc"]) and (teacher["ppl"] <= thresholds["gate_ppl"])
        passed_offline = True
        if str(self.config.gate_profile).lower() == "robust":
            passed_offline = (
                ratios["ppl_ratio"] <= thresholds["gate_offline_ppl_ratio"]
                and ratios["acc_ratio"] >= thresholds["gate_offline_acc_ratio"]
            )
        passed = passed_tf and passed_offline

        return {
            "profile": str(self.config.gate_profile).lower(),
            "TeacherForced": teacher,
            "OfflineBetween": offline,
            "ratios": ratios,
            "thresholds": thresholds,
            "passed_components": {"TeacherForced": passed_tf, "OfflineBetween": passed_offline},
            "passed": passed,
        }

    def _run_gate_eval(self, *, batch_size: int, offline_steps: int, rng: random.Random) -> Dict[str, float]:
        batches = max(1, int(self.config.eval_batches))
        seq_len = int(self.config.sequence_len)
        total_loss = 0.0
        total_acc = 0.0
        for _ in range(batches):
            batch = self._build_eval_batch(batch_size=batch_size, seq_len=seq_len, rng=rng)
            batch = {k: v.to(self.config.device, non_blocking=True) for k, v in batch.items()}
            init = torch.zeros((batch_size, 2 * int(self.config.dim_q) + 1), device=self.config.device)
            loss, diag = self.seq_model(
                context_tokens=batch["context_tokens"],
                query_token=batch["query_token"],
                target_token=batch["target_token"],
                initial_state_flat=init,
                offline_steps=offline_steps,
            )
            total_loss += float(loss.item())
            total_acc += float(diag["acc"].item())
        avg_loss = total_loss / float(batches)
        avg_acc = total_acc / float(batches)
        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "ppl": float(math.exp(avg_loss)),
            "steps": offline_steps,
        }

    def _build_eval_batch(self, *, batch_size: int, seq_len: int, rng: random.Random) -> Dict[str, torch.Tensor]:
        ctx = torch.full((batch_size, seq_len), self.tokenizer.pad_id, dtype=torch.long)
        query = torch.full((batch_size,), self.tokenizer.pad_id, dtype=torch.long)
        target = torch.full((batch_size,), self.tokenizer.pad_id, dtype=torch.long)
        for i in range(batch_size):
            context, query_tok, target_tok = self._eval_corpus.sample_window(seq_len=seq_len, rng=rng)
            ctx_len = min(len(context), seq_len)
            if ctx_len > 0:
                tail = torch.tensor(context[-ctx_len:], dtype=torch.long)
                ctx[i, seq_len - ctx_len :] = tail
            query[i] = int(query_tok)
            target[i] = int(target_tok)
        return {"context_tokens": ctx, "query_token": query, "target_token": target}

    def _gate_rng(self, *, seed: Optional[int], offset: int) -> random.Random:
        if seed is not None:
            return random.Random(seed + offset)
        return random.Random(self._eval_rng.randint(0, 2**31 - 1))


def parse_stage5_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5 language gate helper.")
    parser.add_argument("--text_file", help="Text file for gate evaluation.")
    parser.add_argument("--steps", type=int, help="Number of (notional) training steps.")
    parser.add_argument("--batch_size", type=int, help="Training batch size.")
    parser.add_argument("--sequence_len", type=int, help="Sequence length for context.")
    parser.add_argument("--eval_lines", type=int, help="Eval corpus size.")
    parser.add_argument("--eval_batches", type=int, help="Eval batches per gate run.")
    parser.add_argument("--eval_batch_size", type=int, help="Eval batch size.")
    parser.add_argument("--save_dir", help="Checkpoint directory.")
    parser.add_argument("--cuda_graph", type=int, choices=[0, 1], help="Enable CUDA graph logging placeholder.")
    parser.add_argument("--seed", type=int, help="Random seed override.")
    return parser.parse_args()


def _apply_env_overrides(cfg: Stage5Config) -> Stage5Config:
    def _get_int(name: str, default: int) -> int:
        v = os.getenv(name)
        return default if v is None or v == "" else int(v)

    def _get_float(name: str, default: float) -> float:
        v = os.getenv(name)
        return default if v is None or v == "" else float(v)

    cfg.steps = _get_int("HEI_STEPS", cfg.steps)
    cfg.batch_size = _get_int("HEI_BATCH_SIZE", cfg.batch_size)
    cfg.eval_batch_size = _get_int("HEI_EVAL_BATCH", cfg.eval_batch_size)
    cfg.eval_batches = _get_int("HEI_EVAL_BATCHES", cfg.eval_batches)
    cfg.evolution_steps = _get_int("HEI_EVO", cfg.evolution_steps)
    cfg.dt = _get_float("HEI_DT", cfg.dt)
    cfg.lr = _get_float("HEI_LR", cfg.lr)
    cfg.sequence_len = _get_int("HEI_SEQUENCE_LEN", cfg.sequence_len)
    cfg.context_window_train = cfg.sequence_len
    cfg.max_lines = _get_int("HEI_MAX_LINES", cfg.max_lines)
    cfg.eval_lines = _get_int("HEI_EVAL_LINES", cfg.eval_lines)
    cfg.seed = _get_int("HEI_SEED", cfg.seed)
    cfg.gate_acc = _get_float("HEI_GATE_ACC", cfg.gate_acc)
    cfg.gate_ppl = _get_float("HEI_GATE_PPL", cfg.gate_ppl)
    cfg.gate_offline_acc_ratio = _get_float("HEI_GATE_OFF_ACC", cfg.gate_offline_acc_ratio)
    cfg.gate_offline_ppl_ratio = _get_float("HEI_GATE_OFF_PPL", cfg.gate_offline_ppl_ratio)
    cfg.offline_steps_between = _get_int("HEI_OFFLINE_STEPS", cfg.offline_steps_between)
    if os.getenv("HEI_CUDA_GRAPH") is not None:
        cfg.use_cuda_graph = os.getenv("HEI_CUDA_GRAPH", "0") == "1"
    return cfg


def main() -> int:
    args = parse_stage5_args()
    cfg = _apply_env_overrides(Stage5Config())
    for name, value in vars(args).items():
        if value is None:
            continue
        if name == "cuda_graph":
            cfg.use_cuda_graph = bool(value)
            continue
        setattr(cfg, name, value)
    cfg.text_file = cfg.text_file or _resolve_text_file(cfg.text_file)
    trainer = Stage5Trainer(cfg)
    # Placeholder training entrypoint (currently gate-only).
    gate = trainer.evaluate_L5_gate()
    print("=== Stage5 Gate (bootstrapped) ===")
    print("Profile:", gate["profile"])
    print("TeacherForced:", gate["TeacherForced"])
    print("OfflineBetween:", gate["OfflineBetween"])
    print("Thresholds:", gate["thresholds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
