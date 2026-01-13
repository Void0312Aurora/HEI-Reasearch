"""
Stage 7 runner (v0 active language gate).

This entrypoint orchestrates the existing active-sampling trainer so Stage 7
can reuse the L0-L5 curriculum+diagnostics while remaining faithful to the
L7 specification (closed-loop gate + interface regression).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List


@dataclass
class Stage7Config:
    wiki_path: str = os.getenv("HEI_STAGE7_WIKI_PATH", "HEI/data/wiki/wikipedia-zh-20250901.json")
    clue_path: str = os.getenv("HEI_STAGE7_CLUE_PATH", "HEI/data/CLUE/CLUECorpusSmall.txt")
    max_samples_wiki: int = int(os.getenv("HEI_STAGE7_MAX_WIKI", "100000"))
    max_samples_clue: int = int(os.getenv("HEI_STAGE7_MAX_CLUE", "100000"))
    tokenizer_mode: str = os.getenv("HEI_STAGE7_TOKENIZER_MODE", "byte")
    tokenizer_vocab: int = int(os.getenv("HEI_STAGE7_TOKENIZER_VOCAB", "4096"))
    tokenizer_min_freq: int = int(os.getenv("HEI_STAGE7_TOKENIZER_MIN_FREQ", "2"))
    tokenizer_build_samples: int = int(os.getenv("HEI_STAGE7_TOKENIZER_BUILD_SAMPLES", "20000"))
    eval_text_file: str = os.getenv("HEI_STAGE7_EVAL_FILE", "HEI/data/CLUE/CLUECorpusSmall.txt")
    eval_lines: int = int(os.getenv("HEI_STAGE7_EVAL_LINES", "2000"))
    gate_profile: str = os.getenv("HEI_STAGE7_GATE_PROFILE", "base")
    gate_batch_size: int = int(os.getenv("HEI_STAGE7_GATE_BATCH_SIZE", "64"))
    gate_num_prompts: int = int(os.getenv("HEI_STAGE7_GATE_NUM_PROMPTS", "200"))
    gate_prompt_chars: int = int(os.getenv("HEI_STAGE7_GATE_PROMPT_CHARS", "4"))
    gate_max_new_tokens: int = int(os.getenv("HEI_STAGE7_GATE_MAX_NEW_TOKENS", "64"))
    gate_temperature: float = float(os.getenv("HEI_STAGE7_GATE_TEMPERATURE", "1.0"))
    # Gate decoding: 0 = auto (base: top_k=20, robust: top_k=1)
    gate_top_k: int = int(os.getenv("HEI_STAGE7_GATE_TOP_K", "0"))
    gate_ngram_n: int = int(os.getenv("HEI_STAGE7_GATE_NGRAM_N", "3"))
    gate_cycle_max_period: int = int(os.getenv("HEI_STAGE7_GATE_CYCLE_MAX_PERIOD", "8"))
    gate_cycle_min_repeats: int = int(os.getenv("HEI_STAGE7_GATE_CYCLE_MIN_REPEATS", "3"))
    gate_cycle_tail_window: int = int(os.getenv("HEI_STAGE7_GATE_CYCLE_TAIL_WINDOW", "32"))
    gate_margin_window: int = int(os.getenv("HEI_STAGE7_GATE_MARGIN_WINDOW", "10"))
    steps: int = int(os.getenv("HEI_STAGE7_STEPS", "2000"))
    batch_size: int = int(os.getenv("HEI_STAGE7_BATCH_SIZE", "512"))
    max_seq_len: int = int(os.getenv("HEI_STAGE7_SEQ_LEN", "128"))
    num_charts: int = int(os.getenv("HEI_STAGE7_NUM_CHARTS", "32"))
    lr: float = float(os.getenv("HEI_STAGE7_LR", "0.001"))
    dt: float = float(os.getenv("HEI_STAGE7_DT", "0.2"))
    num_evolution_steps: int = int(os.getenv("HEI_STAGE7_EVO", "2"))
    beta_kl: float = float(os.getenv("HEI_STAGE7_BETA_KL", "0.01"))
    gamma_pred: float = float(os.getenv("HEI_STAGE7_GAMMA_PRED", "1.0"))
    tf32: int = int(os.getenv("HEI_STAGE7_TF32", os.getenv("HEI_TF32", "1")))
    amp: int = int(os.getenv("HEI_STAGE7_AMP", os.getenv("HEI_AMP", "1")))
    amp_dtype: str = os.getenv("HEI_STAGE7_AMP_DTYPE", os.getenv("HEI_AMP_DTYPE", "bf16"))
    fused_adamw: int = int(os.getenv("HEI_STAGE7_FUSED_ADAMW", os.getenv("HEI_FUSED_ADAMW", "1")))
    cuda_graph: int = int(os.getenv("HEI_STAGE7_CUDA_GRAPH", os.getenv("HEI_CUDA_GRAPH", "-1")))
    init_entity_ckpt: str = os.getenv(
        "HEI_STAGE7_INIT_ENTITY_CKPT",
        os.getenv("HEI_STAGE5_CKPT", "checkpoints/curriculum/stage5_final.pt"),
    )
    port_trainable: int = int(os.getenv("HEI_STAGE7_PORT_TRAINABLE", "1"))
    detach_every: int = int(os.getenv("HEI_STAGE7_DETACH_EVERY", "64"))
    scheduled_sampling_prob: float = float(os.getenv("HEI_STAGE7_SS_PROB", "0.0"))
    scheduled_sampling_mode: str = os.getenv("HEI_STAGE7_SS_MODE", "sample")
    scheduled_sampling_top_k: int = int(os.getenv("HEI_STAGE7_SS_TOP_K", "20"))
    scheduled_sampling_temperature: float = float(os.getenv("HEI_STAGE7_SS_TEMP", "1.0"))
    unlikelihood_weight: float = float(os.getenv("HEI_STAGE7_UNLIKELIHOOD_W", "0.0"))
    unlikelihood_window: int = int(os.getenv("HEI_STAGE7_UNLIKELIHOOD_WINDOW", "1"))
    denoise_mode: str = os.getenv("HEI_STAGE7_DENOISE_MODE", "none")
    denoise_prob: float = float(os.getenv("HEI_STAGE7_DENOISE_PROB", "0.0"))
    experience_push_n: int = int(os.getenv("HEI_STAGE7_EXPERIENCE_PUSH_N", "0"))
    save_dir: str = os.getenv("HEI_STAGE7_SAVE_DIR", "checkpoints/curriculum/stage7_active")
    resume: bool = os.getenv("HEI_STAGE7_RESUME", "1") == "1"
    resume_optimizer: int = int(os.getenv("HEI_STAGE7_RESUME_OPTIMIZER", "1"))
    resume_strict: int = int(os.getenv("HEI_STAGE7_RESUME_STRICT", "1"))
    sample_every: int = int(os.getenv("HEI_STAGE7_SAMPLE_EVERY", "200"))
    save_every: int = int(os.getenv("HEI_STAGE7_SAVE_EVERY", "500"))
    device: str = os.getenv("HEI_STAGE7_DEVICE", os.getenv("HEI_DEVICE", "cuda"))
    offline_replay_mode: str = os.getenv("HEI_STAGE7_OFFLINE_REPLAY_MODE", "none")
    sampler_loss_update_every: int = int(os.getenv("HEI_STAGE7_SAMPLER_LOSS_UPDATE_EVERY", "50"))
    sampler_loss_update_mode: str = os.getenv("HEI_STAGE7_SAMPLER_LOSS_UPDATE_MODE", "per_sample")
    sampler_loss_weight: float = float(os.getenv("HEI_STAGE7_SAMPLER_LOSS_WEIGHT", "1.0"))
    sampler_coverage_weight: float = float(os.getenv("HEI_STAGE7_SAMPLER_COVERAGE_WEIGHT", "0.5"))
    sampler_progress_weight: float = float(os.getenv("HEI_STAGE7_SAMPLER_PROGRESS_WEIGHT", "0.5"))
    sampler_temperature: float = float(os.getenv("HEI_STAGE7_SAMPLER_TEMPERATURE", "1.0"))
    sampler_random_ratio: float = float(os.getenv("HEI_STAGE7_SAMPLER_RANDOM_RATIO", "0.1"))
    sampler_ema_rate: float = float(os.getenv("HEI_STAGE7_SAMPLER_EMA_RATE", "0.1"))
    # Default off: keeping a 400k×128 token cache on GPU costs ~0.8GB for char vocab,
    # and CUDA graphs no longer require on-device data. Enable explicitly if desired.
    data_on_device: bool = os.getenv("HEI_STAGE7_DATA_ON_DEVICE", "0") == "1"
    extra_train_args: List[str] = field(default_factory=list)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Stage7Config:
        cfg = cls()
        for name, value in vars(args).items():
            if value is None:
                continue
            setattr(cfg, name, value)
        if args.extra_train_args:
            cfg.extra_train_args = shlex.split(args.extra_train_args)
        return cfg

    def training_command(self) -> List[str]:
        resume_args: List[str] = []
        if self.resume:
            last_ckpt = os.path.join(self.save_dir, "last.pt")
            if os.path.exists(last_ckpt):
                resume_args = ["--resume_dir", self.save_dir]
        base = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "../training/train_active_sampling.py"),
            *resume_args,
            "--resume_optimizer",
            str(int(self.resume_optimizer)),
            "--resume_strict",
            str(int(self.resume_strict)),
            "--device",
            self.device,
            "--wiki_path",
            self.wiki_path,
            "--clue_path",
            self.clue_path,
            "--max_samples_wiki",
            str(self.max_samples_wiki),
            "--max_samples_clue",
            str(self.max_samples_clue),
            "--tokenizer",
            self.tokenizer_mode,
            "--vocab_size",
            str(self.tokenizer_vocab),
            "--vocab_build_samples",
            str(self.tokenizer_build_samples),
            "--max_seq_len",
            str(self.max_seq_len),
            "--num_charts",
            str(self.num_charts),
            "--lr",
            str(self.lr),
            "--dt",
            str(self.dt),
            "--num_evolution_steps",
            str(self.num_evolution_steps),
            "--beta_kl",
            str(self.beta_kl),
            "--gamma_pred",
            str(self.gamma_pred),
            "--tf32",
            str(int(self.tf32)),
            "--amp",
            str(int(self.amp)),
            "--amp_dtype",
            str(self.amp_dtype),
            "--fused_adamw",
            str(int(self.fused_adamw)),
            "--cuda_graph",
            str(int(self.cuda_graph)),
            "--steps",
            str(self.steps),
            "--batch_size",
            str(self.batch_size),
            "--save_dir",
            self.save_dir,
            "--sample_every",
            str(self.sample_every),
            "--save_every",
            str(self.save_every),
            "--port_arch",
            "minimal",
            "--sequence_mode",
            "recurrent",
            "--sampler_loss_update_every",
            str(int(self.sampler_loss_update_every)),
            "--sampler_loss_update_mode",
            str(self.sampler_loss_update_mode),
            "--sampler_loss_weight",
            str(float(self.sampler_loss_weight)),
            "--sampler_coverage_weight",
            str(float(self.sampler_coverage_weight)),
            "--sampler_progress_weight",
            str(float(self.sampler_progress_weight)),
            "--sampler_temperature",
            str(float(self.sampler_temperature)),
            "--sampler_random_ratio",
            str(float(self.sampler_random_ratio)),
            "--sampler_ema_rate",
            str(float(self.sampler_ema_rate)),
            "--port_trainable",
            str(int(self.port_trainable)),
            "--detach_every",
            str(int(self.detach_every)),
            "--scheduled_sampling_prob",
            str(float(self.scheduled_sampling_prob)),
            "--scheduled_sampling_mode",
            str(self.scheduled_sampling_mode),
            "--scheduled_sampling_top_k",
            str(int(self.scheduled_sampling_top_k)),
            "--scheduled_sampling_temperature",
            str(float(self.scheduled_sampling_temperature)),
            "--unlikelihood_weight",
            str(float(self.unlikelihood_weight)),
            "--unlikelihood_window",
            str(int(self.unlikelihood_window)),
            "--denoise_mode",
            str(self.denoise_mode),
            "--denoise_prob",
            str(float(self.denoise_prob)),
            "--router_balance_weight",
            "0.0",
            "--router_entropy_weight",
            "0.0",
            "--experience_push_n",
            str(int(self.experience_push_n)),
            "--offline_replay_mode",
            self.offline_replay_mode,
            "--pretokenize_samples",
            "1",
            "--data_on_device",
            "1" if self.data_on_device else "0",
        ]
        init_ckpt = str(self.init_entity_ckpt or "").strip()
        if init_ckpt:
            base.extend(["--init_entity_ckpt", init_ckpt])
        if self.extra_train_args:
            base.extend(self.extra_train_args)
        return base

    def gate_command(self) -> List[str]:
        ckpt = os.path.join(self.save_dir, "last.pt")
        top_k = int(self.gate_top_k)
        if top_k <= 0:
            top_k = 20 if str(self.gate_profile).lower() == "base" else 1
        return [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "../training/eval_stage7_gate.py"),
            "--ckpt",
            ckpt,
            "--device",
            self.device,
            "--profile",
            self.gate_profile,
            "--batch_size",
            str(self.gate_batch_size),
            "--max_seq_len",
            str(self.max_seq_len),
            "--eval_text_file",
            self.eval_text_file,
            "--wiki_path",
            self.wiki_path,
            "--clue_path",
            self.clue_path,
            "--max_samples_wiki",
            str(max(self.max_samples_wiki, 1)),
            "--max_samples_clue",
            str(max(self.max_samples_clue, 1)),
            "--eval_samples",
            str(min(512, self.eval_lines)),
            "--num_prompts",
            str(self.gate_num_prompts),
            "--prompt_chars",
            str(self.gate_prompt_chars),
            "--max_new_tokens",
            str(self.gate_max_new_tokens),
            "--temperature",
            str(self.gate_temperature),
            "--top_k",
            str(top_k),
            "--ngram_n",
            str(self.gate_ngram_n),
            "--cycle_max_period",
            str(self.gate_cycle_max_period),
            "--cycle_min_repeats",
            str(self.gate_cycle_min_repeats),
            "--cycle_tail_window",
            str(self.gate_cycle_tail_window),
            "--margin_window",
            str(self.gate_margin_window),
        ]


def parse_stage7_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 7 active language training + gate.")
    parser.add_argument("--wiki_path", help="Override wiki json path")
    parser.add_argument("--clue_path", help="Override CLUE text path")
    parser.add_argument("--max_samples_wiki", type=int)
    parser.add_argument("--max_samples_clue", type=int)
    parser.add_argument("--tokenizer_mode", choices=["byte", "char"], help="Tokenizer mode.")
    parser.add_argument("--tokenizer_vocab", type=int)
    parser.add_argument("--tokenizer_min_freq", type=int)
    parser.add_argument("--tokenizer_build_samples", type=int)
    parser.add_argument("--eval_text_file", help="Evaluation text file for gate.")
    parser.add_argument("--eval_lines", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--num_charts", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--num_evolution_steps", type=int)
    parser.add_argument("--beta_kl", type=float)
    parser.add_argument("--gamma_pred", type=float)
    parser.add_argument("--tf32", type=int)
    parser.add_argument("--amp", type=int)
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16"])
    parser.add_argument("--fused_adamw", type=int)
    parser.add_argument("--cuda_graph", type=int)
    parser.add_argument("--init_entity_ckpt", help="Optional entity init checkpoint (e.g. Stage5).")
    parser.add_argument("--port_trainable", type=int)
    parser.add_argument("--detach_every", type=int)
    parser.add_argument("--scheduled_sampling_prob", type=float)
    parser.add_argument("--scheduled_sampling_mode", choices=["sample", "argmax"])
    parser.add_argument("--scheduled_sampling_top_k", type=int)
    parser.add_argument("--scheduled_sampling_temperature", type=float)
    parser.add_argument("--unlikelihood_weight", type=float, help="Protocol-5 hardening: unlikelihood weight.")
    parser.add_argument("--unlikelihood_window", type=int, help="Protocol-5 hardening: unlikelihood window.")
    parser.add_argument("--denoise_mode", choices=["none", "replace", "repeat"], help="B3 denoising mode.")
    parser.add_argument("--denoise_prob", type=float, help="B3 denoising corruption probability.")
    parser.add_argument("--experience_push_n", type=int, help="Push N end states per batch into experience buffer (0=disable).")
    parser.add_argument("--save_dir", help="Checkpoint directory.")
    parser.add_argument("--resume", type=int)
    parser.add_argument("--resume_optimizer", type=int)
    parser.add_argument("--resume_strict", type=int)
    parser.add_argument("--sample_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--device", help="Training device (cuda/cpu).")
    parser.add_argument("--offline_replay_mode", help="Offline replay mode (none/prioritized/...).")
    parser.add_argument("--sampler_loss_update_every", type=int)
    parser.add_argument("--sampler_loss_update_mode", choices=["per_sample", "batch_mean"])
    parser.add_argument("--sampler_loss_weight", type=float)
    parser.add_argument("--sampler_coverage_weight", type=float)
    parser.add_argument("--sampler_progress_weight", type=float)
    parser.add_argument("--sampler_temperature", type=float)
    parser.add_argument("--sampler_random_ratio", type=float)
    parser.add_argument("--sampler_ema_rate", type=float)
    parser.add_argument("--data_on_device", type=int, help="Keep cached token tensors on device (1/0).")
    parser.add_argument("--gate_profile", choices=["base", "robust"])
    parser.add_argument("--gate_batch_size", type=int)
    parser.add_argument("--extra_train_args", help="Additional args for train_active_sampling.")
    return parser.parse_args()


def run_stage7(cfg: Stage7Config) -> int:
    train_cmd = cfg.training_command()
    print("Stage7 Training command:", flush=True)
    print(" ".join(shlex.quote(arg) for arg in train_cmd), flush=True)
    env = os.environ.copy()
    env["HEI_DEVICE"] = cfg.device
    env["HEI_OFFLINE_REPLAY_MODE"] = cfg.offline_replay_mode
    res = subprocess.run(train_cmd, check=False, env=env)
    if res.returncode != 0:
        print(f"[Stage7] Training failed with exit code {res.returncode}")
        return res.returncode

    gate_cmd = cfg.gate_command()
    print("Stage7 Gate command:", flush=True)
    print(" ".join(shlex.quote(arg) for arg in gate_cmd), flush=True)
    res = subprocess.run(gate_cmd, check=False, env=env)
    if res.returncode != 0:
        print(f"[Stage7] Gate evaluation failed with exit code {res.returncode}")
    return res.returncode


def main() -> int:
    args = parse_stage7_args()
    cfg = Stage7Config.from_args(args)
    os.makedirs(cfg.save_dir, exist_ok=True)
    return run_stage7(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
