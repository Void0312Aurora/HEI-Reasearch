"""
Stage 1: Counting & Periodicity (Level 1)

Goal: Learn simple sequence laws (attractors/limit cycles).
This validates that the dynamics can form discrete attractors or cycles
driven by a minimal port.

Feedback Integration (temp-01.md):
- A3: Optimize unified Free Energy F = V + KL + E_pred.
- A1: Use a minimal port to maintain the Markov Blanket.
- GPU: Use large batch sizes for saturation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import time
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import random

# Ensure HEI is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from curriculum.common.base_trainer import BaseCurriculumTrainer, CurriculumConfig
from he_core.state import ContactState

_PRIMITIVE_BINARY_PATTERN_CACHE: dict[int, torch.Tensor] = {}


def _is_primitive_period(bits: list[int], period: int) -> bool:
    if period <= 1:
        return False
    if all(b == bits[0] for b in bits):
        return False
    for d in range(1, period):
        if period % d != 0:
            continue
        ok = True
        for i in range(period):
            if bits[i] != bits[i % d]:
                ok = False
                break
        if ok:
            return False
    return True


def get_primitive_binary_patterns(period: int) -> torch.Tensor:
    """
    Return all {0,1} patterns of length `period` whose *minimal* period is exactly `period`.
    Cached on CPU; move to device once in the trainer.
    """
    p = int(period)
    if p in _PRIMITIVE_BINARY_PATTERN_CACHE:
        return _PRIMITIVE_BINARY_PATTERN_CACHE[p]
    if p < 2:
        raise ValueError(f"Binary-period patterns require period>=2; got {p}.")
    if p > 12:
        raise ValueError(
            f"Refusing to enumerate binary patterns for period={p} (2^{p} too large). "
            "Use a different period_pattern_mode or reduce HEI_PERIOD_MAX."
        )
    patterns: list[list[int]] = []
    for mask in range(1, (1 << p) - 1):  # exclude all-0 and all-1
        bits = [(mask >> i) & 1 for i in range(p)]
        if _is_primitive_period(bits, p):
            patterns.append(bits)
    if not patterns:
        raise RuntimeError(f"No primitive binary patterns found for period={p}.")
    out = torch.tensor(patterns, dtype=torch.long)
    _PRIMITIVE_BINARY_PATTERN_CACHE[p] = out
    return out


@dataclass
class Stage1Config(CurriculumConfig):
    # Task params
    vocab_size: int = 16
    sequence_len: int = 16
    # Period task: cycle length range (temp-04.md requires periods 2–5).
    period_min: int = 2
    period_max: int = 5
    # Period pattern family:
    # - "offset1": deterministic cycle (start,start+1,...,start+P-1,...) — a good L1 default.
    # - "binary": primitive {A,B} cycles (matches temp-04.md examples like A,B,A,B,... / □□○○...).
    # - "offsets"/"random": harder random patterns over the full vocabulary.
    period_pattern_mode: str = "offset1"
    
    # Physics/Model params
    dim_q: int = 64         # Reduced to 64 to allow more branches (O(D^2) memory)
    num_charts: int = 4     # Allow some switching
    stiffness: float = 0.1  # Standard stiffness
    integrator_method: str = 'semi' # Vector-space semi-implicit (stable, O(B·D))
    
    # Training params
    batch_size: int = 128   # Effective batch = 128 * 16 = 2048
    steps: int = 1000        # Short sanity run (avoid long tests by default)
    evolution_steps: int = 16 # Needed for counting dynamics to converge
    dt: float = 0.2         # Standard time step
    lr: float = 1e-3        # Plasticity rate (temp-05.md); lower improves stability for mixed skills
    # Optimizer hygiene: make implicit defaults explicit.
    weight_decay: float = 0.0
    # temp-05.md / 快慢变量：把 lr 视为“可塑性速率”。核心动力学（V、生成元、z）应更慢以减少身份漂移。
    lr_core_scale: float = 0.3
    
    # Parallel Evolution
    num_branches: int = 1   # Stage1 default: keep the training loop deterministic/stable
    branch_noise: float = 0.1 # Exploration noise (disable with 0.0)

    # Resonant Selection training:
    # Instead of updating only the single best branch (argmin F), optionally train on the
    # top-M branches using a Boltzmann (softmin) weighting of their speculative free energy.
    # This increases useful GPU work (more backward compute) and is closer to "expected free energy"
    # style training (Active Inference flavor) while retaining the particle-filter selection
    # semantics for state rollout (we still propagate the best branch state).
    train_top_m: int = 1
    train_top_m_temp: float = 1.0
    
    # A3 Weights
    beta_kl: float = 0.01   # Standard regularization
    gamma_pred: float = 10.0 # Strong signal
    alpha_chart: float = 0.01 # Minimal entropy regularization
    
    # Speed/memory tradeoff
    truncated_bptt_steps: int = 2  # Detach gradients every k evolution steps (0=no truncation)
    truncated_bptt_tokens: int = 0  # Detach across tokens (0=no truncation)

    # Speculative rollout memory control
    speculative_rollout_no_grad: bool = True

    # Performance: optional CUDA Graph capture for the full train step (forward+backward+opt).
    # This does not change the objective; it reduces Python/launch overhead when shapes are static.
    use_cuda_graph: bool = True
    cuda_graph_warmup: int = 3

    # Port stabilization
    port_u_tanh: bool = True
    port_u_scale: float = 1.0
    # Decode from full internal state (q,p,s) to let memory live in momentum/contact scalar.
    # This is still "minimal port" (same interface), but avoids forcing all memory into q only.
    port_decode_state: bool = True

    # Numerics: prefer fail-fast (no projection) for curriculum stages.
    sanitize_nonfinite: bool = False
    strict_nonfinite: bool = True
    # A4 viability as a structural constraint: keep q/p within a safe radius each integrator step.
    # This avoids having the barrier dominate the learning signal (and is CUDA-graph friendly).
    q_clip_norm: float = 25.0
    p_clip_norm: float = 25.0
    s_clip_abs: float = 0.0

    # Damping passed to integrator (group method only)
    damping: float = 0.1

    # Router as slow carrier (B 方案): w̄ dynamics time constant (seconds).
    # 0 disables smoothing (w̄ = w_raw).
    # In Stage1 tasks the context is constant across the whole sequence, so router inertia
    # stabilizes chart usage and reduces destructive switching.
    router_tau: float = 1.0
    # Provide an explicit task-context carrier `c` to the router (理论基础-7 §6.1).
    # This makes mixed-task training well-posed (no ambiguity at t=0).
    router_context_dim: int = 8
    # Disable L2 transport during Stage1 by default (enable later once metrics say it's meaningful).
    transport_threshold: float = 0.0

    # Atlas locality for ports (理论基础-7 §6.1): keep only top-k charts per sample for port coupling + decode.
    # NOTE: period-2 benefits from a small multi-chart activation set; top-2 keeps locality
    # while still allowing a two-phase pattern.
    port_top_k: int = 2
    port_topk_impl: str = "dense"

    # A4: viability barrier (kept inside the single scalar objective)
    viability_q_max: float = 25.0
    viability_p_max: float = 25.0
    lambda_viability: float = 0.0

    # Port small-gain proxy: constrain ||W||_2 to avoid destabilizing the core dynamics.
    port_gain_max: float = 1.5
    # Prefer hard projection (see Stage1Trainer._project_port_gain_) over adding yet another loss weight.
    lambda_port_gain: float = 0.0
    port_gain_pi_iters: int = 2

    # Skill specialization (理论基础-7/训练机制待确认清单.md §6.1.3):
    # Encourage charts to become separable across task context by maximizing MI(task; chart).
    # Default ON for mixed-task stability (prevents router collapse to a single chart across tasks).
    lambda_task_mi: float = 1.0
    
    # Task types
    task_mix: str = "mixed" # Train on all tasks to pass Gate Level 1

    # Diagnostics (off by default; enable via env vars in __main__)
    diag_router: bool = False
    diag_pred_baselines: bool = False
    diag_overlap_gap: float = 0.1  # top2 gap threshold for "overlap event"

    # 快慢变量/诊断协议.md 协议5：端口闭环放大系数（teacher-forced vs open-loop）
    diag_closed_loop: bool = False
    diag_closed_loop_every: int = 50
    diag_closed_loop_len: int = 64
    diag_closed_loop_batch: int = 64
    diag_closed_loop_tail: int = 32
    diag_closed_loop_max_period: int = 8

    # 快慢变量/诊断协议.md 协议1：局部线性化谱隙（近似版，finite-diff JVP）
    diag_spectral_gap: bool = False
    diag_spectral_every: int = 200
    diag_spectral_eps: float = 1e-3
    diag_spectral_vecs: int = 8
    diag_spectral_pi_iters: int = 5

    def __str__(self) -> str:
        # Print a compact config by default; opt into the full dataclass repr when needed.
        if os.getenv("HEI_FULL_CONFIG", "0") == "1":
            return self.__repr__()
        return (
            "Stage1Config("
            f"device={self.device}, steps={self.steps}, batch_size={self.batch_size}, "
            f"seq={self.sequence_len}, vocab={self.vocab_size}, "
            f"per={self.period_min}-{self.period_max}, "
            f"per_mode={self.period_pattern_mode}, "
            f"dim_q={self.dim_q}, dim_z={self.dim_z}, charts={self.num_charts}, "
            f"integrator={self.integrator_method}, dt={self.dt}, evo={self.evolution_steps}, branches={self.num_branches}, "
            f"lr={self.lr}, wd={self.weight_decay}, lr_core={self.lr_core_scale}, "
            f"beta_kl={self.beta_kl}, gamma_pred={self.gamma_pred}, "
            f"q/p_clip={self.q_clip_norm}/{self.p_clip_norm}, "
            f"router_ctx={self.router_context_dim}, router_tau={self.router_tau}, "
            f"port_topk={self.port_top_k}, port_topk_impl={self.port_topk_impl}, "
            f"port_decode_state={int(self.port_decode_state)}, "
            f"port_gain_max={self.port_gain_max}, cuda_graph={self.use_cuda_graph}"
            ")"
        )

class TinySymbolPort(nn.Module):
    """
    Minimal port for Stage 1.
    Maps discrete symbols to/from the internal manifold.
    """
    def __init__(
        self,
        vocab_size: int,
        dim_q: int,
        *,
        num_charts: int = 1,
        u_tanh: bool = True,
        u_scale: float = 1.0,
        decode_state: bool = False,
        decode_top_k: int = 0,
        decode_topk_impl: str = "dense",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_q = dim_q
        self.num_charts = int(num_charts or 1)
        self.u_tanh = bool(u_tanh)
        self.u_scale = float(u_scale)
        self.decode_state = bool(decode_state)
        self.decode_top_k = int(decode_top_k or 0)
        self.decode_topk_impl = str(decode_topk_impl or "dense")
        self.decode_in_dim = (2 * dim_q + 1) if self.decode_state else dim_q
        
        # Read: Symbol -> u (perturbation)
        self.embed = nn.Embedding(vocab_size, dim_q)
        # Write: q -> Logits (chart-conditioned; aligns with "chart as local frame" in 理论基础-7).
        if self.num_charts <= 1:
            self.readout = nn.Linear(self.decode_in_dim, vocab_size)
            self.readout_weight = None
            self.readout_bias = None
        else:
            self.readout = None
            self.readout_weight = nn.Parameter(torch.empty(self.num_charts, vocab_size, self.decode_in_dim))
            self.readout_bias = nn.Parameter(torch.zeros(self.num_charts, vocab_size))
        
        # Init stronger to ensure control authority (the port is the only interface in Stage1).
        nn.init.normal_(self.embed.weight, std=1.0)
        if self.readout is not None:
            nn.init.normal_(self.readout.weight, std=0.1)
            nn.init.zeros_(self.readout.bias)
        else:
            nn.init.normal_(self.readout_weight, std=0.1)
            nn.init.zeros_(self.readout_bias)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        u = self.embed(tokens)
        if self.u_tanh:
            u = torch.tanh(u)
        if self.u_scale != 1.0:
            u = u * self.u_scale
        return u
        
    def decode(self, x: torch.Tensor, chart_weights: torch.Tensor | None = None) -> torch.Tensor:
        if self.readout is not None:
            return self.readout(x)
        # x: [B,D], chart_weights: [B,K]
        B = x.shape[0]
        K = int(self.num_charts)
        logits_k = torch.einsum("bd,kvd->bkv", x, self.readout_weight) + self.readout_bias.unsqueeze(0)  # [B,K,V]
        if chart_weights is None:
            w = x.new_full((B, K), 1.0 / float(K))
        else:
            w = chart_weights
            if self.decode_top_k > 0 and self.decode_top_k < K:
                # Keep only top-k charts per sample (reduces cross-chart gradient leakage).
                k = int(self.decode_top_k)
                if k == 1:
                    idx = w.argmax(dim=1)  # [B]
                    return logits_k.gather(1, idx.view(B, 1, 1).expand(-1, 1, self.vocab_size)).squeeze(1)
                w_vals, idx = torch.topk(w, k=k, dim=1)  # [B,k]
                w_norm = w_vals / w_vals.sum(dim=1, keepdim=True).clamp(min=1e-8)
                w_masked = torch.zeros_like(w)
                w_masked.scatter_(1, idx, w_norm)
                w = w_masked
        return (logits_k * w.unsqueeze(-1)).sum(dim=1)

class SequenceModel(nn.Module):
    """
    Helper module to wrap the sequence loop.
    Implements Parallel Mental Simulation (Resonant Selection).
    """
    def __init__(
        self,
        entity,
        port,
        config,
        task_router_embed: nn.Module | None = None,
        task_u_embed: nn.Module | None = None,
    ):
        super().__init__()
        self.entity = entity
        self.port = port
        self.config = config
        self.task_router_embed = task_router_embed
        self.task_u_embed = task_u_embed
        
    def forward(
        self,
        inputs,
        targets,
        initial_state_flat,
        task_ids,
        branch_noise_buf: torch.Tensor | None = None,
        period_len: torch.Tensor | None = None,
    ):
        # inputs: [B, T]
        # targets: [B, T]
        # initial_state_flat: [B, D]
        # task_ids: [B, T]
        
        B = inputs.shape[0]
        K = getattr(self.config, 'num_branches', 1)
        D = initial_state_flat.shape[1]
        
        state_flat = initial_state_flat
        total_F = torch.zeros((), device=inputs.device)
        total_acc = torch.zeros((), device=inputs.device)
        total_V = torch.zeros((), device=inputs.device)
        total_pred_ce = torch.zeros((), device=inputs.device)
        
        # Per-task accuracy accumulators
        # 0: Counting, 1: Period, 2: Constant
        task_acc_sum = torch.zeros(3, device=inputs.device)
        task_counts = torch.zeros(3, device=inputs.device)
        # Token-position breakdown (t=0 vs t>=1) to study "memory" vs "cold-start" effects.
        task_acc_sum_t0 = torch.zeros(3, device=inputs.device)
        task_counts_t0 = torch.zeros(3, device=inputs.device)
        task_acc_sum_t1 = torch.zeros(3, device=inputs.device)
        task_counts_t1 = torch.zeros(3, device=inputs.device)
        # Period warm-start: for a cycle of length P, the first unambiguous next-token prediction
        # happens at t=P-1 (after observing a full cycle). We track accuracy on tokens where
        # t >= (P-1) for Period samples; other tasks include all tokens.
        task_acc_sum_tw = torch.zeros(3, device=inputs.device)
        task_counts_tw = torch.zeros(3, device=inputs.device)
        
        # Chart usage accumulator used for entropy regularization.
        # Important: keep it tied only to the SELECTED branch to avoid exploding
        # the autograd graph across all K branches.
        total_chart_weights = torch.zeros(self.config.num_charts, device=inputs.device)

        collect_router_diag = bool(getattr(self.config, "diag_router", False))
        collect_baseline_diag = bool(getattr(self.config, "diag_pred_baselines", False))
        lambda_task_mi = float(getattr(self.config, "lambda_task_mi", 0.0) or 0.0)
        collect_task_mi = lambda_task_mi > 0.0

        # Router/task diagnostics (aligned with 理论基础-7/训练机制待确认清单.md §6.1)
        collect_task_chart_stats = collect_router_diag or collect_task_mi
        collect_task_counts = collect_task_chart_stats or collect_baseline_diag
        task_chart_weight_sum = None
        task_token_counts = None
        prev_chart_idx = None
        switch_count = None
        switch_den = None
        top2_gap_sum = None
        overlap_count = None
        overlap_den = None
        if collect_task_chart_stats:
            task_chart_weight_sum = torch.zeros(3, self.config.num_charts, device=inputs.device)
        if collect_task_counts:
            task_token_counts = torch.zeros(3, device=inputs.device)
        if collect_router_diag:
            switch_count = torch.zeros((), device=inputs.device)
            switch_den = torch.zeros((), device=inputs.device)
            top2_gap_sum = torch.zeros((), device=inputs.device)
            overlap_count = torch.zeros((), device=inputs.device)
            overlap_den = torch.zeros((), device=inputs.device)

        # Prediction baseline diagnostics (what heuristic the model is collapsing to)
        task_match_prev_sum = None
        task_match_cur_sum = None
        task_match_plus1_sum = None
        if collect_baseline_diag:
            task_match_prev_sum = torch.zeros(3, device=inputs.device)
            task_match_cur_sum = torch.zeros(3, device=inputs.device)
            task_match_plus1_sum = torch.zeros(3, device=inputs.device)

        # Router state (EMA weights) carried across tokens (detached).
        router_state = None
        
        speculative_no_grad = bool(getattr(self.config, 'speculative_rollout_no_grad', True))
        train_top_m = int(getattr(self.config, "train_top_m", 1) or 1)
        train_top_m = max(1, train_top_m)
        train_top_m_temp = float(getattr(self.config, "train_top_m_temp", 1.0) or 1.0)
        if train_top_m_temp <= 0.0:
            train_top_m_temp = 1.0
        # Viability shaping: align branch selection with the unified scalar objective by
        # penalizing branches that already violate the A4 viability bounds.
        viab_q_max = float(getattr(self.config, "viability_q_max", 0.0) or 0.0)
        viab_p_max = float(getattr(self.config, "viability_p_max", 0.0) or 0.0)
        lambda_viability = float(getattr(self.config, "lambda_viability", 0.0) or 0.0)
        use_viability_in_selection = lambda_viability > 0.0 and (viab_q_max > 0.0 or viab_p_max > 0.0)

        for t in range(self.config.sequence_len):
            token_in = inputs[:, t] # [B]
            target_t = targets[:, t] # [B]
            task_t = task_ids[:, t] # [B]
            
            # 1. Encode Input
            u = self.port.encode(token_in) # [B, Du]
            # Condition the port drive on external context `c` (here: task id).
            # This aligns with 理论基础-7 §6.1: skills are chart/behavior patterns conditioned on c,
            # and it removes the under-determinacy at t=0 under mixed tasks.
            if self.task_u_embed is not None:
                u = u + torch.tanh(self.task_u_embed(task_t))
            router_ctx = None
            if self.task_router_embed is not None:
                router_ctx = self.task_router_embed(task_t)  # [B, C]
            
            # 2. Expand & Perturb (Mental Simulation)
            # [B, D] -> [B, K, D]
            state_k_init = state_flat.unsqueeze(1).expand(-1, K, -1).clone()
            u_k = u.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
            router_ctx_k = None
            if router_ctx is not None:
                router_ctx_k = router_ctx.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)

            # Add noise to q component of state to encourage exploration.
            # In CUDA graph mode we may supply `branch_noise_buf` so randomness happens OUTSIDE capture.
            dim_q = self.entity.dim_q
            if K > 1:
                branch_noise = float(getattr(self.config, 'branch_noise', 0.0) or 0.0)
                if branch_noise != 0.0:
                    if branch_noise_buf is None:
                        noise = torch.randn(B, K, dim_q, device=inputs.device) * branch_noise
                    else:
                        noise = branch_noise_buf
                    state_k_init[:, :, :dim_q] = state_k_init[:, :, :dim_q] + noise

            # Flatten for batch processing
            state_k_flat_init = state_k_init.reshape(B * K, D)
            
            # B方案：使用上一步的 w̄ 作为慢载体输入（不依赖 task_id）。
            prev_weights_input = router_state
            prev_weights_input_k = None
            if prev_weights_input is not None:
                prev_weights_input_k = prev_weights_input.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)

            # 3. Parallel Evolution (two-phase to avoid keeping graphs for ALL branches)
            # Phase A (speculative): evolve all K branches WITHOUT creating autograd graphs.
            # Phase B (train): re-run the selected top-M branches WITH gradients (default M=1).
            current_step_weights = None
            state_train_multi = None
            f_topm_ng = None
            m_used = 1

            trunc_k = getattr(self.config, 'truncated_bptt_steps', 0) or self.config.evolution_steps
            if K > 1 and speculative_no_grad:
                # ---- Phase A: speculative rollout (no grad) ----
                with torch.no_grad():
                    state_ng = state_k_flat_init
                    weights_ng = prev_weights_input_k
                    dead_ng = torch.zeros(B * K, device=inputs.device, dtype=torch.bool)
                    chart_w_ng = None
                    for _ in range(self.config.evolution_steps):
                        out_ng = self.entity.forward_tensor(
                            state_flat=state_ng,
                            u_dict={'symbolic': u_k},
                            dt=self.config.dt,
                            prev_chart_weights=weights_ng,
                            prediction_error=None,
                            detach_next_prev_weights=True,
                            compute_action=False,
                            router_context=router_ctx_k,
                            strict_nonfinite_override=False,
                            sanitize_nonfinite_override=True,
                            return_finite_mask=True,
                            skip_free_energy=True,
                        )
                        state_ng = out_ng['next_state_flat']
                        chart_w_ng = out_ng.get("chart_weights", None)
                        weights_ng = out_ng['next_prev_chart_weights']
                        dead_ng = dead_ng | (~out_ng["finite_mask"])

                    q_ng = state_ng[:, :dim_q]
                    x_ng = state_ng if getattr(self.port, "decode_state", False) else q_ng
                    logits_ng = self.port.decode(x_ng, chart_weights=chart_w_ng)  # [B*K,V]
                    target_k_ng = target_t.unsqueeze(1).expand(-1, K).reshape(-1)  # [B*K]
                    pred_error_ng = F.cross_entropy(logits_ng, target_k_ng, reduction='none')

                    z_batch_ng = self.entity.z.expand(B * K, -1)
                    V_inp_ng = torch.cat([q_ng, z_batch_ng], dim=1)
                    V_k_ng = self.entity.net_V(V_inp_ng).squeeze(-1)
                    if self.entity.internal_gen.stiffness > 0:
                        V_k_ng = V_k_ng + 0.5 * self.entity.internal_gen.stiffness * (q_ng ** 2).sum(dim=1)
                    KL_val_ng = 0.5 * (self.entity.z ** 2).sum() * self.config.beta_kl
                    F_total_ng = V_k_ng + KL_val_ng + self.config.gamma_pred * pred_error_ng
                    if use_viability_in_selection:
                        p_ng = state_ng[:, dim_q : 2 * dim_q]
                        viab_pen = torch.zeros_like(F_total_ng)
                        if viab_q_max > 0.0:
                            viab_pen = viab_pen + torch.relu(q_ng.norm(dim=1) - viab_q_max).pow(2)
                        if viab_p_max > 0.0:
                            viab_pen = viab_pen + torch.relu(p_ng.norm(dim=1) - viab_p_max).pow(2)
                        viab_pen = torch.nan_to_num(viab_pen, nan=1e9, posinf=1e9, neginf=1e9)
                        F_total_ng = F_total_ng + float(lambda_viability) * viab_pen
                    # Treat invalid branches as extremely costly so selection remains stable.
                    F_total_ng = torch.nan_to_num(F_total_ng, nan=1e9, posinf=1e9, neginf=1e9)
                    F_total_ng = F_total_ng + dead_ng.to(dtype=F_total_ng.dtype) * 1e9
                    F_total_ng = F_total_ng.view(B, K)
                    m_used = min(int(train_top_m), int(K))
                    # Pick top-M lowest-energy branches per sample (sorted).
                    topm = torch.topk(F_total_ng, k=m_used, dim=1, largest=False)
                    best_indices = topm.indices[:, 0]
                    f_topm_ng = topm.values  # [B,M]

                # Gather initial state(s) of selected branch(es) so grad-run matches the chosen branch.
                batch_indices = torch.arange(B, device=inputs.device)
                if m_used == 1:
                    state_sel_flat = state_k_init[batch_indices, best_indices].contiguous()  # [B, D]
                else:
                    topm_idx = topm.indices  # [B,M]
                    b_idx = batch_indices.unsqueeze(1).expand(-1, m_used)  # [B,M]
                    state_sel_flat = state_k_init[b_idx, topm_idx].contiguous().reshape(B * m_used, D)  # [B*M,D]

                # ---- Phase B: train rollout (grad enabled) ----
                state_train = state_sel_flat
                current_step_weights = None
                u_train = u
                router_ctx_train = router_ctx
                weights_train = prev_weights_input
                if m_used > 1:
                    u_train = u.unsqueeze(1).expand(-1, m_used, -1).reshape(B * m_used, -1)
                    if router_ctx is not None:
                        router_ctx_train = router_ctx.unsqueeze(1).expand(-1, m_used, -1).reshape(B * m_used, -1)
                    if prev_weights_input is not None:
                        weights_train = prev_weights_input.unsqueeze(1).expand(-1, m_used, -1).reshape(B * m_used, -1)
                for evo_i in range(self.config.evolution_steps):
                    out = self.entity.forward_tensor(
                        state_flat=state_train,
                        u_dict={'symbolic': u_train},
                        dt=self.config.dt,
                        prev_chart_weights=weights_train,
                        prediction_error=None,
                        detach_next_prev_weights=True,
                        compute_action=False,
                        router_context=router_ctx_train,
                        skip_free_energy=True,
                    )
                    state_train = out['next_state_flat']
                    current_step_weights = out['chart_weights']  # [B, Charts]
                    weights_train = out['next_prev_chart_weights']  # detached router state
                    if trunc_k > 0 and (evo_i + 1) % trunc_k == 0 and (evo_i + 1) < self.config.evolution_steps:
                        state_train = state_train.detach().requires_grad_(True)

                if m_used == 1:
                    state_flat = state_train
                    selected_weights = current_step_weights
                    router_state = weights_train
                else:
                    # Keep the full top-M terminal states for a weighted training objective,
                    # but propagate ONLY the best branch state/weights to the next token.
                    state_train_multi = state_train  # [B*M,D]
                    state_view = state_train_multi.view(B, m_used, D)
                    state_flat = state_view[:, 0, :]

                    w_view = current_step_weights.view(B, m_used, -1) if current_step_weights is not None else None
                    selected_weights = w_view[:, 0, :] if w_view is not None else None

                    prev_view = weights_train.view(B, m_used, -1) if weights_train is not None else None
                    router_state = prev_view[:, 0, :] if prev_view is not None else None
            else:
                # Single-phase (legacy): evolve all branches with grad.
                state_k_flat = state_k_flat_init
                current_step_weights = None
                weights_k = prev_weights_input_k
                for evo_i in range(self.config.evolution_steps):
                    out = self.entity.forward_tensor(
                        state_flat=state_k_flat,
                        u_dict={'symbolic': u_k},
                        dt=self.config.dt,
                        prev_chart_weights=weights_k,
                        prediction_error=None,
                        detach_next_prev_weights=True,
                        compute_action=False,
                        router_context=router_ctx_k,
                        skip_free_energy=True,
                    )
                    state_k_flat = out['next_state_flat']
                    current_step_weights = out['chart_weights']  # [B*K, Charts]
                    weights_k = out['next_prev_chart_weights']
                    if trunc_k > 0 and (evo_i + 1) % trunc_k == 0 and (evo_i + 1) < self.config.evolution_steps:
                        state_k_flat = state_k_flat.detach().requires_grad_(True)
            
            # 4. Evaluate + (optional) selection for legacy path
            if K > 1 and speculative_no_grad:
                if m_used == 1 or state_train_multi is None or f_topm_ng is None:
                    # Selected branch already evolved with grad; evaluate on selected only.
                    q_sel = state_flat[:, :dim_q]
                    x_sel = state_flat if getattr(self.port, "decode_state", False) else q_sel
                    logits_sel = self.port.decode(x_sel, chart_weights=selected_weights)  # [B,V]
                    selected_error = F.cross_entropy(logits_sel, target_t, reduction='none')

                    z_batch = self.entity.z.expand(B, -1)
                    V_inp = torch.cat([q_sel, z_batch], dim=1)
                    V_sel = self.entity.net_V(V_inp).squeeze(-1)
                    if self.entity.internal_gen.stiffness > 0:
                        V_sel = V_sel + 0.5 * self.entity.internal_gen.stiffness * (q_sel ** 2).sum(dim=1)
                    KL_val = 0.5 * (self.entity.z ** 2).sum() * self.config.beta_kl
                    selected_F_int = V_sel + KL_val

                    selected_logits = logits_sel
                else:
                    # Evaluate all top-M branches (grad-enabled) and use a Boltzmann-weighted
                    # expected free energy as the training signal while keeping metrics on the best branch.
                    M = int(m_used)
                    q_multi = state_train_multi[:, :dim_q]  # [B*M,Dq]
                    x_multi = state_train_multi if getattr(self.port, "decode_state", False) else q_multi
                    logits_multi = self.port.decode(x_multi, chart_weights=current_step_weights)  # [B*M,V]
                    logits_multi_v = logits_multi.view(B, M, -1)
                    target_multi = target_t.unsqueeze(1).expand(-1, M).reshape(-1)
                    pred_error_multi = F.cross_entropy(logits_multi, target_multi, reduction='none').view(B, M)

                    z_batch = self.entity.z.expand(B * M, -1)
                    V_inp = torch.cat([q_multi, z_batch], dim=1)
                    V_multi = self.entity.net_V(V_inp).squeeze(-1)
                    if self.entity.internal_gen.stiffness > 0:
                        V_multi = V_multi + 0.5 * self.entity.internal_gen.stiffness * (q_multi ** 2).sum(dim=1)
                    KL_val = 0.5 * (self.entity.z ** 2).sum() * self.config.beta_kl
                    F_int_multi = (V_multi + KL_val).view(B, M)

                    F_total_multi = F_int_multi + self.config.gamma_pred * pred_error_multi
                    # weights from speculative energies (no-grad), temperature-scaled.
                    w = torch.softmax(-f_topm_ng / float(train_top_m_temp), dim=1)
                    selected_F_int = (w * F_int_multi).sum(dim=1)
                    selected_error = (w * pred_error_multi).sum(dim=1)

                    # Metrics on the best branch (first in top-k list).
                    selected_logits = logits_multi_v[:, 0, :]
            else:
                # Legacy path: evaluate all K, select by internal energy.
                q_k = state_k_flat[:, :dim_q] # [B*K, dim_q]
                x_k = state_k_flat if getattr(self.port, "decode_state", False) else q_k
                logits_k = self.port.decode(x_k, chart_weights=current_step_weights) # [B*K, V]

                target_k = target_t.unsqueeze(1).expand(-1, K).reshape(-1)
                pred_error_k = F.cross_entropy(logits_k, target_k, reduction='none')

                z_batch = self.entity.z.expand(B*K, -1)
                V_inp = torch.cat([q_k, z_batch], dim=1)
                V_k = self.entity.net_V(V_inp).squeeze(-1) # [B*K]
                if self.entity.internal_gen.stiffness > 0:
                    V_k = V_k + 0.5 * self.entity.internal_gen.stiffness * (q_k ** 2).sum(dim=1)

                KL_val = 0.5 * (self.entity.z ** 2).sum() * self.config.beta_kl
                F_internal = V_k + KL_val # [B*K]
                F_total = F_internal + self.config.gamma_pred * pred_error_k
                if K > 1 and use_viability_in_selection:
                    p_k = state_k_flat[:, dim_q : 2 * dim_q]
                    viab_pen = torch.zeros_like(F_total)
                    if viab_q_max > 0.0:
                        viab_pen = viab_pen + torch.relu(q_k.norm(dim=1) - viab_q_max).pow(2)
                    if viab_p_max > 0.0:
                        viab_pen = viab_pen + torch.relu(p_k.norm(dim=1) - viab_p_max).pow(2)
                    viab_pen = torch.nan_to_num(viab_pen, nan=1e9, posinf=1e9, neginf=1e9)
                    F_total = F_total + float(lambda_viability) * viab_pen
                best_indices = torch.argmin(F_total.view(B, K), dim=1) # [B]

                batch_indices = torch.arange(B, device=inputs.device)
                flat_indices = batch_indices * K + best_indices

                state_flat = state_k_flat[flat_indices] # [B, D]

                if current_step_weights is not None:
                    selected_weights = current_step_weights[flat_indices] # [B, Charts]
                else:
                    selected_weights = None
                if weights_k is not None:
                    router_state = weights_k[flat_indices]  # detached
                else:
                    router_state = None

                selected_F_int = F_internal[flat_indices]
                selected_logits = logits_k[flat_indices]
                selected_error = pred_error_k[flat_indices]

            # Accumulate chart usage from the selected path only (for entropy regularization)
            if selected_weights is not None:
                total_chart_weights = total_chart_weights + selected_weights.sum(dim=0)
            
            # Total Free Energy (Training Objective) = Internal F + Gamma * Prediction Error
            # This forces V to become a proxy for "Expected Success"
            selected_F_total = selected_F_int + self.config.gamma_pred * selected_error
            
            # Accumulate
            total_F = total_F + selected_F_total.mean()
            total_V = total_V + (selected_F_int - KL_val).mean()
            total_pred_ce = total_pred_ce + selected_error.mean()
            
            # Accuracy
            correct = (selected_logits.argmax(dim=-1) == target_t).float()
            acc = correct.mean()
            total_acc = total_acc + acc
            
            # Per-task accuracy
            # Avoid GPU→CPU sync from `mask.any()` by using vectorized masking.
            task_onehot = F.one_hot(task_t, num_classes=3).to(dtype=correct.dtype)  # [B,3]
            task_acc_sum = task_acc_sum + (correct.unsqueeze(1) * task_onehot).sum(dim=0)
            task_counts = task_counts + task_onehot.sum(dim=0)
            if t == 0:
                task_acc_sum_t0 = task_acc_sum_t0 + (correct.unsqueeze(1) * task_onehot).sum(dim=0)
                task_counts_t0 = task_counts_t0 + task_onehot.sum(dim=0)
            else:
                task_acc_sum_t1 = task_acc_sum_t1 + (correct.unsqueeze(1) * task_onehot).sum(dim=0)
                task_counts_t1 = task_counts_t1 + task_onehot.sum(dim=0)

            # Warm-start accuracy (Period only uses per-sample P; others include all tokens).
            if period_len is None:
                warm_ok = (t > 0)  # fallback: match Per1 semantics
                warm_mask = correct.new_full((B, 1), float(warm_ok))
            else:
                p = period_len
                if p.dim() > 1:
                    p = p.view(-1)
                else:
                    p = p
                # Non-period tasks: always include. Period task: include only when t >= P-1.
                warm_ok_vec = (task_t != 1) | (t >= (p - 1))
                warm_mask = warm_ok_vec.to(dtype=correct.dtype).unsqueeze(1)  # [B,1]
            task_acc_sum_tw = task_acc_sum_tw + (correct.unsqueeze(1) * task_onehot * warm_mask).sum(dim=0)
            task_counts_tw = task_counts_tw + (task_onehot * warm_mask).sum(dim=0)
            if task_token_counts is not None:
                task_token_counts = task_token_counts + task_onehot.sum(dim=0).to(dtype=task_token_counts.dtype)

            # Diagnostics: task↔chart statistics, switch rate, overlap events.
            if collect_task_chart_stats and (selected_weights is not None):
                w = selected_weights  # [B,Charts]
                task_chart_weight_sum = task_chart_weight_sum + task_onehot.t().to(dtype=w.dtype) @ w

                if collect_router_diag:
                    chart_idx = w.argmax(dim=1)  # [B]
                    if prev_chart_idx is not None:
                        switch_count = switch_count + (chart_idx != prev_chart_idx).to(dtype=w.dtype).sum()
                        switch_den = switch_den + chart_idx.numel()
                    prev_chart_idx = chart_idx

                    top2 = torch.topk(w, k=2, dim=1).values  # [B,2]
                    gap = top2[:, 0] - top2[:, 1]
                    top2_gap_sum = top2_gap_sum + gap.sum()
                    gap_thr = float(getattr(self.config, "diag_overlap_gap", 0.1) or 0.1)
                    overlap_count = overlap_count + (gap < gap_thr).to(dtype=w.dtype).sum()
                    overlap_den = overlap_den + gap.numel()

            # Diagnostics: does the model behave like copy / prev-token / +1 baseline?
            if collect_baseline_diag:
                pred_tok = selected_logits.argmax(dim=-1)  # [B]
                tok_cur = token_in
                tok_plus1 = (tok_cur + 1) % int(self.config.vocab_size)
                match_cur = (pred_tok == tok_cur).to(dtype=correct.dtype)
                match_plus1 = (pred_tok == tok_plus1).to(dtype=correct.dtype)
                if t > 0:
                    tok_prev = inputs[:, t - 1]
                    match_prev = (pred_tok == tok_prev).to(dtype=correct.dtype)
                else:
                    match_prev = torch.zeros_like(match_cur)

                task_match_prev_sum = task_match_prev_sum + (match_prev.unsqueeze(1) * task_onehot).sum(dim=0)
                task_match_cur_sum = task_match_cur_sum + (match_cur.unsqueeze(1) * task_onehot).sum(dim=0)
                task_match_plus1_sum = task_match_plus1_sum + (match_plus1.unsqueeze(1) * task_onehot).sum(dim=0)
            
        # Normalize per-task acc
        per_task_acc = task_acc_sum / (task_counts + 1e-8)
        per_task_acc_t0 = task_acc_sum_t0 / (task_counts_t0 + 1e-8)
        per_task_acc_t1 = task_acc_sum_t1 / (task_counts_t1 + 1e-8)
        per_task_acc_tw = task_acc_sum_tw / (task_counts_tw + 1e-8)

        diag = None
        mi_task_chart = None
        if task_chart_weight_sum is not None and task_token_counts is not None:
            joint = task_chart_weight_sum / (task_chart_weight_sum.sum() + 1e-8)
            p_task = joint.sum(dim=1, keepdim=True)
            p_chart = joint.sum(dim=0, keepdim=True)
            mi_task_chart = (
                joint
                * (
                    torch.log(joint + 1e-8)
                    - torch.log(p_task + 1e-8)
                    - torch.log(p_chart + 1e-8)
                )
            ).sum()
            mi_task_chart = mi_task_chart.clamp(min=0.0)

        if collect_router_diag or collect_baseline_diag or collect_task_mi:
            diag = {
                "task_chart_weight_sum": task_chart_weight_sum,
                "task_token_counts": task_token_counts,
                "switch_count": switch_count,
                "switch_den": switch_den,
                "top2_gap_sum": top2_gap_sum,
                "overlap_count": overlap_count,
                "overlap_den": overlap_den,
                "task_match_prev_sum": task_match_prev_sum,
                "task_match_cur_sum": task_match_cur_sum,
                "task_match_plus1_sum": task_match_plus1_sum,
                "mean_V": total_V / float(self.config.sequence_len),
                "mean_pred_ce": total_pred_ce / float(self.config.sequence_len),
                "mi_task_chart": mi_task_chart,
            }

        if mi_task_chart is None:
            mi_task_chart = torch.zeros((), device=inputs.device)
        return (
            total_F,
            total_acc,
            total_chart_weights,
            state_flat,
            per_task_acc,
            per_task_acc_t0,
            per_task_acc_t1,
            per_task_acc_tw,
            mi_task_chart,
            total_V,
            total_pred_ce,
            diag,
        )

class Stage1Trainer(BaseCurriculumTrainer):
    def __init__(self, config: Stage1Config):
        super().__init__(config)
        self.config = config

        self._period_patterns: dict[int, torch.Tensor] = {}
        if str(getattr(self.config, "period_pattern_mode", "binary")) == "binary":
            p_min = max(2, int(getattr(self.config, "period_min", 2) or 2))
            p_max = max(p_min, int(getattr(self.config, "period_max", 5) or 5))
            for p in range(p_min, p_max + 1):
                pats = get_primitive_binary_patterns(p).to(device=self.config.device, non_blocking=True)
                self._period_patterns[p] = pats
        
        # Add the minimal port
        self.port = TinySymbolPort(
            config.vocab_size,
            config.dim_q,
            num_charts=config.num_charts,
            u_tanh=getattr(config, 'port_u_tanh', True),
            u_scale=float(getattr(config, 'port_u_scale', 1.0)),
            decode_state=bool(getattr(config, "port_decode_state", False)),
            decode_top_k=int(getattr(config, "port_top_k", 0) or 0),
            decode_topk_impl=str(getattr(config, "port_topk_impl", "dense") or "dense"),
        ).to(config.device)
        self.entity.add_interface('symbolic', config.dim_q)

        # Explicit task context injection into the port drive u (external "c" -> internal forcing).
        # Kept minimal: a 3-way embedding added to u.
        self.task_u_embed = nn.Embedding(3, config.dim_q).to(config.device)
        with torch.no_grad():
            # Match token-embedding scale so the context carrier is not trivially ignored.
            nn.init.normal_(self.task_u_embed.weight, std=1.0)

        # Optional: give the atlas router an explicit task context (c) as input.
        # This aligns Stage1 with 理论基础-7/训练机制待确认清单.md §6.1 where skills are chart patterns conditioned on c.
        router_ctx_dim = int(getattr(config, "router_context_dim", 0) or 0)
        self.task_router_embed = None
        if router_ctx_dim > 0:
            self.task_router_embed = nn.Embedding(3, router_ctx_dim).to(config.device)
        
        # Optimizer: explicit stability–plasticity split (temp-05.md / 理论基础-7 快慢变量).
        # Keep core dynamics (V, internal generator, z) slower; keep skill-bearing carriers (ports/router) faster.
        lr = float(config.lr)
        weight_decay = float(getattr(config, "weight_decay", 0.0) or 0.0)
        lr_core_scale = float(getattr(config, "lr_core_scale", 1.0) or 1.0)
        lr_core = lr * lr_core_scale

        core_ids = {id(p) for p in self.entity.net_V.parameters()}
        core_ids.update(id(p) for p in self.entity.internal_gen.parameters())
        if hasattr(self.entity, "z"):
            core_ids.add(id(self.entity.z))

        core_params: list[torch.nn.Parameter] = []
        fast_entity_params: list[torch.nn.Parameter] = []
        for p in self.entity.parameters():
            if not p.requires_grad:
                continue
            (core_params if id(p) in core_ids else fast_entity_params).append(p)

        fast_params = fast_entity_params + list(self.port.parameters()) + list(self.task_u_embed.parameters())
        if self.task_router_embed is not None:
            fast_params = fast_params + list(self.task_router_embed.parameters())

        param_groups = []
        if core_params:
            param_groups.append({"params": core_params, "lr": lr_core})
        if fast_params:
            param_groups.append({"params": [p for p in fast_params if p.requires_grad], "lr": lr})

        self._train_params = [p for g in param_groups for p in g["params"] if p.requires_grad]
        use_cudagraph = bool(getattr(config, "use_cuda_graph", False)) and torch.cuda.is_available() and str(config.device).startswith("cuda")
        self.optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            capturable=use_cudagraph,
            foreach=True if use_cudagraph else None,
        )
        
        # Prepare compiled model
        self.seq_model = SequenceModel(
            self.entity,
            self.port,
            config,
            task_router_embed=self.task_router_embed,
            task_u_embed=self.task_u_embed,
        )
        
        # Enable TF32 for speed
        torch.set_float32_matmul_precision('high')
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        
        # Compile if available
        # NOTE: torch.compile with aot_autograd currently fails with double backward (RuntimeError)
        # which is required for Hamiltonian dynamics training (grad(H)).
        # We disable it for now and rely on large batch sizes for GPU saturation.
        if hasattr(torch, 'compile') and False: # Disabled
            print("Compiling sequence model with torch.compile...")
            # mode='reduce-overhead' is best for small batches/loops, 
            # but 'default' is safer. Let's try default first.
            self.seq_model = torch.compile(self.seq_model)
            
        # Enable TF32 for speed
        torch.set_float32_matmul_precision('high')

    def _project_port_gain_(self) -> None:
        """Hard small-gain projection for port coupling matrices (spectral-norm cap).

        Keeps the training dynamics stable without adding more weighted loss terms.
        Uses a short, deterministic power-iteration estimate per chart, suitable for CUDA graph capture.
        """
        gain_max = float(getattr(self.config, "port_gain_max", 0.0) or 0.0)
        if gain_max <= 0.0:
            return
        port_mod = self.entity.generator.ports["symbolic"] if "symbolic" in self.entity.generator.ports else None
        W = getattr(port_mod, "W_stack", None) if port_mod is not None else None
        if W is None:
            return
        with torch.no_grad():
            iters = int(getattr(self.config, "port_gain_pi_iters", 2) or 2)
            iters = max(1, iters)
            # Deterministic init (avoid RNG inside CUDA graph capture).
            v = torch.ones(W.shape[0], W.shape[2], device=W.device, dtype=W.dtype)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
            for _ in range(iters):
                u = torch.einsum("kuq,kq->ku", W, v)
                v = torch.einsum("kuq,ku->kq", W, u)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
            u = torch.einsum("kuq,kq->ku", W, v)
            sigma = u.norm(dim=1)  # [K] approx ||W_k||_2
            scale = (gain_max / (sigma + 1e-8)).clamp(max=1.0)
            W.mul_(scale.view(-1, 1, 1))

    def _clip_grad_norm_capturable_(self, max_norm: float = 1.0, eps: float = 1e-6) -> None:
        """
        Capturable grad clipping (L2) suitable for CUDA graph capture.
        Avoids `.item()` and Python-side conditionals on CUDA tensors.
        """
        max_norm = float(max_norm)
        if max_norm <= 0.0:
            return
        grads = [p.grad for p in self._train_params if p.grad is not None]
        if not grads:
            return
        device = grads[0].device
        total_sq = torch.zeros((), device=device, dtype=grads[0].dtype)
        for g in grads:
            total_sq = total_sq + (g * g).sum()
        total_norm = torch.sqrt(total_sq)
        coef = max_norm / (total_norm + float(eps))
        coef = torch.clamp(coef, max=1.0)
        torch._foreach_mul_(grads, coef)

    def _train_loop_cuda_graph(self) -> None:
        """
        CUDA Graph “GPU saturation” mode: capture the full train step once and replay.

        Theory note: this does NOT change the dynamics/objective; it only reduces
        Python + kernel-launch overhead when shapes are static.
        """
        device = self.config.device
        if (not torch.cuda.is_available()) or (not str(device).startswith("cuda")):
            raise RuntimeError("CUDA graph mode requested but CUDA is not available.")

        # Autograd can execute CUDA backward ops from multiple CPU worker threads.
        # With per-thread default streams, that can introduce cross-stream dependencies
        # (Engine InputBuffer stream-blocking) that are NOT capturable under CUDA graphs.
        # Force single-threaded autograd for the duration of graph capture+replay.
        prev_autograd_mt = None
        if hasattr(torch.autograd, "is_multithreading_enabled") and hasattr(torch.autograd, "set_multithreading_enabled"):
            try:
                prev_autograd_mt = bool(torch.autograd.is_multithreading_enabled())
                if prev_autograd_mt:
                    torch.autograd.set_multithreading_enabled(False)
            except Exception:
                prev_autograd_mt = None

        try:
            print(f"Starting Stage 1 Training (CUDA Graph) on {device}...")
            print(f"Config: {self.config}")
            print(f"A3 Alignment: Optimizing Unified Free Energy F")

            os.makedirs(self.config.save_dir, exist_ok=True)
            best_path = os.path.join(self.config.save_dir, "stage1_best.pt")
            best_perw = -1.0
            best_gate: Dict[str, Any] | None = None
            mode = str(getattr(self.config, "period_pattern_mode", "random"))
            gate_alt = "random" if mode in {"offsets", "random"} else mode

            B = int(self.config.batch_size)
            T = int(self.config.sequence_len)
            dq = int(self.config.dim_q)
            K = int(getattr(self.config, "num_branches", 1) or 1)
            D = 2 * dq + 1

            # Static input buffers (filled each step, then graph replay consumes them).
            inputs_buf = torch.empty((B, T), device=device, dtype=torch.long)
            targets_buf = torch.empty((B, T), device=device, dtype=torch.long)
            task_ids_buf = torch.empty((B, T), device=device, dtype=torch.long)
            period_len_buf = torch.empty((B, 1), device=device, dtype=torch.long)
            init_state_buf = torch.empty((B, D), device=device, dtype=torch.float32)
            # Branch-noise buffer: keeps stochastic exploration outside CUDA-graph capture.
            branch_noise_std = float(getattr(self.config, "branch_noise", 0.0) or 0.0)
            branch_noise_buf = None
            if K > 1 and branch_noise_std != 0.0:
                branch_noise_buf = torch.empty((B, K, dq), device=device, dtype=torch.float32)

            # Static output buffers for logging (updated inside the captured graph).
            loss_buf = torch.zeros((), device=device)
            mean_F_buf = torch.zeros((), device=device)
            mean_V_buf = torch.zeros((), device=device)
            mean_ce_buf = torch.zeros((), device=device)
            mean_acc_buf = torch.zeros((), device=device)
            entropy_buf = torch.zeros((), device=device)
            mi_task_chart_buf = torch.zeros((), device=device)
            per_task_acc_buf = torch.zeros(3, device=device)
            per_task_acc_t1_buf = torch.zeros(3, device=device)
            per_task_acc_tw_buf = torch.zeros(3, device=device)

            alpha = float(getattr(self.config, "alpha_chart", 0.0) or 0.0)
            lambda_task_mi = float(getattr(self.config, "lambda_task_mi", 0.0) or 0.0)
            lambda_viability = float(getattr(self.config, "lambda_viability", 0.0) or 0.0)
            lambda_port_gain = float(getattr(self.config, "lambda_port_gain", 0.0) or 0.0)
            q_max = float(getattr(self.config, "viability_q_max", 0.0) or 0.0)
            p_max = float(getattr(self.config, "viability_p_max", 0.0) or 0.0)
            gain_max = float(getattr(self.config, "port_gain_max", 0.0) or 0.0)

            dbg_params = os.getenv("HEI_DEBUG_CG_PARAMS", "0") == "1"
            p_before = None
            if dbg_params:
                try:
                    # Track a couple of representative parameter storages for CUDA-graph correctness.
                    rw = getattr(self.port, "readout_weight", None)
                    ew = getattr(self.port, "embed", None)
                    ewt = ew.weight if ew is not None else None
                    wstack = None
                    try:
                        port_mod = self.entity.generator.ports["symbolic"] if "symbolic" in self.entity.generator.ports else None
                        wstack = getattr(port_mod, "W_stack", None) if port_mod is not None else None
                    except Exception:
                        wstack = None
                    p_before = {
                        "readout_weight_ptr": int(rw.data_ptr()) if rw is not None else -1,
                        "readout_weight_norm": float(rw.detach().norm().item()) if rw is not None else float("nan"),
                        "embed_weight_ptr": int(ewt.data_ptr()) if ewt is not None else -1,
                        "embed_weight_norm": float(ewt.detach().norm().item()) if ewt is not None else float("nan"),
                        "W_stack_ptr": int(wstack.data_ptr()) if wstack is not None else -1,
                        "W_stack_norm": float(wstack.detach().norm().item()) if wstack is not None else float("nan"),
                    }
                except Exception:
                    p_before = None

            def refill_buffers() -> None:
                inputs, targets, task_ids, period_len = self.generate_batch()
                inputs_buf.copy_(inputs)
                targets_buf.copy_(targets)
                task_ids_buf.copy_(task_ids)
                period_len_buf.copy_(period_len)
                # Fresh initial condition (matches entity.reset() scale, without re-allocations).
                with torch.no_grad():
                    init_state_buf.zero_()
                    init_state_buf[:, :dq].normal_(mean=0.0, std=0.1)
                    init_state_buf[:, dq : 2 * dq].normal_(mean=0.0, std=0.1)
                    if branch_noise_buf is not None:
                        branch_noise_buf.normal_(mean=0.0, std=branch_noise_std)

            def fwd_bwd_step() -> None:
                # Capture forward + backward only; keep optimizer.step() OUTSIDE the graph.
                # This avoids optimizer-specific CUDA-graph corner cases where the replayed
                # step can diverge from eager execution / saved weights.
                self.optimizer.zero_grad(set_to_none=False)
                (
                    total_F,
                    total_acc,
                    chart_usage,
                    final_state_flat,
                    per_task_acc,
                    _per_task_acc_t0,
                    per_task_acc_t1,
                    per_task_acc_tw,
                    mi_task_chart,
                    total_V,
                    total_pred_ce,
                    _diag,
                ) = self.seq_model(inputs_buf, targets_buf, init_state_buf, task_ids_buf, branch_noise_buf, period_len_buf)

                mean_F = total_F / float(T)
                mean_V = total_V / float(T)
                mean_ce = total_pred_ce / float(T)
                mean_acc = total_acc / float(T)

                chart_dist = chart_usage / (chart_usage.sum() + 1e-8)
                entropy = -(chart_dist * torch.log(chart_dist + 1e-8)).sum()

                # A4: viability barrier (soft).
                barrier = torch.zeros((), device=device)
                if q_max > 0.0:
                    q = final_state_flat[:, :dq]
                    barrier = barrier + torch.relu(q.norm(dim=1) - q_max).pow(2).mean()
                if p_max > 0.0:
                    p = final_state_flat[:, dq : 2 * dq]
                    barrier = barrier + torch.relu(p.norm(dim=1) - p_max).pow(2).mean()

                # Port small-gain proxy (optional).
                port_gain_pen = torch.zeros((), device=device)
                if lambda_port_gain > 0.0 and gain_max > 0.0:
                    port_mod = self.entity.generator.ports["symbolic"] if "symbolic" in self.entity.generator.ports else None
                    W = getattr(port_mod, "W_stack", None) if port_mod is not None else None
                    if W is not None:
                        iters = int(getattr(self.config, "port_gain_pi_iters", 2) or 2)
                        v = torch.ones(W.shape[0], W.shape[2], device=W.device, dtype=W.dtype)
                        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
                        for _ in range(max(1, iters)):
                            u = torch.einsum("kuq,kq->ku", W, v)
                            v = torch.einsum("kuq,ku->kq", W, u)
                            v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
                        u = torch.einsum("kuq,kq->ku", W, v)
                        sigma_max = u.norm(dim=1)
                        port_gain_pen = torch.relu(sigma_max - gain_max).pow(2).mean()

                loss = (
                    mean_F
                    - alpha * entropy
                    - lambda_task_mi * mi_task_chart
                    + lambda_viability * barrier
                    + lambda_port_gain * port_gain_pen
                )

                # Export scalars for logging.
                # IMPORTANT: detach to keep logging buffers out of the autograd graph;
                # otherwise `.copy_()` makes the destination require grad and breaks CUDA-graph capture.
                loss_buf.copy_(loss.detach())
                mean_F_buf.copy_(mean_F.detach())
                mean_V_buf.copy_(mean_V.detach())
                mean_ce_buf.copy_(mean_ce.detach())
                mean_acc_buf.copy_(mean_acc.detach())
                entropy_buf.copy_(entropy.detach())
                mi_task_chart_buf.copy_(mi_task_chart.detach())
                per_task_acc_buf.copy_(per_task_acc.detach())
                per_task_acc_t1_buf.copy_(per_task_acc_t1.detach())
                per_task_acc_tw_buf.copy_(per_task_acc_tw.detach())

                loss.backward()
                self._clip_grad_norm_capturable_(1.0)

            warmup = int(getattr(self.config, "cuda_graph_warmup", 3) or 3)
            warmup = max(1, warmup)
            for _ in range(warmup):
                refill_buffers()
                fwd_bwd_step()
                self.optimizer.step()
                self._project_port_gain_()

            torch.cuda.synchronize()
            p_warm = None
            if dbg_params and p_before is not None:
                try:
                    rw = getattr(self.port, "readout_weight", None)
                    ew = getattr(self.port, "embed", None)
                    ewt = ew.weight if ew is not None else None
                    wstack = None
                    try:
                        port_mod = self.entity.generator.ports["symbolic"] if "symbolic" in self.entity.generator.ports else None
                        wstack = getattr(port_mod, "W_stack", None) if port_mod is not None else None
                    except Exception:
                        wstack = None
                    p_warm = {
                        "readout_weight_ptr": int(rw.data_ptr()) if rw is not None else -1,
                        "readout_weight_norm": float(rw.detach().norm().item()) if rw is not None else float("nan"),
                        "embed_weight_ptr": int(ewt.data_ptr()) if ewt is not None else -1,
                        "embed_weight_norm": float(ewt.detach().norm().item()) if ewt is not None else float("nan"),
                        "W_stack_ptr": int(wstack.data_ptr()) if wstack is not None else -1,
                        "W_stack_norm": float(wstack.detach().norm().item()) if wstack is not None else float("nan"),
                    }
                except Exception:
                    p_warm = None
            pool = torch.cuda.graphs.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                fwd_bwd_step()

            start_time = time.time()
            for step in range(1, self.config.steps + 1):
                refill_buffers()
                graph.replay()

                if step % self.config.log_every == 0:
                    # Synchronize only when we need to print scalars.
                    torch.cuda.synchronize()
                    elapsed = time.time() - start_time
                    task_acc_str = f"Cnt={per_task_acc_buf[0].item():.2f} Per={per_task_acc_buf[1].item():.2f} Cst={per_task_acc_buf[2].item():.2f}"
                    task_acc_t1_str = (
                        f"PerW={per_task_acc_tw_buf[1].item():.2f}"
                    )
                    io_dbg = ""
                    if os.getenv("HEI_DEBUG_CG_IO", "0") == "1":
                        try:
                            xsum = int(inputs_buf.sum().item())
                            eq = float((inputs_buf == targets_buf).float().mean().item())
                            t0 = task_ids_buf[:, 0]
                            h0 = int((t0 == 0).sum().item())
                            h1 = int((t0 == 1).sum().item())
                            h2 = int((t0 == 2).sum().item())
                            io_dbg = f" IO(sum={xsum} eq={eq:.3f} tasks=[{h0},{h1},{h2}])"
                        except Exception:
                            io_dbg = " IO(err)"
                    print(
                        f"Step {step}: F={mean_F_buf.item():.4f} "
                        f"Acc={mean_acc_buf.item():.4f} ({task_acc_str}) "
                        f"{task_acc_t1_str} "
                        f"V={mean_V_buf.item():.2f} CE={mean_ce_buf.item():.2f} "
                        f"ChartEnt={entropy_buf.item():.2f} "
                        f"MI_tc={mi_task_chart_buf.item():.3f} "
                        f"Time={elapsed:.1f}s"
                        f"{io_dbg}"
                    )
                    if os.getenv("HEI_DEBUG_CG_POST", "0") == "1":
                        try:
                            with torch.no_grad():
                                (
                                    _total_F,
                                    total_acc_post,
                                    _chart_usage,
                                    _final_state_flat,
                                    per_task_acc_post,
                                    _per_task_acc_t0_post,
                                    per_task_acc_t1_post,
                                    per_task_acc_tw_post,
                                    _mi_task_chart_post,
                                    _total_V_post,
                                    total_pred_ce_post,
                                    _diag_post,
                                ) = self.seq_model(
                                    inputs_buf,
                                    targets_buf,
                                    init_state_buf,
                                    task_ids_buf,
                                    branch_noise_buf,
                                    period_len_buf,
                                )
                                mean_acc_post = float((total_acc_post / float(T)).item())
                                mean_ce_post = float((total_pred_ce_post / float(T)).item())
                                print(
                                    "CGPost:"
                                    f" Acc={mean_acc_post:.4f}"
                                    f" Cnt={float(per_task_acc_post[0].item()):.2f}"
                                    f" Per={float(per_task_acc_post[1].item()):.2f}"
                                    f" PerW={float(per_task_acc_tw_post[1].item()):.2f}"
                                    f" Cst={float(per_task_acc_post[2].item()):.2f}"
                                    f" CE={mean_ce_post:.2f}"
                                )
                        except Exception as e:
                            print(f"CGPost: failed ({type(e).__name__}: {e})")

                # Apply the parameter update AFTER logging so debug comparisons can
                # evaluate the same (pre-step) weights used for the logged metrics.
                self.optimizer.step()
                self._project_port_gain_()

                if step % self.config.log_every == 0:
                    try:
                        gate = self.evaluate_L1_gate(
                            batch_size=2048,
                            repeats=1,
                            seed=0,
                            period_alt_mode=gate_alt,
                            require_constant=False,
                            enable_router_diag=False,
                        )
                        perw = float(gate["Per"]["acc_tw"])
                        if perw > best_perw:
                            best_perw = perw
                            best_gate = gate
                            self.save(best_path)
                            print(f"GateBest: PerW={best_perw:.3f} (Alt={gate_alt})")
                    except Exception as e:
                        print(f"GateEval: failed ({type(e).__name__}: {e})")

            if os.getenv("HEI_DEBUG_CG_VALIDATE", "0") == "1":
                try:
                    with torch.no_grad():
                        (
                            _total_F,
                            total_acc,
                            chart_usage,
                            _final_state_flat,
                            per_task_acc,
                            _per_task_acc_t0,
                            per_task_acc_t1,
                            per_task_acc_tw,
                            mi_task_chart,
                            _total_V,
                            total_pred_ce,
                            _diag,
                        ) = self.seq_model(
                            inputs_buf,
                            targets_buf,
                            init_state_buf,
                            task_ids_buf,
                            branch_noise_buf,
                            period_len_buf,
                        )
                        mean_acc = float((total_acc / float(T)).item())
                        mean_ce = float((total_pred_ce / float(T)).item())
                        chart_dist = chart_usage / (chart_usage.sum() + 1e-8)
                        ent = float((-(chart_dist * torch.log(chart_dist + 1e-8)).sum()).item())
                        mi = float(mi_task_chart.item()) if mi_task_chart is not None else 0.0
                        print(
                            "CGValidate:"
                            f" Acc={mean_acc:.4f}"
                            f" Cnt={float(per_task_acc[0].item()):.2f}"
                            f" Per={float(per_task_acc[1].item()):.2f}"
                            f" PerW={float(per_task_acc_tw[1].item()):.2f}"
                            f" Cst={float(per_task_acc[2].item()):.2f}"
                            f" CE={mean_ce:.2f}"
                            f" ChartEnt={ent:.2f}"
                            f" MI_tc={mi:.3f}"
                        )
                except Exception as e:
                    print(f"CGValidate: failed ({type(e).__name__}: {e})")

            if dbg_params and p_before is not None:
                try:
                    rw = getattr(self.port, "readout_weight", None)
                    ew = getattr(self.port, "embed", None)
                    ewt = ew.weight if ew is not None else None
                    wstack = None
                    try:
                        port_mod = self.entity.generator.ports["symbolic"] if "symbolic" in self.entity.generator.ports else None
                        wstack = getattr(port_mod, "W_stack", None) if port_mod is not None else None
                    except Exception:
                        wstack = None
                    p_after = {
                        "readout_weight_ptr": int(rw.data_ptr()) if rw is not None else -1,
                        "readout_weight_norm": float(rw.detach().norm().item()) if rw is not None else float("nan"),
                        "embed_weight_ptr": int(ewt.data_ptr()) if ewt is not None else -1,
                        "embed_weight_norm": float(ewt.detach().norm().item()) if ewt is not None else float("nan"),
                        "W_stack_ptr": int(wstack.data_ptr()) if wstack is not None else -1,
                        "W_stack_norm": float(wstack.detach().norm().item()) if wstack is not None else float("nan"),
                    }
                    print(f"CGParams(before)={p_before}")
                    if p_warm is not None:
                        print(f"CGParams(warm )={p_warm}")
                    print(f"CGParams(after )={p_after}")
                except Exception as e:
                    print(f"CGParams: failed ({type(e).__name__}: {e})")

            if os.path.exists(best_path):
                try:
                    self.load(best_path)
                    if best_gate is not None:
                        print(
                            f"Loaded best checkpoint: PerW={best_perw:.3f} "
                            f"Cnt={best_gate['Cnt']['acc']:.3f} Per={best_gate['Per']['acc']:.3f} "
                            f"(Alt={best_gate['period_alt_mode']})"
                        )
                except Exception as e:
                    print(f"Best checkpoint load failed ({type(e).__name__}: {e})")
            save_path = os.path.join(self.config.save_dir, "stage1_final.pt")
            self.save(save_path)
            print(f"Stage 1 Complete. Saved to {save_path}")
        finally:
            if prev_autograd_mt is not None:
                try:
                    torch.autograd.set_multithreading_enabled(prev_autograd_mt)
                except Exception:
                    pass

    def generate_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate synthetic data on the fly (on GPU).
        Vectorized implementation for high throughput.
        Returns: (inputs, targets, task_ids, period_len) with shapes [B,T], [B,T], [B,T], [B,1].
        """
        B = self.config.batch_size
        T = self.config.sequence_len
        V = self.config.vocab_size
        device = self.config.device
        
        # [B, 1]
        starts = torch.randint(0, V, (B, 1), device=device)
        
        if self.config.task_mix == "counting":
            task_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
        elif self.config.task_mix == "period":
            task_ids = torch.ones((B, 1), dtype=torch.long, device=device)
        elif self.config.task_mix == "constant": # Added for completeness
            task_ids = torch.full((B, 1), 2, dtype=torch.long, device=device)
        else: # mixed (balanced per batch to reduce metric variance / task interference)
            # Period(2–5) is the hardest L1 skill; oversample it to encourage true multi-step memory.
            n1 = B // 2
            n0 = B // 4
            n2 = B - n0 - n1
            task_ids = torch.cat(
                [
                    torch.zeros((n0, 1), dtype=torch.long, device=device),
                    torch.ones((n1, 1), dtype=torch.long, device=device),
                    torch.full((n2, 1), 2, dtype=torch.long, device=device),
                ],
                dim=0,
            )
            task_ids = task_ids[torch.randperm(B, device=device)]
        
        # [1, T+1]
        time_steps = torch.arange(T + 1, device=device).unsqueeze(0) # [1, T+1]
        
        # 1. Counting: (start + t) % V
        # starts: [B, 1] -> broadcast to [B, T+1]
        seq_count = (starts + time_steps) % V
        
        # 2. Periodic cycle: period length sampled in [period_min, period_max] for task==Period.
        p_min = int(getattr(self.config, "period_min", 2) or 2)
        p_max = int(getattr(self.config, "period_max", 5) or 5)
        p_min = max(2, p_min)
        p_max = max(p_min, p_max)
        # Use a fixed max pattern length for tensorization.
        pat_len = p_max

        # Period length per sample (only relevant when task_id==1).
        period_len = torch.full((B, 1), p_min, dtype=torch.long, device=device)
        if p_max > p_min:
            period_len_rand = torch.randint(p_min, p_max + 1, (B, 1), device=device, dtype=torch.long)
            period_len = torch.where(task_ids == 1, period_len_rand, period_len)
        else:
            period_len = torch.where(task_ids == 1, period_len, period_len)

        period_mode = str(getattr(self.config, "period_pattern_mode", "binary"))
        if period_mode == "binary":
            # Choose two distinct symbols per sample (A=start, B!=A), and a primitive {A,B} pattern of length P.
            a = starts  # [B,1]
            b_raw = torch.randint(0, V - 1, (B, 1), device=device, dtype=torch.long)
            b = b_raw + (b_raw >= a).to(dtype=torch.long)

            pattern_bits = torch.zeros((B, pat_len), device=device, dtype=torch.long)
            task_is_period = (task_ids[:, 0] == 1)
            p_vec = period_len[:, 0]
            for p in range(p_min, p_max + 1):
                rows = (task_is_period & (p_vec == p)).nonzero(as_tuple=False).view(-1)
                if rows.numel() == 0:
                    continue
                pats = self._period_patterns.get(p, None)
                if pats is None:
                    pats = get_primitive_binary_patterns(p).to(device=device, non_blocking=True)
                    self._period_patterns[p] = pats
                idx = torch.randint(0, pats.shape[0], (rows.shape[0],), device=device)
                bits = pats[idx]  # [n,p]
                pattern_bits[rows, :p] = bits
            pattern = torch.where(pattern_bits == 0, a, b)  # [B,pat_len]
        elif period_mode == "offset1":
            offsets = torch.arange(pat_len, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            pattern = (starts + offsets) % V  # [B,pat_len]
        elif period_mode in {"offsets", "random"}:
            # Harder: random *non-zero* offsets from start so the cycle is not trivially constant.
            # offsets[:, 0] = 0 ensures pattern starts at `starts`.
            offsets = torch.randint(1, V, (B, pat_len), device=device, dtype=torch.long)
            offsets[:, 0] = 0
            pattern = (starts + offsets) % V
        else:
            raise ValueError(f"Unknown period_pattern_mode={period_mode!r}.")

        # Build periodic sequence by indexing pattern[t mod P].
        idx = torch.remainder(time_steps, period_len)  # [B,T+1]
        seq_period = pattern.gather(1, idx)
        
        # 3. Constant
        seq_const = starts.expand(-1, T + 1)
        
        # Select
        # task_ids: [B, 1] -> expand to [B, T+1]
        task_mask = task_ids.expand(-1, T + 1)
        
        data = torch.where(task_mask == 0, seq_count,
               torch.where(task_mask == 1, seq_period, seq_const))
               
        return data[:, :-1], data[:, 1:], task_ids.expand(-1, T), period_len

    def _generate_batch_custom(
        self,
        batch_size: int,
        sequence_len: int,
        task_id: int | None,
        period_len: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a synthetic batch with a configurable length and (optional) fixed task id.
        Returns (inputs, targets, task_ids) with shapes [B,T].
        """
        B = int(batch_size)
        T = int(sequence_len)
        V = int(self.config.vocab_size)
        device = self.config.device

        starts = torch.randint(0, V, (B, 1), device=device)
        if task_id is None:
            task_ids = torch.randint(0, 3, (B, 1), device=device)
        else:
            task_ids = torch.full((B, 1), int(task_id), dtype=torch.long, device=device)

        time_steps = torch.arange(T + 1, device=device).unsqueeze(0)  # [1,T+1]

        seq_count = (starts + time_steps) % V
        # Default: period-2 diagnostic batch unless overridden.
        p = int(period_len) if period_len is not None else 2
        p = max(2, p)
        pat_len = p
        offsets = torch.randint(1, V, (B, pat_len), device=device, dtype=torch.long)
        offsets[:, 0] = 0
        pattern = (starts + offsets) % V
        idx = torch.remainder(time_steps, p)  # [1,T+1]
        seq_period = pattern.gather(1, idx.expand(B, -1))
        seq_const = starts.expand(-1, T + 1)

        task_mask = task_ids.expand(-1, T + 1)
        data = torch.where(task_mask == 0, seq_count, torch.where(task_mask == 1, seq_period, seq_const))
        return data[:, :-1], data[:, 1:], task_ids.expand(-1, T)

    @torch.no_grad()
    def _closed_loop_diag(self) -> str:
        """
        Protocol 5 (快慢变量/诊断协议.md): quantify closed-loop amplification.

        For Stage1 tasks, "bad collapse" means locking into an *unexpected* short cycle:
        - counting: any period <= max_period is considered bad
        - periodicity: period==2 is expected; other short cycles are bad
        - constant: period==1 is expected; other short cycles are bad
        """
        B = int(getattr(self.config, "diag_closed_loop_batch", 64) or 64)
        L = int(getattr(self.config, "diag_closed_loop_len", 64) or 64)
        tail = int(getattr(self.config, "diag_closed_loop_tail", 32) or 32)
        max_p = int(getattr(self.config, "diag_closed_loop_max_period", 8) or 8)
        tail = min(tail, L)

        def detect_period(seq: torch.Tensor) -> torch.Tensor:
            # seq: [B,L] tokens
            # returns: period int in [0..max_p] (0 means "no short cycle detected")
            if L <= 1 or max_p <= 0:
                return torch.zeros(seq.shape[0], device=seq.device, dtype=torch.long)
            tail_seq = seq[:, -tail:]  # [B,tail]
            periods = torch.zeros(seq.shape[0], device=seq.device, dtype=torch.long)
            unresolved = torch.ones(seq.shape[0], device=seq.device, dtype=torch.bool)
            for p in range(1, max_p + 1):
                if p >= tail:
                    break
                ok = (tail_seq[:, p:] == tail_seq[:, :-p]).all(dim=1)
                hit = unresolved & ok
                if hit.any():
                    periods[hit] = p
                    unresolved = unresolved & (~hit)
                if not unresolved.any():
                    break
            return periods

        def rollout(task_id: int, open_loop: bool) -> Tuple[torch.Tensor, torch.Tensor]:
            inputs, targets, task_ids = self._generate_batch_custom(B, L, task_id)
            self.entity.reset(B, self.config.device)
            state_flat = self.entity.state.flat.clone()
            router_state = None
            router_ctx = None
            task_vec = torch.full((B,), int(task_id), device=self.config.device, dtype=torch.long)
            if self.task_router_embed is not None:
                router_ctx = self.task_router_embed(task_vec)

            # Start token for open-loop is the first symbol of the ground-truth sequence.
            tok = inputs[:, 0]
            preds = []
            correct = []
            for t in range(L):
                token_in = tok if open_loop else inputs[:, t]
                target_t = targets[:, t]
                u = self.port.encode(token_in)
                u = u + torch.tanh(self.task_u_embed(task_vec))

                weights_in = router_state
                chart_w = None
                for _ in range(self.config.evolution_steps):
                    out = self.entity.forward_tensor(
                        state_flat=state_flat,
                        u_dict={"symbolic": u},
                        dt=self.config.dt,
                        prev_chart_weights=weights_in,
                        prediction_error=None,
                        detach_next_prev_weights=True,
                        compute_action=False,
                        router_context=router_ctx,
                    )
                    state_flat = out["next_state_flat"]
                    chart_w = out.get("chart_weights", None)
                    weights_in = out["next_prev_chart_weights"]
                router_state = weights_in

                q = state_flat[:, : self.config.dim_q]
                x = state_flat if getattr(self.port, "decode_state", False) else q
                logits = self.port.decode(x, chart_weights=chart_w)
                pred = logits.argmax(dim=-1)

                preds.append(pred)
                correct.append((pred == target_t))
                tok = pred

            pred_seq = torch.stack(preds, dim=1)  # [B,L]
            acc = torch.stack(correct, dim=1).float().mean()
            return pred_seq, acc

        parts = []
        for task_id, name, expected_p in [
            (0, "Cnt", None),
            (1, "Per", 2),
            (2, "Cst", 1),
        ]:
            _, acc_tf = rollout(task_id, open_loop=False)
            seq_ol, _ = rollout(task_id, open_loop=True)
            periods = detect_period(seq_ol)
            locked = periods > 0
            if expected_p is None:
                bad = locked
            else:
                bad = locked & (periods != int(expected_p))

            lock_rate = locked.float().mean().item()
            bad_rate = bad.float().mean().item()
            mean_p = periods[locked].float().mean().item() if locked.any() else 0.0
            parts.append(f"{name}:TFAcc={acc_tf.item():.2f} Lock={lock_rate:.2f} Bad={bad_rate:.2f} P={mean_p:.1f}")

        return "CL(" + " ".join(parts) + ")"

    @torch.no_grad()
    def _spectral_gap_diag(self) -> str:
        """
        Protocol 1 (快慢变量/诊断协议.md): estimate a spectral-gap proxy for the one-token map.

        We avoid higher-order autograd by using finite-difference JVPs:
          Jv ≈ (f(x + eps*v) - f(x)) / eps
        Report:
          - sigma_max (power iteration)
          - random-direction contraction stats (median/p90)
          - gap ratio (sigma_max / median)
        """
        eps = float(getattr(self.config, "diag_spectral_eps", 1e-3) or 1e-3)
        n_vec = int(getattr(self.config, "diag_spectral_vecs", 8) or 8)
        iters = int(getattr(self.config, "diag_spectral_pi_iters", 5) or 5)

        B = 1
        self.entity.reset(B, self.config.device)
        x0 = self.entity.state.flat.clone()
        tok0 = torch.randint(0, int(self.config.vocab_size), (B,), device=self.config.device)
        u0 = self.port.encode(tok0)

        def f(x_flat: torch.Tensor) -> torch.Tensor:
            state = x_flat
            prev_w = None
            for _ in range(self.config.evolution_steps):
                out = self.entity.forward_tensor(
                    state_flat=state,
                    u_dict={"symbolic": u0},
                    dt=self.config.dt,
                    prev_chart_weights=prev_w,
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                )
                state = out["next_state_flat"]
                prev_w = out["next_prev_chart_weights"]
            return state

        y0 = f(x0)
        d = x0.shape[1]
        dq = int(self.config.dim_q)

        def jvp(v: torch.Tensor) -> torch.Tensor:
            v = v.view_as(x0)
            v = v / (v.norm() + 1e-8)
            y1 = f(x0 + eps * v)
            return (y1 - y0) / eps

        # Power iteration for sigma_max
        v = torch.randn(d, device=self.config.device)
        v = v / (v.norm() + 1e-8)
        sigma = 0.0
        for _ in range(max(1, iters)):
            jv = jvp(v)
            sigma = float(jv.norm().item())
            v = (jv.view(-1) / (jv.norm() + 1e-8)).detach()

        # Random-direction stats
        norms = []
        for _ in range(max(1, n_vec)):
            vr = torch.randn(d, device=self.config.device)
            jv = jvp(vr)
            norms.append(float(jv.norm().item()))
        norms_t = torch.tensor(norms, device=self.config.device)
        med = float(norms_t.median().item())
        p90 = float(torch.quantile(norms_t, 0.9).item()) if norms_t.numel() > 1 else med
        gap = float(sigma / (med + 1e-8))

        # Subspace probes (q-only and p-only perturbations)
        device = self.config.device
        mask_q = torch.zeros(d, device=device)
        mask_p = torch.zeros(d, device=device)
        mask_s = torch.zeros(d, device=device)
        mask_q[:dq] = 1.0
        mask_p[dq : 2 * dq] = 1.0
        mask_s[2 * dq :] = 1.0

        def probe(mask: torch.Tensor) -> tuple[float, float, float]:
            v0 = torch.randn(d, device=device) * mask
            jv = jvp(v0).view(-1)
            return (
                float(jv[:dq].norm().item()),
                float(jv[dq : 2 * dq].norm().item()),
                float(jv[2 * dq :].abs().mean().item()),
            )

        q_to_q, q_to_p, q_to_s = probe(mask_q)
        p_to_q, p_to_p, p_to_s = probe(mask_p)

        return (
            f"SGap(smax={sigma:.3f} med={med:.3f} p90={p90:.3f} gap={gap:.2f} eps={eps:g} "
            f"Jq=[{q_to_q:.3f},{q_to_p:.3f}] Jp=[{p_to_q:.3f},{p_to_p:.3f}])"
        )

    @staticmethod
    def _make_eval_batch_cpu(
        *,
        batch_size: int,
        sequence_len: int,
        vocab_size: int,
        task_mix: str,
        seed: int,
        period_alt_mode: str = "random",
        period_min: int = 2,
        period_max: int = 5,
        period_fixed: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deterministic (CPU) batch generator for evaluation.

        period_alt_mode:
          - "random": random cycle pattern (task difficulty matches training).
          - "binary": primitive {A,B} cycles (matches temp-04.md examples).
          - "offset1": deterministic cycle pattern: token_i = (start + i) mod V.
        """
        B = int(batch_size)
        T = int(sequence_len)
        V = int(vocab_size)
        if B <= 0 or T <= 0 or V <= 1:
            raise ValueError("Invalid eval batch shape/vocab.")

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        starts = torch.randint(0, V, (B, 1), generator=gen, dtype=torch.long)

        task_mix = str(task_mix)
        if task_mix == "counting":
            task_ids = torch.zeros((B, 1), dtype=torch.long)
        elif task_mix == "period":
            task_ids = torch.ones((B, 1), dtype=torch.long)
        elif task_mix == "constant":
            task_ids = torch.full((B, 1), 2, dtype=torch.long)
        elif task_mix == "mixed":
            n0 = B // 3
            n1 = B // 3
            n2 = B - n0 - n1
            task_ids = torch.cat(
                [
                    torch.zeros((n0, 1), dtype=torch.long),
                    torch.ones((n1, 1), dtype=torch.long),
                    torch.full((n2, 1), 2, dtype=torch.long),
                ],
                dim=0,
            )
            perm = torch.randperm(B, generator=gen)
            task_ids = task_ids[perm]
        else:
            raise ValueError(f"Unknown task_mix={task_mix!r} for eval batch.")

        time_steps = torch.arange(T + 1, dtype=torch.long).unsqueeze(0)  # [1,T+1]
        seq_count = (starts + time_steps) % V

        p_min = max(2, int(period_min))
        p_max = max(p_min, int(period_max))
        pat_len = max(p_max, int(period_fixed) if period_fixed is not None else p_max)

        # Period length per sample (only relevant when task_id==1).
        period_len = torch.full((B, 1), p_min, dtype=torch.long)
        if period_fixed is not None:
            period_len = torch.where(task_ids == 1, torch.full((B, 1), int(period_fixed), dtype=torch.long), period_len)
        elif p_max > p_min:
            period_len_rand = torch.randint(p_min, p_max + 1, (B, 1), generator=gen, dtype=torch.long)
            period_len = torch.where(task_ids == 1, period_len_rand, period_len)

        period_alt_mode = str(period_alt_mode)
        if period_alt_mode == "random":
            offsets = torch.randint(1, V, (B, pat_len), generator=gen, dtype=torch.long)
            offsets[:, 0] = 0
            pattern = (starts + offsets) % V  # [B,pat_len]
        elif period_alt_mode == "binary":
            a = starts
            b_raw = torch.randint(0, V - 1, (B, 1), generator=gen, dtype=torch.long)
            b = b_raw + (b_raw >= a).to(dtype=torch.long)

            pattern_bits = torch.zeros((B, pat_len), dtype=torch.long)
            task_is_period = (task_ids[:, 0] == 1)
            p_vec = period_len[:, 0]
            for p in range(p_min, p_max + 1):
                rows = (task_is_period & (p_vec == p)).nonzero(as_tuple=False).view(-1)
                if rows.numel() == 0:
                    continue
                pats = get_primitive_binary_patterns(p)  # [Np,p] (CPU)
                idx = torch.randint(0, pats.shape[0], (rows.shape[0],), generator=gen, dtype=torch.long)
                bits = pats[idx]
                pattern_bits[rows, :p] = bits
            pattern = torch.where(pattern_bits == 0, a, b)  # [B,pat_len]
        elif period_alt_mode == "offset1":
            offsets = torch.arange(pat_len, dtype=torch.long).unsqueeze(0).expand(B, -1)
            pattern = (starts + offsets) % V  # [B,pat_len]
        else:
            raise ValueError(f"Unknown period_alt_mode={period_alt_mode!r}.")
        idx = torch.remainder(time_steps, period_len)  # [B,T+1]
        seq_period = pattern.gather(1, idx)
        seq_const = starts.expand(-1, T + 1)

        task_mask = task_ids.expand(-1, T + 1)
        data = torch.where(task_mask == 0, seq_count, torch.where(task_mask == 1, seq_period, seq_const))
        return data[:, :-1], data[:, 1:], task_ids.expand(-1, T), period_len

    @torch.no_grad()
    def evaluate_L1_gate(
        self,
        *,
        batch_size: int = 4096,
        repeats: int = 4,
        seed: int = 0,
        period_alt_mode: str = "random",
        require_constant: bool = True,
        enable_router_diag: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate Level-1 (Stage1) skill gate on fresh batches.

        Key point: for a period-P cycle, the first unambiguous next-token prediction
        occurs at t=P-1 after observing a full cycle. We therefore gate Period with
        PerW (accuracy on tokens where t >= P-1), matching the "…, ?, predict" intuition
        in temp-04.md for periods 2–5.
        """
        device = self.config.device
        B = int(batch_size)
        T = int(self.config.sequence_len)
        V = int(self.config.vocab_size)
        repeats = max(1, int(repeats))

        prev_diag_router = bool(getattr(self.config, "diag_router", False))
        if enable_router_diag:
            self.config.diag_router = True

        # Accumulators
        sums = {
            "Cnt": {"acc": 0.0, "acc_t0": 0.0, "acc_t1": 0.0, "acc_tw": 0.0},
            "Per": {"acc": 0.0, "acc_t0": 0.0, "acc_t1": 0.0, "acc_tw": 0.0},
            "Cst": {"acc": 0.0, "acc_t0": 0.0, "acc_t1": 0.0, "acc_tw": 0.0},
            "aux": {"chart_ent": 0.0, "mi_tc": 0.0},
        }

        for r in range(repeats):
            inputs_cpu, targets_cpu, task_ids_cpu, period_len_cpu = self._make_eval_batch_cpu(
                batch_size=B,
                sequence_len=T,
                vocab_size=V,
                task_mix=str(getattr(self.config, "task_mix", "mixed")),
                seed=int(seed) + r,
                period_alt_mode=period_alt_mode,
                period_min=int(getattr(self.config, "period_min", 2) or 2),
                period_max=int(getattr(self.config, "period_max", 5) or 5),
            )

            inputs = inputs_cpu.to(device=device, non_blocking=True)
            targets = targets_cpu.to(device=device, non_blocking=True)
            task_ids = task_ids_cpu.to(device=device, non_blocking=True)
            period_len = period_len_cpu.to(device=device, non_blocking=True)

            # Deterministic init-state distribution (matches CUDA-graph init).
            dq = int(self.config.dim_q)
            D = 2 * dq + 1
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed) + 10_000 + r)
            init_cpu = torch.zeros((B, D), dtype=torch.float32)
            init_cpu[:, :dq] = torch.randn((B, dq), generator=gen) * 0.1
            init_cpu[:, dq : 2 * dq] = torch.randn((B, dq), generator=gen) * 0.1
            init_state_flat = init_cpu.to(device=device, non_blocking=True)

            (
                _total_F,
                _total_acc,
                chart_usage,
                _final_state_flat,
                per_task_acc,
                per_task_acc_t0,
                per_task_acc_t1,
                per_task_acc_tw,
                mi_task_chart,
                _total_V,
                _total_pred_ce,
                _diag,
            ) = self.seq_model(inputs, targets, init_state_flat, task_ids, None, period_len)

            # Per-task acc vectors are [Cnt,Per,Cst].
            sums["Cnt"]["acc"] += float(per_task_acc[0].item())
            sums["Per"]["acc"] += float(per_task_acc[1].item())
            sums["Cst"]["acc"] += float(per_task_acc[2].item())

            sums["Cnt"]["acc_t0"] += float(per_task_acc_t0[0].item())
            sums["Per"]["acc_t0"] += float(per_task_acc_t0[1].item())
            sums["Cst"]["acc_t0"] += float(per_task_acc_t0[2].item())

            sums["Cnt"]["acc_t1"] += float(per_task_acc_t1[0].item())
            sums["Per"]["acc_t1"] += float(per_task_acc_t1[1].item())
            sums["Cst"]["acc_t1"] += float(per_task_acc_t1[2].item())

            sums["Cnt"]["acc_tw"] += float(per_task_acc_tw[0].item())
            sums["Per"]["acc_tw"] += float(per_task_acc_tw[1].item())
            sums["Cst"]["acc_tw"] += float(per_task_acc_tw[2].item())

            # Chart usage entropy (collapse proxy).
            dist = chart_usage / (chart_usage.sum() + 1e-8)
            ent = -(dist * torch.log(dist + 1e-8)).sum()
            sums["aux"]["chart_ent"] += float(ent.item())
            sums["aux"]["mi_tc"] += float(mi_task_chart.item()) if mi_task_chart is not None else 0.0

        # Restore config
        self.config.diag_router = prev_diag_router

        rep = float(repeats)
        metrics = {
            "Cnt": {k: v / rep for k, v in sums["Cnt"].items()},
            "Per": {k: v / rep for k, v in sums["Per"].items()},
            "Cst": {k: v / rep for k, v in sums["Cst"].items()},
            "chart_entropy": sums["aux"]["chart_ent"] / rep,
            "mi_task_chart": sums["aux"]["mi_tc"] / rep,
            "period_alt_mode": str(period_alt_mode),
            "period_min": int(getattr(self.config, "period_min", 2) or 2),
            "period_max": int(getattr(self.config, "period_max", 5) or 5),
            "repeats": repeats,
            "batch_size": B,
            "sequence_len": T,
        }

        # Gate thresholds from temp-04.md (Level 1) + constant as an extra sanity check.
        passed_cnt = metrics["Cnt"]["acc"] >= 0.95
        passed_per = metrics["Per"]["acc_tw"] >= 0.90  # PerW
        passed_cst = (metrics["Cst"]["acc"] >= 0.95) if require_constant else True

        metrics["passed"] = bool(passed_cnt and passed_per and passed_cst)
        metrics["passed_components"] = {
            "Cnt>=0.95": bool(passed_cnt),
            "PerW>=0.90": bool(passed_per),
            "Cst>=0.95": bool(passed_cst) if require_constant else True,
        }
        return metrics

    def train_loop(self):
        if bool(getattr(self.config, "use_cuda_graph", False)) and torch.cuda.is_available() and str(self.config.device).startswith("cuda"):
            try:
                self._train_loop_cuda_graph()
                return
            except Exception as e:
                print(f"CUDA graph disabled (capture failed): {type(e).__name__}: {e}")
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
        print(f"Starting Stage 1 Training on {self.config.device}...")
        print(f"Config: {self.config}")
        print(f"A3 Alignment: Optimizing Unified Free Energy F")

        os.makedirs(self.config.save_dir, exist_ok=True)
        best_path = os.path.join(self.config.save_dir, "stage1_best.pt")
        best_perw = -1.0
        best_gate: Dict[str, Any] | None = None
        mode = str(getattr(self.config, "period_pattern_mode", "random"))
        gate_alt = "random" if mode in {"offsets", "random"} else mode
        
        start_time = time.time()
        
        debug_mem = os.getenv("HEI_DEBUG_MEM", "0") == "1"
        # Important performance note:
        # `if not torch.isfinite(t).all():` forces a GPU→CPU sync each step.
        # Keep this check opt-in; the core already has async asserts when `strict_nonfinite=True`.
        debug_fail_nonfinite = os.getenv("HEI_FAIL_NONFINITE", "0") == "1"

        for step in range(1, self.config.steps + 1):
            self.optimizer.zero_grad(set_to_none=True)
            if debug_mem and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # 1. Data
            inputs, targets, task_ids, period_len = self.generate_batch() # [B, T], [B,1]
            
            # 2. Reset Entity (Just get initial state tensor)
            self.entity.reset(self.config.batch_size, self.config.device)
            initial_state_flat = self.entity.state.flat.clone()
            
            # 3. Run Compiled Sequence
            (
                total_F,
                total_acc,
                chart_usage,
                final_state_flat,
                per_task_acc,
                per_task_acc_t0,
                per_task_acc_t1,
                per_task_acc_tw,
                mi_task_chart,
                total_V,
                total_pred_ce,
                diag,
            ) = self.seq_model(inputs, targets, initial_state_flat, task_ids, None, period_len)
            
            # Update entity state for consistency (though not strictly needed for training loop)
            # We need to detach to stop gradient flow into next batch if we were doing BPTT across batches
            # But here we reset every batch.
            with torch.no_grad():
                self.entity.state.flat.copy_(final_state_flat)
            
            # Average over time
            mean_F = total_F / self.config.sequence_len
            mean_acc = total_acc / self.config.sequence_len
            mean_V = total_V / self.config.sequence_len
            mean_ce = total_pred_ce / self.config.sequence_len
            
            # Chart Entropy Regularization
            # Maximize entropy of average usage (use all charts)
            # Minimize entropy of individual usage (be decisive) - implicitly handled by router softmax
            # Here we just want to prevent collapse to single chart
            chart_dist = chart_usage / (chart_usage.sum() + 1e-8)
            entropy = -(chart_dist * torch.log(chart_dist + 1e-8)).sum()
            
            # Add entropy bonus (maximize entropy -> minimize -entropy)
            # alpha_chart defaults to 0.0 if not set
            alpha = getattr(self.config, 'alpha_chart', 0.0)

            # Skill specialization: maximize MI(task;chart) (optional).
            lambda_task_mi = float(getattr(self.config, "lambda_task_mi", 0.0) or 0.0)

            # A4: viability barrier (soft, part of the unified scalar).
            q = final_state_flat[:, : self.config.dim_q]
            p = final_state_flat[:, self.config.dim_q : 2 * self.config.dim_q]
            q_norms = q.norm(dim=1)
            p_norms = p.norm(dim=1)
            q_max = float(getattr(self.config, "viability_q_max", 0.0) or 0.0)
            p_max = float(getattr(self.config, "viability_p_max", 0.0) or 0.0)
            barrier = torch.zeros((), device=final_state_flat.device)
            if q_max > 0.0:
                barrier = barrier + torch.relu(q_norms - q_max).pow(2).mean()
            if p_max > 0.0:
                barrier = barrier + torch.relu(p_norms - p_max).pow(2).mean()
            lambda_viability = float(getattr(self.config, "lambda_viability", 0.0) or 0.0)

            # Port small-gain regularizer (spectral norm per chart).
            port_gain_pen = torch.zeros((), device=final_state_flat.device)
            port_gain_sig = None
            lambda_port_gain = float(getattr(self.config, "lambda_port_gain", 0.0) or 0.0)
            gain_max = float(getattr(self.config, "port_gain_max", 0.0) or 0.0)
            if lambda_port_gain > 0.0 and gain_max > 0.0:
                try:
                    port_mod = self.entity.generator.ports["symbolic"] if "symbolic" in self.entity.generator.ports else None
                    W = getattr(port_mod, "W_stack", None) if port_mod is not None else None
                    if W is not None:
                        # Fast spectral-norm estimate (power iteration per chart).
                        iters = int(getattr(self.config, "port_gain_pi_iters", 2) or 2)
                        v = torch.randn(W.shape[0], W.shape[2], device=W.device, dtype=W.dtype)
                        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
                        for _ in range(max(1, iters)):
                            u = torch.einsum("kuq,kq->ku", W, v)
                            v = torch.einsum("kuq,ku->kq", W, u)
                            v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
                        u = torch.einsum("kuq,kq->ku", W, v)
                        sigma_max = u.norm(dim=1)  # [K]
                        port_gain_sig = sigma_max.detach()
                        port_gain_pen = torch.relu(sigma_max - gain_max).pow(2).mean()
                except Exception:
                    port_gain_pen = torch.zeros((), device=final_state_flat.device)
                    port_gain_sig = None

            loss = (
                mean_F
                - alpha * entropy
                - lambda_task_mi * mi_task_chart
                + lambda_viability * barrier
                + lambda_port_gain * port_gain_pen
            )

            if debug_fail_nonfinite:
                if loss.is_cuda:
                    torch._assert_async(torch.isfinite(loss).all(), "Non-finite loss encountered; aborting for diagnosis.")
                else:
                    if not torch.isfinite(loss).all():
                        raise RuntimeError("Non-finite loss encountered; aborting for diagnosis.")
            
            # 4. Backprop
            loss.backward()
            # Clip ALL trainable parameters (including task embeddings) for stability.
            torch.nn.utils.clip_grad_norm_(self._train_params, 1.0)
            self.optimizer.step()
            self._project_port_gain_()
            
            # 5. Logging & Diagnostics
            if step % self.config.log_every == 0:
                elapsed = time.time() - start_time
                
                # Format per-task acc
                task_acc_str = f"Cnt={per_task_acc[0]:.2f} Per={per_task_acc[1]:.2f} Cst={per_task_acc[2]:.2f}"
                task_acc_t1_str = f"PerW={per_task_acc_tw[1]:.2f}"

                mem_str = ""
                if debug_mem and torch.cuda.is_available():
                    alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                    reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    mem_str = f" Mem={alloc_mb:.0f}MiB Res={reserved_mb:.0f}MiB Peak={peak_mb:.0f}MiB"

                extra = ""
                if diag is not None and getattr(self.config, "diag_router", False):
                    task_chart = diag.get("task_chart_weight_sum", None)
                    task_counts = diag.get("task_token_counts", None)
                    if task_chart is not None and task_counts is not None:
                        joint = task_chart / (task_chart.sum() + 1e-8)
                        p_task = joint.sum(dim=1, keepdim=True)
                        p_chart = joint.sum(dim=0, keepdim=True)
                        mi_tc = (joint * (torch.log(joint + 1e-8) - torch.log(p_task + 1e-8) - torch.log(p_chart + 1e-8))).sum()

                        cond = task_chart / (task_counts.unsqueeze(1) + 1e-8)  # P(chart|task)
                        top_chart = cond.argmax(dim=1)
                        neff = 1.0 / (chart_dist.pow(2).sum() + 1e-8)

                        switch_den = diag.get("switch_den", None)
                        switch_rate = None
                        if switch_den is not None and float(switch_den.detach().item()) > 0:
                            switch_rate = (diag["switch_count"] / switch_den).clamp(min=0.0, max=1.0)

                        overlap_den = diag.get("overlap_den", None)
                        overlap_rate = None
                        top2_gap = None
                        if overlap_den is not None and float(overlap_den.detach().item()) > 0:
                            overlap_rate = (diag["overlap_count"] / overlap_den).clamp(min=0.0, max=1.0)
                            top2_gap = diag["top2_gap_sum"] / overlap_den

                        extra_parts = [
                            f"N_eff={neff.item():.2f}",
                            f"TopChart=[{int(top_chart[0])},{int(top_chart[1])},{int(top_chart[2])}]",
                        ]
                        if switch_rate is not None:
                            extra_parts.append(f"Switch={switch_rate.item():.2f}")
                        if overlap_rate is not None and top2_gap is not None:
                            extra_parts.append(f"Overlap={overlap_rate.item():.2f}")
                            extra_parts.append(f"Gap={top2_gap.item():.2f}")
                        extra = " " + " ".join(extra_parts)

                if diag is not None and getattr(self.config, "diag_pred_baselines", False):
                    task_counts = diag.get("task_token_counts", None)
                    if task_counts is not None:
                        denom = task_counts + 1e-8
                        prev_r = (diag["task_match_prev_sum"] / denom) if diag.get("task_match_prev_sum", None) is not None else None
                        cur_r = (diag["task_match_cur_sum"] / denom) if diag.get("task_match_cur_sum", None) is not None else None
                        plus1_r = (diag["task_match_plus1_sum"] / denom) if diag.get("task_match_plus1_sum", None) is not None else None
                        if prev_r is not None and cur_r is not None and plus1_r is not None:
                            extra = extra + (
                                f" Base(prev)=[{prev_r[0]:.2f},{prev_r[1]:.2f},{prev_r[2]:.2f}]"
                                f" Base(cur)=[{cur_r[0]:.2f},{cur_r[1]:.2f},{cur_r[2]:.2f}]"
                                f" Base(+1)=[{plus1_r[0]:.2f},{plus1_r[1]:.2f},{plus1_r[2]:.2f}]"
                            )

                # Port coupling gain diagnostics (small-gain proxy).
                gain_str = ""
                if getattr(self.config, "diag_router", False):
                    try:
                        port_mod = self.entity.generator.ports["symbolic"] if "symbolic" in self.entity.generator.ports else None
                        W = getattr(port_mod, "W_stack", None)
                        if W is not None:
                            w_flat = W.reshape(self.config.num_charts, -1)
                            w_norms = w_flat.norm(dim=1)
                            gain_str = f" Wmax={w_norms.max().item():.2f} Wmean={w_norms.mean().item():.2f}"
                    except Exception:
                        gain_str = ""

                # A4 barrier + small-gain regularizer values (for monitoring only)
                reg_str = ""
                if float(getattr(self.config, "lambda_viability", 0.0) or 0.0) > 0.0:
                    try:
                        reg_str += f" Barrier={float(barrier.detach().item()):.3g}"
                    except Exception:
                        pass
                if float(getattr(self.config, "lambda_port_gain", 0.0) or 0.0) > 0.0:
                    try:
                        reg_str += f" GainPen={float(port_gain_pen.detach().item()):.3g}"
                        if port_gain_sig is not None:
                            reg_str += f" SigMax={float(port_gain_sig.max().item()):.2f}"
                    except Exception:
                        pass

                mi_str = (
                    f" MI_tc={float(mi_task_chart.detach().item()):.3f}" if torch.is_tensor(mi_task_chart) else ""
                )
                vc_str = ""
                try:
                    vc_str = f" V={float(mean_V.detach().item()):.2f} CE={float(mean_ce.detach().item()):.2f}"
                except Exception:
                    vc_str = ""
                print(
                    f"Step {step}: F={mean_F.item():.4f} "
                    f"Acc={mean_acc.item():.4f} ({task_acc_str}) "
                    + f"{task_acc_t1_str} "
                    f"ChartEnt={entropy.item():.2f}"
                    + mi_str
                    + vc_str
                    + f" Time={elapsed:.1f}s"
                    + mem_str
                    + extra
                    + gain_str
                    + reg_str
                )

                try:
                    gate = self.evaluate_L1_gate(
                        batch_size=2048,
                        repeats=1,
                        seed=0,
                        period_alt_mode=gate_alt,
                        require_constant=False,
                        enable_router_diag=False,
                    )
                    perw = float(gate["Per"]["acc_tw"])
                    if perw > best_perw:
                        best_perw = perw
                        best_gate = gate
                        self.save(best_path)
                        print(f"GateBest: PerW={best_perw:.3f} (Alt={gate_alt})")
                except Exception as e:
                    print(f"GateEval: failed ({type(e).__name__}: {e})")

                if getattr(self.config, "diag_closed_loop", False) and (
                    step % int(getattr(self.config, "diag_closed_loop_every", 50) or 50) == 0
                ):
                    try:
                        print(self._closed_loop_diag())
                    except Exception as e:
                        print(f"CL(diag_failed): {type(e).__name__}: {e}")

                if getattr(self.config, "diag_spectral_gap", False) and (
                    step % int(getattr(self.config, "diag_spectral_every", 200) or 200) == 0
                ):
                    try:
                        print(self._spectral_gap_diag())
                    except Exception as e:
                        print(f"SGap(diag_failed): {type(e).__name__}: {e}")
                
                # Check viability (A4)
                q_norm = self.entity.state.q.norm(dim=1).mean()
                if q_norm.item() > 100 or torch.isnan(q_norm).item():
                    print("WARNING: Viability violation (Divergence)")
                    break

        # Save
        if os.path.exists(best_path):
            try:
                self.load(best_path)
                if best_gate is not None:
                    print(
                        f"Loaded best checkpoint: PerW={best_perw:.3f} "
                        f"Cnt={best_gate['Cnt']['acc']:.3f} Per={best_gate['Per']['acc']:.3f} "
                        f"(Alt={best_gate['period_alt_mode']})"
                    )
            except Exception as e:
                print(f"Best checkpoint load failed ({type(e).__name__}: {e})")
        save_path = os.path.join(self.config.save_dir, "stage1_final.pt")
        self.save(save_path)
        print(f"Stage 1 Complete. Saved to {save_path}")

if __name__ == "__main__":
    config = Stage1Config()
    # Optional reproducibility
    if os.getenv("HEI_SEED") is not None:
        seed = int(os.getenv("HEI_SEED", "0"))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    # Debug helpers (no effect unless env vars are set)
    if os.getenv("HEI_DEBUG_STEPS") is not None:
        config.steps = int(os.getenv("HEI_DEBUG_STEPS", "5"))
        config.log_every = 1
    if os.getenv("HEI_STEPS") is not None:
        config.steps = int(os.getenv("HEI_STEPS", str(config.steps)))
    if os.getenv("HEI_LOG_EVERY") is not None:
        config.log_every = int(os.getenv("HEI_LOG_EVERY", str(config.log_every)))
    env_cudagraph = os.getenv("HEI_CUDA_GRAPH")
    if env_cudagraph is not None:
        config.use_cuda_graph = str(env_cudagraph) == "1"
    if os.getenv("HEI_CUDA_GRAPH_WARMUP") is not None:
        config.cuda_graph_warmup = int(os.getenv("HEI_CUDA_GRAPH_WARMUP", str(getattr(config, "cuda_graph_warmup", 3))))
    if os.getenv("HEI_BATCH_SIZE") is not None:
        config.batch_size = int(os.getenv("HEI_BATCH_SIZE", str(config.batch_size)))
    if os.getenv("HEI_SEQ_LEN") is not None:
        config.sequence_len = int(os.getenv("HEI_SEQ_LEN", str(config.sequence_len)))
    if os.getenv("HEI_PERIOD_MIN") is not None:
        config.period_min = int(os.getenv("HEI_PERIOD_MIN", str(getattr(config, "period_min", 2))))
    if os.getenv("HEI_PERIOD_MAX") is not None:
        config.period_max = int(os.getenv("HEI_PERIOD_MAX", str(getattr(config, "period_max", 5))))
    if os.getenv("HEI_PERIOD_MODE") is not None:
        config.period_pattern_mode = str(os.getenv("HEI_PERIOD_MODE", str(getattr(config, "period_pattern_mode", "binary"))))
    if os.getenv("HEI_DIM_Q") is not None:
        config.dim_q = int(os.getenv("HEI_DIM_Q", str(config.dim_q)))
    if os.getenv("HEI_NUM_CHARTS") is not None:
        config.num_charts = int(os.getenv("HEI_NUM_CHARTS", str(config.num_charts)))
    if os.getenv("HEI_LR") is not None:
        config.lr = float(os.getenv("HEI_LR", str(config.lr)))
    if os.getenv("HEI_WEIGHT_DECAY") is not None:
        config.weight_decay = float(os.getenv("HEI_WEIGHT_DECAY", str(getattr(config, "weight_decay", 0.0))))
    if os.getenv("HEI_WD") is not None:
        config.weight_decay = float(os.getenv("HEI_WD", str(getattr(config, "weight_decay", 0.0))))
    if os.getenv("HEI_LR_CORE_SCALE") is not None:
        config.lr_core_scale = float(os.getenv("HEI_LR_CORE_SCALE", str(getattr(config, "lr_core_scale", 1.0))))
    if os.getenv("HEI_DT") is not None:
        config.dt = float(os.getenv("HEI_DT", str(config.dt)))
    if os.getenv("HEI_DAMPING") is not None:
        config.damping = float(os.getenv("HEI_DAMPING", str(config.damping)))
    if os.getenv("HEI_SUBSTEPS") is not None:
        config.integrator_substeps = int(os.getenv("HEI_SUBSTEPS", str(config.integrator_substeps)))
    if os.getenv("HEI_STIFFNESS") is not None:
        config.stiffness = float(os.getenv("HEI_STIFFNESS", str(config.stiffness)))
    if os.getenv("HEI_ALPHA_MIN") is not None:
        config.alpha_min = float(os.getenv("HEI_ALPHA_MIN", str(getattr(config, "alpha_min", 0.2))))
    if os.getenv("HEI_ALPHA_MAX") is not None:
        config.alpha_max = float(os.getenv("HEI_ALPHA_MAX", str(getattr(config, "alpha_max", 1.0))))
    if os.getenv("HEI_PORT_TOP_K") is not None:
        config.port_top_k = int(os.getenv("HEI_PORT_TOP_K", str(getattr(config, "port_top_k", 0))))
    if os.getenv("HEI_PORT_TOPK_IMPL") is not None:
        config.port_topk_impl = str(os.getenv("HEI_PORT_TOPK_IMPL", str(getattr(config, "port_topk_impl", "dense"))))
    if os.getenv("HEI_BRANCHES") is not None:
        config.num_branches = int(os.getenv("HEI_BRANCHES", str(config.num_branches)))
    if os.getenv("HEI_BRANCH_NOISE") is not None:
        config.branch_noise = float(os.getenv("HEI_BRANCH_NOISE", str(config.branch_noise)))
    if os.getenv("HEI_PORT_U_SCALE") is not None:
        config.port_u_scale = float(os.getenv("HEI_PORT_U_SCALE", str(getattr(config, "port_u_scale", 1.0))))
    if os.getenv("HEI_PORT_U_TANH") is not None:
        config.port_u_tanh = str(os.getenv("HEI_PORT_U_TANH", "1")) == "1"
    if os.getenv("HEI_PORT_DECODE_STATE") is not None:
        config.port_decode_state = str(os.getenv("HEI_PORT_DECODE_STATE", "1")) == "1"
    if os.getenv("HEI_TRAIN_TOP_M") is not None:
        config.train_top_m = int(os.getenv("HEI_TRAIN_TOP_M", str(getattr(config, "train_top_m", 1))))
    if os.getenv("HEI_TRAIN_TOP_M_TEMP") is not None:
        config.train_top_m_temp = float(os.getenv("HEI_TRAIN_TOP_M_TEMP", str(getattr(config, "train_top_m_temp", 1.0))))
    if os.getenv("HEI_EVOLUTION_STEPS") is not None:
        config.evolution_steps = int(os.getenv("HEI_EVOLUTION_STEPS", str(config.evolution_steps)))
    if os.getenv("HEI_GAMMA_PRED") is not None:
        config.gamma_pred = float(os.getenv("HEI_GAMMA_PRED", str(config.gamma_pred)))
    if os.getenv("HEI_ALPHA_CHART") is not None:
        config.alpha_chart = float(os.getenv("HEI_ALPHA_CHART", str(config.alpha_chart)))
    if os.getenv("HEI_LAMBDA_TASK_MI") is not None:
        config.lambda_task_mi = float(os.getenv("HEI_LAMBDA_TASK_MI", str(config.lambda_task_mi)))
    if os.getenv("HEI_TASK_MIX") is not None:
        config.task_mix = str(os.getenv("HEI_TASK_MIX", str(config.task_mix)))
    if os.getenv("HEI_VIAB_QMAX") is not None:
        config.viability_q_max = float(os.getenv("HEI_VIAB_QMAX", str(config.viability_q_max)))
    if os.getenv("HEI_VIAB_PMAX") is not None:
        config.viability_p_max = float(os.getenv("HEI_VIAB_PMAX", str(config.viability_p_max)))
    if os.getenv("HEI_LAMBDA_VIAB") is not None:
        config.lambda_viability = float(os.getenv("HEI_LAMBDA_VIAB", str(config.lambda_viability)))
    if os.getenv("HEI_PORT_GAIN_MAX") is not None:
        config.port_gain_max = float(os.getenv("HEI_PORT_GAIN_MAX", str(config.port_gain_max)))
    if os.getenv("HEI_LAMBDA_PORT_GAIN") is not None:
        config.lambda_port_gain = float(os.getenv("HEI_LAMBDA_PORT_GAIN", str(config.lambda_port_gain)))
    if os.getenv("HEI_PORT_GAIN_ITERS") is not None:
        config.port_gain_pi_iters = int(os.getenv("HEI_PORT_GAIN_ITERS", str(config.port_gain_pi_iters)))
    if os.getenv("HEI_ROUTER_TAU") is not None:
        config.router_tau = float(os.getenv("HEI_ROUTER_TAU", "0.0"))
    if os.getenv("HEI_ROUTER_CONTEXT_DIM") is not None:
        config.router_context_dim = int(os.getenv("HEI_ROUTER_CONTEXT_DIM", "0"))
    if os.getenv("HEI_TRANSPORT_THRESHOLD") is not None:
        config.transport_threshold = float(os.getenv("HEI_TRANSPORT_THRESHOLD", "0.0"))
    if os.getenv("HEI_DIAG_ROUTER", "0") == "1":
        config.diag_router = True
    if os.getenv("HEI_DIAG_BASELINES", "0") == "1":
        config.diag_pred_baselines = True
    if os.getenv("HEI_DIAG_CLOSED_LOOP", "0") == "1":
        config.diag_closed_loop = True
    if os.getenv("HEI_CLOSED_LOOP_EVERY") is not None:
        config.diag_closed_loop_every = int(os.getenv("HEI_CLOSED_LOOP_EVERY", str(config.diag_closed_loop_every)))
    if os.getenv("HEI_CLOSED_LOOP_LEN") is not None:
        config.diag_closed_loop_len = int(os.getenv("HEI_CLOSED_LOOP_LEN", str(config.diag_closed_loop_len)))
    if os.getenv("HEI_CLOSED_LOOP_BATCH") is not None:
        config.diag_closed_loop_batch = int(os.getenv("HEI_CLOSED_LOOP_BATCH", str(config.diag_closed_loop_batch)))
    if os.getenv("HEI_CLOSED_LOOP_TAIL") is not None:
        config.diag_closed_loop_tail = int(os.getenv("HEI_CLOSED_LOOP_TAIL", str(config.diag_closed_loop_tail)))
    if os.getenv("HEI_CLOSED_LOOP_MAXP") is not None:
        config.diag_closed_loop_max_period = int(os.getenv("HEI_CLOSED_LOOP_MAXP", str(config.diag_closed_loop_max_period)))
    if os.getenv("HEI_DIAG_SPECTRAL", "0") == "1":
        config.diag_spectral_gap = True
    if os.getenv("HEI_SPECTRAL_EVERY") is not None:
        config.diag_spectral_every = int(os.getenv("HEI_SPECTRAL_EVERY", str(config.diag_spectral_every)))
    if os.getenv("HEI_SPECTRAL_EPS") is not None:
        config.diag_spectral_eps = float(os.getenv("HEI_SPECTRAL_EPS", str(config.diag_spectral_eps)))
    if os.getenv("HEI_SPECTRAL_VECS") is not None:
        config.diag_spectral_vecs = int(os.getenv("HEI_SPECTRAL_VECS", str(config.diag_spectral_vecs)))
    if os.getenv("HEI_SPECTRAL_PI_ITERS") is not None:
        config.diag_spectral_pi_iters = int(os.getenv("HEI_SPECTRAL_PI_ITERS", str(config.diag_spectral_pi_iters)))
    trainer = Stage1Trainer(config)
    trainer.train_loop()

    if os.getenv("HEI_EVAL_L1", "0") == "1":
        eval_batch = int(os.getenv("HEI_EVAL_BATCH", "4096"))
        eval_repeats = int(os.getenv("HEI_EVAL_REPEATS", "4"))
        eval_seed = int(os.getenv("HEI_EVAL_SEED", "0"))
        env_eval_alt = os.getenv("HEI_EVAL_PERIOD_ALT")
        if env_eval_alt is None:
            mode = str(getattr(config, "period_pattern_mode", "random"))
            eval_period_alt = "random" if mode in {"offsets", "random"} else mode
        else:
            eval_period_alt = str(env_eval_alt)
        eval_require_cst = os.getenv("HEI_EVAL_REQUIRE_CST", "1") == "1"
        gate = trainer.evaluate_L1_gate(
            batch_size=eval_batch,
            repeats=eval_repeats,
            seed=eval_seed,
            period_alt_mode=eval_period_alt,
            require_constant=eval_require_cst,
            enable_router_diag=True,
        )
        print(
            "=== L1 Gate (Eval) ===\n"
            f"Cnt={gate['Cnt']['acc']:.3f} "
            f"Per={gate['Per']['acc']:.3f} "
            f"PerW={gate['Per']['acc_tw']:.3f} "
            f"Cst={gate['Cst']['acc']:.3f} "
            f"ChartEnt={gate['chart_entropy']:.2f} "
            f"MI_tc={gate['mi_task_chart']:.3f} "
            f"P={gate['period_min']}-{gate['period_max']} Alt={gate['period_alt_mode']} "
            f"Result={'PASS' if gate['passed'] else 'FAIL'}"
        )
