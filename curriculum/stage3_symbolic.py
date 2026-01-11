"""
Stage 3: Symbolic Manipulation (Level 3)

Reference: HEI/docs/temp/2026/1/A/1·4/01/temp-04.md

Tasks:
- Copy:    ABC -> ABC
- Reverse: ABC -> CBA
- Replace: A->X, B->Y, ABC -> XYC
- Unparen: (AB) -> AB, ((AB)) -> AB   (nesting depth <= 3)

Acceptance (temp-04.md):
- Length <= 10: transform accuracy > 95%
- Nesting depth <= 3: accuracy > 85%
- Port switch (input↔output) is smooth

Notes:
- We model each sample as a single token stream:
    [BOS, OP, SEP, <input>, SEP, <output>, EOS, PAD...]
  and train next-token prediction.
- "Port switch" is implemented via two SoulEntity interfaces:
    - symbol_in: used up to and including the second SEP (reading / transition)
    - symbol_out: used for output tokens and EOS
"""

from __future__ import annotations

import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ensure HEI is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from curriculum.common.base_trainer import BaseCurriculumTrainer, CurriculumConfig


# =====================
# Token vocabulary
# =====================

TOK_PAD = 0
TOK_BOS = 1
TOK_EOS = 2
TOK_SEP = 3

TOK_OP_COPY = 4
TOK_OP_REV = 5
TOK_OP_SUB = 6
TOK_OP_UNPAREN = 7

TOK_LPAREN = 8
TOK_RPAREN = 9

TOK_A = 10
TOK_B = 11
TOK_C = 12
TOK_D = 13

TOK_X = 14
TOK_Y = 15


def _op_to_token(op_id: int) -> int:
    if op_id == 0:
        return TOK_OP_COPY
    if op_id == 1:
        return TOK_OP_REV
    if op_id == 2:
        return TOK_OP_SUB
    if op_id == 3:
        return TOK_OP_UNPAREN
    raise ValueError(f"Unknown op_id={op_id}")


@dataclass
class Stage3Config(CurriculumConfig):
    # Data / tasks
    vocab_size: int = 32
    sequence_len: int = 32
    max_str_len: int = 10
    max_nesting: int = 3

    # Model / geometry
    dim_q: int = 64
    dim_z: int = 16
    num_charts: int = 8
    integrator_method: str = "semi"
    dt: float = 0.2
    evolution_steps: int = 2
    damping: float = 0.1

    router_context_dim: int = 8
    router_tau: float = 0.0
    transport_threshold: float = 0.0

    port_top_k: int = 2
    port_topk_impl: str = "dense"
    port_decode_state: bool = True
    port_decode_top_k: int = 0

    # Training
    batch_size: int = 256
    steps: int = 1500
    log_every: int = 200
    lr: float = 1e-2
    weight_decay: float = 0.0
    lr_core_scale: float = 0.3

    gamma_pred: float = 1.0
    loss_in_weight: float = 0.1
    loss_out_weight: float = 1.0
    alpha_chart: float = 0.0

    use_cuda_graph: bool = False
    cuda_graph_warmup: int = 3

    # Gate thresholds (temp-04.md)
    gate_transform_acc: float = 0.95
    gate_nesting_acc: float = 0.85
    gate_switch_ratio: float = 2.0
    gate_switch_abs: float = 1.0

    def __str__(self) -> str:
        if os.getenv("HEI_FULL_CONFIG", "0") == "1":
            return self.__repr__()
        return (
            "Stage3Config("
            f"device={self.device}, steps={self.steps}, batch_size={self.batch_size}, "
            f"seq={self.sequence_len}, vocab={self.vocab_size}, charts={self.num_charts}, "
            f"dt={self.dt}, evo={self.evolution_steps}, lr={self.lr}, tau={self.router_tau}, "
            f"topk={self.port_top_k}, cuda_graph={int(self.use_cuda_graph)}"
            ")"
        )


class SymbolPort(nn.Module):
    """
    Token -> u (dim_q) with separate embeddings for input/output modes.
    Decode: chart-conditioned linear readout mixed by chart weights.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        dim_q: int,
        num_charts: int,
        decode_state: bool = True,
        decode_top_k: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim_q = int(dim_q)
        self.num_charts = int(num_charts)
        self.decode_state = bool(decode_state)
        self.decode_top_k = int(decode_top_k or 0)
        self.decode_in_dim = (2 * self.dim_q + 1) if self.decode_state else self.dim_q

        self.embed_in = nn.Embedding(self.vocab_size, self.dim_q)
        self.embed_out = nn.Embedding(self.vocab_size, self.dim_q)
        nn.init.normal_(self.embed_in.weight, std=1.0)
        nn.init.normal_(self.embed_out.weight, std=1.0)

        self.readout_weight = nn.Parameter(torch.empty(self.num_charts, self.vocab_size, self.decode_in_dim))
        self.readout_bias = nn.Parameter(torch.zeros(self.num_charts, self.vocab_size))
        nn.init.normal_(self.readout_weight, std=0.1)
        nn.init.zeros_(self.readout_bias)

        # Guardrail: keep the language-side port lightweight to avoid "port domination"
        # (the dynamics core should carry the computation, not an MLP in the readout).
        self.decode_norm = nn.Identity()
        self.decode_mlp = nn.Identity()

    def encode(self, tokens: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        # mode: 0=input, 1=output
        e_in = self.embed_in(tokens)
        e_out = self.embed_out(tokens)
        m = mode.to(dtype=e_in.dtype).unsqueeze(-1)
        e = (1.0 - m) * e_in + m * e_out
        return torch.tanh(e)

    def decode(self, x: torch.Tensor, chart_weights: torch.Tensor | None) -> torch.Tensor:
        # x: [B,D], chart_weights: [B,K]
        B = x.shape[0]
        K = self.num_charts
        x = self.decode_mlp(self.decode_norm(x))
        logits_k = torch.einsum("bd,kvd->bkv", x, self.readout_weight) + self.readout_bias.unsqueeze(0)  # [B,K,V]
        if chart_weights is None:
            w = x.new_full((B, K), 1.0 / float(K))
        else:
            w = chart_weights
            if self.decode_top_k > 0 and self.decode_top_k < K:
                k = int(self.decode_top_k)
                if k == 1:
                    idx = w.argmax(dim=1)
                    return logits_k.gather(1, idx.view(B, 1, 1).expand(-1, 1, self.vocab_size)).squeeze(1)
                w_vals, idx = torch.topk(w, k=k, dim=1)
                w_norm = w_vals / w_vals.sum(dim=1, keepdim=True).clamp(min=1e-8)
                w_masked = torch.zeros_like(w)
                w_masked.scatter_(1, idx, w_norm)
                w = w_masked
        return (logits_k * w.unsqueeze(-1)).sum(dim=1)


class MemoryDecoder(nn.Module):
    """
    Lightweight pointer-style attention over a dynamics-derived memory bank.

    Guardrail: the decoder does NOT receive privileged signals like (op_id, out_pos, mem_len),
    and the memory bank is NOT raw token embeddings; it must be produced by the dynamics core.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        dim_state: int,
        dim_q: int,
        max_mem_len: int,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim_state = int(dim_state)
        self.dim_q = int(dim_q)
        self.max_mem_len = int(max_mem_len)

        self.mempos_embed = nn.Embedding(self.max_mem_len, self.dim_q)

        self.query_in = nn.Linear(self.dim_state, self.dim_q)
        self.mem_in = nn.Linear(self.dim_state, self.dim_q)
        self.q_proj = nn.Linear(self.dim_q, self.dim_q)
        self.k_proj = nn.Linear(self.dim_q, self.dim_q)
        self.v_proj = nn.Linear(self.dim_q, self.dim_q)
        self.readout = nn.Sequential(
            nn.LayerNorm(self.dim_q),
            nn.Linear(self.dim_q, self.dim_q),
            nn.Tanh(),
            nn.Linear(self.dim_q, self.vocab_size),
        )

        with torch.no_grad():
            nn.init.normal_(self.mempos_embed.weight, std=1.0)

    def forward(
        self,
        *,
        state: torch.Tensor,  # [B,Ds]
        mem_state: torch.Tensor,  # [B,M,Ds]
        mem_mask: torch.Tensor,  # [B,M] bool
    ) -> torch.Tensor:
        B, M, _ = mem_state.shape

        mem = torch.tanh(self.mem_in(mem_state))  # [B,M,D]
        pos_ids = torch.arange(M, device=mem.device, dtype=torch.long).view(1, M).expand(B, -1)
        k_in = mem + self.mempos_embed(pos_ids)
        K = self.k_proj(k_in)  # [B,M,D]
        V = self.v_proj(mem)  # [B,M,D]

        qv = torch.tanh(self.q_proj(torch.tanh(self.query_in(state))))  # [B,D]

        D = int(self.dim_q)
        scale = float(D) ** -0.5
        scores = (K * qv.unsqueeze(1)).sum(dim=2) * scale  # [B,M]
        scores = scores.masked_fill(~mem_mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), V).squeeze(1)  # [B,D]

        h = torch.tanh(ctx + qv)
        return self.readout(h)


class SequenceModel(nn.Module):
    def __init__(
        self,
        *,
        entity,
        port: SymbolPort,
        mem_decoder: MemoryDecoder,
        config: Stage3Config,
        op_embed: nn.Embedding | None,
        mode_embed: nn.Embedding | None,
        pos_embed: nn.Embedding | None,
    ):
        super().__init__()
        self.entity = entity
        self.port = port
        self.mem_decoder = mem_decoder
        self.config = config
        self.op_embed = op_embed
        self.mode_embed = mode_embed
        self.pos_embed = pos_embed
        # Ablation toggles (kept as env flags to avoid threading through configs everywhere).
        self.disable_mem_decoder = os.getenv("HEI_DISABLE_MEM_DECODER", "0") == "1"
        self.disable_mem_bank = os.getenv("HEI_DISABLE_MEM_BANK", "0") == "1"

    def forward(
        self,
        *,
        tokens: torch.Tensor,  # [B,T]
        modes: torch.Tensor,  # [B,T] 0=input, 1=output
        op_ids: torch.Tensor,  # [B]
        out_start: torch.Tensor,  # [B] token index of first output token
        out_len: torch.Tensor,  # [B] output token count (excluding EOS)
        seq_len: torch.Tensor,  # [B] actual sequence length (incl EOS)
        initial_state_flat: torch.Tensor,  # [B,D]
        return_chart_hist: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = tokens.device
        B, T = tokens.shape
        dq = int(self.config.dim_q)
        evo = int(self.config.evolution_steps)
        max_mem = int(getattr(self.config, "max_str_len", 10)) + 2 * int(getattr(self.config, "max_nesting", 3))

        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        modes_in = modes[:, :-1]

        # Output prediction steps: predict output tokens + EOS
        out_start_step = out_start - 1  # SEP2 step predicts first output token
        out_end_step = out_start + out_len - 1  # last output token step predicts EOS
        out_start_step_f = out_start_step.to(dtype=torch.long)
        mem_len = torch.clamp(out_start - 4, min=0, max=max_mem).to(dtype=torch.long)  # inp length
        mem_mask = (torch.arange(max_mem, device=device, dtype=torch.long).view(1, -1) < mem_len.view(-1, 1))
        mem_bank = torch.zeros((B, max_mem, int(initial_state_flat.shape[1])), device=device, dtype=initial_state_flat.dtype)

        state_flat = initial_state_flat
        prev_w = None

        loss_sum = inputs.new_zeros(())
        acc_out_sum = inputs.new_zeros(())
        acc_out_den = inputs.new_zeros(())
        chart_usage = inputs.new_zeros((int(self.config.num_charts),), dtype=torch.float32)

        chart_hist = None
        if return_chart_hist:
            chart_hist = inputs.new_zeros((B, T - 1, int(self.config.num_charts)), dtype=torch.float32)

        in_w = float(getattr(self.config, "loss_in_weight", 0.1) or 0.0)
        out_w = float(getattr(self.config, "loss_out_weight", 1.0) or 1.0)
        gamma_pred = float(getattr(self.config, "gamma_pred", 1.0) or 1.0)

        if self.op_embed is not None:
            op_ctx = torch.tanh(self.op_embed(op_ids))  # [B,C]
        else:
            op_ctx = None

        for t in range(T - 1):
            tok_t = inputs[:, t]
            mode_t = modes_in[:, t]
            tgt_t = targets[:, t]

            # Context for the router: (op, mode)
            router_ctx = None
            if self.mode_embed is not None and op_ctx is not None:
                router_ctx = torch.tanh(op_ctx + self.mode_embed(mode_t))
            elif op_ctx is not None:
                router_ctx = op_ctx
            elif self.mode_embed is not None:
                router_ctx = torch.tanh(self.mode_embed(mode_t))
            if self.pos_embed is not None:
                # Position since output switch (0..max_len+1), available as a time-like context.
                # pos = 0 before output; pos increases on output/EOS prediction steps.
                pos = torch.clamp((t - out_start_step_f), min=0)
                pos = torch.clamp(pos, max=int(self.pos_embed.num_embeddings - 1))
                pos_ctx = torch.tanh(self.pos_embed(pos))
                router_ctx = pos_ctx if router_ctx is None else torch.tanh(router_ctx + pos_ctx)

            u = self.port.encode(tok_t, mode_t)
            # Mixed mode within batch is normal; avoid Python branching per sample by always passing both ports.
            u_in = u * (mode_t == 0).to(dtype=u.dtype).unsqueeze(1)
            u_out = u * (mode_t == 1).to(dtype=u.dtype).unsqueeze(1)
            u_dict = {"symbol_in": u_in, "symbol_out": u_out}

            chart_w = None
            for _ in range(evo):
                out = self.entity.forward_tensor(
                    state_flat=state_flat,
                    u_dict=u_dict,
                    dt=float(self.config.dt),
                    prev_chart_weights=prev_w,
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                    router_context=router_ctx,
                    skip_free_energy=True,
                )
                state_flat = out["next_state_flat"]
                chart_w = out.get("chart_weights", None)
                prev_w = out.get("next_prev_chart_weights", None)

            q = state_flat[:, :dq]
            mem_pos = t - 3  # token index 3 is the first input symbol
            if (not self.disable_mem_bank) and (0 <= mem_pos < max_mem):
                Ds = int(state_flat.shape[1])
                idx = torch.full((B, 1, Ds), int(mem_pos), device=device, dtype=torch.long)
                src = state_flat.unsqueeze(1) * (mem_pos < mem_len).to(dtype=state_flat.dtype).view(B, 1, 1)
                mem_bank = mem_bank.scatter(1, idx, src)
            x = state_flat if bool(getattr(self.port, "decode_state", False)) else q
            logits_state = self.port.decode(x, chart_weights=chart_w)

            is_out = (t >= out_start_step) & (t <= out_end_step)
            if self.disable_mem_decoder:
                logits = logits_state
            else:
                logits_mem = self.mem_decoder(
                    state=state_flat,
                    mem_state=mem_bank,
                    mem_mask=mem_mask,
                )
                logits = torch.where(is_out.unsqueeze(1), logits_mem, logits_state)
            ce = F.cross_entropy(logits, tgt_t, reduction="none")

            # Valid steps stop at (seq_len-1)
            valid = (t < (seq_len - 1)).to(dtype=ce.dtype)
            is_out_f = is_out.to(dtype=ce.dtype)
            w = (in_w + (out_w - in_w) * is_out_f) * valid
            denom = w.sum().clamp(min=1e-8)
            loss_sum = loss_sum + gamma_pred * (w * ce).sum() / denom

            pred = logits.argmax(dim=-1)
            correct = (pred == tgt_t).to(dtype=torch.float32) * valid.to(dtype=torch.float32)
            acc_out_sum = acc_out_sum + (correct * is_out_f.to(dtype=correct.dtype)).sum()
            acc_out_den = acc_out_den + (is_out_f.to(dtype=correct.dtype) * valid.to(dtype=correct.dtype)).sum()

            if chart_w is not None:
                chart_usage = chart_usage + chart_w.sum(dim=0).to(dtype=chart_usage.dtype)
                if chart_hist is not None:
                    chart_hist[:, t, :] = chart_w.to(dtype=chart_hist.dtype)

        mean_loss = loss_sum / float(T - 1)

        dist = chart_usage / (chart_usage.sum() + 1e-8)
        chart_ent = -(dist * torch.log(dist + 1e-8)).sum()
        alpha_chart = float(getattr(self.config, "alpha_chart", 0.0) or 0.0)
        mean_loss = mean_loss - alpha_chart * chart_ent

        diag = {
            "chart_usage": chart_usage,
            "chart_entropy": chart_ent,
            "acc_out": acc_out_sum / (acc_out_den + 1e-8),
        }
        if chart_hist is not None:
            diag["chart_hist"] = chart_hist
        return mean_loss, diag


class Stage3Trainer(BaseCurriculumTrainer):
    def __init__(self, config: Stage3Config):
        super().__init__(config)
        self.config = config

        self.port = SymbolPort(
            vocab_size=int(config.vocab_size),
            dim_q=int(config.dim_q),
            num_charts=int(config.num_charts),
            decode_state=bool(getattr(config, "port_decode_state", True)),
            decode_top_k=int(getattr(config, "port_decode_top_k", 0) or 0),
        ).to(config.device)

        # Two interfaces to make input↔output switching explicit.
        self.entity.add_interface("symbol_in", int(config.dim_q))
        self.entity.add_interface("symbol_out", int(config.dim_q))

        # Router context: operation id + mode id.
        ctx_dim = int(getattr(config, "router_context_dim", 0) or 0)
        self.op_embed = None
        self.mode_embed = None
        self.pos_embed = None
        if ctx_dim > 0:
            self.op_embed = nn.Embedding(4, ctx_dim).to(config.device)
            self.mode_embed = nn.Embedding(2, ctx_dim).to(config.device)
            self.pos_embed = nn.Embedding(int(getattr(config, "max_str_len", 10)) + 2, ctx_dim).to(config.device)
            with torch.no_grad():
                nn.init.normal_(self.op_embed.weight, std=1.0)
                nn.init.normal_(self.mode_embed.weight, std=1.0)
                nn.init.normal_(self.pos_embed.weight, std=1.0)

        # Keep the connection fixed: L3 doesn't require re-learning L2 geometry.
        for p in self.entity.connection.parameters():
            p.requires_grad_(False)

        max_mem_len = int(getattr(config, "max_str_len", 10)) + 2 * int(getattr(config, "max_nesting", 3))
        dim_state = 2 * int(config.dim_q) + 1
        self.mem_decoder = MemoryDecoder(
            vocab_size=int(config.vocab_size),
            dim_state=dim_state,
            dim_q=int(config.dim_q),
            max_mem_len=max_mem_len,
        ).to(config.device)

        lr = float(config.lr)
        weight_decay = float(getattr(config, "weight_decay", 0.0) or 0.0)
        lr_core_scale = float(getattr(config, "lr_core_scale", 1.0) or 1.0)
        lr_core = lr * lr_core_scale

        core_ids = {id(p) for p in self.entity.net_V.parameters()}
        core_ids.update(id(p) for p in self.entity.internal_gen.parameters())
        core_ids.add(id(self.entity.z))

        core_params: List[nn.Parameter] = []
        fast_entity_params: List[nn.Parameter] = []
        for p in self.entity.parameters():
            if not p.requires_grad:
                continue
            (core_params if id(p) in core_ids else fast_entity_params).append(p)

        fast_params: List[nn.Parameter] = []
        fast_params.extend(list(self.port.parameters()))
        fast_params.extend(list(self.mem_decoder.parameters()))
        if self.op_embed is not None:
            fast_params.extend(list(self.op_embed.parameters()))
        if self.mode_embed is not None:
            fast_params.extend(list(self.mode_embed.parameters()))
        if self.pos_embed is not None:
            fast_params.extend(list(self.pos_embed.parameters()))

        param_groups = []
        if core_params:
            param_groups.append({"params": core_params, "lr": lr_core})
        if fast_entity_params:
            param_groups.append({"params": fast_entity_params, "lr": lr})
        if fast_params:
            param_groups.append({"params": fast_params, "lr": lr})

        self._train_params = [p for g in param_groups for p in g["params"] if p.requires_grad]

        use_cudagraph = (
            bool(getattr(config, "use_cuda_graph", False))
            and torch.cuda.is_available()
            and str(config.device).startswith("cuda")
        )
        self.optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            capturable=use_cudagraph,
            foreach=True,
        )

        self.seq_model = SequenceModel(
            entity=self.entity,
            port=self.port,
            mem_decoder=self.mem_decoder,
            config=config,
            op_embed=self.op_embed,
            mode_embed=self.mode_embed,
            pos_embed=self.pos_embed,
        )

        torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def _clip_grad_norm_capturable_(self, max_norm: float = 1.0, eps: float = 1e-6) -> None:
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
        device = self.config.device
        if (not torch.cuda.is_available()) or (not str(device).startswith("cuda")):
            raise RuntimeError("CUDA graph mode requested but CUDA is not available.")

        # Autograd can execute CUDA backward ops from multiple CPU worker threads; per-thread
        # streams can introduce cross-stream dependencies that break CUDA graph capture.
        prev_autograd_mt = None
        if hasattr(torch.autograd, "is_multithreading_enabled") and hasattr(torch.autograd, "set_multithreading_enabled"):
            try:
                prev_autograd_mt = bool(torch.autograd.is_multithreading_enabled())
                if prev_autograd_mt:
                    torch.autograd.set_multithreading_enabled(False)
            except Exception:
                prev_autograd_mt = None

        try:
            print(f"Starting Stage 3 Training (CUDA Graph) on {device}...")
            print(f"Config: {self.config}")
            print("L3: symbolic manipulation")

            os.makedirs(self.config.save_dir, exist_ok=True)
            best_path = os.path.join(self.config.save_dir, "stage3_best.pt")
            best_score = -1.0

            B = int(self.config.batch_size)
            T = int(self.config.sequence_len)
            dq = int(self.config.dim_q)
            D = 2 * dq + 1

            tokens_buf = torch.empty((B, T), device=device, dtype=torch.long)
            modes_buf = torch.empty((B, T), device=device, dtype=torch.long)
            op_ids_buf = torch.empty((B,), device=device, dtype=torch.long)
            out_start_buf = torch.empty((B,), device=device, dtype=torch.long)
            out_len_buf = torch.empty((B,), device=device, dtype=torch.long)
            seq_len_buf = torch.empty((B,), device=device, dtype=torch.long)
            init_state_buf = torch.zeros((B, D), device=device, dtype=torch.float32)

            loss_buf = torch.zeros((), device=device)
            outacc_buf = torch.zeros((), device=device)
            chartent_buf = torch.zeros((), device=device)

            def refill_buffers() -> None:
                batch = self.generate_batch(eval_mode=False)
                tokens_buf.copy_(batch["tokens"])
                modes_buf.copy_(batch["modes"])
                op_ids_buf.copy_(batch["op_ids"])
                out_start_buf.copy_(batch["out_start"])
                out_len_buf.copy_(batch["out_len"])
                seq_len_buf.copy_(batch["seq_len"])

            def fwd_bwd_step() -> None:
                # Keep grads allocated (set_to_none=False) for stable CUDA-graph replay.
                self.optimizer.zero_grad(set_to_none=False)
                loss, diag = self.seq_model(
                    tokens=tokens_buf,
                    modes=modes_buf,
                    op_ids=op_ids_buf,
                    out_start=out_start_buf,
                    out_len=out_len_buf,
                    seq_len=seq_len_buf,
                    initial_state_flat=init_state_buf,
                    return_chart_hist=False,
                )
                loss_buf.copy_(loss.detach())
                outacc_buf.copy_(diag["acc_out"].detach())
                chartent_buf.copy_(diag["chart_entropy"].detach())
                loss.backward()
                self._clip_grad_norm_capturable_(1.0)

            warmup = int(getattr(self.config, "cuda_graph_warmup", 3) or 3)
            warmup = max(1, warmup)
            for _ in range(warmup):
                refill_buffers()
                fwd_bwd_step()
                self.optimizer.step()

            torch.cuda.synchronize()
            pool = torch.cuda.graphs.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                fwd_bwd_step()

            start = time.time()
            for step in range(1, int(self.config.steps) + 1):
                refill_buffers()
                graph.replay()

                if step % int(self.config.log_every) == 0:
                    torch.cuda.synchronize()
                    elapsed = time.time() - start
                    print(
                        f"Step {step}: Loss={float(loss_buf.item()):.4f} "
                        f"OutAcc={float(outacc_buf.item()):.3f} ChartEnt={float(chartent_buf.item()):.2f} "
                        f"Time={elapsed:.1f}s"
                    )
                    try:
                        gate = self.evaluate_L3_gate(
                            batch_size=int(os.getenv("HEI_EVAL_BATCH", "2048")),
                            seed=int(os.getenv("HEI_EVAL_SEED", "0")),
                        )
                        score = float(
                            gate["acc_copy"] + gate["acc_reverse"] + gate["acc_substitute"] + gate["acc_unparen"]
                        )
                        if score > best_score:
                            best_score = score
                            torch.save(self.state_dict(), best_path)
                            print(f"GateBest: score={best_score:.3f} passed={gate['passed']}")
                    except Exception as e:
                        print(f"GateEval: failed ({type(e).__name__}: {e})")

                # Apply the parameter update after (optional) logging.
                self.optimizer.step()

            if os.path.exists(best_path):
                try:
                    self.load_state_dict(torch.load(best_path, map_location=self.config.device))
                    print(f"Loaded best checkpoint: {best_path}")
                except Exception as e:
                    print(f"Best checkpoint load failed ({type(e).__name__}: {e})")

            save_path = os.path.join(self.config.save_dir, "stage3_final.pt")
            torch.save(self.state_dict(), save_path)
            print(f"Stage 3 Complete. Saved to {save_path}")
        finally:
            if prev_autograd_mt is not None:
                try:
                    torch.autograd.set_multithreading_enabled(prev_autograd_mt)
                except Exception:
                    pass

    def load_stage2_entity(self, path: str) -> None:
        sd = torch.load(path, map_location=self.config.device)
        ent_sd = {}
        for k, v in sd.items():
            if k.startswith("entity."):
                ent_sd[k[len("entity.") :]] = v
        cur = self.entity.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        for k, v in ent_sd.items():
            if k not in cur:
                continue
            if tuple(cur[k].shape) != tuple(v.shape):
                continue
            filtered[k] = v
        self.entity.load_state_dict(filtered, strict=False)

    def generate_batch(self, *, eval_mode: bool = False) -> Dict[str, torch.Tensor]:
        B = int(self.config.batch_size)
        T = int(self.config.sequence_len)
        max_len = int(getattr(self.config, "max_str_len", 10))
        max_nest = int(getattr(self.config, "max_nesting", 3))
        max_mem = max_len + 2 * max_nest

        device = self.config.device

        # Balanced ops for eval, mixed for training.
        if eval_mode:
            op_ids = (torch.arange(B, device="cpu") % 4).to(dtype=torch.long)
            op_ids = op_ids[torch.randperm(B)]
        else:
            ops_env = os.getenv("HEI_OPS", "").strip()
            if ops_env:
                allowed = [int(x.strip()) for x in ops_env.split(",") if x.strip() != ""]
                if not allowed:
                    raise RuntimeError("HEI_OPS is set but empty after parsing.")
                allowed_t = torch.tensor(allowed, dtype=torch.long, device="cpu")
                op_ids = allowed_t[torch.randint(0, len(allowed), (B,), device="cpu")]
            else:
                op_ids = torch.randint(0, 4, (B,), device="cpu", dtype=torch.long)

        tokens = torch.full((B, T), TOK_PAD, dtype=torch.long, device="cpu")
        modes = torch.ones((B, T), dtype=torch.long, device="cpu")  # default to output
        out_start = torch.zeros((B,), dtype=torch.long, device="cpu")
        out_len = torch.zeros((B,), dtype=torch.long, device="cpu")
        seq_len = torch.zeros((B,), dtype=torch.long, device="cpu")
        nesting = torch.zeros((B,), dtype=torch.long, device="cpu")

        base_symbols = [TOK_A, TOK_B, TOK_C]
        sub_map = {TOK_A: TOK_X, TOK_B: TOK_Y}

        for i in range(B):
            op = int(op_ids[i].item())
            L = random.randint(1, max_len)
            base = [random.choice(base_symbols) for _ in range(L)]

            if op == 0:  # copy
                inp = base
                out = list(base)
            elif op == 1:  # reverse
                inp = base
                out = list(reversed(base))
            elif op == 2:  # substitute
                inp = base
                out = [sub_map.get(x, x) for x in base]
            else:  # unparen
                d = random.randint(1, max_nest)
                nesting[i] = d
                inp = [TOK_LPAREN] * d + base + [TOK_RPAREN] * d
                out = list(base)

            seq: List[int] = [TOK_BOS, _op_to_token(op), TOK_SEP]
            seq.extend(inp)
            seq.append(TOK_SEP)
            seq.extend(out)
            seq.append(TOK_EOS)

            if len(seq) > T:
                raise RuntimeError(f"sequence_len too small: need={len(seq)} have={T}")

            tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            # Port mode: input until (and incl) second SEP, output after.
            inp_len = len(inp)
            sep2_pos = 3 + inp_len  # token index of the second SEP
            modes[i, : sep2_pos + 1] = 0
            modes[i, sep2_pos + 1 : len(seq)] = 1

            out_start[i] = sep2_pos + 1
            out_len[i] = len(out)
            seq_len[i] = len(seq)

        batch = {
            "tokens": tokens.to(device=device, non_blocking=True),
            "modes": modes.to(device=device, non_blocking=True),
            "op_ids": op_ids.to(device=device, non_blocking=True),
            "out_start": out_start.to(device=device, non_blocking=True),
            "out_len": out_len.to(device=device, non_blocking=True),
            "seq_len": seq_len.to(device=device, non_blocking=True),
            "nesting": nesting.to(device=device, non_blocking=True),
        }
        return batch

    @torch.no_grad()
    def evaluate_L3_gate(
        self,
        *,
        batch_size: int = 4096,
        seed: int = 0,
    ) -> Dict[str, Any]:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        # Build one large eval batch for stable metrics.
        B_old = int(self.config.batch_size)
        try:
            self.config.batch_size = int(batch_size)
            batch = self.generate_batch(eval_mode=True)
        finally:
            self.config.batch_size = B_old

        tokens = batch["tokens"]
        modes = batch["modes"]
        op_ids = batch["op_ids"]
        out_start = batch["out_start"]
        out_len = batch["out_len"]
        seq_len = batch["seq_len"]

        dq = int(self.config.dim_q)
        D = 2 * dq + 1
        init = torch.zeros((tokens.shape[0], D), device=self.config.device, dtype=torch.float32)

        loss, diag = self.seq_model(
            tokens=tokens,
            modes=modes,
            op_ids=op_ids,
            out_start=out_start,
            out_len=out_len,
            seq_len=seq_len,
            initial_state_flat=init,
            return_chart_hist=True,
        )

        # Teacher-forced predictions
        # Re-run with logits to compute per-step argmax without storing logits: we reuse the same run,
        # but SequenceModel only returns metrics. So we approximate by using chart_hist as a proxy
        # for switch smoothness, and re-run a lightweight pass to collect predictions.
        chart_hist = diag["chart_hist"]  # [B,T-1,K]
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        B, TT = inputs.shape

        # Collect predictions under the same rollout (no grad); for simplicity, run a second pass.
        # This keeps the code compact and avoids caching logits.
        state_flat = init
        prev_w = None
        preds = torch.zeros((B, TT), device=tokens.device, dtype=torch.long)
        max_mem = int(getattr(self.config, "max_str_len", 10)) + 2 * int(getattr(self.config, "max_nesting", 3))
        mem_len = torch.clamp(out_start - 4, min=0, max=max_mem).to(dtype=torch.long)
        mem_mask = (torch.arange(max_mem, device=tokens.device, dtype=torch.long).view(1, -1) < mem_len.view(-1, 1))
        mem_bank = torch.zeros((B, max_mem, int(init.shape[1])), device=tokens.device, dtype=init.dtype)

        evo = int(self.config.evolution_steps)
        if self.op_embed is not None:
            op_ctx = torch.tanh(self.op_embed(op_ids))
        else:
            op_ctx = None

        for t in range(TT):
            tok_t = inputs[:, t]
            mode_t = modes[:, t]

            router_ctx = None
            if self.mode_embed is not None and op_ctx is not None:
                router_ctx = torch.tanh(op_ctx + self.mode_embed(mode_t))
            elif op_ctx is not None:
                router_ctx = op_ctx
            elif self.mode_embed is not None:
                router_ctx = torch.tanh(self.mode_embed(mode_t))
            if self.pos_embed is not None:
                pos = torch.clamp((t - (out_start - 1)), min=0)
                pos = torch.clamp(pos, max=int(self.pos_embed.num_embeddings - 1))
                pos_ctx = torch.tanh(self.pos_embed(pos))
                router_ctx = pos_ctx if router_ctx is None else torch.tanh(router_ctx + pos_ctx)

            u = self.port.encode(tok_t, mode_t)
            u_in = u * (mode_t == 0).to(dtype=u.dtype).unsqueeze(1)
            u_out = u * (mode_t == 1).to(dtype=u.dtype).unsqueeze(1)
            u_dict = {"symbol_in": u_in, "symbol_out": u_out}

            chart_w = None
            for _ in range(evo):
                out = self.entity.forward_tensor(
                    state_flat=state_flat,
                    u_dict=u_dict,
                    dt=float(self.config.dt),
                    prev_chart_weights=prev_w,
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                    router_context=router_ctx,
                    skip_free_energy=True,
                )
                state_flat = out["next_state_flat"]
                chart_w = out.get("chart_weights", None)
                prev_w = out.get("next_prev_chart_weights", None)

            q = state_flat[:, :dq]
            mem_pos = t - 3
            if (not getattr(self.seq_model, "disable_mem_bank", False)) and (0 <= mem_pos < max_mem):
                Ds = int(state_flat.shape[1])
                idx = torch.full((B, 1, Ds), int(mem_pos), device=tokens.device, dtype=torch.long)
                src = state_flat.unsqueeze(1) * (mem_pos < mem_len).to(dtype=state_flat.dtype).view(B, 1, 1)
                mem_bank = mem_bank.scatter(1, idx, src)
            x = state_flat if bool(getattr(self.port, "decode_state", False)) else q
            logits_state = self.port.decode(x, chart_weights=chart_w)
            is_out = (t >= (out_start - 1)) & (t <= (out_start + out_len - 1))
            if getattr(self.seq_model, "disable_mem_decoder", False):
                logits = logits_state
            else:
                logits_mem = self.mem_decoder(
                    state=state_flat,
                    mem_state=mem_bank,
                    mem_mask=mem_mask,
                )
                logits = torch.where(is_out.unsqueeze(1), logits_mem, logits_state)
            preds[:, t] = logits.argmax(dim=-1)

        # Sequence-level transform correctness on output region (output tokens + EOS)
        out_start_step = out_start - 1
        out_end_step = out_start + out_len - 1
        steps = torch.arange(TT, device=tokens.device).view(1, -1)  # [1,TT]
        valid_steps = steps < (seq_len - 1).view(-1, 1)
        out_steps = (steps >= out_start_step.view(-1, 1)) & (steps <= out_end_step.view(-1, 1))
        mask = valid_steps & out_steps

        match = (preds == targets) & mask
        # Each sample passes if all masked positions match and mask has expected count (out_len+1).
        needed = (out_len + 1).view(-1)
        got = mask.to(dtype=torch.long).sum(dim=1)
        all_match = match.to(dtype=torch.long).sum(dim=1) == got
        ok = all_match & (got == needed)

        # Per-op accuracies
        acc_by_op = {}
        exact_by_op = {}
        for op in range(4):
            sel = op_ids == op
            if int(sel.sum().item()) == 0:
                acc_by_op[op] = float("nan")
                exact_by_op[op] = float("nan")
            else:
                tok_total = mask[sel].to(dtype=torch.float32).sum().clamp(min=1.0)
                tok_correct = match[sel].to(dtype=torch.float32).sum()
                acc_by_op[op] = float((tok_correct / tok_total).item())
                exact_by_op[op] = float(ok[sel].to(dtype=torch.float32).mean().item())

        # Port switch smoothness: compare chart weights at the switch boundary.
        # switch_t is the first output token index => input step index is switch_t (feeding token[switch_t]).
        switch_t = out_start  # token index
        # chart_hist is [B,T-1,K] for input steps 0..T-2, so index switch_t is valid.
        w_before = chart_hist.gather(1, (switch_t - 1).view(-1, 1, 1).expand(-1, 1, chart_hist.shape[2])).squeeze(1)
        w_after = chart_hist.gather(1, switch_t.view(-1, 1, 1).expand(-1, 1, chart_hist.shape[2])).squeeze(1)
        delta_switch = (w_after - w_before).abs().sum(dim=1)  # [B]

        # Mean adjacent delta within the meaningful prefix+output window (up to EOS prediction step).
        deltas_all = (chart_hist[:, 1:, :] - chart_hist[:, :-1, :]).abs().sum(dim=2)  # [B,TT-1]
        win_end = out_end_step.clamp(min=1).view(-1, 1)  # step index
        idx = torch.arange(deltas_all.shape[1], device=tokens.device).view(1, -1)
        win_mask = idx < win_end
        delta_mean = (deltas_all * win_mask.to(dtype=deltas_all.dtype)).sum(dim=1) / win_mask.sum(dim=1).clamp(min=1)
        ratio = delta_switch / (delta_mean + 1e-8)

        delta_switch_mean = float(delta_switch.mean().item())
        ratio_mean = float(ratio.mean().item())

        # Gate checks
        gate_transform = float(getattr(self.config, "gate_transform_acc", 0.95))
        gate_nesting = float(getattr(self.config, "gate_nesting_acc", 0.85))
        gate_switch_ratio = float(getattr(self.config, "gate_switch_ratio", 2.0))
        gate_switch_abs = float(getattr(self.config, "gate_switch_abs", 1.0))

        acc_copy = acc_by_op[0]
        acc_rev = acc_by_op[1]
        acc_sub = acc_by_op[2]
        acc_unp = acc_by_op[3]

        passed_transform = (acc_copy >= gate_transform) and (acc_rev >= gate_transform) and (acc_sub >= gate_transform)
        passed_nesting = acc_unp >= gate_nesting
        passed_switch = (delta_switch_mean <= gate_switch_abs) and (ratio_mean <= gate_switch_ratio)

        return {
            "acc_copy": acc_copy,
            "acc_reverse": acc_rev,
            "acc_substitute": acc_sub,
            "acc_unparen": acc_unp,
            "exact_copy": exact_by_op[0],
            "exact_reverse": exact_by_op[1],
            "exact_substitute": exact_by_op[2],
            "exact_unparen": exact_by_op[3],
            "delta_switch_mean_l1": delta_switch_mean,
            "delta_switch_ratio": ratio_mean,
            "loss": float(loss.item()),
            "passed_components": {
                "transform": bool(passed_transform),
                "nesting": bool(passed_nesting),
                "port_switch": bool(passed_switch),
            },
            "passed": bool(passed_transform and passed_nesting and passed_switch),
        }

    def train_loop(self) -> None:
        use_cudagraph = (
            bool(getattr(self.config, "use_cuda_graph", False))
            and torch.cuda.is_available()
            and str(self.config.device).startswith("cuda")
        )
        if use_cudagraph:
            return self._train_loop_cuda_graph()

        print(f"Starting Stage 3 Training on {self.config.device}...")
        print(f"Config: {self.config}")
        print("L3: symbolic manipulation")

        best_path = os.path.join(self.config.save_dir, "stage3_best.pt")
        best_score = -1.0

        start = time.time()
        for step in range(1, int(self.config.steps) + 1):
            self.optimizer.zero_grad(set_to_none=True)
            batch = self.generate_batch(eval_mode=False)

            dq = int(self.config.dim_q)
            D = 2 * dq + 1
            init = torch.zeros((int(self.config.batch_size), D), device=self.config.device, dtype=torch.float32)

            loss, diag = self.seq_model(
                tokens=batch["tokens"],
                modes=batch["modes"],
                op_ids=batch["op_ids"],
                out_start=batch["out_start"],
                out_len=batch["out_len"],
                seq_len=batch["seq_len"],
                initial_state_flat=init,
                return_chart_hist=False,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._train_params, 1.0)
            self.optimizer.step()

            if step % int(self.config.log_every) == 0:
                elapsed = time.time() - start
                print(
                    f"Step {step}: Loss={float(loss.item()):.4f} "
                    f"OutAcc={float(diag['acc_out'].item()):.3f} ChartEnt={float(diag['chart_entropy'].item()):.2f} "
                    f"Time={elapsed:.1f}s"
                )
                try:
                    gate = self.evaluate_L3_gate(
                        batch_size=int(os.getenv("HEI_EVAL_BATCH", "2048")),
                        seed=int(os.getenv("HEI_EVAL_SEED", "0")),
                    )
                    score = float(gate["acc_copy"] + gate["acc_reverse"] + gate["acc_substitute"] + gate["acc_unparen"])
                    if score > best_score:
                        best_score = score
                        torch.save(self.state_dict(), best_path)
                        print(f"GateBest: score={best_score:.3f} passed={gate['passed']}")
                except Exception as e:
                    print(f"GateEval: failed ({type(e).__name__}: {e})")

        if os.path.exists(best_path):
            try:
                self.load_state_dict(torch.load(best_path, map_location=self.config.device))
                print(f"Loaded best checkpoint: {best_path}")
            except Exception as e:
                print(f"Best checkpoint load failed ({type(e).__name__}: {e})")

        os.makedirs(self.config.save_dir, exist_ok=True)
        save_path = os.path.join(self.config.save_dir, "stage3_final.pt")
        torch.save(self.state_dict(), save_path)
        print(f"Stage 3 Complete. Saved to {save_path}")


def _apply_env_overrides(config: Stage3Config) -> Stage3Config:
    if os.getenv("HEI_SEED") is not None:
        seed = int(os.getenv("HEI_SEED", "0"))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if os.getenv("HEI_STEPS") is not None:
        config.steps = int(os.getenv("HEI_STEPS", str(config.steps)))
    if os.getenv("HEI_LOG_EVERY") is not None:
        config.log_every = int(os.getenv("HEI_LOG_EVERY", str(config.log_every)))
    if os.getenv("HEI_BATCH_SIZE") is not None:
        config.batch_size = int(os.getenv("HEI_BATCH_SIZE", str(config.batch_size)))
    if os.getenv("HEI_SEQ_LEN") is not None:
        config.sequence_len = int(os.getenv("HEI_SEQ_LEN", str(config.sequence_len)))
    if os.getenv("HEI_MAX_STR_LEN") is not None:
        config.max_str_len = int(os.getenv("HEI_MAX_STR_LEN", str(config.max_str_len)))
    if os.getenv("HEI_MAX_NESTING") is not None:
        config.max_nesting = int(os.getenv("HEI_MAX_NESTING", str(config.max_nesting)))
    if os.getenv("HEI_LR") is not None:
        config.lr = float(os.getenv("HEI_LR", str(config.lr)))
    if os.getenv("HEI_LR_CORE_SCALE") is not None:
        config.lr_core_scale = float(os.getenv("HEI_LR_CORE_SCALE", str(config.lr_core_scale)))
    if os.getenv("HEI_DT") is not None:
        config.dt = float(os.getenv("HEI_DT", str(config.dt)))
    if os.getenv("HEI_EVO") is not None:
        config.evolution_steps = int(os.getenv("HEI_EVO", str(config.evolution_steps)))
    if os.getenv("HEI_DAMPING") is not None:
        config.damping = float(os.getenv("HEI_DAMPING", str(config.damping)))
    if os.getenv("HEI_ROUTER_TAU") is not None:
        config.router_tau = float(os.getenv("HEI_ROUTER_TAU", str(config.router_tau)))
    if os.getenv("HEI_TRANSPORT_THRESHOLD") is not None:
        config.transport_threshold = float(os.getenv("HEI_TRANSPORT_THRESHOLD", str(config.transport_threshold)))
    if os.getenv("HEI_ROUTER_CTX") is not None:
        config.router_context_dim = int(os.getenv("HEI_ROUTER_CTX", str(config.router_context_dim)))
    if os.getenv("HEI_PORT_TOP_K") is not None:
        config.port_top_k = int(os.getenv("HEI_PORT_TOP_K", str(config.port_top_k)))
    if os.getenv("HEI_PORT_DECODE_TOP_K") is not None:
        config.port_decode_top_k = int(os.getenv("HEI_PORT_DECODE_TOP_K", str(config.port_decode_top_k)))
    if os.getenv("HEI_ALPHA_CHART") is not None:
        config.alpha_chart = float(os.getenv("HEI_ALPHA_CHART", str(config.alpha_chart)))
    if os.getenv("HEI_LOSS_IN_W") is not None:
        config.loss_in_weight = float(os.getenv("HEI_LOSS_IN_W", str(config.loss_in_weight)))
    if os.getenv("HEI_LOSS_OUT_W") is not None:
        config.loss_out_weight = float(os.getenv("HEI_LOSS_OUT_W", str(config.loss_out_weight)))
    if os.getenv("HEI_CUDA_GRAPH") is not None:
        config.use_cuda_graph = os.getenv("HEI_CUDA_GRAPH", "0") == "1"
    if os.getenv("HEI_CUDA_GRAPH_WARMUP") is not None:
        config.cuda_graph_warmup = int(os.getenv("HEI_CUDA_GRAPH_WARMUP", str(config.cuda_graph_warmup)))
    return config


if __name__ == "__main__":
    config = _apply_env_overrides(Stage3Config())
    trainer = Stage3Trainer(config)

    resume = os.getenv("HEI_STAGE3_RESUME", "")
    if resume and os.path.exists(resume):
        trainer.load_state_dict(torch.load(resume, map_location=config.device), strict=False)
        print(f"Resumed Stage3 from {resume}")
    else:
        ckpt = os.getenv("HEI_STAGE2_CKPT", "checkpoints/curriculum/stage2_final.pt")
        if ckpt and os.path.exists(ckpt):
            trainer.load_stage2_entity(ckpt)
            print(f"Loaded Stage2 entity init from {ckpt}")
        else:
            print("Stage2 init not found; training Stage3 from scratch.")

    trainer.train_loop()

    if os.getenv("HEI_EVAL_L3", "1") == "1":
        gate = trainer.evaluate_L3_gate(
            batch_size=int(os.getenv("HEI_EVAL_BATCH", "4096")),
            seed=int(os.getenv("HEI_EVAL_SEED", "0")),
        )
        pc = gate["passed_components"]
        print(
            "=== L3 Gate (Eval) ===\n"
            f"Copy TokAcc={gate['acc_copy']:.3f} Exact={gate['exact_copy']:.3f} (thr={config.gate_transform_acc:.2f})\n"
            f"Reverse TokAcc={gate['acc_reverse']:.3f} Exact={gate['exact_reverse']:.3f} (thr={config.gate_transform_acc:.2f})\n"
            f"Substitute TokAcc={gate['acc_substitute']:.3f} Exact={gate['exact_substitute']:.3f} (thr={config.gate_transform_acc:.2f})\n"
            f"Unparen TokAcc={gate['acc_unparen']:.3f} Exact={gate['exact_unparen']:.3f} (thr={config.gate_nesting_acc:.2f})\n"
            f"PortSwitch ΔL1={gate['delta_switch_mean_l1']:.3f} ratio={gate['delta_switch_ratio']:.2f} "
            f"(thr_abs={config.gate_switch_abs:.2f} thr_ratio={config.gate_switch_ratio:.2f})\n"
            f"Passed: {gate['passed']} Components={pc}"
        )
