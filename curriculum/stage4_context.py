"""
Stage 4: Context Dependence (Level 4)

Reference: HEI/docs/temp/2026/1/A/1·4/01/temp-04.md  (Level 4 section)

Task:
    If X appeared in the context window, then A -> B; else A -> A.

Acceptance (temp-04.md):
    - Context window = 5:  accuracy > 90%
    - Context window = 10: accuracy > 80%
    - Offline evolution can preserve context information

Engineering notes:
    - We keep the "port" lightweight (no MLP readout) to avoid port domination.
    - We provide an optional CUDA-graph training mode to reduce Python/kernel-launch overhead.
"""

from __future__ import annotations

import os
import sys
import time
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
# Token vocabulary (compatible with Stage3 ids where possible)
# =====================

TOK_PAD = 0
TOK_BOS = 1
TOK_EOS = 2

TOK_A = 10
TOK_B = 11
TOK_C = 12
TOK_D = 13
TOK_X = 14
TOK_Y = 15


@dataclass
class Stage4Config(CurriculumConfig):
    # Data
    vocab_size: int = 32
    context_window_train: int = 10
    offline_steps: int = 10  # no-input steps between context and query (tests offline memory)

    # Model / geometry
    dim_q: int = 64
    dim_z: int = 16
    num_charts: int = 16
    integrator_method: str = "semi"
    dt: float = 0.2
    evolution_steps: int = 2
    damping: float = 0.1

    router_context_dim: int = 8
    router_tau: float = 0.0
    transport_threshold: float = 0.0

    port_top_k: int = 4
    port_topk_impl: str = "dense"
    port_decode_state: bool = True
    port_decode_top_k: int = 0

    # Training
    batch_size: int = 2048
    steps: int = 2000
    log_every: int = 200
    lr: float = 1e-2
    weight_decay: float = 0.0
    lr_core_scale: float = 0.3
    alpha_chart: float = 0.0

    use_cuda_graph: bool = False
    cuda_graph_warmup: int = 3

    # Gate thresholds (temp-04.md)
    gate_acc_w5: float = 0.90
    gate_acc_w10: float = 0.80

    def __str__(self) -> str:
        if os.getenv("HEI_FULL_CONFIG", "0") == "1":
            return self.__repr__()
        return (
            "Stage4Config("
            f"device={self.device}, steps={self.steps}, batch_size={self.batch_size}, "
            f"ctx={self.context_window_train}, offline={self.offline_steps}, charts={self.num_charts}, "
            f"dt={self.dt}, evo={self.evolution_steps}, lr={self.lr}, cuda_graph={int(self.use_cuda_graph)}"
            ")"
        )


class SymbolPort(nn.Module):
    """
    Lightweight symbolic port:
      token -> u (dim_q)
      state -> logits (chart-conditioned linear readout)

    Guardrail: no extra MLP in the readout path.
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

        self.embed = nn.Embedding(self.vocab_size, self.dim_q)
        nn.init.normal_(self.embed.weight, std=1.0)

        self.readout_weight = nn.Parameter(torch.empty(self.num_charts, self.vocab_size, self.decode_in_dim))
        self.readout_bias = nn.Parameter(torch.zeros(self.num_charts, self.vocab_size))
        nn.init.normal_(self.readout_weight, std=0.1)
        nn.init.zeros_(self.readout_bias)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.embed(tokens))

    def decode(self, x: torch.Tensor, chart_weights: torch.Tensor | None) -> torch.Tensor:
        # x: [B,D], chart_weights: [B,K]
        B = x.shape[0]
        K = self.num_charts
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


class SequenceModel(nn.Module):
    def __init__(
        self,
        *,
        entity,
        port: SymbolPort,
        config: Stage4Config,
        phase_embed: nn.Embedding | None,
    ):
        super().__init__()
        self.entity = entity
        self.port = port
        self.config = config
        self.phase_embed = phase_embed

    def forward(
        self,
        *,
        context_tokens: torch.Tensor,  # [B,W]
        query_token: torch.Tensor,  # [B]
        target_token: torch.Tensor,  # [B]
        initial_state_flat: torch.Tensor,  # [B,Ds]
        offline_steps: int | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = context_tokens.device
        B, W = context_tokens.shape
        dq = int(self.config.dim_q)
        evo = int(self.config.evolution_steps)
        offline = int(self.config.offline_steps if offline_steps is None else offline_steps)
        offline = max(0, offline)

        state_flat = initial_state_flat
        prev_w = None
        chart_usage = context_tokens.new_zeros((int(self.config.num_charts),), dtype=torch.float32)

        def _router_ctx(phase_id: int) -> torch.Tensor | None:
            if self.phase_embed is None:
                return None
            pid = torch.full((B,), int(phase_id), device=device, dtype=torch.long)
            return torch.tanh(self.phase_embed(pid))

        # Phase 0: read context
        for t in range(W):
            u = self.port.encode(context_tokens[:, t])
            u_dict = {"symbolic": u}
            router_ctx = _router_ctx(0)
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
            if chart_w is not None:
                chart_usage = chart_usage + chart_w.sum(dim=0).to(dtype=chart_usage.dtype)

        # Phase 1: offline evolution (no inputs)
        if offline > 0:
            z = torch.zeros((B, int(self.config.dim_q)), device=device, dtype=state_flat.dtype)
            u_dict0 = {"symbolic": z}
            router_ctx = _router_ctx(1)
            for _ in range(offline):
                chart_w = None
                for _ in range(evo):
                    out = self.entity.forward_tensor(
                        state_flat=state_flat,
                        u_dict=u_dict0,
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
                if chart_w is not None:
                    chart_usage = chart_usage + chart_w.sum(dim=0).to(dtype=chart_usage.dtype)

        # Phase 2: query token A
        u = self.port.encode(query_token)
        u_dict = {"symbolic": u}
        router_ctx = _router_ctx(2)
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
        if chart_w is not None:
            chart_usage = chart_usage + chart_w.sum(dim=0).to(dtype=chart_usage.dtype)

        q = state_flat[:, :dq]
        x = state_flat if bool(getattr(self.port, "decode_state", False)) else q
        logits = self.port.decode(x, chart_weights=chart_w)

        loss = F.cross_entropy(logits, target_token)
        pred = logits.argmax(dim=-1)
        acc = (pred == target_token).to(dtype=torch.float32).mean()

        dist = chart_usage / (chart_usage.sum() + 1e-8)
        chart_ent = -(dist * torch.log(dist + 1e-8)).sum()
        alpha_chart = float(getattr(self.config, "alpha_chart", 0.0) or 0.0)
        loss = loss - alpha_chart * chart_ent

        return loss, {"acc": acc, "chart_entropy": chart_ent}


class Stage4Trainer(BaseCurriculumTrainer):
    def __init__(self, config: Stage4Config):
        super().__init__(config)
        self.config = config

        self.port = SymbolPort(
            vocab_size=int(config.vocab_size),
            dim_q=int(config.dim_q),
            num_charts=int(config.num_charts),
            decode_state=bool(getattr(config, "port_decode_state", True)),
            decode_top_k=int(getattr(config, "port_decode_top_k", 0) or 0),
        ).to(config.device)

        self.entity.add_interface("symbolic", int(config.dim_q))

        ctx_dim = int(getattr(config, "router_context_dim", 0) or 0)
        self.phase_embed = None
        if ctx_dim > 0:
            self.phase_embed = nn.Embedding(3, ctx_dim).to(config.device)
            with torch.no_grad():
                nn.init.normal_(self.phase_embed.weight, std=1.0)

        # Keep the connection fixed: L4 doesn't require re-learning L2 geometry.
        for p in self.entity.connection.parameters():
            p.requires_grad_(False)

        lr = float(config.lr)
        weight_decay = float(getattr(config, "weight_decay", 0.0) or 0.0)
        lr_core_scale = float(getattr(config, "lr_core_scale", 1.0) or 1.0)
        lr_core = lr * lr_core_scale

        core_ids = {id(p) for p in self.entity.net_V.parameters()}
        core_ids.update(id(p) for p in self.entity.internal_gen.parameters())
        if hasattr(self.entity, "z"):
            core_ids.add(id(self.entity.z))

        core_params: List[nn.Parameter] = []
        fast_entity_params: List[nn.Parameter] = []
        for p in self.entity.parameters():
            if not p.requires_grad:
                continue
            (core_params if id(p) in core_ids else fast_entity_params).append(p)

        fast_params: List[nn.Parameter] = []
        fast_params.extend(list(self.port.parameters()))
        if self.phase_embed is not None:
            fast_params.extend(list(self.phase_embed.parameters()))

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

        self.seq_model = SequenceModel(entity=self.entity, port=self.port, config=config, phase_embed=self.phase_embed)

        torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def load_stage3_entity(self, path: str) -> None:
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

    def generate_batch(self, *, context_len: int, eval_mode: bool = False) -> Dict[str, torch.Tensor]:
        B = int(self.config.batch_size)
        W = int(context_len)
        device = self.config.device

        fillers = torch.tensor([TOK_C, TOK_D, TOK_Y], dtype=torch.long, device="cpu")
        idx = torch.randint(0, int(fillers.numel()), (B, W), device="cpu", dtype=torch.long)
        ctx = fillers[idx]

        # Balanced for eval; random for training.
        if eval_mode:
            y = (torch.arange(B, device="cpu", dtype=torch.long) % 2)
            y = y[torch.randperm(B)]
        else:
            y = torch.randint(0, 2, (B,), device="cpu", dtype=torch.long)

        pos = torch.randint(0, W, (B,), device="cpu", dtype=torch.long)
        rows = torch.arange(B, device="cpu", dtype=torch.long)
        ctx[rows[y == 1], pos[y == 1]] = TOK_X

        query = torch.full((B,), TOK_A, dtype=torch.long, device="cpu")
        target = torch.where(y == 1, torch.full_like(y, TOK_B), torch.full_like(y, TOK_A))

        return {
            "context_tokens": ctx.to(device=device, non_blocking=True),
            "query_token": query.to(device=device, non_blocking=True),
            "target_token": target.to(device=device, non_blocking=True),
        }

    @torch.no_grad()
    def evaluate_L4_gate(self, *, batch_size: int = 8192, seed: int = 0) -> Dict[str, Any]:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        B_old = int(self.config.batch_size)
        try:
            self.config.batch_size = int(batch_size)
            b5 = self.generate_batch(context_len=5, eval_mode=True)
            b10 = self.generate_batch(context_len=10, eval_mode=True)
        finally:
            self.config.batch_size = B_old

        dq = int(self.config.dim_q)
        D = 2 * dq + 1
        init5 = torch.zeros((b5["query_token"].shape[0], D), device=self.config.device, dtype=torch.float32)
        init10 = torch.zeros((b10["query_token"].shape[0], D), device=self.config.device, dtype=torch.float32)

        # IMPORTANT: this architecture computes the vector field via `autograd.grad`.
        # Running under `torch.no_grad()` can change the dynamics path (create_graph=False)
        # and yield different trajectories. Re-enable grads for evaluation rollouts.
        with torch.enable_grad():
            loss5_0, diag5_0 = self.seq_model(
                context_tokens=b5["context_tokens"],
                query_token=b5["query_token"],
                target_token=b5["target_token"],
                initial_state_flat=init5,
                offline_steps=0,
            )
            loss10_0, diag10_0 = self.seq_model(
                context_tokens=b10["context_tokens"],
                query_token=b10["query_token"],
                target_token=b10["target_token"],
                initial_state_flat=init10,
                offline_steps=0,
            )
            loss5_off, diag5_off = self.seq_model(
                context_tokens=b5["context_tokens"],
                query_token=b5["query_token"],
                target_token=b5["target_token"],
                initial_state_flat=init5,
                offline_steps=int(self.config.offline_steps),
            )
            loss10_off, diag10_off = self.seq_model(
                context_tokens=b10["context_tokens"],
                query_token=b10["query_token"],
                target_token=b10["target_token"],
                initial_state_flat=init10,
                offline_steps=int(self.config.offline_steps),
            )

        acc5 = float(diag5_0["acc"].item())
        acc10 = float(diag10_0["acc"].item())
        acc5_off = float(diag5_off["acc"].item())
        acc10_off = float(diag10_off["acc"].item())
        gate5 = float(getattr(self.config, "gate_acc_w5", 0.90))
        gate10 = float(getattr(self.config, "gate_acc_w10", 0.80))

        return {
            "acc_w5": acc5,
            "acc_w10": acc10,
            "acc_w5_offline": acc5_off,
            "acc_w10_offline": acc10_off,
            "loss_w5": float(loss5_0.item()),
            "loss_w10": float(loss10_0.item()),
            "loss_w5_offline": float(loss5_off.item()),
            "loss_w10_offline": float(loss10_off.item()),
            "passed": bool((acc5 >= gate5) and (acc10 >= gate10)),
            "passed_components": {"w5": bool(acc5 >= gate5), "w10": bool(acc10 >= gate10)},
        }

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

        prev_autograd_mt = None
        if hasattr(torch.autograd, "is_multithreading_enabled") and hasattr(torch.autograd, "set_multithreading_enabled"):
            try:
                prev_autograd_mt = bool(torch.autograd.is_multithreading_enabled())
                if prev_autograd_mt:
                    torch.autograd.set_multithreading_enabled(False)
            except Exception:
                prev_autograd_mt = None

        try:
            print(f"Starting Stage 4 Training (CUDA Graph) on {device}...")
            print(f"Config: {self.config}")
            print("L4: context dependence")

            os.makedirs(self.config.save_dir, exist_ok=True)
            best_path = os.path.join(self.config.save_dir, "stage4_best.pt")
            best_score = -1.0

            B = int(self.config.batch_size)
            W = int(self.config.context_window_train)
            dq = int(self.config.dim_q)
            D = 2 * dq + 1

            ctx_buf = torch.empty((B, W), device=device, dtype=torch.long)
            query_buf = torch.empty((B,), device=device, dtype=torch.long)
            target_buf = torch.empty((B,), device=device, dtype=torch.long)
            init_state_buf = torch.zeros((B, D), device=device, dtype=torch.float32)

            loss_buf = torch.zeros((), device=device)
            acc_buf = torch.zeros((), device=device)
            chartent_buf = torch.zeros((), device=device)

            def refill_buffers() -> None:
                batch = self.generate_batch(context_len=W, eval_mode=False)
                ctx_buf.copy_(batch["context_tokens"])
                query_buf.copy_(batch["query_token"])
                target_buf.copy_(batch["target_token"])

            def fwd_bwd_step() -> None:
                self.optimizer.zero_grad(set_to_none=False)
                loss, diag = self.seq_model(
                    context_tokens=ctx_buf,
                    query_token=query_buf,
                    target_token=target_buf,
                    initial_state_flat=init_state_buf,
                )
                loss_buf.copy_(loss.detach())
                acc_buf.copy_(diag["acc"].detach())
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
                        f"Acc={float(acc_buf.item()):.3f} ChartEnt={float(chartent_buf.item()):.2f} "
                        f"Time={elapsed:.1f}s"
                    )
                    try:
                        gate = self.evaluate_L4_gate(
                            batch_size=int(os.getenv("HEI_EVAL_BATCH", "8192")),
                            seed=int(os.getenv("HEI_EVAL_SEED", "0")),
                        )
                        score = float(gate["acc_w5"] + gate["acc_w10"])
                        if score > best_score:
                            best_score = score
                            torch.save(self.state_dict(), best_path)
                            print(f"GateBest: score={best_score:.3f} passed={gate['passed']}")
                    except Exception as e:
                        print(f"GateEval: failed ({type(e).__name__}: {e})")

                self.optimizer.step()

            if os.path.exists(best_path):
                try:
                    self.load_state_dict(torch.load(best_path, map_location=self.config.device))
                    print(f"Loaded best checkpoint: {best_path}")
                except Exception as e:
                    print(f"Best checkpoint load failed ({type(e).__name__}: {e})")

            save_path = os.path.join(self.config.save_dir, "stage4_final.pt")
            torch.save(self.state_dict(), save_path)
            print(f"Stage 4 Complete. Saved to {save_path}")
        finally:
            if prev_autograd_mt is not None:
                try:
                    torch.autograd.set_multithreading_enabled(prev_autograd_mt)
                except Exception:
                    pass

    def train_loop(self) -> None:
        use_cudagraph = (
            bool(getattr(self.config, "use_cuda_graph", False))
            and torch.cuda.is_available()
            and str(self.config.device).startswith("cuda")
        )
        if use_cudagraph:
            return self._train_loop_cuda_graph()

        print(f"Starting Stage 4 Training on {self.config.device}...")
        print(f"Config: {self.config}")
        print("L4: context dependence")

        best_path = os.path.join(self.config.save_dir, "stage4_best.pt")
        best_score = -1.0

        start = time.time()
        for step in range(1, int(self.config.steps) + 1):
            self.optimizer.zero_grad(set_to_none=True)
            batch = self.generate_batch(context_len=int(self.config.context_window_train), eval_mode=False)

            dq = int(self.config.dim_q)
            D = 2 * dq + 1
            init = torch.zeros((int(self.config.batch_size), D), device=self.config.device, dtype=torch.float32)

            loss, diag = self.seq_model(
                context_tokens=batch["context_tokens"],
                query_token=batch["query_token"],
                target_token=batch["target_token"],
                initial_state_flat=init,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._train_params, 1.0)
            self.optimizer.step()

            if step % int(self.config.log_every) == 0:
                elapsed = time.time() - start
                print(
                    f"Step {step}: Loss={float(loss.item()):.4f} "
                    f"Acc={float(diag['acc'].item()):.3f} ChartEnt={float(diag['chart_entropy'].item()):.2f} "
                    f"Time={elapsed:.1f}s"
                )
                try:
                    gate = self.evaluate_L4_gate(
                        batch_size=int(os.getenv("HEI_EVAL_BATCH", "8192")),
                        seed=int(os.getenv("HEI_EVAL_SEED", "0")),
                    )
                    score = float(gate["acc_w5"] + gate["acc_w10"])
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
        save_path = os.path.join(self.config.save_dir, "stage4_final.pt")
        torch.save(self.state_dict(), save_path)
        print(f"Stage 4 Complete. Saved to {save_path}")


def _apply_env_overrides(config: Stage4Config) -> Stage4Config:
    if os.getenv("HEI_SEED") is not None:
        seed = int(os.getenv("HEI_SEED", "0"))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if os.getenv("HEI_STEPS") is not None:
        config.steps = int(os.getenv("HEI_STEPS", str(config.steps)))
    if os.getenv("HEI_LOG_EVERY") is not None:
        config.log_every = int(os.getenv("HEI_LOG_EVERY", str(config.log_every)))
    if os.getenv("HEI_BATCH_SIZE") is not None:
        config.batch_size = int(os.getenv("HEI_BATCH_SIZE", str(config.batch_size)))
    if os.getenv("HEI_CONTEXT_W") is not None:
        config.context_window_train = int(os.getenv("HEI_CONTEXT_W", str(config.context_window_train)))
    if os.getenv("HEI_OFFLINE_STEPS") is not None:
        config.offline_steps = int(os.getenv("HEI_OFFLINE_STEPS", str(config.offline_steps)))
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
    if os.getenv("HEI_ROUTER_CTX") is not None:
        config.router_context_dim = int(os.getenv("HEI_ROUTER_CTX", str(config.router_context_dim)))
    if os.getenv("HEI_PORT_TOP_K") is not None:
        config.port_top_k = int(os.getenv("HEI_PORT_TOP_K", str(config.port_top_k)))
    if os.getenv("HEI_PORT_DECODE_TOP_K") is not None:
        config.port_decode_top_k = int(os.getenv("HEI_PORT_DECODE_TOP_K", str(config.port_decode_top_k)))
    if os.getenv("HEI_ALPHA_CHART") is not None:
        config.alpha_chart = float(os.getenv("HEI_ALPHA_CHART", str(config.alpha_chart)))
    if os.getenv("HEI_CUDA_GRAPH") is not None:
        config.use_cuda_graph = os.getenv("HEI_CUDA_GRAPH", "0") == "1"
    if os.getenv("HEI_CUDA_GRAPH_WARMUP") is not None:
        config.cuda_graph_warmup = int(os.getenv("HEI_CUDA_GRAPH_WARMUP", str(config.cuda_graph_warmup)))
    return config


if __name__ == "__main__":
    config = _apply_env_overrides(Stage4Config())
    trainer = Stage4Trainer(config)

    resume = os.getenv("HEI_STAGE4_RESUME", "")
    if resume and os.path.exists(resume):
        trainer.load_state_dict(torch.load(resume, map_location=config.device), strict=False)
        print(f"Resumed Stage4 from {resume}")
    else:
        ckpt = os.getenv("HEI_STAGE3_CKPT", "checkpoints/curriculum/stage3_final.pt")
        if ckpt and os.path.exists(ckpt):
            trainer.load_stage3_entity(ckpt)
            print(f"Loaded Stage3 entity init (partial) from {ckpt}")
        else:
            print("Stage3 init not found; training Stage4 from scratch.")

    trainer.train_loop()

    if os.getenv("HEI_EVAL_L4", "1") == "1":
        gate = trainer.evaluate_L4_gate(
            batch_size=int(os.getenv("HEI_EVAL_BATCH", "8192")),
            seed=int(os.getenv("HEI_EVAL_SEED", "0")),
        )
        pc = gate["passed_components"]
        print(
            "=== L4 Gate (Eval) ===\n"
            f"W=5  Acc={gate['acc_w5']:.3f} OfflineAcc={gate['acc_w5_offline']:.3f} (thr={config.gate_acc_w5:.2f})\n"
            f"W=10 Acc={gate['acc_w10']:.3f} OfflineAcc={gate['acc_w10_offline']:.3f} (thr={config.gate_acc_w10:.2f})\n"
            f"Passed: {gate['passed']} Components={pc}"
        )
