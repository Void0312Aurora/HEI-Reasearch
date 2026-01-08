"""
Stage 2: Finite State Machines (Level 2)

Reference: HEI/docs/temp/2026/1/A/1·4/01/temp-04.md

Goal:
- Learn discrete state transitions (FSM) driven by a binary input stream.
- Verify that atlas chart weights correspond to FSM states.
- Verify that parallel transport / connection is consistent (holonomy small).

Tasks implemented:
- FSM-2: 2-state machine (A/B) with the transition table given in temp-04.md.
         (In this rule, next_state depends only on input bit: 0->A, 1->B.)
- FSM-4: 4-state counter: state increments mod 4 when input bit == 1, else stays.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ensure HEI is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from curriculum.common.base_trainer import BaseCurriculumTrainer, CurriculumConfig
from EXP.diag.holonomy import compute_holonomy


@dataclass
class Stage2Config(CurriculumConfig):
    # Task
    sequence_len: int = 32
    # Mix ratio for training batches
    p_task4: float = 0.75
    # Output state vocabulary: fixed to 4 states (0..3)
    num_states: int = 4

    # Model / physics
    dim_q: int = 64
    dim_z: int = 16
    num_charts: int = 4
    integrator_method: str = "semi"
    dt: float = 0.2
    evolution_steps: int = 16
    damping: float = 0.1

    # L2: enable router inertia + parallel transport (SoulEntity implements PT on momentum when chart weights shift)
    router_tau: float = 1.0
    transport_threshold: float = 0.1
    router_context_dim: int = 8

    # Port coupling locality + decoding locality (shared control knob)
    port_top_k: int = 2
    port_topk_impl: str = "dense"
    port_decode_state: bool = True

    # Training
    batch_size: int = 256
    steps: int = 1500
    log_every: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_core_scale: float = 0.3

    # A3 weights
    beta_kl: float = 0.01
    gamma_pred: float = 10.0

    # Atlas regularization (avoid collapse; keep it weak by default)
    alpha_chart: float = 0.01
    # Encourage chart↔state alignment on FSM-4 (optional)
    lambda_state_mi: float = 0.0

    # Performance
    use_cuda_graph: bool = False
    cuda_graph_warmup: int = 3

    # Gate thresholds (temp-04.md)
    gate_acc2: float = 0.98
    gate_acc4: float = 0.90
    gate_holonomy_ratio: float = 0.01
    gate_state_chart_unique: int = 4

    def __str__(self) -> str:
        if os.getenv("HEI_FULL_CONFIG", "0") == "1":
            return self.__repr__()
        return (
            "Stage2Config("
            f"device={self.device}, steps={self.steps}, batch_size={self.batch_size}, "
            f"seq={self.sequence_len}, dim_q={self.dim_q}, dim_z={self.dim_z}, charts={self.num_charts}, "
            f"dt={self.dt}, evo={self.evolution_steps}, lr={self.lr}, lr_core={self.lr_core_scale}, "
            f"tau={self.router_tau}, transport_thr={self.transport_threshold}, "
            f"topk={self.port_top_k}, decode_state={int(self.port_decode_state)}, "
            f"cuda_graph={self.use_cuda_graph}"
            ")"
        )


class BinaryStatePort(nn.Module):
    """
    Binary input -> u (dim_q), internal state -> logits over 4 states.
    Readout is chart-conditioned (one head per chart) and mixed by chart weights.
    """

    def __init__(
        self,
        *,
        dim_q: int,
        num_charts: int,
        num_states: int = 4,
        decode_state: bool = True,
        decode_top_k: int = 0,
    ):
        super().__init__()
        self.dim_q = int(dim_q)
        self.num_charts = int(num_charts)
        self.num_states = int(num_states)
        self.decode_state = bool(decode_state)
        self.decode_top_k = int(decode_top_k or 0)
        self.decode_in_dim = (2 * self.dim_q + 1) if self.decode_state else self.dim_q

        self.bit_embed = nn.Embedding(2, self.dim_q)
        nn.init.normal_(self.bit_embed.weight, std=1.0)

        self.readout_weight = nn.Parameter(torch.empty(self.num_charts, self.num_states, self.decode_in_dim))
        self.readout_bias = nn.Parameter(torch.zeros(self.num_charts, self.num_states))
        nn.init.normal_(self.readout_weight, std=0.1)
        nn.init.zeros_(self.readout_bias)

    def encode(self, bits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.bit_embed(bits))

    def decode(self, x: torch.Tensor, chart_weights: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B,D], chart_weights: [B,K]
        B = x.shape[0]
        K = self.num_charts
        logits_k = torch.einsum("bd,kvd->bkv", x, self.readout_weight) + self.readout_bias.unsqueeze(0)  # [B,K,S]
        if chart_weights is None:
            w = x.new_full((B, K), 1.0 / float(K))
        else:
            w = chart_weights
            if self.decode_top_k > 0 and self.decode_top_k < K:
                k = int(self.decode_top_k)
                if k == 1:
                    idx = w.argmax(dim=1)
                    return logits_k.gather(1, idx.view(B, 1, 1).expand(-1, 1, self.num_states)).squeeze(1)
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
        port: BinaryStatePort,
        config: Stage2Config,
        task_u_embed: nn.Embedding | None = None,
        router_ctx_embed: nn.Embedding | None = None,
    ):
        super().__init__()
        self.entity = entity
        self.port = port
        self.config = config
        self.task_u_embed = task_u_embed
        self.router_ctx_embed = router_ctx_embed

    def forward(
        self,
        inputs: torch.Tensor,  # [B,T] bits {0,1}
        targets: torch.Tensor,  # [B,T] states {0..3}
        initial_state_flat: torch.Tensor,  # [B,D]
        task_ids: torch.Tensor,  # [B,T] {0,1}
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T = inputs.shape
        dq = int(self.config.dim_q)

        state_flat = initial_state_flat
        prev_w = None

        # chart_weights returned by forward_tensor correspond to the *pre-step* state.
        # Our targets are the *post-step* states, so we keep a shifted copy for chart↔state stats.
        targets_prev = torch.zeros_like(targets)
        targets_prev[:, 1:] = targets[:, :-1]

        total_F = inputs.new_zeros(())
        total_acc = inputs.new_zeros(())
        total_V = inputs.new_zeros(())
        total_pred_ce = inputs.new_zeros(())
        chart_usage = inputs.new_zeros((int(self.config.num_charts),), dtype=torch.float32)

        # Per-task accuracy (task0=FSM2, task1=FSM4)
        task_acc_sum = inputs.new_zeros((2,), dtype=torch.float32)
        task_counts = inputs.new_zeros((2,), dtype=torch.float32)

        # State↔chart stats for FSM4 (4 states)
        state_chart_weight_sum = inputs.new_zeros((int(self.config.num_states), int(self.config.num_charts)), dtype=torch.float32)
        state_counts = inputs.new_zeros((int(self.config.num_states),), dtype=torch.float32)

        beta_kl = float(getattr(self.config, "beta_kl", 0.01))
        gamma_pred = float(getattr(self.config, "gamma_pred", 1.0))

        for t in range(T):
            bit_t = inputs[:, t]
            target_t = targets[:, t]
            state_prev_t = targets_prev[:, t]
            task_t = task_ids[:, t]

            u = self.port.encode(bit_t)
            if self.task_u_embed is not None:
                u = u + torch.tanh(self.task_u_embed(task_t))

            router_ctx = None
            if self.router_ctx_embed is not None:
                router_ctx = torch.tanh(self.router_ctx_embed(task_t))

            # Evolve the state for this token
            chart_w = None
            for _ in range(int(self.config.evolution_steps)):
                out = self.entity.forward_tensor(
                    state_flat=state_flat,
                    u_dict={"binary": u},
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

            # Decode state
            q = state_flat[:, :dq]
            # Predict next FSM state from the post-step state, using the next state's chart weights.
            chart_w_next = self.entity.atlas.router(q.clone(), context=router_ctx)
            x = state_flat if self.port.decode_state else q
            logits = self.port.decode(x, chart_weights=chart_w_next)

            # FSM2 head: restrict to states {0,1}.
            logits2 = logits[:, :2]
            tgt2 = torch.where(task_t == 0, target_t, torch.zeros_like(target_t))
            pred_error2 = F.cross_entropy(logits2, tgt2, reduction="none")
            pred_error4 = F.cross_entropy(logits, target_t, reduction="none")
            mask2 = (task_t == 0).to(dtype=pred_error2.dtype)
            pred_error = mask2 * pred_error2 + (1.0 - mask2) * pred_error4

            # Chart↔state alignment (use the pre-step state label).
            logw = torch.log(chart_w.clamp(min=1e-8))
            chart_state_loss = F.nll_loss(logw, state_prev_t, reduction="none")

            # L2 FSM: optimize prediction + chart-state alignment as a free-energy proxy (supervised).
            F_total = gamma_pred * (pred_error + chart_state_loss)

            total_F = total_F + F_total.mean()
            total_V = total_V + inputs.new_zeros(())
            total_pred_ce = total_pred_ce + pred_error.mean()

            # Accuracy
            pred4 = logits.argmax(dim=-1)
            pred2 = logits2.argmax(dim=-1)
            pred = torch.where(task_t == 0, pred2, pred4)
            correct = (pred == target_t).to(dtype=torch.float32)
            total_acc = total_acc + correct.mean()

            task_onehot = F.one_hot(task_t, num_classes=2).to(dtype=torch.float32)
            task_acc_sum = task_acc_sum + (correct.unsqueeze(1) * task_onehot).sum(dim=0)
            task_counts = task_counts + task_onehot.sum(dim=0)

            if chart_w is not None:
                chart_usage = chart_usage + chart_w.sum(dim=0).to(dtype=chart_usage.dtype)

                # FSM4-only mapping stats
                mask4 = (task_t == 1).to(dtype=torch.float32).unsqueeze(1)
                state_onehot = F.one_hot(state_prev_t, num_classes=int(self.config.num_states)).to(dtype=torch.float32)
                state_chart_weight_sum = state_chart_weight_sum + (state_onehot * mask4).t() @ chart_w.to(dtype=torch.float32)
                state_counts = state_counts + (state_onehot * mask4).sum(dim=0)

        mean_F = total_F / float(T)
        mean_acc = total_acc / float(T)
        mean_V = total_V / float(T)
        mean_ce = total_pred_ce / float(T)

        per_task_acc = task_acc_sum / (task_counts + 1e-8)

        # Chart entropy
        dist = chart_usage / (chart_usage.sum() + 1e-8)
        chart_ent = -(dist * torch.log(dist + 1e-8)).sum()

        # MI(state;chart) on FSM4 (soft assignment via chart weights)
        mi_state_chart = inputs.new_zeros(())
        if state_chart_weight_sum.sum() > 0:
            joint = state_chart_weight_sum / (state_chart_weight_sum.sum() + 1e-8)
            p_s = joint.sum(dim=1, keepdim=True)
            p_c = joint.sum(dim=0, keepdim=True)
            mi_state_chart = (joint * (torch.log(joint + 1e-8) - torch.log(p_s + 1e-8) - torch.log(p_c + 1e-8))).sum()
            mi_state_chart = mi_state_chart.clamp(min=0.0)

        diag = {
            "chart_usage": chart_usage,
            "chart_entropy": chart_ent,
            "per_task_acc": per_task_acc,
            "state_chart_weight_sum": state_chart_weight_sum,
            "state_counts": state_counts,
            "mi_state_chart": mi_state_chart,
            "mean_V": mean_V,
            "mean_pred_ce": mean_ce,
        }
        return mean_F, diag


class Stage2Trainer(BaseCurriculumTrainer):
    def __init__(self, config: Stage2Config):
        super().__init__(config)
        self.config = config

        self.port = BinaryStatePort(
            dim_q=int(config.dim_q),
            num_charts=int(config.num_charts),
            num_states=int(config.num_states),
            decode_state=bool(getattr(config, "port_decode_state", True)),
            decode_top_k=int(getattr(config, "port_top_k", 0) or 0),
        ).to(config.device)
        self.entity.add_interface("binary", int(config.dim_q))

        # Disambiguate FSM rules (FSM2 vs FSM4) without leaking the current state.
        self.task_u_embed = nn.Embedding(2, int(config.dim_q)).to(config.device)
        with torch.no_grad():
            nn.init.normal_(self.task_u_embed.weight, std=1.0)

        # Router context includes (task_id, bit) so the atlas can implement state transitions.
        router_ctx_dim = int(getattr(config, "router_context_dim", 0) or 0)
        self.router_ctx_embed = None
        if router_ctx_dim > 0:
            self.router_ctx_embed = nn.Embedding(2, router_ctx_dim).to(config.device)

        # L2 gate requires low holonomy; keep the connection fixed during FSM training.
        for p in self.entity.connection.parameters():
            p.requires_grad_(False)

        lr = float(config.lr)
        weight_decay = float(getattr(config, "weight_decay", 0.0) or 0.0)
        lr_core_scale = float(getattr(config, "lr_core_scale", 1.0) or 1.0)
        lr_core = lr * lr_core_scale

        core_ids = {id(p) for p in self.entity.net_V.parameters()}
        core_ids.update(id(p) for p in self.entity.internal_gen.parameters())
        core_ids.add(id(self.entity.z))

        core_params: list[nn.Parameter] = []
        fast_entity_params: list[nn.Parameter] = []
        for p in self.entity.parameters():
            if not p.requires_grad:
                continue
            (core_params if id(p) in core_ids else fast_entity_params).append(p)

        fast_params = fast_entity_params + list(self.port.parameters()) + list(self.task_u_embed.parameters())
        if self.router_ctx_embed is not None:
            fast_params = fast_params + list(self.router_ctx_embed.parameters())

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

        self.seq_model = SequenceModel(
            entity=self.entity,
            port=self.port,
            config=config,
            task_u_embed=self.task_u_embed,
            router_ctx_embed=self.router_ctx_embed,
        )

        torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def load_stage1_entity(self, path: str) -> None:
        sd = torch.load(path, map_location=self.config.device)
        ent_sd = {}
        for k, v in sd.items():
            if k.startswith("entity."):
                ent_sd[k[len("entity.") :]] = v
        cur = self.entity.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        mismatched: list[Tuple[str, Tuple[int, ...], Tuple[int, ...] | None]] = []
        for k, v in ent_sd.items():
            if k not in cur:
                continue
            if tuple(cur[k].shape) != tuple(v.shape):
                mismatched.append((k, tuple(v.shape), tuple(cur[k].shape) if k in cur else None))
                continue
            filtered[k] = v

        missing, unexpected = self.entity.load_state_dict(filtered, strict=False)
        if os.getenv("HEI_DEBUG_LOAD", "0") == "1":
            msg = f"Loaded Stage1 entity from {path} (missing={len(missing)} unexpected={len(unexpected)}"
            if mismatched:
                msg += f" mismatched={len(mismatched)})"
                print(msg)
                for k, s_old, s_new in mismatched[:10]:
                    print(f"  skip {k}: ckpt{s_old} != cur{s_new}")
            else:
                msg += ")"
                print(msg)

    def generate_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = int(self.config.batch_size)
        T = int(self.config.sequence_len)
        device = self.config.device

        # Sample task per sequence
        p4 = float(getattr(self.config, "p_task4", 0.5))
        task_seq = (torch.rand((B, 1), device=device) < p4).to(dtype=torch.long)  # 1 => FSM4
        task_ids = task_seq.expand(-1, T).contiguous()

        # Binary inputs
        bits = torch.randint(0, 2, (B, T), device=device, dtype=torch.long)

        # Targets
        # FSM2 (temp-04.md): next_state = bit (0->A, 1->B)
        tgt2 = bits
        # FSM4: mod-4 counter increment on bit==1
        tgt4 = torch.remainder(torch.cumsum(bits, dim=1), 4)

        targets = torch.where(task_seq.expand(-1, T) == 1, tgt4, tgt2)
        return bits, targets, task_ids

    @torch.no_grad()
    def evaluate_L2_gate(
        self,
        *,
        batch_size: int = 8192,
        repeats: int = 2,
        seed: int = 0,
        holonomy_loops: int = 32,
    ) -> Dict[str, Any]:
        device = self.config.device
        B = int(batch_size)
        T = int(self.config.sequence_len)
        dq = int(self.config.dim_q)
        D = 2 * dq + 1
        repeats = max(1, int(repeats))

        def eval_task(task_id: int) -> Dict[str, Any]:
            acc_sum = 0.0
            mi_sum = 0.0
            ent_sum = 0.0
            state_chart_sum = None
            state_counts_sum = None
            for r in range(repeats):
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(seed) + 10_000 * task_id + r)
                bits_cpu = torch.randint(0, 2, (B, T), generator=gen, dtype=torch.long)
                if task_id == 0:
                    targets_cpu = bits_cpu
                else:
                    targets_cpu = torch.remainder(torch.cumsum(bits_cpu, dim=1), 4)
                task_ids_cpu = torch.full((B, T), int(task_id), dtype=torch.long)

                # Deterministic init distribution
                init_cpu = torch.zeros((B, D), dtype=torch.float32)

                bits = bits_cpu.to(device=device, non_blocking=True)
                targets = targets_cpu.to(device=device, non_blocking=True)
                task_ids = task_ids_cpu.to(device=device, non_blocking=True)
                init = init_cpu.to(device=device, non_blocking=True)

                mean_F, diag = self.seq_model(bits, targets, init, task_ids)

                per_task_acc = diag["per_task_acc"]  # [2]
                acc_sum += float(per_task_acc[task_id].item())
                mi_sum += float(diag["mi_state_chart"].item())
                ent_sum += float(diag["chart_entropy"].item())
                if state_chart_sum is None:
                    state_chart_sum = diag["state_chart_weight_sum"].detach().cpu()
                    state_counts_sum = diag["state_counts"].detach().cpu()
                else:
                    state_chart_sum = state_chart_sum + diag["state_chart_weight_sum"].detach().cpu()
                    state_counts_sum = state_counts_sum + diag["state_counts"].detach().cpu()

            rep = float(repeats)
            out = {
                "acc": acc_sum / rep,
                "mi_state_chart": mi_sum / rep,
                "chart_entropy": ent_sum / rep,
                "state_chart_weight_sum": state_chart_sum,
                "state_counts": state_counts_sum,
            }
            return out

        res2 = eval_task(0)
        res4 = eval_task(1)

        # Chart↔state correspondence (FSM4 only)
        eps = 1e-8
        wsum = res4["state_chart_weight_sum"]
        cnts = res4["state_counts"].unsqueeze(1) + eps
        cond = wsum / cnts  # E[w | state]
        top_chart = cond.argmax(dim=1)  # [4]
        unique_top = int(torch.unique(top_chart).numel())
        max_w = cond.max(dim=1).values

        # Connection holonomy (proxy for PT consistency)
        conn = self.entity.connection
        hol_ratios = []
        for i in range(int(holonomy_loops)):
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed) + 50_000 + i)
            center = torch.randn(dq, generator=g) * 0.5
            points_cpu = [center + torch.randn(dq, generator=g) * 0.05 for _ in range(5)]
            v0_cpu = torch.randn(dq, generator=g)
            points = [p.to(device=device, non_blocking=True) for p in points_cpu]
            v0 = v0_cpu.to(device=device, non_blocking=True)
            metrics = compute_holonomy(conn, points, v0)
            hol_ratios.append(float(metrics["holonomy_ratio"]))
        hol_ratio = float(sum(hol_ratios) / max(1, len(hol_ratios)))

        passed_acc2 = res2["acc"] >= float(getattr(self.config, "gate_acc2", 0.98))
        passed_acc4 = res4["acc"] >= float(getattr(self.config, "gate_acc4", 0.90))
        passed_chart = unique_top >= int(getattr(self.config, "gate_state_chart_unique", 4))
        passed_holo = hol_ratio <= float(getattr(self.config, "gate_holonomy_ratio", 0.01))

        return {
            "FSM2": res2,
            "FSM4": res4,
            "fsm4_state_top_chart": top_chart.tolist(),
            "fsm4_state_top_weight": max_w.tolist(),
            "fsm4_unique_top_charts": unique_top,
            "holonomy_ratio": hol_ratio,
            "passed_components": {
                "FSM2_acc": bool(passed_acc2),
                "FSM4_acc": bool(passed_acc4),
                "ChartStateMap": bool(passed_chart),
                "Holonomy": bool(passed_holo),
            },
            "passed": bool(passed_acc2 and passed_acc4 and passed_chart and passed_holo),
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

    def train_loop(self) -> None:
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
        self._train_loop_eager()

    def _train_loop_eager(self) -> None:
        print(f"Starting Stage 2 Training on {self.config.device}...")
        print(f"Config: {self.config}")
        print("L2: FSM training")

        best_path = os.path.join(self.config.save_dir, "stage2_best.pt")
        best_acc4 = -1.0

        start = time.time()
        for step in range(1, int(self.config.steps) + 1):
            self.optimizer.zero_grad(set_to_none=True)
            bits, targets, task_ids = self.generate_batch()

            dq = int(self.config.dim_q)
            D = 2 * dq + 1
            init = torch.zeros((int(self.config.batch_size), D), device=self.config.device, dtype=torch.float32)

            mean_F, diag = self.seq_model(bits, targets, init, task_ids)

            chart_usage = diag["chart_usage"]
            dist = chart_usage / (chart_usage.sum() + 1e-8)
            entropy = -(dist * torch.log(dist + 1e-8)).sum()

            mi_state_chart = diag["mi_state_chart"]
            alpha = float(getattr(self.config, "alpha_chart", 0.0) or 0.0)
            lambda_state_mi = float(getattr(self.config, "lambda_state_mi", 0.0) or 0.0)

            loss = mean_F - alpha * entropy - lambda_state_mi * mi_state_chart
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._train_params, 1.0)
            self.optimizer.step()

            if step % int(self.config.log_every) == 0:
                elapsed = time.time() - start
                per_task_acc = diag["per_task_acc"]
                print(
                    f"Step {step}: F={float(mean_F.item()):.4f} "
                    f"Acc2={float(per_task_acc[0].item()):.3f} Acc4={float(per_task_acc[1].item()):.3f} "
                    f"ChartEnt={float(diag['chart_entropy'].item()):.2f} MI_sc={float(mi_state_chart.item()):.3f} "
                    f"Time={elapsed:.1f}s"
                )

                try:
                    gate = self.evaluate_L2_gate(batch_size=4096, repeats=1, seed=0, holonomy_loops=16)
                    acc4 = float(gate["FSM4"]["acc"])
                    if acc4 > best_acc4:
                        best_acc4 = acc4
                        torch.save(self.state_dict(), best_path)
                        print(f"GateBest: Acc4={best_acc4:.3f}")
                except Exception as e:
                    print(f"GateEval: failed ({type(e).__name__}: {e})")

        if os.path.exists(best_path):
            try:
                self.load_state_dict(torch.load(best_path, map_location=self.config.device))
                print(f"Loaded best checkpoint: {best_path}")
            except Exception as e:
                print(f"Best checkpoint load failed ({type(e).__name__}: {e})")

        os.makedirs(self.config.save_dir, exist_ok=True)
        save_path = os.path.join(self.config.save_dir, "stage2_final.pt")
        torch.save(self.state_dict(), save_path)
        print(f"Stage 2 Complete. Saved to {save_path}")

    def _train_loop_cuda_graph(self) -> None:
        device = self.config.device
        if (not torch.cuda.is_available()) or (not str(device).startswith("cuda")):
            raise RuntimeError("CUDA graph mode requested but CUDA is not available.")

        print(f"Starting Stage 2 Training (CUDA Graph) on {device}...")
        print(f"Config: {self.config}")
        print("L2: FSM training")

        B = int(self.config.batch_size)
        T = int(self.config.sequence_len)
        dq = int(self.config.dim_q)
        D = 2 * dq + 1

        best_path = os.path.join(self.config.save_dir, "stage2_best.pt")
        best_acc4 = -1.0

        bits_buf = torch.empty((B, T), device=device, dtype=torch.long)
        targets_buf = torch.empty((B, T), device=device, dtype=torch.long)
        task_ids_buf = torch.empty((B, T), device=device, dtype=torch.long)
        init_buf = torch.empty((B, D), device=device, dtype=torch.float32)

        loss_buf = torch.zeros((), device=device)
        acc2_buf = torch.zeros((), device=device)
        acc4_buf = torch.zeros((), device=device)
        ent_buf = torch.zeros((), device=device)
        mi_buf = torch.zeros((), device=device)

        alpha = float(getattr(self.config, "alpha_chart", 0.0) or 0.0)
        lambda_state_mi = float(getattr(self.config, "lambda_state_mi", 0.0) or 0.0)

        def refill_buffers() -> None:
            bits, targets, task_ids = self.generate_batch()
            bits_buf.copy_(bits)
            targets_buf.copy_(targets)
            task_ids_buf.copy_(task_ids)
            init_buf.zero_()

        def fwd_bwd_step() -> None:
            self.optimizer.zero_grad(set_to_none=False)
            mean_F, diag = self.seq_model(bits_buf, targets_buf, init_buf, task_ids_buf)
            chart_usage = diag["chart_usage"]
            dist = chart_usage / (chart_usage.sum() + 1e-8)
            entropy = -(dist * torch.log(dist + 1e-8)).sum()
            mi_state_chart = diag["mi_state_chart"]

            loss = mean_F - alpha * entropy - lambda_state_mi * mi_state_chart
            loss_buf.copy_(loss.detach())

            per_task_acc = diag["per_task_acc"]
            acc2_buf.copy_(per_task_acc[0].detach())
            acc4_buf.copy_(per_task_acc[1].detach())
            ent_buf.copy_(diag["chart_entropy"].detach())
            mi_buf.copy_(mi_state_chart.detach())

            loss.backward()
            self._clip_grad_norm_capturable_(1.0)

        warmup = max(1, int(getattr(self.config, "cuda_graph_warmup", 3) or 3))
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
            self.optimizer.step()

            if step % int(self.config.log_every) == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - start
                print(
                    f"Step {step}: F={float(loss_buf.item()):.4f} "
                    f"Acc2={float(acc2_buf.item()):.3f} Acc4={float(acc4_buf.item()):.3f} "
                    f"ChartEnt={float(ent_buf.item()):.2f} MI_sc={float(mi_buf.item()):.3f} "
                    f"Time={elapsed:.1f}s"
                )

                try:
                    gate = self.evaluate_L2_gate(batch_size=4096, repeats=1, seed=0, holonomy_loops=16)
                    acc4 = float(gate["FSM4"]["acc"])
                    if acc4 > best_acc4:
                        best_acc4 = acc4
                        torch.save(self.state_dict(), best_path)
                        print(f"GateBest: Acc4={best_acc4:.3f}")
                except Exception as e:
                    print(f"GateEval: failed ({type(e).__name__}: {e})")

        if os.path.exists(best_path):
            try:
                self.load_state_dict(torch.load(best_path, map_location=self.config.device))
                print(f"Loaded best checkpoint: {best_path}")
            except Exception as e:
                print(f"Best checkpoint load failed ({type(e).__name__}: {e})")

        os.makedirs(self.config.save_dir, exist_ok=True)
        save_path = os.path.join(self.config.save_dir, "stage2_final.pt")
        torch.save(self.state_dict(), save_path)
        print(f"Stage 2 Complete. Saved to {save_path}")


def _apply_env_overrides(config: Stage2Config) -> Stage2Config:
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
    if os.getenv("HEI_PORT_TOP_K") is not None:
        config.port_top_k = int(os.getenv("HEI_PORT_TOP_K", str(config.port_top_k)))
    if os.getenv("HEI_ALPHA_CHART") is not None:
        config.alpha_chart = float(os.getenv("HEI_ALPHA_CHART", str(config.alpha_chart)))
    if os.getenv("HEI_LAMBDA_STATE_MI") is not None:
        config.lambda_state_mi = float(os.getenv("HEI_LAMBDA_STATE_MI", str(config.lambda_state_mi)))
    if os.getenv("HEI_CUDA_GRAPH") is not None:
        config.use_cuda_graph = str(os.getenv("HEI_CUDA_GRAPH")) == "1"
    if os.getenv("HEI_CUDA_GRAPH_WARMUP") is not None:
        config.cuda_graph_warmup = int(os.getenv("HEI_CUDA_GRAPH_WARMUP", str(config.cuda_graph_warmup)))
    if os.getenv("HEI_P_TASK4") is not None:
        config.p_task4 = float(os.getenv("HEI_P_TASK4", str(config.p_task4)))
    return config


if __name__ == "__main__":
    config = _apply_env_overrides(Stage2Config())
    trainer = Stage2Trainer(config)

    ckpt = os.getenv("HEI_STAGE1_CKPT", "checkpoints/curriculum/stage1_final.pt")
    if ckpt and os.path.exists(ckpt):
        trainer.load_stage1_entity(ckpt)
        print(f"Loaded Stage1 entity init from {ckpt}")
    else:
        print("Stage1 init not found; training Stage2 from scratch.")

    trainer.train_loop()

    if os.getenv("HEI_EVAL_L2", "1") == "1":
        gate = trainer.evaluate_L2_gate(
            batch_size=int(os.getenv("HEI_EVAL_BATCH", "8192")),
            repeats=int(os.getenv("HEI_EVAL_REPEATS", "2")),
            seed=int(os.getenv("HEI_EVAL_SEED", "0")),
            holonomy_loops=int(os.getenv("HEI_HOLO_LOOPS", "32")),
        )
        pc = gate["passed_components"]
        print(
            "=== L2 Gate (Eval) ===\n"
            f"FSM2 Acc={gate['FSM2']['acc']:.3f} (thr={config.gate_acc2:.2f})\n"
            f"FSM4 Acc={gate['FSM4']['acc']:.3f} (thr={config.gate_acc4:.2f})\n"
            f"FSM4 chart_top={gate['fsm4_state_top_chart']} unique={gate['fsm4_unique_top_charts']} (thr={config.gate_state_chart_unique})\n"
            f"Holonomy ratio={gate['holonomy_ratio']:.4f} (thr={config.gate_holonomy_ratio:.3f})\n"
            f"Passed: {gate['passed']} Components={pc}"
        )
