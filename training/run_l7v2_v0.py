#!/usr/bin/env python3
"""
L7v2 v0 verification runner (E0/E1/E2).

This is intentionally a *verification-first* script:
- Uses the v0 defaults from temp-07:
  - F_pred: same-step reconstruction/denoising
  - offline trigger: fixed rhythm (N online + M offline)
  - read kernel: Gaussian + truncated renormalization (R=3σ)
  - control: only pointer s (σ fixed)

Outputs a minimal "Run Card" style record:
- run_config.json
- log.jsonl
- summary.json

NOTE: This is not a language-training replacement for Stage7; it is a mechanism testbed
for A1/A2/A3/A4-aligned active sampling with a continuous port.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
import shlex
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Ensure HEI is on path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from he_core.soul_entity import create_soul_entity
from he_core.state import ContactState


@dataclass(frozen=True)
class V0Defaults:
    f_pred: str = "reconstruct_denoise"
    offline_trigger: str = "fixed_rhythm"
    kernel: str = "gaussian_trunc_renorm"
    control: str = "s_only"


def _now_run_id(seed: int) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"l7v2_{ts}_seed{seed}"


def _git_commit(repo_root: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or None
    except Exception:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_text_file(path: str, *, max_chars: int) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read()
    if max_chars > 0:
        s = s[:max_chars]
    return s.replace("\r\n", "\n")


def reset_experience(entity) -> None:
    """
    SoulEntity.reset() does not clear the ExperienceBuffer. For A2 experiments we
    want per-episode experience-conditioning, so we hard-reset it here.
    """
    exp = getattr(entity, "experience", None)
    if exp is None:
        return
    exp.ptr = 0
    exp.size = 0
    exp.sensory = None
    exp.active = None
    exp.states = None
    exp.rewards = None


@dataclass
class CharVocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id[self.pad_token])

    @property
    def unk_id(self) -> int:
        return int(self.token_to_id[self.unk_token])

    def encode(self, text: str) -> List[int]:
        unk = self.unk_id
        t2i = self.token_to_id
        return [t2i.get(ch, unk) for ch in text]


def build_char_vocab(texts: Sequence[str], *, vocab_size: int, min_freq: int = 1) -> CharVocab:
    if vocab_size < 4:
        raise ValueError("vocab_size too small")
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(t)

    specials = ["<PAD>", "<UNK>"]
    id_to_token: List[str] = list(specials)
    token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(id_to_token)}

    # Most frequent chars
    for ch, freq in counter.most_common():
        if freq < min_freq:
            continue
        if ch in token_to_id:
            continue
        token_to_id[ch] = len(id_to_token)
        id_to_token.append(ch)
        if len(id_to_token) >= vocab_size:
            break

    return CharVocab(token_to_id=token_to_id, id_to_token=id_to_token)


class GaussianTruncRenormReadPort(nn.Module):
    def __init__(
        self,
        *,
        token_ids: torch.Tensor,  # [L] long
        embedding: nn.Embedding,  # [V, D], frozen
        sigma: float,
        trunc_factor: float = 3.0,
    ):
        super().__init__()
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be [L]")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.register_buffer("token_ids", token_ids)
        self.embedding = embedding
        self.sigma = float(sigma)
        self.trunc_factor = float(trunc_factor)
        self.radius = int(math.ceil(self.trunc_factor * self.sigma))

    @property
    def length(self) -> int:
        return int(self.token_ids.numel())

    def read(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [B] float pointer in [0, L-1].
        Returns:
            y: [B, D] continuous observation.
        """
        if s.ndim != 1:
            raise ValueError("s must be [B]")
        device = s.device
        L = self.length
        if L <= 0:
            raise RuntimeError("empty token sequence")

        radius = int(self.radius)
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.long)  # [K]
        base = torch.floor(s).to(dtype=torch.long)  # [B]
        idx = base.unsqueeze(1) + offsets.unsqueeze(0)  # [B,K]

        valid = (idx >= 0) & (idx < L)
        idx_clamped = idx.clamp(0, L - 1)

        dist = idx.to(dtype=s.dtype) - s.unsqueeze(1)
        w = torch.exp(-0.5 * (dist / self.sigma) ** 2)
        w = w * valid.to(dtype=w.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)

        emb = self.embedding(self.token_ids[idx_clamped])  # [B,K,D]
        return (w.unsqueeze(-1) * emb).sum(dim=1)


class ObsDecoder(nn.Module):
    def __init__(self, dim_q: int, dim_y: int, hidden: int = 0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(dim_q, hidden),
                nn.SiLU(),
                nn.Linear(hidden, dim_y),
            )
        else:
            self.net = nn.Linear(dim_q, dim_y)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.net(q)


def compute_v_term_per_sample(entity, state: ContactState) -> torch.Tensor:
    z_batch = entity.z.expand(state.batch_size, -1)
    V_inp = torch.cat([state.q, z_batch], dim=1)
    V = entity.net_V(V_inp)  # [B,1]
    stiffness = float(getattr(entity.internal_gen, "stiffness", 0.0) or 0.0)
    if stiffness > 0:
        V = V + 0.5 * stiffness * (state.q ** 2).sum(dim=1, keepdim=True)
    return V.squeeze(1)


def compute_loss(
    *,
    entity,
    state: ContactState,
    pred_err_per_sample: torch.Tensor,
    action_delta: Optional[torch.Tensor],
    beta_kl: float,
    gamma_pred: float,
    cost_weight: float,
    cost_power: float = 2.0,
    cost_offset: float = 0.0,
    anti_lock_weight: float = 0.0,
    anti_lock_sigma: float = 1.0,
    revisit_penalty_per_sample: Optional[torch.Tensor] = None,
    pref_penalty_per_sample: Optional[torch.Tensor] = None,
    cover_penalty_per_sample: Optional[torch.Tensor] = None,
    epi_penalty_per_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    V_ps = compute_v_term_per_sample(entity, state)  # [B]
    V_term = V_ps.mean()
    KL = 0.5 * (entity.z ** 2).sum() * float(beta_kl)
    E_pred = float(gamma_pred) * pred_err_per_sample.mean()

    cost = V_term.new_zeros(())
    if action_delta is not None:
        p = float(cost_power)
        if not math.isfinite(p) or p <= 0:
            p = 2.0
        off = float(cost_offset)
        if not math.isfinite(off) or off < 0:
            off = 0.0
        d = action_delta.to(dtype=V_term.dtype).abs()
        cost_term = (d + off).pow(p)
        cost = float(cost_weight) * cost_term.mean()

    anti_lock = V_term.new_zeros(())
    if action_delta is not None and float(anti_lock_weight) > 0.0:
        sigma = max(float(anti_lock_sigma), 1e-6)
        d = action_delta.to(dtype=V_term.dtype)
        anti_lock = float(anti_lock_weight) * torch.exp(-0.5 * (d / sigma) ** 2).mean()

    revisit = V_term.new_zeros(())
    if revisit_penalty_per_sample is not None:
        revisit = revisit_penalty_per_sample.to(dtype=V_term.dtype).mean()

    pref = V_term.new_zeros(())
    if pref_penalty_per_sample is not None:
        pref = pref_penalty_per_sample.to(dtype=V_term.dtype).mean()

    cover = V_term.new_zeros(())
    if cover_penalty_per_sample is not None:
        cover = cover_penalty_per_sample.to(dtype=V_term.dtype).mean()

    epi = V_term.new_zeros(())
    if epi_penalty_per_sample is not None:
        epi = epi_penalty_per_sample.to(dtype=V_term.dtype).mean()

    F = V_term + KL + E_pred + cost + anti_lock + revisit + pref + cover + epi
    # Important: return tensors to avoid per-step GPU sync via `.item()`.
    return F, {
        "V_term": V_term.detach(),
        "KL": KL.detach(),
        "E_pred": E_pred.detach(),
        "cost": cost.detach(),
        "anti_lock": anti_lock.detach(),
        "revisit": revisit.detach(),
        "pref": pref.detach(),
        "cover": cover.detach(),
        "epi": epi.detach(),
        "F": F.detach(),
    }


def _parts_to_float(parts: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().item()) for k, v in parts.items()}


def _apply_pointer_bounds(s: torch.Tensor, *, max_s: float, mode: str) -> torch.Tensor:
    max_s = float(max_s)
    if not math.isfinite(max_s) or max_s <= 0.0:
        return s.clamp(0.0, max(0.0, max_s))

    m = str(mode or "clamp").strip().lower()
    if m == "clamp":
        return s.clamp(0.0, max_s)
    if m == "reflect":
        # Reflect into [0, max_s] without a hard clamp-to-constant at the boundary.
        # For max_s=L, period is 2L: values beyond L are mirrored back.
        max_t = s.new_tensor(max_s)
        period = max_t * 2.0
        x = torch.remainder(s, period)
        return torch.where(x <= max_t, x, period - x)
    if m == "wrap":
        # Ring pointer: wrap into [0, max_s] (period is max_s+1 because bounds are inclusive).
        period = s.new_tensor(max_s + 1.0)
        return torch.remainder(s, period)
    raise ValueError(f"unknown pointer_bounds mode: {mode}")


def _chart_entropy(weights: torch.Tensor, *, normalize: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """
    Args:
        weights: [..., K] non-negative, rows sum to ~1.
    Returns:
        ent: [...] entropy (optionally normalized by log(K)).
    """
    if weights.ndim < 1:
        raise ValueError("weights must have at least 1 dimension")
    K = int(weights.shape[-1])
    if K <= 0:
        raise ValueError("empty chart dimension")
    w = weights.clamp(min=float(eps))
    ent = -(w * torch.log(w)).sum(dim=-1)
    if normalize and K > 1:
        ent = ent / float(math.log(float(K)))
    return ent


@torch.no_grad()
def choose_delta_one_step(
    *,
    entity,
    decoder: nn.Module,
    read_port: GaussianTruncRenormReadPort,
    s: torch.Tensor,  # [B]
    candidates: torch.Tensor,  # [C] float
    dt: float,
    beta_kl: float,
    gamma_pred: float,
    cost_weight: float,
    cost_power: float = 2.0,
    cost_offset: float = 0.0,
    noise_std: float,
    pointer_drift: float = 0.0,
    pointer_bounds: str = "clamp",
    anti_lock_weight: float = 0.0,
    anti_lock_sigma: float = 1.0,
    revisit_weight: float = 0.0,
    recent_s: Optional[torch.Tensor] = None,  # [K,B] long (quantized pointer history)
    lookahead_steps: int = 1,
    lookahead_discount: float = 1.0,
    pref_repeat_weight: float = 0.0,
    token_ids: Optional[torch.Tensor] = None,  # [L] long on same device as s
    prev_token_ids: Optional[torch.Tensor] = None,  # [B] long
    cover_weight: float = 0.0,
    cover_sigma: float = 2.0,
    cover_recent_s: Optional[torch.Tensor] = None,  # [K,B] float pointer history
    epi_weight: float = 0.0,
    epi_normalize: int = 1,
    epi_mode: str = "chart_entropy",
    epi_obs_eps: float = 1.0,
    epi_pred_floor: float = 0.0,
) -> torch.Tensor:
    """
    One-step lookahead: pick delta that minimizes (V + gamma*E_pred + cost).
    This is a v0 approximation of argmin_a E[F_{t+1}|a] + cost(a).
    """
    device = s.device
    B = int(s.shape[0])
    C = int(candidates.numel())
    if C <= 0:
        return torch.zeros_like(s)

    lookahead_steps = int(max(1, int(lookahead_steps)))
    lookahead_discount = float(lookahead_discount)
    if not math.isfinite(lookahead_discount):
        lookahead_discount = 1.0
    lookahead_discount = float(max(0.0, min(lookahead_discount, 1.0)))

    # Vectorized candidate evaluation (reduces Python overhead & improves GPU utilization).
    L = float(read_port.length - 1)
    cand = candidates.to(device=device, dtype=torch.float32)  # [C]
    s0 = s.to(dtype=torch.float32)  # [B]
    state_flat = entity.state.flat  # [B,S]
    state_rep = state_flat.unsqueeze(0).expand(C, B, -1).reshape(C * B, -1)
    prev_w = getattr(entity, "_prev_chart_weights", None)
    prev_rep = None
    if prev_w is not None:
        prev_rep = prev_w.unsqueeze(0).expand(C, B, -1).reshape(C * B, -1)

    total = torch.zeros((C, B), device=device, dtype=torch.float32)
    s_cur = s0.unsqueeze(0).expand(C, B)  # [C,B]
    state_cur = state_rep  # [C*B,S]
    prev_cur = prev_rep  # [C*B,num_charts] or None

    # Terms that do not depend on the simulated state (apply each step).
    p = float(cost_power)
    if not math.isfinite(p) or p <= 0:
        p = 2.0
    off = float(cost_offset)
    if not math.isfinite(off) or off < 0:
        off = 0.0
    step_base = float(cost_weight) * (cand.abs() + off).pow(p)  # [C]
    if float(anti_lock_weight) > 0.0:
        sigma = max(float(anti_lock_sigma), 1e-6)
        step_base = step_base + float(anti_lock_weight) * torch.exp(-0.5 * (cand / sigma).pow(2))

    # Revisit penalty is evaluated on the *first* step (w.r.t. actual recent history).
    revisit_first: Optional[torch.Tensor] = None
    if float(revisit_weight) > 0.0 and recent_s is not None and int(recent_s.numel()) > 0:
        s_new_first = _apply_pointer_bounds(
            s_cur + float(pointer_drift) + cand.unsqueeze(1),
            max_s=L,
            mode=str(pointer_bounds),
        )  # [C,B]
        s_new_round = s_new_first.round().to(dtype=torch.long)
        hit = (recent_s.unsqueeze(0) == s_new_round.unsqueeze(1)).any(dim=1).to(dtype=torch.float32)  # [C,B]
        revisit_first = float(revisit_weight) * hit

    # Outcome preference (v0.3): discourage immediate token repetition (a minimal non-degeneracy prior).
    repeat_first: Optional[torch.Tensor] = None
    if float(pref_repeat_weight) > 0.0 and token_ids is not None and prev_token_ids is not None:
        if token_ids.ndim == 1 and prev_token_ids.ndim == 1 and int(prev_token_ids.numel()) == B:
            s_new_first = _apply_pointer_bounds(
                s_cur + float(pointer_drift) + cand.unsqueeze(1),
                max_s=L,
                mode=str(pointer_bounds),
            )  # [C,B]
            s_new_round = s_new_first.round().to(dtype=torch.long).clamp(0, int(L))
            tok_new = token_ids[s_new_round]  # [C,B]
            repeat = (tok_new == prev_token_ids.unsqueeze(0)).to(dtype=torch.float32)
            repeat_first = float(pref_repeat_weight) * repeat

    # Continuous repulsion / coverage potential: discourage concentration in pointer-space.
    cover_first: Optional[torch.Tensor] = None
    if float(cover_weight) > 0.0 and cover_recent_s is not None and int(cover_recent_s.numel()) > 0:
        sigma = max(float(cover_sigma), 1e-6)
        s_new_first = _apply_pointer_bounds(
            s_cur + float(pointer_drift) + cand.unsqueeze(1),
            max_s=L,
            mode=str(pointer_bounds),
        )  # [C,B]
        hist = cover_recent_s.to(device=device, dtype=torch.float32)  # [K,B]
        sim = torch.exp(-0.5 * ((s_new_first.unsqueeze(1) - hist.unsqueeze(0)) / sigma).pow(2))
        # Sum matches the "repulsion potential" definition: Σ exp(-||Δs||^2 / 2σ^2).
        cover_first = float(cover_weight) * sim.sum(dim=1)

    epi_mode_s = str(epi_mode or "chart_entropy").strip().lower()
    epi_first: Optional[torch.Tensor] = None
    discount = 1.0
    for k in range(lookahead_steps):
        state_pre = state_cur
        prev_pre = prev_cur
        s_cur = _apply_pointer_bounds(
            s_cur + float(pointer_drift) + cand.unsqueeze(1),
            max_s=L,
            mode=str(pointer_bounds),
        )  # [C,B]

        y_flat = read_port.read(s_cur.reshape(-1))  # [C*B,D]
        D = int(y_flat.shape[1])
        y = y_flat.view(C, B, D)

        if k == 0 and float(epi_weight) != 0.0 and epi_mode_s == "obs_grad":
            eps = float(epi_obs_eps)
            if not math.isfinite(eps) or eps <= 0.0:
                eps = 1.0
            s_plus = _apply_pointer_bounds(s_cur + eps, max_s=L, mode=str(pointer_bounds))
            s_minus = _apply_pointer_bounds(s_cur - eps, max_s=L, mode=str(pointer_bounds))
            y_plus = read_port.read(s_plus.reshape(-1))  # [C*B,D]
            y_minus = read_port.read(s_minus.reshape(-1))  # [C*B,D]
            dy = (y_plus - y_minus).view(C, B, D)
            # A simple observability proxy: local sensory gradient magnitude.
            dy_norm = dy.pow(2).mean(dim=2)  # [C,B]
            epi_first = float(epi_weight) * dy_norm

        y_in = y
        noise: Optional[torch.Tensor] = None
        if noise_std > 0:
            noise = torch.randn_like(y)
            y_in = y + float(noise_std) * noise

        out = entity.forward_tensor(
            state_flat=state_pre,
            u_dict={"default": y_in.reshape(C * B, D)},
            dt=float(dt),
            prev_chart_weights=prev_pre,
            prediction_error=None,
            detach_next_prev_weights=True,
            compute_action=False,
            skip_free_energy=True,
        )
        state_cur = out["next_state_flat"]
        prev_cur = out["next_prev_chart_weights"]
        if k == 0 and float(epi_weight) != 0.0 and epi_mode_s == "chart_entropy":
            w = out.get("chart_weights", None)
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                w = w.view(C, B, -1)
                ent = _chart_entropy(w, normalize=(int(epi_normalize) != 0))  # [C,B]
                epi_first = float(epi_weight) * ent

        if k == 0 and float(epi_weight) != 0.0 and epi_mode_s == "jac_e":
            eps = float(epi_obs_eps)
            if not math.isfinite(eps) or eps <= 0.0:
                eps = 1.0
            s_plus = _apply_pointer_bounds(s_cur + eps, max_s=L, mode=str(pointer_bounds))
            s_minus = _apply_pointer_bounds(s_cur - eps, max_s=L, mode=str(pointer_bounds))
            y_plus = read_port.read(s_plus.reshape(-1)).view(C, B, D)
            y_minus = read_port.read(s_minus.reshape(-1)).view(C, B, D)
            y_plus_in = y_plus
            y_minus_in = y_minus
            if noise is not None:
                y_plus_in = y_plus + float(noise_std) * noise
                y_minus_in = y_minus + float(noise_std) * noise

            out_plus = entity.forward_tensor(
                state_flat=state_pre,
                u_dict={"default": y_plus_in.reshape(C * B, D)},
                dt=float(dt),
                prev_chart_weights=prev_pre,
                prediction_error=None,
                detach_next_prev_weights=True,
                compute_action=False,
                skip_free_energy=True,
            )
            out_minus = entity.forward_tensor(
                state_flat=state_pre,
                u_dict={"default": y_minus_in.reshape(C * B, D)},
                dt=float(dt),
                prev_chart_weights=prev_pre,
                prediction_error=None,
                detach_next_prev_weights=True,
                compute_action=False,
                skip_free_energy=True,
            )
            next_plus = ContactState(entity.dim_q, C * B, entity.state.device, out_plus["next_state_flat"])
            next_minus = ContactState(entity.dim_q, C * B, entity.state.device, out_minus["next_state_flat"])
            y_hat_plus = decoder(next_plus.q).view(C, B, D)
            y_hat_minus = decoder(next_minus.q).view(C, B, D)
            pred_plus = (y_hat_plus - y_plus).pow(2).mean(dim=2)
            pred_minus = (y_hat_minus - y_minus).pow(2).mean(dim=2)
            # Dimensionless relative sensitivity: |d log(E_pred) / dδ| (finite difference).
            floor = float(epi_pred_floor)
            if not math.isfinite(floor) or floor <= 0.0:
                floor = 1e-8
            floor = float(max(floor, 1e-8))
            denom = (0.5 * (pred_plus + pred_minus)).clamp(min=floor)
            jac = (pred_plus - pred_minus).abs() / (float(2.0 * eps) * denom)
            epi_first = float(epi_weight) * jac

        next_state = ContactState(entity.dim_q, C * B, entity.state.device, state_cur)
        y_hat = decoder(next_state.q).view(C, B, D)
        pred_err = (y_hat - y).pow(2).mean(dim=2)  # [C,B]

        V_ps = compute_v_term_per_sample(entity, next_state).view(C, B)  # [C,B]
        step_score = V_ps + float(gamma_pred) * pred_err + step_base.unsqueeze(1)
        if k == 0 and revisit_first is not None:
            step_score = step_score + revisit_first
        if k == 0 and repeat_first is not None:
            step_score = step_score + repeat_first
        if k == 0 and cover_first is not None:
            step_score = step_score + cover_first
        if k == 0 and epi_first is not None:
            step_score = step_score + epi_first

        total = total + float(discount) * step_score
        discount *= float(lookahead_discount)
        if discount <= 0.0:
            break

    best_idx = total.argmin(dim=0)  # [B]
    return cand[best_idx]


def _entity_forward_step_fast(
    *,
    entity,
    u_dict: Dict[str, torch.Tensor],
    dt: float,
    skip_free_energy: bool = True,
    compute_action: bool = False,
) -> None:
    out = entity.forward_tensor(
        state_flat=entity.state.flat,
        u_dict=u_dict,
        dt=float(dt),
        prev_chart_weights=getattr(entity, "_prev_chart_weights", None),
        prediction_error=None,
        detach_next_prev_weights=True,
        compute_action=bool(compute_action),
        skip_free_energy=bool(skip_free_energy),
    )
    entity.state = ContactState(entity.dim_q, entity.state.batch_size, entity.state.device, out["next_state_flat"])
    entity._prev_chart_weights = out["next_prev_chart_weights"]


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes ROC-AUC for binary labels {0,1} via rank statistic.
    """
    scores = scores.detach().flatten().float().cpu()
    labels = labels.detach().flatten().long().cpu()
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Mann–Whitney / rank-sum AUC expects ranks in ascending order (low→high score).
    order = torch.argsort(scores, descending=False)
    ranked_labels = labels[order]
    # ranks are 1..N
    ranks = torch.arange(1, ranked_labels.numel() + 1, dtype=torch.float32)
    rank_sum_pos = ranks[ranked_labels == 1].sum().item()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
) -> Dict[str, float]:
    set_seed(seed)
    X = features.detach().float()
    y = labels.detach().long()
    n = int(X.shape[0])
    if n < 10:
        return {"acc": float("nan"), "auc": float("nan")}

    perm = torch.randperm(n)
    X = X[perm]
    y = y[perm]
    n_train = int(0.8 * n)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]

    clf = nn.Linear(int(X.shape[1]), 2)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(int(epochs)):
        logits = clf(X_tr)
        loss = torch.nn.functional.cross_entropy(logits, y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = clf(X_te)
        pred = logits.argmax(dim=1)
        acc = float((pred == y_te).float().mean().item())
        scores = logits[:, 1]
        auc = _binary_auc(scores, y_te)
    return {"acc": acc, "auc": float(auc)}


def make_optimizer(
    *,
    entity,
    decoder: nn.Module,
    lr: float,
    weight_decay: float,
    train_v: bool,
    lr_v: Optional[float] = None,
) -> torch.optim.Optimizer:
    lr = float(lr)
    wd = float(weight_decay)
    if lr_v is None:
        lr_v = lr
    lr_v = float(lr_v)

    v_params = list(getattr(entity, "net_V", nn.Identity()).parameters())
    v_param_ids = {id(p) for p in v_params}
    other_entity_params = [p for p in entity.parameters() if id(p) not in v_param_ids]
    main_params = other_entity_params + list(decoder.parameters())

    if not train_v:
        for p in v_params:
            p.requires_grad_(False)
        return torch.optim.AdamW(main_params, lr=lr, weight_decay=wd)

    groups = [{"params": main_params, "lr": lr, "weight_decay": wd}]
    if v_params:
        groups.append({"params": v_params, "lr": lr_v, "weight_decay": wd})
    return torch.optim.AdamW(groups)


def run_online_steps(
    *,
    entity,
    decoder: nn.Module,
    read_port: GaussianTruncRenormReadPort,
    optimizer: torch.optim.Optimizer,
    device: str,
    batch_size: int,
    steps: int,
    dt: float,
    mode: str,
    step_size: float,
    candidates: Sequence[float],
    beta_kl: float,
    gamma_pred: float,
    cost_weight: float,
    cost_power: float,
    cost_offset: float,
    noise_std: float,
    pointer_drift: float,
    pointer_bounds: str,
    anti_lock_weight: float,
    anti_lock_sigma: float,
    revisit_weight: float,
    revisit_window: int,
    lookahead_steps: int,
    lookahead_discount: float,
    pref_repeat_weight: float,
    cover_weight: float,
    cover_sigma: float,
    cover_window: int,
    cover_warmup_frac: float,
    epi_weight: float,
    epi_normalize: int,
    epi_mode: str,
    epi_obs_eps: float,
    epi_pred_floor: float,
    log_f,
    tag: str,
    log_every: int,
) -> Dict[str, List[float]]:
    entity.train()
    decoder.train()
    reset_experience(entity)

    # External pointer state
    s = torch.rand(batch_size, device=device) * float(read_port.length - 1)
    cand = torch.tensor(list(candidates), device=device, dtype=torch.float32) * float(step_size)
    token_ids = getattr(read_port, "token_ids", None)
    prev_tok: Optional[torch.Tensor] = None
    if float(pref_repeat_weight) > 0.0 and isinstance(token_ids, torch.Tensor) and token_ids.ndim == 1:
        idx0 = s.detach().round().to(dtype=torch.long).clamp(0, int(read_port.length - 1))
        prev_tok = token_ids[idx0]

    entity.enter_online()

    recent_s: Optional[torch.Tensor] = None
    recent_len = 0
    recent_ptr = 0
    if float(revisit_weight) > 0.0 and int(revisit_window) > 0:
        K = int(revisit_window)
        recent_s = torch.full((K, batch_size), -1, device=device, dtype=torch.long)
        recent_s[0] = s.detach().round().to(dtype=torch.long)
        recent_len = 1
        recent_ptr = 1 % K

    cover_s: Optional[torch.Tensor] = None
    cover_len = 0
    cover_ptr = 0
    if float(cover_weight) > 0.0 and int(cover_window) > 0:
        Kc = int(cover_window)
        cover_s = torch.empty((Kc, batch_size), device=device, dtype=torch.float32)
        cover_s[0] = s.detach().to(dtype=torch.float32)
        cover_len = 1
        cover_ptr = 1 % Kc

    warmup_frac = float(cover_warmup_frac)
    if not math.isfinite(warmup_frac) or warmup_frac <= 0.0:
        warmup_steps = 0
    else:
        warmup_steps = int(round(float(steps) * max(0.0, min(warmup_frac, 1.0))))

    pred_curve = torch.empty(int(steps), device=device, dtype=torch.float32)
    F_curve = torch.empty(int(steps), device=device, dtype=torch.float32)
    delta_curve = torch.empty(int(steps), device=device, dtype=torch.float32)
    last_parts: Dict[str, torch.Tensor] = {}
    last_s_mean = torch.tensor(float("nan"), device=device)
    last_s_std = torch.tensor(float("nan"), device=device)
    last_pred_err = torch.tensor(float("nan"), device=device)
    last_delta = torch.tensor(float("nan"), device=device)
    last_H = torch.tensor(float("nan"), device=device)
    last_epi_proxy = torch.tensor(float("nan"), device=device)
    epi_mode_s = str(epi_mode or "chart_entropy").strip().lower()

    for t in range(int(steps)):
        if warmup_steps > 0:
            cover_scale = min(1.0, float(t + 1) / float(max(1, warmup_steps)))
        else:
            cover_scale = 1.0
        eff_cover_weight = float(cover_weight) * float(cover_scale)

        if mode == "passive":
            delta = torch.full_like(s, float(step_size))
        elif mode == "active":
            # Important for E0 fairness: candidate evaluation may sample noise (randn) and would otherwise
            # perturb the global RNG stream, changing the subsequent training noise even if delta ends up
            # identical to passive. We isolate the RNG usage of planning via fork_rng.
            rng_devices: List[int] = []
            if str(device).startswith("cuda") and torch.cuda.is_available():
                rng_devices = [int(torch.cuda.current_device())]
            with torch.random.fork_rng(devices=rng_devices, enabled=(float(noise_std) > 0.0)):
                delta = choose_delta_one_step(
                    entity=entity,
                    decoder=decoder,
                    read_port=read_port,
                    s=s,
                    candidates=cand,
                    dt=dt,
                    beta_kl=beta_kl,
                    gamma_pred=gamma_pred,
                    cost_weight=cost_weight,
                    cost_power=float(cost_power),
                    cost_offset=float(cost_offset),
                    noise_std=noise_std,
                    pointer_drift=float(pointer_drift),
                    pointer_bounds=str(pointer_bounds),
                    anti_lock_weight=float(anti_lock_weight),
                    anti_lock_sigma=float(anti_lock_sigma),
                    revisit_weight=float(revisit_weight),
                    recent_s=(recent_s[:recent_len] if recent_s is not None and recent_len > 0 else None),
                    lookahead_steps=int(lookahead_steps),
                    lookahead_discount=float(lookahead_discount),
                    pref_repeat_weight=float(pref_repeat_weight),
                    token_ids=token_ids,
                    prev_token_ids=prev_tok,
                    cover_weight=float(eff_cover_weight),
                    cover_sigma=float(cover_sigma),
                    cover_recent_s=(cover_s[:cover_len] if cover_s is not None and cover_len > 0 else None),
                    epi_weight=float(epi_weight),
                    epi_normalize=int(epi_normalize),
                    epi_mode=str(epi_mode_s),
                    epi_obs_eps=float(epi_obs_eps),
                    epi_pred_floor=float(epi_pred_floor),
                )
        else:
            raise ValueError(f"unknown mode: {mode}")

        s_new = _apply_pointer_bounds(
            s + float(pointer_drift) + delta,
            max_s=float(read_port.length - 1),
            mode=str(pointer_bounds),
        )
        pref_penalty_ps: Optional[torch.Tensor] = None
        if prev_tok is not None and isinstance(token_ids, torch.Tensor):
            idx_new = s_new.detach().round().to(dtype=torch.long).clamp(0, int(read_port.length - 1))
            tok_new = token_ids[idx_new]
            pref_penalty_ps = float(pref_repeat_weight) * (tok_new == prev_tok).to(dtype=s_new.dtype)
            prev_tok = tok_new
        revisit_penalty_ps: Optional[torch.Tensor] = None
        if float(revisit_weight) > 0.0 and recent_s is not None and recent_len > 0:
            s_new_round = s_new.detach().round().to(dtype=torch.long)
            hit = (recent_s[:recent_len] == s_new_round.unsqueeze(0)).any(dim=0).to(dtype=s_new.dtype)
            revisit_penalty_ps = float(revisit_weight) * hit
            # push new pointer into ring buffer
            recent_s[recent_ptr] = s_new_round
            recent_ptr = (recent_ptr + 1) % int(recent_s.shape[0])
            recent_len = min(recent_len + 1, int(recent_s.shape[0]))

        cover_penalty_ps: Optional[torch.Tensor] = None
        if eff_cover_weight > 0.0 and cover_s is not None and cover_len > 0:
            sigma = max(float(cover_sigma), 1e-6)
            sim = torch.exp(-0.5 * ((s_new.unsqueeze(0) - cover_s[:cover_len]) / sigma).pow(2))
            cover_penalty_ps = float(eff_cover_weight) * sim.sum(dim=0)
            cover_s[cover_ptr] = s_new.detach().to(dtype=torch.float32)
            cover_ptr = (cover_ptr + 1) % int(cover_s.shape[0])
            cover_len = min(cover_len + 1, int(cover_s.shape[0]))
        epi_penalty_ps: Optional[torch.Tensor] = None
        y = read_port.read(s_new)
        noise: Optional[torch.Tensor] = None
        y_in = y
        if noise_std > 0:
            noise = torch.randn_like(y)
            y_in = y + float(noise_std) * noise

        if epi_mode_s == "jac_e":
            compute_proxy = (
                float(epi_weight) != 0.0
                or (t == int(steps) - 1)
                or (log_f is not None and int(log_every) > 0 and t % int(log_every) == 0)
            )
            if compute_proxy:
                eps = float(epi_obs_eps)
                if not math.isfinite(eps) or eps <= 0.0:
                    eps = 1.0
                floor = float(epi_pred_floor)
                if not math.isfinite(floor) or floor <= 0.0:
                    floor = 1e-8
                floor = float(max(floor, 1e-8))
                with torch.no_grad():
                    state_pre = entity.state.flat
                    prev_pre = getattr(entity, "_prev_chart_weights", None)
                    s_plus = _apply_pointer_bounds(
                        s_new + eps,
                        max_s=float(read_port.length - 1),
                        mode=str(pointer_bounds),
                    )
                    s_minus = _apply_pointer_bounds(
                        s_new - eps,
                        max_s=float(read_port.length - 1),
                        mode=str(pointer_bounds),
                    )
                    y_plus = read_port.read(s_plus)
                    y_minus = read_port.read(s_minus)
                    y_plus_in = y_plus
                    y_minus_in = y_minus
                    if noise is not None:
                        y_plus_in = y_plus + float(noise_std) * noise
                        y_minus_in = y_minus + float(noise_std) * noise

                    out_plus = entity.forward_tensor(
                        state_flat=state_pre,
                        u_dict={"default": y_plus_in},
                        dt=float(dt),
                        prev_chart_weights=prev_pre,
                        prediction_error=None,
                        detach_next_prev_weights=True,
                        compute_action=False,
                        skip_free_energy=True,
                    )
                    out_minus = entity.forward_tensor(
                        state_flat=state_pre,
                        u_dict={"default": y_minus_in},
                        dt=float(dt),
                        prev_chart_weights=prev_pre,
                        prediction_error=None,
                        detach_next_prev_weights=True,
                        compute_action=False,
                        skip_free_energy=True,
                    )
                    next_plus = ContactState(entity.dim_q, batch_size, entity.state.device, out_plus["next_state_flat"])
                    next_minus = ContactState(
                        entity.dim_q, batch_size, entity.state.device, out_minus["next_state_flat"]
                    )
                    y_hat_plus = decoder(next_plus.q)
                    y_hat_minus = decoder(next_minus.q)
                    pred_plus = (y_hat_plus - y_plus).pow(2).mean(dim=1)
                    pred_minus = (y_hat_minus - y_minus).pow(2).mean(dim=1)
                    denom = (0.5 * (pred_plus + pred_minus)).clamp(min=floor)
                    jac_ps = (pred_plus - pred_minus).abs() / (float(2.0 * eps) * denom)
                last_epi_proxy = jac_ps.detach().mean().to(dtype=torch.float32)
                if float(epi_weight) != 0.0:
                    epi_penalty_ps = float(epi_weight) * jac_ps.to(dtype=s_new.dtype)

        if epi_mode_s == "obs_grad":
            eps = float(epi_obs_eps)
            if not math.isfinite(eps) or eps <= 0.0:
                eps = 1.0
            s_plus = _apply_pointer_bounds(
                s_new + eps,
                max_s=float(read_port.length - 1),
                mode=str(pointer_bounds),
            )
            s_minus = _apply_pointer_bounds(
                s_new - eps,
                max_s=float(read_port.length - 1),
                mode=str(pointer_bounds),
            )
            y_plus = read_port.read(s_plus)
            y_minus = read_port.read(s_minus)
            obs_grad_ps = (y_plus - y_minus).pow(2).mean(dim=1)
            last_epi_proxy = obs_grad_ps.detach().mean().to(dtype=torch.float32)
            if float(epi_weight) != 0.0:
                epi_penalty_ps = float(epi_weight) * obs_grad_ps.to(dtype=s_new.dtype)
        s = s_new

        _entity_forward_step_fast(entity=entity, u_dict={"default": y_in}, dt=float(dt))
        if epi_mode_s == "chart_entropy":
            w = getattr(entity, "_prev_chart_weights", None)
            if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[0]) == batch_size:
                ent = _chart_entropy(w, normalize=(int(epi_normalize) != 0))
                last_epi_proxy = ent.detach().mean().to(dtype=torch.float32)
                if float(epi_weight) != 0.0:
                    epi_penalty_ps = float(epi_weight) * ent.to(dtype=s.dtype)

        # Reconstruction (same-step)
        y_hat = decoder(entity.state.q)
        pred_err_ps = (y_hat - y).pow(2).mean(dim=1)

        loss, parts = compute_loss(
            entity=entity,
            state=entity.state,
            pred_err_per_sample=pred_err_ps,
            action_delta=delta,
            beta_kl=beta_kl,
            gamma_pred=gamma_pred,
            cost_weight=cost_weight,
            cost_power=float(cost_power),
            cost_offset=float(cost_offset),
            anti_lock_weight=float(anti_lock_weight),
            anti_lock_sigma=float(anti_lock_sigma),
            revisit_penalty_per_sample=revisit_penalty_ps,
            pref_penalty_per_sample=pref_penalty_ps,
            cover_penalty_per_sample=cover_penalty_ps,
            epi_penalty_per_sample=epi_penalty_ps,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(entity.parameters()) + list(decoder.parameters()), max_norm=1.0)
        optimizer.step()

        # Truncate BPTT: detach state
        entity.state = ContactState(entity.dim_q, batch_size, device, entity.state.flat.detach())

        pred_mean = pred_err_ps.detach().mean().to(dtype=torch.float32)
        pred_curve[int(t)] = pred_mean
        F_curve[int(t)] = loss.detach().to(dtype=torch.float32)
        delta_curve[int(t)] = delta.detach().mean().to(dtype=torch.float32)
        last_parts = parts
        last_s_mean = s.detach().mean()
        last_s_std = s.detach().std()
        last_pred_err = pred_mean
        last_delta = delta.detach().mean()
        w = getattr(entity, "_prev_chart_weights", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[0]) == batch_size:
            last_H = _chart_entropy(w, normalize=True).mean().to(dtype=torch.float32)
        else:
            last_H = torch.tensor(float("nan"), device=device)

        if log_f is not None and (int(log_every) > 0) and (t % int(log_every) == 0 or t == int(steps) - 1):
            parts_f = _parts_to_float(parts)
            rec = {
                "tag": tag,
                "phase": "online",
                "mode": mode,
                "t": int(t),
                "s_mean": float(s.detach().mean().item()),
                "s_std": float(s.detach().std().item()),
                "delta_mean": float(delta.detach().mean().item()),
                "pred_err": float(pred_mean.detach().item()),
                "H_chart": float(last_H.detach().item()),
                "epi_proxy": float(last_epi_proxy.detach().item()),
                "pointer_drift": float(pointer_drift),
                "pointer_bounds": str(pointer_bounds),
                "cover_w": float(eff_cover_weight),
                **parts_f,
            }
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    curves = {
        "pred_err": pred_curve.detach().cpu().tolist(),
        "F": F_curve.detach().cpu().tolist(),
        "delta": delta_curve.detach().cpu().tolist(),
        "final": {
            "s_mean": float(last_s_mean.detach().cpu().item()),
            "s_std": float(last_s_std.detach().cpu().item()),
            "delta_mean": float(last_delta.detach().cpu().item()),
            "pred_err": float(last_pred_err.detach().cpu().item()),
            "H_chart": float(last_H.detach().cpu().item()),
            "epi_proxy": float(last_epi_proxy.detach().cpu().item()),
            **_parts_to_float(last_parts),
        },
    }
    return {
        **curves,
    }


def run_fixed_rhythm_episodes(
    *,
    entity,
    decoder: nn.Module,
    read_port: GaussianTruncRenormReadPort,
    optimizer: torch.optim.Optimizer,
    device: str,
    batch_size: int,
    episodes: int,
    online_steps: int,
    offline_steps: int,
    dt: float,
    step_size: float,
    candidates: Sequence[float],
    beta_kl: float,
    gamma_pred: float,
    cost_weight: float,
    cost_power: float,
    cost_offset: float,
    noise_std: float,
    pointer_drift: float,
    pointer_bounds: str,
    anti_lock_weight: float,
    anti_lock_sigma: float,
    revisit_weight: float,
    revisit_window: int,
    lookahead_steps: int,
    lookahead_discount: float,
    pref_repeat_weight: float,
    cover_weight: float,
    cover_sigma: float,
    cover_window: int,
    cover_warmup_frac: float,
    epi_weight: float,
    epi_normalize: int,
    epi_mode: str,
    epi_obs_eps: float,
    epi_pred_floor: float,
    replay_mode: str,
    log_f,
    label: int,
    tag: str,
    log_every: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Returns:
      features: [episodes*B, dim_q] (mean offline q)
      labels:   [episodes*B]
      e2_diag:  basic cycle metrics on pointer s (quantized)
    """
    entity.train()
    decoder.train()
    epi_mode_s = str(epi_mode or "chart_entropy").strip().lower()

    cand = torch.tensor(list(candidates), device=device, dtype=torch.float32) * float(step_size)
    token_ids = getattr(read_port, "token_ids", None)
    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    # E2: pointer cycle diagnostics (quantize to int)
    cycle_hits = 0
    boundary_cycle_hits = 0
    cycle_total = 0
    max_period = 8
    min_repeats = 3
    tail_window = 32
    boundary_low = 0
    boundary_high = int(read_port.length - 1)

    for ep in range(int(episodes)):
        entity.reset(batch_size=batch_size, device=device)
        reset_experience(entity)
        # Fresh pointer per episode, away from edges
        s = torch.rand(batch_size, device=device) * float(read_port.length - 1)
        prev_tok: Optional[torch.Tensor] = None
        if float(pref_repeat_weight) > 0.0 and isinstance(token_ids, torch.Tensor) and token_ids.ndim == 1:
            idx0 = s.detach().round().to(dtype=torch.long).clamp(0, int(read_port.length - 1))
            prev_tok = token_ids[idx0]
        # Keep only the last `tail_window` pointer positions on-device; copy once per episode.
        s_tail = torch.empty((tail_window, batch_size), device=device, dtype=torch.long)

        recent_s: Optional[torch.Tensor] = None
        recent_len = 0
        recent_ptr = 0
        if float(revisit_weight) > 0.0 and int(revisit_window) > 0:
            K = int(revisit_window)
            recent_s = torch.full((K, batch_size), -1, device=device, dtype=torch.long)
            recent_s[0] = s.detach().round().to(dtype=torch.long)
            recent_len = 1
            recent_ptr = 1 % K

        cover_s: Optional[torch.Tensor] = None
        cover_len = 0
        cover_ptr = 0
        if float(cover_weight) > 0.0 and int(cover_window) > 0:
            Kc = int(cover_window)
            cover_s = torch.empty((Kc, batch_size), device=device, dtype=torch.float32)
            cover_s[0] = s.detach().to(dtype=torch.float32)
            cover_len = 1
            cover_ptr = 1 % Kc

        warmup_frac = float(cover_warmup_frac)
        if not math.isfinite(warmup_frac) or warmup_frac <= 0.0:
            warmup_steps = 0
        else:
            warmup_steps = int(round(float(online_steps) * max(0.0, min(warmup_frac, 1.0))))

        # Online
        entity.enter_online()
        for t in range(int(online_steps)):
            if warmup_steps > 0:
                cover_scale = min(1.0, float(t + 1) / float(max(1, warmup_steps)))
            else:
                cover_scale = 1.0
            eff_cover_weight = float(cover_weight) * float(cover_scale)

            rng_devices: List[int] = []
            if str(device).startswith("cuda") and torch.cuda.is_available():
                rng_devices = [int(torch.cuda.current_device())]
            with torch.random.fork_rng(devices=rng_devices, enabled=(float(noise_std) > 0.0)):
                delta = choose_delta_one_step(
                    entity=entity,
                    decoder=decoder,
                    read_port=read_port,
                    s=s,
                    candidates=cand,
                    dt=dt,
                    beta_kl=beta_kl,
                    gamma_pred=gamma_pred,
                    cost_weight=cost_weight,
                    cost_power=float(cost_power),
                    cost_offset=float(cost_offset),
                    noise_std=noise_std,
                    pointer_drift=float(pointer_drift),
                    pointer_bounds=str(pointer_bounds),
                    anti_lock_weight=float(anti_lock_weight),
                    anti_lock_sigma=float(anti_lock_sigma),
                    revisit_weight=float(revisit_weight),
                    recent_s=(recent_s[:recent_len] if recent_s is not None and recent_len > 0 else None),
                    lookahead_steps=int(lookahead_steps),
                    lookahead_discount=float(lookahead_discount),
                    pref_repeat_weight=float(pref_repeat_weight),
                    token_ids=token_ids,
                    prev_token_ids=prev_tok,
                    cover_weight=float(eff_cover_weight),
                    cover_sigma=float(cover_sigma),
                    cover_recent_s=(cover_s[:cover_len] if cover_s is not None and cover_len > 0 else None),
                    epi_weight=float(epi_weight),
                    epi_normalize=int(epi_normalize),
                    epi_mode=str(epi_mode_s),
                    epi_obs_eps=float(epi_obs_eps),
                    epi_pred_floor=float(epi_pred_floor),
                )
            s_new = _apply_pointer_bounds(
                s + float(pointer_drift) + delta,
                max_s=float(read_port.length - 1),
                mode=str(pointer_bounds),
            )
            pref_penalty_ps: Optional[torch.Tensor] = None
            if prev_tok is not None and isinstance(token_ids, torch.Tensor):
                idx_new = s_new.detach().round().to(dtype=torch.long).clamp(0, int(read_port.length - 1))
                tok_new = token_ids[idx_new]
                pref_penalty_ps = float(pref_repeat_weight) * (tok_new == prev_tok).to(dtype=s_new.dtype)
                prev_tok = tok_new
            revisit_penalty_ps: Optional[torch.Tensor] = None
            if float(revisit_weight) > 0.0 and recent_s is not None and recent_len > 0:
                s_new_round = s_new.detach().round().to(dtype=torch.long)
                hit = (recent_s[:recent_len] == s_new_round.unsqueeze(0)).any(dim=0).to(dtype=s_new.dtype)
                revisit_penalty_ps = float(revisit_weight) * hit
                recent_s[recent_ptr] = s_new_round
                recent_ptr = (recent_ptr + 1) % int(recent_s.shape[0])
                recent_len = min(recent_len + 1, int(recent_s.shape[0]))
            cover_penalty_ps: Optional[torch.Tensor] = None
            if eff_cover_weight > 0.0 and cover_s is not None and cover_len > 0:
                sigma = max(float(cover_sigma), 1e-6)
                sim = torch.exp(-0.5 * ((s_new.unsqueeze(0) - cover_s[:cover_len]) / sigma).pow(2))
                cover_penalty_ps = float(eff_cover_weight) * sim.sum(dim=0)
                cover_s[cover_ptr] = s_new.detach().to(dtype=torch.float32)
                cover_ptr = (cover_ptr + 1) % int(cover_s.shape[0])
                cover_len = min(cover_len + 1, int(cover_s.shape[0]))
            s = s_new
            epi_penalty_ps: Optional[torch.Tensor] = None
            epi_proxy_mean = float("nan")
            y = read_port.read(s)
            noise: Optional[torch.Tensor] = None
            y_in = y
            if noise_std > 0:
                noise = torch.randn_like(y)
                y_in = y + float(noise_std) * noise

            if epi_mode_s == "jac_e":
                compute_proxy = (
                    float(epi_weight) != 0.0
                    or (log_f is not None and int(log_every) > 0 and t % int(log_every) == 0)
                    or (t == online_steps - 1)
                )
                if compute_proxy:
                    eps = float(epi_obs_eps)
                    if not math.isfinite(eps) or eps <= 0.0:
                        eps = 1.0
                    floor = float(epi_pred_floor)
                    if not math.isfinite(floor) or floor <= 0.0:
                        floor = 1e-8
                    floor = float(max(floor, 1e-8))
                    with torch.no_grad():
                        state_pre = entity.state.flat
                        prev_pre = getattr(entity, "_prev_chart_weights", None)
                        s_plus = _apply_pointer_bounds(
                            s + eps,
                            max_s=float(read_port.length - 1),
                            mode=str(pointer_bounds),
                        )
                        s_minus = _apply_pointer_bounds(
                            s - eps,
                            max_s=float(read_port.length - 1),
                            mode=str(pointer_bounds),
                        )
                        y_plus = read_port.read(s_plus)
                        y_minus = read_port.read(s_minus)
                        y_plus_in = y_plus
                        y_minus_in = y_minus
                        if noise is not None:
                            y_plus_in = y_plus + float(noise_std) * noise
                            y_minus_in = y_minus + float(noise_std) * noise

                        out_plus = entity.forward_tensor(
                            state_flat=state_pre,
                            u_dict={"default": y_plus_in},
                            dt=float(dt),
                            prev_chart_weights=prev_pre,
                            prediction_error=None,
                            detach_next_prev_weights=True,
                            compute_action=False,
                            skip_free_energy=True,
                        )
                        out_minus = entity.forward_tensor(
                            state_flat=state_pre,
                            u_dict={"default": y_minus_in},
                            dt=float(dt),
                            prev_chart_weights=prev_pre,
                            prediction_error=None,
                            detach_next_prev_weights=True,
                            compute_action=False,
                            skip_free_energy=True,
                        )
                        next_plus = ContactState(
                            entity.dim_q, batch_size, entity.state.device, out_plus["next_state_flat"]
                        )
                        next_minus = ContactState(
                            entity.dim_q, batch_size, entity.state.device, out_minus["next_state_flat"]
                        )
                        y_hat_plus = decoder(next_plus.q)
                        y_hat_minus = decoder(next_minus.q)
                        pred_plus = (y_hat_plus - y_plus).pow(2).mean(dim=1)
                        pred_minus = (y_hat_minus - y_minus).pow(2).mean(dim=1)
                        denom = (0.5 * (pred_plus + pred_minus)).clamp(min=floor)
                        jac_ps = (pred_plus - pred_minus).abs() / (float(2.0 * eps) * denom)
                    epi_proxy_mean = float(jac_ps.detach().mean().item())
                    if float(epi_weight) != 0.0:
                        epi_penalty_ps = float(epi_weight) * jac_ps.to(dtype=s.dtype)
            if epi_mode_s == "obs_grad":
                eps = float(epi_obs_eps)
                if not math.isfinite(eps) or eps <= 0.0:
                    eps = 1.0
                s_plus = _apply_pointer_bounds(
                    s + eps,
                    max_s=float(read_port.length - 1),
                    mode=str(pointer_bounds),
                )
                s_minus = _apply_pointer_bounds(
                    s - eps,
                    max_s=float(read_port.length - 1),
                    mode=str(pointer_bounds),
                )
                y_plus = read_port.read(s_plus)
                y_minus = read_port.read(s_minus)
                obs_grad_ps = (y_plus - y_minus).pow(2).mean(dim=1)
                epi_proxy_mean = float(obs_grad_ps.detach().mean().item())
                if float(epi_weight) != 0.0:
                    epi_penalty_ps = float(epi_weight) * obs_grad_ps.to(dtype=s.dtype)
            s_tail[int(t) % int(tail_window)] = s.detach().round().to(dtype=torch.long)

            _entity_forward_step_fast(entity=entity, u_dict={"default": y_in}, dt=float(dt))
            # Fill experience buffer for offline replay conditioning (reward is a dummy scalar in this verifier).
            entity.experience.push(y_in, None, entity.state.flat, 0.0)

            y_hat = decoder(entity.state.q)
            pred_err_ps = (y_hat - y).pow(2).mean(dim=1)
            if epi_mode_s == "chart_entropy":
                w = getattr(entity, "_prev_chart_weights", None)
                if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[0]) == batch_size:
                    ent = _chart_entropy(w, normalize=(int(epi_normalize) != 0))
                    epi_proxy_mean = float(ent.detach().mean().item())
                    if float(epi_weight) != 0.0:
                        epi_penalty_ps = float(epi_weight) * ent.to(dtype=s.dtype)
            loss, parts = compute_loss(
                entity=entity,
                state=entity.state,
                pred_err_per_sample=pred_err_ps,
                action_delta=delta,
                beta_kl=beta_kl,
                gamma_pred=gamma_pred,
                cost_weight=cost_weight,
                cost_power=float(cost_power),
                cost_offset=float(cost_offset),
                anti_lock_weight=float(anti_lock_weight),
                anti_lock_sigma=float(anti_lock_sigma),
                revisit_penalty_per_sample=revisit_penalty_ps,
                pref_penalty_per_sample=pref_penalty_ps,
                cover_penalty_per_sample=cover_penalty_ps,
                epi_penalty_per_sample=epi_penalty_ps,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(entity.parameters()) + list(decoder.parameters()), max_norm=1.0)
            optimizer.step()
            entity.state = ContactState(entity.dim_q, batch_size, device, entity.state.flat.detach())

            if log_f is not None and (int(log_every) > 0) and (t % int(log_every) == 0 or t == online_steps - 1):
                w = getattr(entity, "_prev_chart_weights", None)
                if isinstance(w, torch.Tensor) and w.ndim == 2 and int(w.shape[0]) == batch_size:
                    H_chart = float(_chart_entropy(w, normalize=True).mean().detach().item())
                else:
                    H_chart = float("nan")
                parts_f = _parts_to_float(parts)
                rec = {
                    "tag": tag,
                    "episode": int(ep),
                    "label": int(label),
                    "phase": "online",
                    "t": int(t),
                    "s_mean": float(s.detach().mean().item()),
                    "delta_mean": float(delta.detach().mean().item()),
                    "pred_err": float(pred_err_ps.detach().mean().item()),
                    "H_chart": float(H_chart),
                    "epi_proxy": float(epi_proxy_mean),
                    "pointer_drift": float(pointer_drift),
                    "pointer_bounds": str(pointer_bounds),
                    "cover_w": float(eff_cover_weight),
                    **parts_f,
                }
                log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # E2: detect short cycles in pointer tail (per sample)
        T = int(online_steps)
        W = min(int(tail_window), max(0, T))
        if W > 0:
            start = int(T - W)
            idx = [(start + k) % int(tail_window) for k in range(W)]
            tail = s_tail[idx].detach().cpu().tolist()  # List[W][B]
            for b in range(batch_size):
                tokens = [int(tail[k][b]) for k in range(W)]
                cycle_total += 1
                cyc = _detect_tail_cycle(tokens, max_period=max_period, min_repeats=min_repeats)
                if cyc is not None:
                    cycle_hits += 1
                    _, pattern = cyc
                    if pattern and all((x == boundary_low) or (x == boundary_high) for x in pattern):
                        boundary_cycle_hits += 1

        # Offline (freeze env)
        entity.enter_offline()
        offline_q: List[torch.Tensor] = []
        # Offline is not trained in this verifier; avoid building autograd graphs.
        with torch.no_grad():
            for t in range(int(offline_steps)):
                u_off: Dict[str, torch.Tensor] = {}
                exp = getattr(entity, "experience", None)
                exp_size = int(getattr(exp, "size", 0) or 0) if exp is not None else 0
                if replay_mode != "none" and exp is not None and exp_size > 0:
                    replay = exp.sample_replay(batch_size, mode=replay_mode)
                    if replay is not None:
                        replay_state = replay["states"].to(device)
                        if replay_state.shape[0] < batch_size:
                            pad = replay_state[0:1].expand(batch_size - replay_state.shape[0], -1)
                            replay_state = torch.cat([replay_state, pad], dim=0)
                        replay_q = replay_state[:, : entity.dim_q]
                        u_off["replay"] = 0.1 * (replay_q - entity.state.q)

                _entity_forward_step_fast(entity=entity, u_dict=u_off, dt=float(dt))
                offline_q.append(entity.state.q.detach())
                if log_f is not None and (int(log_every) > 0) and (t % int(log_every) == 0 or t == offline_steps - 1):
                    rec = {
                        "tag": tag,
                        "episode": int(ep),
                        "label": int(label),
                        "phase": "offline",
                        "t": int(t),
                        "q_norm": float(entity.state.q.detach().norm(dim=1).mean().item()),
                        "p_norm": float(entity.state.p.detach().norm(dim=1).mean().item()),
                    }
                    log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        q_mean = torch.stack(offline_q, dim=0).mean(dim=0).detach().cpu() if offline_q else entity.state.q.detach().cpu()
        all_feats.append(q_mean)
        all_labels.append(torch.full((batch_size,), int(label), dtype=torch.long))

    feats = torch.cat(all_feats, dim=0) if all_feats else torch.zeros((0, entity.dim_q))
    labs = torch.cat(all_labels, dim=0) if all_labels else torch.zeros((0,), dtype=torch.long)

    non_boundary_hits = int(cycle_hits - boundary_cycle_hits)
    denom = max(1, int(cycle_total))
    e2 = {
        # Backward compatible aggregate (includes boundary cycles).
        "ptr_cycle_rate": float(cycle_hits / denom),
        "ptr_cycle_hits": int(cycle_hits),
        "ptr_cycle_total": int(cycle_total),
        # New: separate boundary pseudo-cycles for cleaner gating & comparisons.
        "ptr_cycle_rate_non_boundary": float(non_boundary_hits / denom),
        "ptr_cycle_hits_non_boundary": int(non_boundary_hits),
        "ptr_cycle_rate_boundary": float(boundary_cycle_hits / denom),
        "ptr_cycle_hits_boundary": int(boundary_cycle_hits),
        "ptr_cycle_boundary_low": int(boundary_low),
        "ptr_cycle_boundary_high": int(boundary_high),
    }
    return feats, labs, e2


def _detect_tail_cycle(tokens: Sequence[int], *, max_period: int, min_repeats: int) -> Optional[Tuple[int, Tuple[int, ...]]]:
    if max_period <= 0 or min_repeats <= 1:
        return None
    tail = list(tokens)
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
            return p, pattern
    return None


def _pick_vocab_token_id(vocab: CharVocab, candidates: Sequence[str]) -> Optional[int]:
    for tok in candidates:
        idx = vocab.token_to_id.get(str(tok))
        if idx is None:
            continue
        idx_i = int(idx)
        if idx_i in (vocab.pad_id, vocab.unk_id):
            continue
        return idx_i
    return None


def _pick_e3_markers(vocab: CharVocab) -> Tuple[int, int, str, str]:
    """
    Pick two distinct marker token ids for E3 that are likely to exist in the vocabulary.
    Falls back to the first available non-special tokens.
    """
    a_pref = ["A", "甲", "①", "Ⅰ", "一", "0", "◇", "□", "◎", "※"]
    b_pref = ["B", "乙", "②", "Ⅱ", "二", "1", "◆", "■", "●", "※"]
    a_id = _pick_vocab_token_id(vocab, a_pref)
    b_id = _pick_vocab_token_id(vocab, b_pref)
    if a_id is not None and b_id is not None and a_id != b_id:
        return a_id, b_id, str(vocab.id_to_token[a_id]), str(vocab.id_to_token[b_id])

    # Fallback: first two non-special ids.
    ids = [i for i in range(len(vocab.id_to_token)) if i not in (vocab.pad_id, vocab.unk_id)]
    if len(ids) < 2:
        raise RuntimeError("vocab too small for E3 markers")
    a_id = int(ids[0] if a_id is None else a_id)
    b_id = int(ids[1] if b_id is None or b_id == a_id else b_id)
    if a_id == b_id:
        b_id = int(ids[1])
    return a_id, b_id, str(vocab.id_to_token[a_id]), str(vocab.id_to_token[b_id])


def _pick_e3_kv_keys(vocab: CharVocab, *, exclude_ids: Sequence[int]) -> Tuple[int, int, str, str]:
    """
    Pick two distinct token ids to act as keys in E3+ (key-value binding).
    """
    exclude = {int(vocab.pad_id), int(vocab.unk_id)}
    exclude.update(int(x) for x in exclude_ids)
    k1_pref = ["K", "键", "Ⓚ", "κ", "甲", "X", "☆", "○", "◇", "□"]
    k2_pref = ["Q", "钥", "Ⓠ", "η", "乙", "Y", "★", "●", "◆", "■"]
    k1 = _pick_vocab_token_id(vocab, k1_pref)
    k2 = _pick_vocab_token_id(vocab, k2_pref)
    if k1 is not None and int(k1) in exclude:
        k1 = None
    if k2 is not None and int(k2) in exclude:
        k2 = None
    if k1 is not None and k2 is not None and int(k1) != int(k2):
        return int(k1), int(k2), str(vocab.id_to_token[int(k1)]), str(vocab.id_to_token[int(k2)])

    # Fallback: first two non-special ids not in exclude.
    ids = [i for i in range(len(vocab.id_to_token)) if i not in exclude]
    if len(ids) < 2:
        raise RuntimeError("vocab too small for E3+ keys")
    k1_id = int(ids[0] if k1 is None else int(k1))
    k2_id = int(ids[1] if k2 is None or int(k2) == k1_id else int(k2))
    if k2_id == k1_id:
        k2_id = int(ids[1])
    return k1_id, k2_id, str(vocab.id_to_token[k1_id]), str(vocab.id_to_token[k2_id])


def _make_allowed_token_ids(
    *,
    vocab: CharVocab,
    marker_ids: Sequence[int],
    device: str,
) -> torch.Tensor:
    exclude = {int(vocab.pad_id), int(vocab.unk_id)}
    exclude.update(int(x) for x in marker_ids)
    ids = [i for i in range(len(vocab.id_to_token)) if i not in exclude]
    if not ids:
        raise RuntimeError("no allowed token ids (check vocab_size / exclusions)")
    return torch.tensor(ids, device=device, dtype=torch.long)


def run_e3_marker_reading(
    *,
    entity,
    decoder: nn.Module,
    embedding: nn.Embedding,
    vocab: CharVocab,
    optimizer: torch.optim.Optimizer,
    device: str,
    batch_size: int,
    pairs: int,
    seq_len: int,
    clue_len: int,
    clue_margin: int,
    policy: str,
    fixate_steps: int,
    fixate_mad_mult: float,
    boundary_window: int,
    read_sigma: float,
    online_steps: int,
    offline_steps: int,
    dt: float,
    step_size: float,
    candidates: Sequence[float],
    beta_kl: float,
    gamma_pred: float,
    cost_weight: float,
    cost_power: float,
    cost_offset: float,
    noise_std: float,
    pointer_drift: float,
    pointer_bounds: str,
    anti_lock_weight: float,
    anti_lock_sigma: float,
    revisit_weight: float,
    revisit_window: int,
    lookahead_steps: int,
    lookahead_discount: float,
    pref_repeat_weight: float,
    cover_weight: float,
    cover_sigma: float,
    cover_window: int,
    cover_warmup_frac: float,
    epi_weight: float,
    epi_normalize: int,
    epi_mode: str,
    epi_obs_eps: float,
    epi_pred_floor: float,
    replay_mode: str,
    seed: int,
    log_f,
    tag: str,
    log_every: int,
) -> Dict[str, object]:
    """
    E3 (minimal reading/understanding closed-loop):
      - Build a synthetic token sequence where the ONLY class difference is a localized "clue patch".
      - Agent must actively sample (pointer) to observe the patch.
      - Verifier is automatic: we know the clue span.

    Returns summary dict with:
      - hit_rate / first_hit stats
      - linear probe metrics on mean offline q
      - pointer short-cycle diagnostics (non-boundary separated)
    """
    entity.train()
    decoder.train()
    epi_mode_s = str(epi_mode or "chart_entropy").strip().lower()
    policy_s = str(policy or "active").strip().lower()
    if policy_s not in ("active", "active_fixate", "passive", "random", "frozen"):
        raise ValueError(f"unknown e3 policy: {policy}")
    fixate_steps_i = int(max(0, int(fixate_steps))) if policy_s == "active_fixate" else 0
    fixate_mad_mult_f = float(fixate_mad_mult) if policy_s == "active_fixate" else 0.0
    if not math.isfinite(fixate_mad_mult_f) or fixate_mad_mult_f <= 0.0:
        fixate_mad_mult_f = 0.0

    marker_a, marker_b, tok_a, tok_b = _pick_e3_markers(vocab)
    allowed_ids = _make_allowed_token_ids(vocab=vocab, marker_ids=[marker_a, marker_b], device=device)

    CAND = torch.tensor(list(candidates), device=device, dtype=torch.float32) * float(step_size)
    C = int(CAND.numel())
    if C <= 0:
        raise ValueError("empty candidates")

    pairs = int(max(1, int(pairs)))
    seq_len = int(max(32, int(seq_len)))
    clue_len = int(max(1, int(clue_len)))
    clue_margin = int(max(0, int(clue_margin)))
    online_steps = int(max(1, int(online_steps)))
    offline_steps = int(max(0, int(offline_steps)))
    if clue_margin * 2 + clue_len >= seq_len:
        clue_margin = max(0, (seq_len - clue_len) // 4)
    read_sigma = float(read_sigma)
    if not math.isfinite(read_sigma) or read_sigma <= 0.0:
        read_sigma = 2.0
    boundary_window = int(boundary_window)
    if boundary_window <= 0:
        boundary_window = int(max(1, round(3.0 * float(read_sigma))))
    boundary_window = int(min(boundary_window, max(1, (seq_len - 1) // 2)))

    # E2-like pointer cycle diagnostics (quantize to int)
    cycle_hits = 0
    boundary_cycle_hits = 0
    cycle_total = 0
    max_period = 8
    min_repeats = 3
    tail_window = 32
    boundary_low = 0
    boundary_high = int(seq_len - 1)

    feats_all: List[torch.Tensor] = []
    labs_all: List[torch.Tensor] = []
    hit_any_all: List[torch.Tensor] = []
    first_hit_all: List[torch.Tensor] = []
    dwell_all: List[torch.Tensor] = []
    span_all: List[torch.Tensor] = []

    base_marker_violations = 0
    base_marker_count_total = 0
    clue_start_list: List[int] = []
    clue_edge_min = int(1e9)

    delta_sum = 0.0
    delta_abs_sum = 0.0
    delta_count = 0
    boundary_steps = 0
    boundary_total = 0
    fixate_trigger_total = 0
    fixate_step_total = 0

    # For fairness, each pair shares the same base sequence & clue position; only the marker id differs.
    for ep in range(pairs):
        base = allowed_ids[torch.randint(0, int(allowed_ids.numel()), (seq_len,), device=device)]
        base_marker_count = int(((base == int(marker_a)) | (base == int(marker_b))).sum().item())
        base_marker_count_total += base_marker_count
        if base_marker_count != 0:
            base_marker_violations += 1
        if clue_len < seq_len:
            pos_hi = int(seq_len - clue_margin - clue_len)
            pos_lo = int(clue_margin)
            if pos_hi <= pos_lo:
                pos_lo = 0
                pos_hi = int(seq_len - clue_len)
            clue_start = int(torch.randint(pos_lo, pos_hi + 1, (1,), device=device).item())
        else:
            clue_start = 0
        clue_end = int(min(seq_len, clue_start + clue_len))
        clue_start_list.append(int(clue_start))
        clue_edge_min = int(min(clue_edge_min, clue_start, int(seq_len - clue_end)))

        label_order = [(0, marker_a), (1, marker_b)]
        random.shuffle(label_order)
        for label, marker_id in label_order:
            token_ids = base.clone()
            token_ids[clue_start:clue_end] = int(marker_id)
            read_port = GaussianTruncRenormReadPort(token_ids=token_ids, embedding=embedding, sigma=float(read_sigma)).to(
                device
            )

            entity.reset(batch_size=batch_size, device=device)
            reset_experience(entity)

            # Start pointer away from edges to reduce boundary artifacts.
            start_margin = int(max(1, round(3.0 * float(read_port.sigma))))
            s = torch.rand(batch_size, device=device) * float(max(1.0, read_port.length - 1 - 2 * start_margin)) + float(
                start_margin
            )

            token_ids_port = getattr(read_port, "token_ids", None)
            prev_tok: Optional[torch.Tensor] = None
            if float(pref_repeat_weight) > 0.0 and isinstance(token_ids_port, torch.Tensor) and token_ids_port.ndim == 1:
                idx0 = s.detach().round().to(dtype=torch.long).clamp(0, int(read_port.length - 1))
                prev_tok = token_ids_port[idx0]

            recent_s: Optional[torch.Tensor] = None
            recent_len = 0
            recent_ptr = 0
            if float(revisit_weight) > 0.0 and int(revisit_window) > 0:
                K = int(revisit_window)
                recent_s = torch.full((K, batch_size), -1, device=device, dtype=torch.long)
                recent_s[0] = s.detach().round().to(dtype=torch.long)
                recent_len = 1
                recent_ptr = 1 % K

            cover_s: Optional[torch.Tensor] = None
            cover_len = 0
            cover_ptr = 0
            if float(cover_weight) > 0.0 and int(cover_window) > 0:
                Kc = int(cover_window)
                cover_s = torch.empty((Kc, batch_size), device=device, dtype=torch.float32)
                cover_s[0] = s.detach().to(dtype=torch.float32)
                cover_len = 1
                cover_ptr = 1 % Kc

            warmup_frac = float(cover_warmup_frac)
            if not math.isfinite(warmup_frac) or warmup_frac <= 0.0:
                warmup_steps = 0
            else:
                warmup_steps = int(round(float(online_steps) * max(0.0, min(warmup_frac, 1.0))))

            hit_any = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            first_hit = torch.full((batch_size,), -1, device=device, dtype=torch.int32)
            dwell = torch.zeros((batch_size,), device=device, dtype=torch.int32)
            s_min = s.detach().round().to(dtype=torch.long)
            s_max = s_min.clone()

            # Pointer tail for cycle diagnostics
            s_tail = torch.empty((tail_window, batch_size), device=device, dtype=torch.long)

            fixate_left: Optional[torch.Tensor] = None
            if fixate_steps_i > 0:
                fixate_left = torch.zeros((batch_size,), device=device, dtype=torch.int32)

            entity.enter_online()
            for t in range(int(online_steps)):
                fixate_mask: Optional[torch.Tensor] = None
                if fixate_left is not None:
                    fixate_mask = fixate_left > 0
                    fixate_step_total += int(fixate_mask.sum().item())

                # Coverage warmup scaling (optional)
                if warmup_steps > 0:
                    cover_scale = min(1.0, float(t + 1) / float(max(1, warmup_steps)))
                else:
                    cover_scale = 1.0
                eff_cover_weight = float(cover_weight) * float(cover_scale)

                if policy_s in ("active", "active_fixate"):
                    rng_devices: List[int] = []
                    if str(device).startswith("cuda") and torch.cuda.is_available():
                        rng_devices = [int(torch.cuda.current_device())]
                    with torch.random.fork_rng(devices=rng_devices, enabled=(float(noise_std) > 0.0)):
                        delta = choose_delta_one_step(
                            entity=entity,
                            decoder=decoder,
                            read_port=read_port,
                            s=s,
                            candidates=CAND,
                            dt=dt,
                            beta_kl=beta_kl,
                            gamma_pred=gamma_pred,
                            cost_weight=cost_weight,
                            cost_power=float(cost_power),
                            cost_offset=float(cost_offset),
                            noise_std=noise_std,
                            pointer_drift=float(pointer_drift),
                            pointer_bounds=str(pointer_bounds),
                            anti_lock_weight=float(anti_lock_weight),
                            anti_lock_sigma=float(anti_lock_sigma),
                            revisit_weight=float(revisit_weight),
                            recent_s=(recent_s[:recent_len] if recent_s is not None and recent_len > 0 else None),
                            lookahead_steps=int(lookahead_steps),
                            lookahead_discount=float(lookahead_discount),
                            pref_repeat_weight=float(pref_repeat_weight),
                            token_ids=token_ids_port,
                            prev_token_ids=prev_tok,
                            cover_weight=float(eff_cover_weight),
                            cover_sigma=float(cover_sigma),
                            cover_recent_s=(cover_s[:cover_len] if cover_s is not None and cover_len > 0 else None),
                            epi_weight=float(epi_weight),
                            epi_normalize=int(epi_normalize),
                            epi_mode=str(epi_mode_s),
                            epi_obs_eps=float(epi_obs_eps),
                            epi_pred_floor=float(epi_pred_floor),
                        )
                elif policy_s == "random":
                    delta = CAND[torch.randint(0, C, (batch_size,), device=device)]
                else:
                    delta = torch.zeros((batch_size,), device=device, dtype=s.dtype)

                if fixate_mask is not None:
                    delta = delta * (~fixate_mask).to(dtype=delta.dtype)

                drift_eff = 0.0 if policy_s == "frozen" else float(pointer_drift)
                if fixate_mask is not None:
                    drift = float(drift_eff) * (~fixate_mask).to(dtype=s.dtype)
                else:
                    drift = float(drift_eff)
                s_new = _apply_pointer_bounds(
                    s + drift + delta,
                    max_s=float(read_port.length - 1),
                    mode=str(pointer_bounds),
                )
                s_round = s_new.detach().round().to(dtype=torch.long).clamp(0, int(read_port.length - 1))
                s_tail[int(t) % int(tail_window)] = s_round
                s_min = torch.minimum(s_min, s_round)
                s_max = torch.maximum(s_max, s_round)

                delta_sum += float(delta.detach().sum().item())
                delta_abs_sum += float(delta.detach().abs().sum().item())
                delta_count += int(delta.numel())

                near_boundary = (s_round <= int(boundary_window)) | (
                    s_round >= int(read_port.length - 1 - boundary_window)
                )
                boundary_steps += int(near_boundary.sum().item())
                boundary_total += int(near_boundary.numel())

                hit = (s_round >= int(clue_start)) & (s_round < int(clue_end))
                dwell = dwell + hit.to(dtype=dwell.dtype)
                first_hit = torch.where((first_hit < 0) & hit, torch.full_like(first_hit, int(t)), first_hit)
                hit_any = hit_any | hit

                # Book-keeping for penalties
                if recent_s is not None:
                    recent_s[recent_ptr] = s_round
                    recent_ptr = (recent_ptr + 1) % int(revisit_window)
                    recent_len = min(int(revisit_window), recent_len + 1)
                if cover_s is not None:
                    cover_s[cover_ptr] = s_new.detach().to(dtype=torch.float32)
                    cover_ptr = (cover_ptr + 1) % int(cover_window)
                    cover_len = min(int(cover_window), cover_len + 1)
                if prev_tok is not None and token_ids_port is not None:
                    prev_tok = token_ids_port[s_round]

                # Observation and training step
                y = read_port.read(s_new)  # [B,D]
                y_in = y
                if float(noise_std) > 0.0:
                    y_in = y + float(noise_std) * torch.randn_like(y)

                out = entity.forward_tensor(
                    state_flat=entity.state.flat,
                    u_dict={"default": y_in},
                    dt=float(dt),
                    prev_chart_weights=getattr(entity, "_prev_chart_weights", None),
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                    skip_free_energy=True,
                )
                entity.state = ContactState(entity.dim_q, batch_size, device, out["next_state_flat"])
                entity._prev_chart_weights = out["next_prev_chart_weights"]

                y_hat = decoder(entity.state.q)
                pred_err_ps = (y_hat - y).pow(2).mean(dim=1)

                loss, parts = compute_loss(
                    entity=entity,
                    state=entity.state,
                    pred_err_per_sample=pred_err_ps,
                    action_delta=delta,
                    beta_kl=beta_kl,
                    gamma_pred=gamma_pred,
                    cost_weight=cost_weight,
                    cost_power=float(cost_power),
                    cost_offset=float(cost_offset),
                    anti_lock_weight=float(anti_lock_weight),
                    anti_lock_sigma=float(anti_lock_sigma),
                    revisit_penalty_per_sample=None,
                    pref_penalty_per_sample=None,
                    cover_penalty_per_sample=None,
                    epi_penalty_per_sample=None,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(entity.parameters()) + list(decoder.parameters()), max_norm=1.0)
                optimizer.step()

                entity.state = ContactState(entity.dim_q, batch_size, device, entity.state.flat.detach())
                s = s_new.detach()

                if fixate_left is not None:
                    fixate_left = torch.clamp(fixate_left - 1, min=0)
                    pe = pred_err_ps.detach()
                    med = pe.median()
                    mad = (pe - med).abs().median()
                    mad_mult = float(fixate_mad_mult_f)
                    if math.isfinite(mad_mult) and mad_mult > 0.0 and float(mad.item()) > 1e-12:
                        thr = med + mad_mult * mad
                        trigger = pe > thr
                        if trigger.any():
                            fixate_trigger_total += int(trigger.sum().item())
                            fixate_left = torch.where(
                                trigger,
                                torch.full_like(fixate_left, fixate_steps_i),
                                fixate_left,
                            )

                if log_f is not None and (int(log_every) > 0) and (t % int(log_every) == 0 or t == int(online_steps) - 1):
                    rec = {
                        "tag": str(tag),
                        "phase": "e3_online",
                        "episode": int(ep),
                        "label": int(label),
                        "policy": str(policy_s),
                        "t": int(t),
                        "clue_start": int(clue_start),
                        "clue_end": int(clue_end),
                        "base_marker_count": int(base_marker_count),
                        "hit_rate_so_far": float(hit_any.float().mean().item()),
                        "delta_mean": float(delta.detach().mean().item()),
                        "delta_abs_mean": float(delta.detach().abs().mean().item()),
                        "near_boundary_rate": float(near_boundary.float().mean().item()),
                        "pred_err": float(pred_err_ps.detach().mean().item()),
                        "F": float(loss.detach().item()),
                        "fixate_rate": (
                            float(fixate_mask.float().mean().item()) if fixate_mask is not None else float("nan")
                        ),
                        **_parts_to_float(parts),
                    }
                    log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # E2-like cycle metrics at end of episode (online only)
            W = min(int(online_steps), int(tail_window))
            if W > 0:
                start = int(int(online_steps) - W)
                idx = [(start + k) % int(tail_window) for k in range(W)]
                tail = s_tail[idx].detach().cpu().tolist()
                for b in range(batch_size):
                    tokens = [int(tail[k][b]) for k in range(W)]
                    cycle_total += 1
                    cyc = _detect_tail_cycle(tokens, max_period=max_period, min_repeats=min_repeats)
                    if cyc is not None:
                        cycle_hits += 1
                        _, pattern = cyc
                        if pattern and all((x == boundary_low) or (x == boundary_high) for x in pattern):
                            boundary_cycle_hits += 1

            # Offline: freeze env, optionally inject replay.
            entity.enter_offline()
            offline_q: List[torch.Tensor] = []
            with torch.no_grad():
                for t in range(int(offline_steps)):
                    u_off: Dict[str, torch.Tensor] = {}
                    exp = getattr(entity, "experience", None)
                    exp_size = int(getattr(exp, "size", 0) or 0) if exp is not None else 0
                    if replay_mode != "none" and exp is not None and exp_size > 0:
                        replay = exp.sample_replay(batch_size, mode=replay_mode)
                        if replay is not None:
                            replay_state = replay["states"].to(device)
                            if replay_state.shape[0] < batch_size:
                                pad = replay_state[0:1].expand(batch_size - replay_state.shape[0], -1)
                                replay_state = torch.cat([replay_state, pad], dim=0)
                            replay_q = replay_state[:, : entity.dim_q]
                            u_off["replay"] = 0.1 * (replay_q - entity.state.q)

                    _entity_forward_step_fast(entity=entity, u_dict=u_off, dt=float(dt))
                    offline_q.append(entity.state.q.detach())

            q_mean = torch.stack(offline_q, dim=0).mean(dim=0).detach().cpu() if offline_q else entity.state.q.detach().cpu()
            feats_all.append(q_mean)
            labs_all.append(torch.full((batch_size,), int(label), dtype=torch.long))
            hit_any_all.append(hit_any.detach().cpu())
            first_hit_all.append(first_hit.detach().cpu())
            dwell_all.append(dwell.detach().cpu())
            span_all.append((s_max - s_min).detach().cpu().to(dtype=torch.int32))

    feats = torch.cat(feats_all, dim=0) if feats_all else torch.zeros((0, entity.dim_q))
    labs = torch.cat(labs_all, dim=0) if labs_all else torch.zeros((0,), dtype=torch.long)
    hit_any_cat = torch.cat(hit_any_all, dim=0) if hit_any_all else torch.zeros((0,), dtype=torch.bool)
    first_hit_cat = torch.cat(first_hit_all, dim=0) if first_hit_all else torch.zeros((0,), dtype=torch.int32)
    dwell_cat = torch.cat(dwell_all, dim=0) if dwell_all else torch.zeros((0,), dtype=torch.int32)
    span_cat = torch.cat(span_all, dim=0) if span_all else torch.zeros((0,), dtype=torch.int32)

    # Probe separability
    probe = train_linear_probe(feats, labs, seed=int(seed))
    labs_shuf = labs[torch.randperm(int(labs.numel()))] if int(labs.numel()) > 0 else labs
    probe_shuf = train_linear_probe(feats, labs_shuf, seed=int(seed) + 1)

    # Hit stats
    total = int(hit_any_cat.numel())
    hit_rate = float(hit_any_cat.float().mean().item()) if total > 0 else float("nan")
    hit_mask = hit_any_cat.bool()
    if int(hit_mask.sum().item()) > 0:
        first_hits = first_hit_cat[hit_mask].to(dtype=torch.float32)
        first_hit_mean = float(first_hits.mean().item())
        first_hit_med = float(first_hits.median().item())
        dwell_hits = dwell_cat[hit_mask].to(dtype=torch.float32)
        dwell_hit_mean = float(dwell_hits.mean().item())
        dwell_hit_med = float(dwell_hits.median().item())
        span_hits = span_cat[hit_mask].to(dtype=torch.float32)
        span_hit_mean = float(span_hits.mean().item())
        span_hit_med = float(span_hits.median().item())
    else:
        first_hit_mean = float("nan")
        first_hit_med = float("nan")
        dwell_hit_mean = float("nan")
        dwell_hit_med = float("nan")
        span_hit_mean = float("nan")
        span_hit_med = float("nan")

    # Probe separability conditional on exposure (hit/no-hit), to separate "can write when hit"
    # from "overall diluted by low hit_rate".
    n_hit = int(hit_mask.sum().item()) if total > 0 else 0
    n_nohit = int(total - n_hit)
    probe_hit = {"acc": float("nan"), "auc": float("nan")}
    probe_hit_shuf = {"acc": float("nan"), "auc": float("nan")}
    probe_nohit = {"acc": float("nan"), "auc": float("nan")}
    probe_nohit_shuf = {"acc": float("nan"), "auc": float("nan")}
    if n_hit >= 10:
        feats_hit = feats[hit_mask]
        labs_hit = labs[hit_mask]
        probe_hit = train_linear_probe(feats_hit, labs_hit, seed=int(seed) + 100)
        labs_hit_shuf = labs_hit[torch.randperm(int(labs_hit.numel()))]
        probe_hit_shuf = train_linear_probe(feats_hit, labs_hit_shuf, seed=int(seed) + 101)
    if n_nohit >= 10:
        feats_nohit = feats[~hit_mask]
        labs_nohit = labs[~hit_mask]
        probe_nohit = train_linear_probe(feats_nohit, labs_nohit, seed=int(seed) + 200)
        labs_nohit_shuf = labs_nohit[torch.randperm(int(labs_nohit.numel()))]
        probe_nohit_shuf = train_linear_probe(feats_nohit, labs_nohit_shuf, seed=int(seed) + 201)

    non_boundary_hits = int(cycle_hits - boundary_cycle_hits)
    denom = max(1, int(cycle_total))
    e2 = {
        "ptr_cycle_rate": float(cycle_hits / denom),
        "ptr_cycle_hits": int(cycle_hits),
        "ptr_cycle_total": int(cycle_total),
        "ptr_cycle_rate_non_boundary": float(non_boundary_hits / denom),
        "ptr_cycle_hits_non_boundary": int(non_boundary_hits),
        "ptr_cycle_rate_boundary": float(boundary_cycle_hits / denom),
        "ptr_cycle_hits_boundary": int(boundary_cycle_hits),
        "ptr_cycle_boundary_low": int(boundary_low),
        "ptr_cycle_boundary_high": int(boundary_high),
    }

    return {
        "task": "marker_patch",
        "policy": str(policy_s),
        "markers": {"A": {"id": int(marker_a), "token": tok_a}, "B": {"id": int(marker_b), "token": tok_b}},
        "seq_len": int(seq_len),
        "clue_len": int(clue_len),
        "clue_margin": int(clue_margin),
        "boundary_window": int(boundary_window),
        "fixate_steps": int(fixate_steps_i),
        "fixate_mad_mult": float(fixate_mad_mult_f),
        "fixate_trigger_rate": float(fixate_trigger_total / max(1, int(batch_size) * int(online_steps) * 2 * int(pairs))),
        "fixate_step_rate": float(fixate_step_total / max(1, int(batch_size) * int(online_steps) * 2 * int(pairs))),
        "base_marker_violations": int(base_marker_violations),
        "base_marker_count_total": int(base_marker_count_total),
        "clue_edge_min": int(clue_edge_min if clue_edge_min < int(1e8) else -1),
        "clue_start_min": int(min(clue_start_list) if clue_start_list else -1),
        "clue_start_max": int(max(clue_start_list) if clue_start_list else -1),
        "hit_rate": float(hit_rate),
        "first_hit_mean": float(first_hit_mean),
        "first_hit_median": float(first_hit_med),
        "dwell_mean": float(dwell_cat.to(dtype=torch.float32).mean().item()) if total > 0 else float("nan"),
        "dwell_hit_mean": float(dwell_hit_mean),
        "dwell_hit_median": float(dwell_hit_med),
        "span_mean": float(span_cat.to(dtype=torch.float32).mean().item()) if total > 0 else float("nan"),
        "span_hit_mean": float(span_hit_mean),
        "span_hit_median": float(span_hit_med),
        "delta_mean": float(delta_sum / max(1, int(delta_count))),
        "delta_abs_mean": float(delta_abs_sum / max(1, int(delta_count))),
        "near_boundary_rate": float(boundary_steps / max(1, int(boundary_total))),
        "probe_acc": float(probe["acc"]),
        "probe_auc": float(probe["auc"]),
        "probe_acc_shuffled": float(probe_shuf["acc"]),
        "probe_auc_shuffled": float(probe_shuf["auc"]),
        "probe_n_hit": int(n_hit),
        "probe_acc_hit": float(probe_hit["acc"]),
        "probe_auc_hit": float(probe_hit["auc"]),
        "probe_acc_hit_shuffled": float(probe_hit_shuf["acc"]),
        "probe_auc_hit_shuffled": float(probe_hit_shuf["auc"]),
        "probe_n_nohit": int(n_nohit),
        "probe_acc_nohit": float(probe_nohit["acc"]),
        "probe_auc_nohit": float(probe_nohit["auc"]),
        "probe_acc_nohit_shuffled": float(probe_nohit_shuf["acc"]),
        "probe_auc_nohit_shuffled": float(probe_nohit_shuf["auc"]),
        "n_samples": int(total),
        "feature": "mean_offline_q",
        "ptr_cycles": e2,
    }


def run_e3_kv_swap(
    *,
    entity,
    decoder: nn.Module,
    embedding: nn.Embedding,
    vocab: CharVocab,
    optimizer: torch.optim.Optimizer,
    device: str,
    batch_size: int,
    pairs: int,
    seq_len: int,
    key_len: int,
    value_len: int,
    clue_margin: int,
    policy: str,
    fixate_steps: int,
    fixate_mad_mult: float,
    fixate_mode: str,
    fixate_scan_radius: int,
    boundary_window: int,
    read_sigma: float,
    online_steps: int,
    offline_steps: int,
    dt: float,
    step_size: float,
    candidates: Sequence[float],
    beta_kl: float,
    gamma_pred: float,
    cost_weight: float,
    cost_power: float,
    cost_offset: float,
    noise_std: float,
    pointer_drift: float,
    pointer_bounds: str,
    anti_lock_weight: float,
    anti_lock_sigma: float,
    revisit_weight: float,
    revisit_window: int,
    lookahead_steps: int,
    lookahead_discount: float,
    pref_repeat_weight: float,
    cover_weight: float,
    cover_sigma: float,
    cover_window: int,
    cover_warmup_frac: float,
    epi_weight: float,
    epi_normalize: int,
    epi_mode: str,
    epi_obs_eps: float,
    epi_pred_floor: float,
    replay_mode: str,
    query_steps: int,
    query_key: str,
    val_align: int,
    speak: int,
    speak_vocab: int,
    speak_loss_weight: float,
    speak_use: int,
    speak_back_threshold: float,
    seed: int,
    log_f,
    tag: str,
    log_every: int,
) -> Dict[str, object]:
    """
    E3+ (key→value binding; minimal structure):
      - Two key-value pairs are embedded at random positions:
          K1 -> V?, K2 -> V?
      - Label is whether the mapping is (K1->A,K2->B) vs (K1->B,K2->A).
      - Base sequence and positions are shared between the two labels (only values are swapped).
      - Verifier is automatic: we know the key/value spans.

    NOTE: Without an explicit "query/readout" phase, this task is intentionally close to a negative control:
    both labels contain the same multiset of tokens (A and B), so the latent may not linearly separate.

    When `query_steps>0`, we add a query-driven readout phase: we inject the query key embedding as input,
    and train the decoder to reconstruct the corresponding value embedding. This makes key→value binding
    enter the prediction-error term, creating a learnable signal.

    hit_rate refers to `resolved_any_rate`: observed at least one complete (key,value) pair (any key).
    resolved_rate refers to a stricter, query-relevant notion when `query_steps>0` (queried pair complete).
    """
    entity.train()
    decoder.train()
    epi_mode_s = str(epi_mode or "chart_entropy").strip().lower()
    policy_s = str(policy or "active").strip().lower()
    if policy_s not in ("active", "active_fixate", "passive", "random", "frozen"):
        raise ValueError(f"unknown e3 policy: {policy}")
    fixate_steps_i = int(max(0, int(fixate_steps))) if policy_s == "active_fixate" else 0
    fixate_mad_mult_f = float(fixate_mad_mult) if policy_s == "active_fixate" else 0.0
    if not math.isfinite(fixate_mad_mult_f) or fixate_mad_mult_f <= 0.0:
        fixate_mad_mult_f = 0.0
    fixate_mode_s = str(fixate_mode or "freeze").strip().lower()
    if policy_s != "active_fixate" or fixate_steps_i <= 0:
        fixate_mode_s = "freeze"
    if fixate_mode_s in ("", "0", "off", "false", "freeze"):
        fixate_mode_s = "freeze"
    if fixate_mode_s not in ("freeze", "backward_uniform", "both_uniform"):
        raise ValueError(f"unknown fixate_mode: {fixate_mode}")
    fixate_scan_radius_i = int(max(0, int(fixate_scan_radius)))

    query_steps_i = int(max(0, int(query_steps)))
    query_key_s = str(query_key or "none").strip().lower()
    if query_steps_i <= 0:
        query_key_s = "none"
    if query_key_s in ("", "0", "off", "false", "none", "no"):
        query_key_s = "none"
    if query_key_s not in ("none", "k1", "k2", "random"):
        raise ValueError(f"unknown query_key: {query_key}")
    if query_key_s == "none":
        query_steps_i = 0

    val_align_i = int(val_align) if isinstance(val_align, (int, float, str)) else 0
    val_align_i = 1 if val_align_i != 0 else 0

    speak_i = int(speak) if isinstance(speak, (int, float, str)) else 0
    speak_i = 1 if speak_i != 0 else 0
    speak_use_i = int(speak_use) if isinstance(speak_use, (int, float, str)) else 0
    speak_use_i = 1 if speak_use_i != 0 else 0
    speak_vocab_i = int(max(0, int(speak_vocab)))
    speak_loss_w = float(speak_loss_weight)
    if not math.isfinite(speak_loss_w) or speak_loss_w < 0.0:
        speak_loss_w = 0.0
    speak_back_thr = float(speak_back_threshold)
    if not math.isfinite(speak_back_thr):
        speak_back_thr = 0.9
    speak_back_thr = float(max(0.0, min(1.0, speak_back_thr)))
    if speak_vocab_i < 2:
        speak_i = 0
        speak_use_i = 0
        speak_loss_w = 0.0
    if speak_i == 0:
        speak_use_i = 0
        speak_loss_w = 0.0

    speak_head: Optional[nn.Module] = None
    speak_emb: Optional[nn.Module] = None
    if speak_i != 0:
        speak_head = getattr(entity, "e3_speak_head", None)
        speak_emb = getattr(entity, "e3_speak_emb", None)
        if not isinstance(speak_head, nn.Module) or not isinstance(speak_emb, nn.Module):
            raise RuntimeError(
                "E3 speak enabled but missing modules on entity. "
                "Expected `entity.e3_speak_head` and `entity.e3_speak_emb`."
            )

    val_a, val_b, tok_a, tok_b = _pick_e3_markers(vocab)
    key1, key2, tok_k1, tok_k2 = _pick_e3_kv_keys(vocab, exclude_ids=[val_a, val_b])
    allowed_ids = _make_allowed_token_ids(vocab=vocab, marker_ids=[val_a, val_b, key1, key2], device=device)

    CAND = torch.tensor(list(candidates), device=device, dtype=torch.float32) * float(step_size)
    C = int(CAND.numel())
    if C <= 0:
        raise ValueError("empty candidates")

    pairs = int(max(1, int(pairs)))
    seq_len = int(max(64, int(seq_len)))
    key_len = int(max(1, int(key_len)))
    value_len = int(max(1, int(value_len)))
    clue_margin = int(max(0, int(clue_margin)))
    online_steps = int(max(1, int(online_steps)))
    offline_steps = int(max(0, int(offline_steps)))
    pair_span = int(key_len + value_len)
    if clue_margin * 2 + pair_span * 2 >= seq_len:
        clue_margin = max(0, (seq_len - pair_span * 2) // 4)

    read_sigma = float(read_sigma)
    if not math.isfinite(read_sigma) or read_sigma <= 0.0:
        read_sigma = 2.0
    boundary_window = int(boundary_window)
    if boundary_window <= 0:
        boundary_window = int(max(1, round(3.0 * float(read_sigma))))
    boundary_window = int(min(boundary_window, max(1, (seq_len - 1) // 2)))

    # E2-like pointer cycle diagnostics (quantize to int)
    cycle_hits = 0
    boundary_cycle_hits = 0
    cycle_total = 0
    max_period = 8
    min_repeats = 3
    tail_window = 32
    boundary_low = 0
    boundary_high = int(seq_len - 1)

    feats_all: List[torch.Tensor] = []
    labs_all: List[torch.Tensor] = []
    resolved_all: List[torch.Tensor] = []
    resolved_any_all: List[torch.Tensor] = []
    val_hit_all: List[torch.Tensor] = []
    first_hit_all: List[torch.Tensor] = []
    dwell_all: List[torch.Tensor] = []
    span_all: List[torch.Tensor] = []

    base_marker_violations = 0
    base_marker_count_total = 0
    key1_pos_list: List[int] = []
    key2_pos_list: List[int] = []
    edge_min = int(1e9)

    delta_sum = 0.0
    delta_abs_sum = 0.0
    delta_count = 0
    boundary_steps = 0
    boundary_total = 0
    fixate_trigger_total = 0
    fixate_step_total = 0
    val_align_trigger_total = 0
    val_align_step_total = 0
    query_k1_total = 0
    query_total = 0
    query_hit_total = 0
    query_correct_total = 0
    query_correct_hit_total = 0
    speak_total = 0
    speak_correct_total = 0
    speak_back_total = 0
    speak_ce_sum = 0.0
    speak_ce_count = 0

    # For fairness, each pair shares the same base sequence & positions; only the value assignment differs.
    for ep in range(pairs):
        base = allowed_ids[torch.randint(0, int(allowed_ids.numel()), (seq_len,), device=device)]
        base_bad = int(((base == int(val_a)) | (base == int(val_b)) | (base == int(key1)) | (base == int(key2))).sum().item())
        base_marker_count_total += base_bad
        if base_bad != 0:
            base_marker_violations += 1

        pos_lo = int(clue_margin)
        pos_hi = int(seq_len - clue_margin - pair_span)
        if pos_hi <= pos_lo:
            pos_lo = 0
            pos_hi = int(max(0, seq_len - pair_span))

        k1_pos = int(torch.randint(pos_lo, pos_hi + 1, (1,), device=device).item()) if pos_hi >= pos_lo else 0
        tries = 0
        k2_pos = k1_pos
        while tries < 64:
            cand_pos = int(torch.randint(pos_lo, pos_hi + 1, (1,), device=device).item()) if pos_hi >= pos_lo else 0
            a0, a1 = k1_pos, k1_pos + pair_span
            b0, b1 = cand_pos, cand_pos + pair_span
            if (b1 <= a0 - clue_margin) or (a1 <= b0 - clue_margin):
                k2_pos = cand_pos
                break
            tries += 1
        if k2_pos == k1_pos:
            k2_pos = int(max(pos_lo, min(pos_hi, k1_pos + pair_span + clue_margin)))

        key1_pos_list.append(int(k1_pos))
        key2_pos_list.append(int(k2_pos))
        v1_start = int(k1_pos + key_len)
        v1_end = int(min(seq_len, v1_start + value_len))
        v2_start = int(k2_pos + key_len)
        v2_end = int(min(seq_len, v2_start + value_len))
        edge_min = int(min(edge_min, k1_pos, k2_pos, seq_len - v1_end, seq_len - v2_end))

        label_order = [(0, int(val_a), int(val_b)), (1, int(val_b), int(val_a))]
        random.shuffle(label_order)
        for label, v_for_k1, v_for_k2 in label_order:
            token_ids = base.clone()
            token_ids[int(k1_pos) : int(k1_pos + key_len)] = int(key1)
            token_ids[int(k2_pos) : int(k2_pos + key_len)] = int(key2)
            token_ids[v1_start:v1_end] = int(v_for_k1)
            token_ids[v2_start:v2_end] = int(v_for_k2)

            read_port = GaussianTruncRenormReadPort(token_ids=token_ids, embedding=embedding, sigma=float(read_sigma)).to(
                device
            )

            entity.reset(batch_size=batch_size, device=device)
            reset_experience(entity)

            # Start pointer away from edges to reduce boundary artifacts.
            start_margin = int(max(1, round(3.0 * float(read_port.sigma))))
            s = torch.rand(batch_size, device=device) * float(max(1.0, read_port.length - 1 - 2 * start_margin)) + float(
                start_margin
            )

            token_ids_port = getattr(read_port, "token_ids", None)
            prev_tok: Optional[torch.Tensor] = None
            if float(pref_repeat_weight) > 0.0 and isinstance(token_ids_port, torch.Tensor) and token_ids_port.ndim == 1:
                idx0 = s.detach().round().to(dtype=torch.long).clamp(0, int(read_port.length - 1))
                prev_tok = token_ids_port[idx0]

            recent_s: Optional[torch.Tensor] = None
            recent_len = 0
            recent_ptr = 0
            if float(revisit_weight) > 0.0 and int(revisit_window) > 0:
                K = int(revisit_window)
                recent_s = torch.full((K, batch_size), -1, device=device, dtype=torch.long)
                recent_s[0] = s.detach().round().to(dtype=torch.long)
                recent_len = 1
                recent_ptr = 1 % K

            cover_s: Optional[torch.Tensor] = None
            cover_len = 0
            cover_ptr = 0
            if float(cover_weight) > 0.0 and int(cover_window) > 0:
                Kc = int(cover_window)
                cover_s = torch.empty((Kc, batch_size), device=device, dtype=torch.float32)
                cover_s[0] = s.detach().to(dtype=torch.float32)
                cover_len = 1
                cover_ptr = 1 % Kc

            warmup_frac = float(cover_warmup_frac)
            if not math.isfinite(warmup_frac) or warmup_frac <= 0.0:
                warmup_steps = 0
            else:
                warmup_steps = int(round(float(online_steps) * max(0.0, min(warmup_frac, 1.0))))

            k1_seen = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            k2_seen = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            v1_seen = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            v2_seen = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            val_any = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            first_hit = torch.full((batch_size,), -1, device=device, dtype=torch.int32)
            dwell = torch.zeros((batch_size,), device=device, dtype=torch.int32)
            s_min = s.detach().round().to(dtype=torch.long)
            s_max = s_min.clone()

            # Pointer tail for cycle diagnostics
            s_tail = torch.empty((tail_window, batch_size), device=device, dtype=torch.long)

            query_is_k1: Optional[torch.Tensor] = None
            query_ids: Optional[torch.Tensor] = None
            query_target_ids: Optional[torch.Tensor] = None
            if query_key_s != "none":
                if query_key_s == "k1":
                    query_is_k1 = torch.ones((batch_size,), device=device, dtype=torch.bool)
                elif query_key_s == "k2":
                    query_is_k1 = torch.zeros((batch_size,), device=device, dtype=torch.bool)
                else:
                    query_is_k1 = torch.rand((batch_size,), device=device) < 0.5
                query_k1_total += int(query_is_k1.sum().item())
                query_total += int(query_is_k1.numel())
                query_ids = torch.where(
                    query_is_k1,
                    torch.full((batch_size,), int(key1), device=device, dtype=torch.long),
                    torch.full((batch_size,), int(key2), device=device, dtype=torch.long),
                )
                query_target_ids = torch.where(
                    query_is_k1,
                    torch.full((batch_size,), int(v_for_k1), device=device, dtype=torch.long),
                    torch.full((batch_size,), int(v_for_k2), device=device, dtype=torch.long),
                )

            fixate_left: Optional[torch.Tensor] = None
            if fixate_steps_i > 0:
                fixate_left = torch.zeros((batch_size,), device=device, dtype=torch.int32)

            val_align_prev_in_val: Optional[torch.Tensor] = None
            if val_align_i != 0:
                val_align_prev_in_val = torch.zeros((batch_size,), device=device, dtype=torch.bool)

            entity.enter_online()
            for t in range(int(online_steps)):
                fixate_mask: Optional[torch.Tensor] = None
                if fixate_left is not None:
                    fixate_mask = fixate_left > 0
                    fixate_step_total += int(fixate_mask.sum().item())

                in_val_mask: Optional[torch.Tensor] = None
                if (
                    isinstance(token_ids_port, torch.Tensor)
                    and token_ids_port.ndim == 1
                    and (val_align_prev_in_val is not None or speak_i != 0)
                ):
                    s_round_cur = s.detach().round().to(dtype=torch.long).clamp(0, int(read_port.length - 1))
                    tok_cur = token_ids_port[s_round_cur]
                    in_val_mask = (tok_cur == int(val_a)) | (tok_cur == int(val_b))

                val_align_mask: Optional[torch.Tensor] = None
                if val_align_prev_in_val is not None and in_val_mask is not None:
                    val_align_mask = in_val_mask
                    val_align_step_total += int(val_align_mask.sum().item())
                    start = val_align_mask & (~val_align_prev_in_val)
                    if start.any():
                        val_align_trigger_total += int(start.sum().item())
                    val_align_prev_in_val = val_align_mask

                speak_probs: Optional[torch.Tensor] = None
                speak_ce: Optional[torch.Tensor] = None
                speak_back_mask: Optional[torch.Tensor] = None
                if speak_i != 0 and speak_head is not None and speak_emb is not None and in_val_mask is not None:
                    speak_logits = speak_head(entity.state.q.detach())  # [B,V]
                    speak_probs = torch.softmax(speak_logits, dim=1)
                    if speak_loss_w > 0.0:
                        speak_ce = torch.nn.functional.cross_entropy(speak_logits, in_val_mask.to(dtype=torch.long))
                    if speak_use_i != 0:
                        back_prob = speak_probs[:, 1]  # class 1 == BACK
                        speak_back_mask = back_prob >= float(speak_back_thr)
                    speak_pred = speak_probs.argmax(dim=1)
                    speak_total += int(speak_pred.numel())
                    speak_correct_total += int((speak_pred == in_val_mask.to(dtype=torch.long)).sum().item())
                    if speak_ce is not None:
                        speak_ce_sum += float(speak_ce.detach().item())
                        speak_ce_count += 1

                # Coverage warmup scaling (optional)
                if warmup_steps > 0:
                    cover_scale = min(1.0, float(t + 1) / float(max(1, warmup_steps)))
                else:
                    cover_scale = 1.0
                eff_cover_weight = float(cover_weight) * float(cover_scale)

                if policy_s in ("active", "active_fixate"):
                    rng_devices: List[int] = []
                    if str(device).startswith("cuda") and torch.cuda.is_available():
                        rng_devices = [int(torch.cuda.current_device())]
                    with torch.random.fork_rng(devices=rng_devices, enabled=(float(noise_std) > 0.0)):
                        delta = choose_delta_one_step(
                            entity=entity,
                            decoder=decoder,
                            read_port=read_port,
                            s=s,
                            candidates=CAND,
                            dt=dt,
                            beta_kl=beta_kl,
                            gamma_pred=gamma_pred,
                            cost_weight=cost_weight,
                            cost_power=float(cost_power),
                            cost_offset=float(cost_offset),
                            noise_std=noise_std,
                            pointer_drift=float(pointer_drift),
                            pointer_bounds=str(pointer_bounds),
                            anti_lock_weight=float(anti_lock_weight),
                            anti_lock_sigma=float(anti_lock_sigma),
                            revisit_weight=float(revisit_weight),
                            recent_s=(recent_s[:recent_len] if recent_s is not None and recent_len > 0 else None),
                            lookahead_steps=int(lookahead_steps),
                            lookahead_discount=float(lookahead_discount),
                            pref_repeat_weight=float(pref_repeat_weight),
                            token_ids=token_ids_port,
                            prev_token_ids=prev_tok,
                            cover_weight=float(eff_cover_weight),
                            cover_sigma=float(cover_sigma),
                            cover_recent_s=(cover_s[:cover_len] if cover_s is not None and cover_len > 0 else None),
                            epi_weight=float(epi_weight),
                            epi_normalize=int(epi_normalize),
                            epi_mode=str(epi_mode_s),
                            epi_obs_eps=float(epi_obs_eps),
                            epi_pred_floor=float(epi_pred_floor),
                        )
                elif policy_s == "random":
                    delta = CAND[torch.randint(0, C, (batch_size,), device=device)]
                else:
                    delta = torch.zeros((batch_size,), device=device, dtype=s.dtype)

                if fixate_mask is not None:
                    if fixate_mode_s == "freeze":
                        delta = delta * (~fixate_mask).to(dtype=delta.dtype)
                    else:
                        # Local micro-scan during fixation (E3+): attempt to resolve key/value alignment.
                        # By default, use a backwards scan window tied to value_len.
                        radius = int(fixate_scan_radius_i) if int(fixate_scan_radius_i) > 0 else int(value_len)
                        radius = int(max(0, radius))
                        if radius <= 0:
                            delta_fix = torch.zeros_like(delta)
                        elif fixate_mode_s == "backward_uniform":
                            back = torch.randint(0, radius + 1, (batch_size,), device=device, dtype=torch.long)
                            delta_fix = (-back).to(dtype=delta.dtype) * float(step_size)
                        else:
                            span = int(2 * radius + 1)
                            offs = torch.randint(0, span, (batch_size,), device=device, dtype=torch.long) - int(radius)
                            delta_fix = offs.to(dtype=delta.dtype) * float(step_size)
                        delta = torch.where(fixate_mask, delta_fix, delta)

                drift_eff = 0.0 if policy_s == "frozen" else float(pointer_drift)
                if fixate_mask is not None:
                    drift = float(drift_eff) * (~fixate_mask).to(dtype=s.dtype)
                else:
                    drift = s.new_full((batch_size,), float(drift_eff))

                back_mask: Optional[torch.Tensor] = None
                if val_align_mask is not None:
                    # v0p11: de-privileged alignment.
                    # If currently inside a value span (A/B), step backward by one token until exiting the span.
                    # Since base sequences exclude A/B, exiting the span lands on the (unknown) key token.
                    # Apply after fixation so it can override freeze; cancel drift so stepping is exact.
                    back_mask = val_align_mask

                if speak_back_mask is not None:
                    speak_back_total += int(speak_back_mask.sum().item())
                    back_mask = speak_back_mask if back_mask is None else (back_mask | speak_back_mask)

                if back_mask is not None:
                    drift = drift * (~back_mask).to(dtype=s.dtype)
                    delta = torch.where(back_mask, delta.new_full((batch_size,), -float(step_size)), delta)

                s_new = _apply_pointer_bounds(
                    s + drift + delta,
                    max_s=float(read_port.length - 1),
                    mode=str(pointer_bounds),
                )
                s_round = s_new.detach().round().to(dtype=torch.long).clamp(0, int(read_port.length - 1))
                s_tail[int(t) % int(tail_window)] = s_round
                s_min = torch.minimum(s_min, s_round)
                s_max = torch.maximum(s_max, s_round)

                delta_sum += float(delta.detach().sum().item())
                delta_abs_sum += float(delta.detach().abs().sum().item())
                delta_count += int(delta.numel())

                near_boundary = (s_round <= int(boundary_window)) | (
                    s_round >= int(read_port.length - 1 - boundary_window)
                )
                boundary_steps += int(near_boundary.sum().item())
                boundary_total += int(near_boundary.numel())

                k1_hit = (s_round >= int(k1_pos)) & (s_round < int(k1_pos + key_len))
                k2_hit = (s_round >= int(k2_pos)) & (s_round < int(k2_pos + key_len))
                k1_seen = k1_seen | k1_hit
                k2_seen = k2_seen | k2_hit
                v1_hit = (s_round >= int(v1_start)) & (s_round < int(v1_end))
                v2_hit = (s_round >= int(v2_start)) & (s_round < int(v2_end))
                v1_seen = v1_seen | v1_hit
                v2_seen = v2_seen | v2_hit
                hit_val = v1_hit | v2_hit
                val_any = val_any | hit_val
                dwell = dwell + hit_val.to(dtype=dwell.dtype)
                first_hit = torch.where((first_hit < 0) & hit_val, torch.full_like(first_hit, int(t)), first_hit)

                # Book-keeping for penalties
                if recent_s is not None:
                    recent_s[recent_ptr] = s_round
                    recent_ptr = (recent_ptr + 1) % int(revisit_window)
                    recent_len = min(int(revisit_window), recent_len + 1)
                if cover_s is not None:
                    cover_s[cover_ptr] = s_new.detach().to(dtype=torch.float32)
                    cover_ptr = (cover_ptr + 1) % int(cover_window)
                    cover_len = min(int(cover_window), cover_len + 1)
                if prev_tok is not None and token_ids_port is not None:
                    prev_tok = token_ids_port[s_round]

                # Observation and training step
                y = read_port.read(s_new)  # [B,D]
                y_in = y
                if float(noise_std) > 0.0:
                    y_in = y + float(noise_std) * torch.randn_like(y)

                u_dict = {"default": y_in}
                if speak_probs is not None and speak_emb is not None:
                    u_dict["speak"] = speak_probs @ speak_emb.weight
                out = entity.forward_tensor(
                    state_flat=entity.state.flat,
                    u_dict=u_dict,
                    dt=float(dt),
                    prev_chart_weights=getattr(entity, "_prev_chart_weights", None),
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                    skip_free_energy=True,
                )
                entity.state = ContactState(entity.dim_q, batch_size, device, out["next_state_flat"])
                entity._prev_chart_weights = out["next_prev_chart_weights"]

                y_hat = decoder(entity.state.q)
                pred_err_ps = (y_hat - y).pow(2).mean(dim=1)

                loss, parts = compute_loss(
                    entity=entity,
                    state=entity.state,
                    pred_err_per_sample=pred_err_ps,
                    action_delta=delta,
                    beta_kl=beta_kl,
                    gamma_pred=gamma_pred,
                    cost_weight=cost_weight,
                    cost_power=float(cost_power),
                    cost_offset=float(cost_offset),
                    anti_lock_weight=float(anti_lock_weight),
                    anti_lock_sigma=float(anti_lock_sigma),
                    revisit_penalty_per_sample=None,
                    pref_penalty_per_sample=None,
                    cover_penalty_per_sample=None,
                    epi_penalty_per_sample=None,
                )
                loss_total = loss
                if speak_ce is not None and speak_loss_w > 0.0:
                    loss_total = loss_total + float(speak_loss_w) * speak_ce
                optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(list(entity.parameters()) + list(decoder.parameters()), max_norm=1.0)
                optimizer.step()

                entity.state = ContactState(entity.dim_q, batch_size, device, entity.state.flat.detach())
                s = s_new.detach()

                if fixate_left is not None:
                    fixate_left = torch.clamp(fixate_left - 1, min=0)
                    pe = pred_err_ps.detach()
                    med = pe.median()
                    mad = (pe - med).abs().median()
                    mad_mult = float(fixate_mad_mult_f)
                    if math.isfinite(mad_mult) and mad_mult > 0.0 and float(mad.item()) > 1e-12:
                        thr = med + mad_mult * mad
                        trigger = pe > thr
                        if trigger.any():
                            fixate_trigger_total += int(trigger.sum().item())
                            fixate_left = torch.where(
                                trigger,
                                torch.full_like(fixate_left, fixate_steps_i),
                                fixate_left,
                            )

                if log_f is not None and (int(log_every) > 0) and (t % int(log_every) == 0 or t == int(online_steps) - 1):
                    resolved_so_far = ((k1_seen & v1_seen) | (k2_seen & v2_seen)).float().mean().item()
                    speak_acc = float("nan")
                    if speak_probs is not None and in_val_mask is not None:
                        speak_acc = float((speak_probs.argmax(dim=1) == in_val_mask.to(dtype=torch.long)).float().mean().item())
                    rec = {
                        "tag": str(tag),
                        "phase": "e3kv_online",
                        "episode": int(ep),
                        "label": int(label),
                        "policy": str(policy_s),
                        "t": int(t),
                        "k1_pos": int(k1_pos),
                        "k2_pos": int(k2_pos),
                        "key_len": int(key_len),
                        "v1_span": [int(v1_start), int(v1_end)],
                        "v2_span": [int(v2_start), int(v2_end)],
                        "val_hit_rate_so_far": float(val_any.float().mean().item()),
                        "resolved_rate_so_far": float(resolved_so_far),
                        "delta_mean": float(delta.detach().mean().item()),
                        "delta_abs_mean": float(delta.detach().abs().mean().item()),
                        "near_boundary_rate": float(near_boundary.float().mean().item()),
                        "pred_err": float(pred_err_ps.detach().mean().item()),
                        "F": float(loss.detach().item()),
                        "loss_total": float(loss_total.detach().item()),
                        "fixate_rate": (
                            float(fixate_mask.float().mean().item()) if fixate_mask is not None else float("nan")
                        ),
                        "val_align_rate": (
                            float(val_align_mask.float().mean().item()) if val_align_mask is not None else float("nan")
                        ),
                        "speak_ce": (float(speak_ce.detach().item()) if speak_ce is not None else float("nan")),
                        "speak_acc": float(speak_acc),
                        "speak_back_rate": (
                            float(speak_back_mask.float().mean().item()) if speak_back_mask is not None else float("nan")
                        ),
                        **_parts_to_float(parts),
                    }
                    log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            resolved_any = (k1_seen & v1_seen) | (k2_seen & v2_seen)
            if query_is_k1 is not None:
                resolved_query = torch.where(query_is_k1, k1_seen & v1_seen, k2_seen & v2_seen)
            else:
                resolved_query = resolved_any

            # E2-like cycle metrics at end of episode (online only)
            W = min(int(online_steps), int(tail_window))
            if W > 0:
                start = int(int(online_steps) - W)
                idx = [(start + k) % int(tail_window) for k in range(W)]
                tail = s_tail[idx].detach().cpu().tolist()
                for b in range(batch_size):
                    tokens = [int(tail[k][b]) for k in range(W)]
                    cycle_total += 1
                    cyc = _detect_tail_cycle(tokens, max_period=max_period, min_repeats=min_repeats)
                    if cyc is not None:
                        cycle_hits += 1
                        _, pattern = cyc
                        if pattern and all((x == boundary_low) or (x == boundary_high) for x in pattern):
                            boundary_cycle_hits += 1

            # Offline: freeze env, optionally inject replay.
            entity.enter_offline()
            offline_q: List[torch.Tensor] = []
            with torch.no_grad():
                for _ in range(int(offline_steps)):
                    u_off: Dict[str, torch.Tensor] = {}
                    exp = getattr(entity, "experience", None)
                    exp_size = int(getattr(exp, "size", 0) or 0) if exp is not None else 0
                    if replay_mode != "none" and exp is not None and exp_size > 0:
                        replay = exp.sample_replay(batch_size, mode=replay_mode)
                        if replay is not None:
                            replay_state = replay["states"].to(device)
                            if replay_state.shape[0] < batch_size:
                                pad = replay_state[0:1].expand(batch_size - replay_state.shape[0], -1)
                                replay_state = torch.cat([replay_state, pad], dim=0)
                            replay_q = replay_state[:, : entity.dim_q]
                            u_off["replay"] = 0.1 * (replay_q - entity.state.q)
                    _entity_forward_step_fast(entity=entity, u_dict=u_off, dt=float(dt))
                    offline_q.append(entity.state.q.detach())

            q_off_mean = (
                torch.stack(offline_q, dim=0).mean(dim=0).detach().cpu() if offline_q else entity.state.q.detach().cpu()
            )

            q_feat = q_off_mean
            if query_steps_i > 0 and query_ids is not None and query_target_ids is not None:
                # Query/readout phase: input=key embedding, target=value embedding.
                entity.enter_online()
                query_in = embedding(query_ids).detach()
                query_target = embedding(query_target_ids).detach()
                train_mask = resolved_query.detach()
                for _ in range(query_steps_i):
                    out = entity.forward_tensor(
                        state_flat=entity.state.flat,
                        u_dict={"default": query_in},
                        dt=float(dt),
                        prev_chart_weights=getattr(entity, "_prev_chart_weights", None),
                        prediction_error=None,
                        detach_next_prev_weights=True,
                        compute_action=False,
                        skip_free_energy=True,
                    )
                    entity.state = ContactState(entity.dim_q, batch_size, device, out["next_state_flat"])
                    entity._prev_chart_weights = out["next_prev_chart_weights"]

                    y_hat = decoder(entity.state.q)
                    pred_err_ps = (y_hat - query_target).pow(2).mean(dim=1)
                    if train_mask.any():
                        pred_err_hit = pred_err_ps[train_mask].mean()
                        optimizer.zero_grad(set_to_none=True)
                        pred_err_hit.backward()
                        torch.nn.utils.clip_grad_norm_(list(entity.parameters()) + list(decoder.parameters()), max_norm=1.0)
                        optimizer.step()
                    entity.state = ContactState(entity.dim_q, batch_size, device, entity.state.flat.detach())

                with torch.no_grad():
                    emb_a = embedding.weight[int(val_a)].detach().view(1, -1)
                    emb_b = embedding.weight[int(val_b)].detach().view(1, -1)
                    dist_a = (y_hat.detach() - emb_a).pow(2).mean(dim=1)
                    dist_b = (y_hat.detach() - emb_b).pow(2).mean(dim=1)
                    pred_is_b = dist_b < dist_a
                    tgt_is_b = query_target_ids == int(val_b)
                    correct = pred_is_b == tgt_is_b
                    query_correct_total += int(correct.sum().item())
                    if train_mask.any():
                        query_hit_total += int(train_mask.sum().item())
                        query_correct_hit_total += int(correct[train_mask].sum().item())

                # For a stable, cross-episode diagnostic, use the predicted observation (y_hat) as the feature
                # (embedding space is fixed; q space can drift while weights learn).
                q_feat = y_hat.detach().cpu()

            feats_all.append(q_feat)
            labs_all.append(torch.full((batch_size,), int(label), dtype=torch.long))
            resolved_all.append(resolved_query.detach().cpu() if query_steps_i > 0 else resolved_any.detach().cpu())
            resolved_any_all.append(resolved_any.detach().cpu())
            val_hit_all.append(val_any.detach().cpu())
            first_hit_all.append(first_hit.detach().cpu())
            dwell_all.append(dwell.detach().cpu())
            span_all.append((s_max - s_min).detach().cpu().to(dtype=torch.int32))

    feats = torch.cat(feats_all, dim=0) if feats_all else torch.zeros((0, entity.dim_q))
    labs = torch.cat(labs_all, dim=0) if labs_all else torch.zeros((0,), dtype=torch.long)
    resolved_cat = torch.cat(resolved_all, dim=0) if resolved_all else torch.zeros((0,), dtype=torch.bool)
    resolved_any_cat = torch.cat(resolved_any_all, dim=0) if resolved_any_all else torch.zeros((0,), dtype=torch.bool)
    val_hit_cat = torch.cat(val_hit_all, dim=0) if val_hit_all else torch.zeros((0,), dtype=torch.bool)
    first_hit_cat = torch.cat(first_hit_all, dim=0) if first_hit_all else torch.zeros((0,), dtype=torch.int32)
    dwell_cat = torch.cat(dwell_all, dim=0) if dwell_all else torch.zeros((0,), dtype=torch.int32)
    span_cat = torch.cat(span_all, dim=0) if span_all else torch.zeros((0,), dtype=torch.int32)

    probe = train_linear_probe(feats, labs, seed=int(seed))
    labs_shuf = labs[torch.randperm(int(labs.numel()))] if int(labs.numel()) > 0 else labs
    probe_shuf = train_linear_probe(feats, labs_shuf, seed=int(seed) + 1)

    total = int(resolved_cat.numel())
    resolved_rate = float(resolved_cat.float().mean().item()) if total > 0 else float("nan")
    resolved_any_rate = (
        float(resolved_any_cat.float().mean().item()) if int(resolved_any_cat.numel()) > 0 else float("nan")
    )
    val_hit_rate = float(val_hit_cat.float().mean().item()) if total > 0 else float("nan")
    hit_mask = resolved_cat.bool()

    if int(hit_mask.sum().item()) > 0:
        first_hits = first_hit_cat[hit_mask].to(dtype=torch.float32)
        first_hit_mean = float(first_hits.mean().item())
        first_hit_med = float(first_hits.median().item())
        dwell_hits = dwell_cat[hit_mask].to(dtype=torch.float32)
        dwell_hit_mean = float(dwell_hits.mean().item())
        dwell_hit_med = float(dwell_hits.median().item())
        span_hits = span_cat[hit_mask].to(dtype=torch.float32)
        span_hit_mean = float(span_hits.mean().item())
        span_hit_med = float(span_hits.median().item())
    else:
        first_hit_mean = float("nan")
        first_hit_med = float("nan")
        dwell_hit_mean = float("nan")
        dwell_hit_med = float("nan")
        span_hit_mean = float("nan")
        span_hit_med = float("nan")

    n_hit = int(hit_mask.sum().item()) if total > 0 else 0
    n_nohit = int(total - n_hit)
    probe_hit = {"acc": float("nan"), "auc": float("nan")}
    probe_hit_shuf = {"acc": float("nan"), "auc": float("nan")}
    probe_nohit = {"acc": float("nan"), "auc": float("nan")}
    probe_nohit_shuf = {"acc": float("nan"), "auc": float("nan")}
    if n_hit >= 10:
        feats_hit = feats[hit_mask]
        labs_hit = labs[hit_mask]
        probe_hit = train_linear_probe(feats_hit, labs_hit, seed=int(seed) + 100)
        labs_hit_shuf = labs_hit[torch.randperm(int(labs_hit.numel()))]
        probe_hit_shuf = train_linear_probe(feats_hit, labs_hit_shuf, seed=int(seed) + 101)
    if n_nohit >= 10:
        feats_nohit = feats[~hit_mask]
        labs_nohit = labs[~hit_mask]
        probe_nohit = train_linear_probe(feats_nohit, labs_nohit, seed=int(seed) + 200)
        labs_nohit_shuf = labs_nohit[torch.randperm(int(labs_nohit.numel()))]
        probe_nohit_shuf = train_linear_probe(feats_nohit, labs_nohit_shuf, seed=int(seed) + 201)

    non_boundary_hits = int(cycle_hits - boundary_cycle_hits)
    denom = max(1, int(cycle_total))
    e2 = {
        "ptr_cycle_rate": float(cycle_hits / denom),
        "ptr_cycle_hits": int(cycle_hits),
        "ptr_cycle_total": int(cycle_total),
        "ptr_cycle_rate_non_boundary": float(non_boundary_hits / denom),
        "ptr_cycle_hits_non_boundary": int(non_boundary_hits),
        "ptr_cycle_rate_boundary": float(boundary_cycle_hits / denom),
        "ptr_cycle_hits_boundary": int(boundary_cycle_hits),
        "ptr_cycle_boundary_low": int(boundary_low),
        "ptr_cycle_boundary_high": int(boundary_high),
    }

    return {
        "task": "kv_swap",
        "policy": str(policy_s),
        "query_key": str(query_key_s),
        "query_steps": int(query_steps_i),
        "query_k1_rate": (float(query_k1_total / max(1, int(query_total))) if query_steps_i > 0 else float("nan")),
        "query_acc": (float(query_correct_total / max(1, int(query_total))) if query_steps_i > 0 else float("nan")),
        "query_acc_hit": (
            float(query_correct_hit_total / max(1, int(query_hit_total))) if query_steps_i > 0 else float("nan")
        ),
        "keys": {"K1": {"id": int(key1), "token": tok_k1}, "K2": {"id": int(key2), "token": tok_k2}},
        "values": {"A": {"id": int(val_a), "token": tok_a}, "B": {"id": int(val_b), "token": tok_b}},
        "seq_len": int(seq_len),
        "key_len": int(key_len),
        "value_len": int(value_len),
        "clue_margin": int(clue_margin),
        "boundary_window": int(boundary_window),
        "fixate_steps": int(fixate_steps_i),
        "fixate_mad_mult": float(fixate_mad_mult_f),
        "fixate_mode": str(fixate_mode_s),
        "fixate_scan_radius": int(fixate_scan_radius_i),
        "fixate_trigger_rate": float(fixate_trigger_total / max(1, int(batch_size) * int(online_steps) * 2 * int(pairs))),
        "fixate_step_rate": float(fixate_step_total / max(1, int(batch_size) * int(online_steps) * 2 * int(pairs))),
        "val_align": bool(val_align_i != 0),
        "val_align_trigger_rate": float(
            val_align_trigger_total / max(1, int(batch_size) * int(online_steps) * 2 * int(pairs))
        ),
        "val_align_step_rate": float(val_align_step_total / max(1, int(batch_size) * int(online_steps) * 2 * int(pairs))),
        "speak": bool(speak_i != 0),
        "speak_use": bool(speak_use_i != 0),
        "speak_vocab": int(speak_vocab_i),
        "speak_loss_weight": float(speak_loss_w),
        "speak_back_threshold": float(speak_back_thr),
        "speak_acc": (float(speak_correct_total / max(1, int(speak_total))) if speak_i != 0 else float("nan")),
        "speak_ce": (float(speak_ce_sum / max(1, int(speak_ce_count))) if speak_i != 0 else float("nan")),
        "speak_back_rate": float(speak_back_total / max(1, int(batch_size) * int(online_steps) * 2 * int(pairs))),
        "base_marker_violations": int(base_marker_violations),
        "base_marker_count_total": int(base_marker_count_total),
        "edge_min": int(edge_min if edge_min < int(1e8) else -1),
        "k1_pos_min": int(min(key1_pos_list) if key1_pos_list else -1),
        "k1_pos_max": int(max(key1_pos_list) if key1_pos_list else -1),
        "k2_pos_min": int(min(key2_pos_list) if key2_pos_list else -1),
        "k2_pos_max": int(max(key2_pos_list) if key2_pos_list else -1),
        "val_hit_rate": float(val_hit_rate),
        "resolved_any_rate": float(resolved_any_rate),
        "resolved_rate": float(resolved_rate),
        "first_hit_mean": float(first_hit_mean),
        "first_hit_median": float(first_hit_med),
        "dwell_mean": float(dwell_cat.to(dtype=torch.float32).mean().item()) if total > 0 else float("nan"),
        "dwell_hit_mean": float(dwell_hit_mean),
        "dwell_hit_median": float(dwell_hit_med),
        "span_mean": float(span_cat.to(dtype=torch.float32).mean().item()) if total > 0 else float("nan"),
        "span_hit_mean": float(span_hit_mean),
        "span_hit_median": float(span_hit_med),
        "delta_mean": float(delta_sum / max(1, int(delta_count))),
        "delta_abs_mean": float(delta_abs_sum / max(1, int(delta_count))),
        "near_boundary_rate": float(boundary_steps / max(1, int(boundary_total))),
        "probe_acc": float(probe["acc"]),
        "probe_auc": float(probe["auc"]),
        "probe_acc_shuffled": float(probe_shuf["acc"]),
        "probe_auc_shuffled": float(probe_shuf["auc"]),
        "probe_n_hit": int(n_hit),
        "probe_acc_hit": float(probe_hit["acc"]),
        "probe_auc_hit": float(probe_hit["auc"]),
        "probe_acc_hit_shuffled": float(probe_hit_shuf["acc"]),
        "probe_auc_hit_shuffled": float(probe_hit_shuf["auc"]),
        "probe_n_nohit": int(n_nohit),
        "probe_acc_nohit": float(probe_nohit["acc"]),
        "probe_auc_nohit": float(probe_nohit["auc"]),
        "probe_acc_nohit_shuffled": float(probe_nohit_shuf["acc"]),
        "probe_auc_nohit_shuffled": float(probe_nohit_shuf["auc"]),
        "n_samples": int(total),
        "feature": ("query_pred_y_hat" if query_steps_i > 0 else "mean_offline_q"),
        "ptr_cycles": e2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="checkpoints/l7v2_v0_verify")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--text_a", type=str, default="HEI/data/CLUE/CLUECorpusSmall_head5000.txt")
    parser.add_argument("--text_b", type=str, default="HEI/data/cilin/new_cilin.txt")
    parser.add_argument("--max_chars", type=int, default=200000)
    parser.add_argument("--vocab_size", type=int, default=8000)

    parser.add_argument("--dim_q", type=int, default=64)
    parser.add_argument("--num_charts", type=int, default=4)

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta_kl", type=float, default=0.01)
    parser.add_argument("--gamma_pred", type=float, default=1.0)
    parser.add_argument("--stiffness", type=float, default=0.01)
    parser.add_argument("--contact_stiffness", type=float, default=0.1)
    parser.add_argument("--train_v", type=int, default=1, help="Whether to train V(q,z) (net_V).")
    parser.add_argument("--lr_v", type=float, default=None, help="Learning rate for net_V when --train_v=1.")
    parser.add_argument("--cost_weight", type=float, default=1e-3)
    parser.add_argument("--cost_power", type=float, default=2.0)
    parser.add_argument("--cost_offset", type=float, default=0.0)
    parser.add_argument("--noise_std", type=float, default=0.05)

    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--step_size", type=float, default=1.0)
    parser.add_argument("--candidates", type=str, default="-2,-1,0,1,2")

    # v0.1: anti dark-room (still within a single scalar F; optional, default off for v0 reproducibility)
    parser.add_argument("--pointer_drift", type=float, default=0.0)
    parser.add_argument("--pointer_bounds", type=str, default="clamp", choices=["clamp", "reflect", "wrap"])
    parser.add_argument("--anti_lock_weight", type=float, default=0.0)
    parser.add_argument("--anti_lock_sigma", type=float, default=1.0)
    parser.add_argument("--revisit_weight", type=float, default=0.0)
    parser.add_argument("--revisit_window", type=int, default=0)
    parser.add_argument("--lookahead_steps", type=int, default=1)
    parser.add_argument("--lookahead_discount", type=float, default=0.9)
    parser.add_argument("--pref_repeat_weight", type=float, default=0.0)
    parser.add_argument("--cover_weight", type=float, default=0.0)
    parser.add_argument("--cover_sigma", type=float, default=2.0)
    parser.add_argument("--cover_window", type=int, default=0)
    parser.add_argument("--cover_warmup_frac", type=float, default=0.3)
    parser.add_argument("--epi_weight", type=float, default=0.0)
    parser.add_argument("--epi_normalize", type=int, default=1)
    parser.add_argument(
        "--epi_mode",
        type=str,
        default="chart_entropy",
        choices=["chart_entropy", "obs_grad", "jac_e", "jac_E"],
    )
    parser.add_argument("--epi_obs_eps", type=float, default=1.0)
    parser.add_argument("--epi_pred_floor", type=float, default=0.0)

    parser.add_argument("--e0_steps", type=int, default=200)
    parser.add_argument("--episodes_per_class", type=int, default=8)
    parser.add_argument("--online_steps", type=int, default=80)
    parser.add_argument("--offline_steps", type=int, default=20)
    parser.add_argument("--replay_mode", type=str, default="random", choices=["none", "random", "recent", "prioritized"])

    parser.add_argument("--run_e0", type=int, default=1)
    parser.add_argument("--run_e1e2", type=int, default=1)
    parser.add_argument("--run_e3", type=int, default=0)
    parser.add_argument("--e3_task", type=str, default="marker_patch", choices=["marker_patch", "kv_swap"])
    parser.add_argument("--e3_pairs", type=int, default=8, help="Number of base episodes (each yields 2 classes).")
    parser.add_argument("--e3_seq_len", type=int, default=4096)
    parser.add_argument("--e3_clue_len", type=int, default=32)
    parser.add_argument("--e3_key_len", type=int, default=1, help="E3 kv_swap: key span length (default 1).")
    parser.add_argument("--e3_clue_margin", type=int, default=64)
    parser.add_argument(
        "--e3_policy",
        type=str,
        default="active",
        choices=["active", "active_fixate", "passive", "random", "frozen"],
    )
    parser.add_argument("--e3_fixate_steps", type=int, default=8, help="active_fixate: fixation steps after surprise trigger.")
    parser.add_argument(
        "--e3_fixate_mad_mult",
        type=float,
        default=6.0,
        help="active_fixate: trigger if pred_err > median + k*MAD (k=this).",
    )
    parser.add_argument(
        "--e3_fixate_mode",
        type=str,
        default="freeze",
        choices=["freeze", "backward_uniform", "both_uniform"],
        help="active_fixate: fixation behavior (freeze pointer or run a local micro-scan).",
    )
    parser.add_argument(
        "--e3_fixate_scan_radius",
        type=int,
        default=0,
        help="active_fixate: scan radius for non-freeze modes (0 -> use value_len for kv_swap).",
    )
    parser.add_argument(
        "--e3_query_steps",
        type=int,
        default=0,
        help="E3 kv_swap: number of query/readout training steps (0 disables query readout).",
    )
    parser.add_argument(
        "--e3_query_key",
        type=str,
        default="none",
        choices=["none", "k1", "k2", "random"],
        help="E3 kv_swap: which key to query during query/readout phase.",
    )
    parser.add_argument(
        "--e3_val_align",
        type=int,
        default=0,
        help="E3 kv_swap: if 1, value-hit triggers a one-step alignment to the corresponding key on the next step.",
    )
    parser.add_argument("--e3_speak", type=int, default=0, help="E3 kv_swap: enable a discrete 'speak' (inner-language) port.")
    parser.add_argument(
        "--e3_speak_vocab",
        type=int,
        default=2,
        help="E3 kv_swap: speak vocab size (>=2). Index 1 is treated as BACK for e3_speak_use.",
    )
    parser.add_argument(
        "--e3_speak_loss_weight",
        type=float,
        default=0.1,
        help="E3 kv_swap: supervised speak CE loss weight (label = in_val_mask). 0 disables.",
    )
    parser.add_argument(
        "--e3_speak_use",
        type=int,
        default=1,
        help="E3 kv_swap: if 1, use speak token BACK to override pointer delta (non-differentiable threshold).",
    )
    parser.add_argument(
        "--e3_speak_back_threshold",
        type=float,
        default=0.9,
        help="E3 kv_swap: threshold for BACK probability when e3_speak_use=1.",
    )
    parser.add_argument("--e3_boundary_window", type=int, default=0, help="0 = auto (3*sigma).")
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    set_seed(int(args.seed))

    run_id = _now_run_id(int(args.seed))
    save_dir = os.path.join(str(args.save_dir), run_id)
    os.makedirs(save_dir, exist_ok=True)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg = {
        "run_id": run_id,
        "defaults": asdict(V0Defaults()),
        "git_commit": _git_commit(repo_root),
        "device": device,
        "seed": int(args.seed),
        "text_a": args.text_a,
        "text_b": args.text_b,
        "max_chars": int(args.max_chars),
        "vocab_size": int(args.vocab_size),
        "dim_q": int(args.dim_q),
        "num_charts": int(args.num_charts),
        "dt": float(args.dt),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "beta_kl": float(args.beta_kl),
        "gamma_pred": float(args.gamma_pred),
        "stiffness": float(args.stiffness),
        "contact_stiffness": float(args.contact_stiffness),
        "train_v": int(args.train_v),
        "lr_v": None if args.lr_v is None else float(args.lr_v),
        "cost_weight": float(args.cost_weight),
        "cost_power": float(args.cost_power),
        "cost_offset": float(args.cost_offset),
        "noise_std": float(args.noise_std),
        "sigma": float(args.sigma),
        "step_size": float(args.step_size),
        "candidates": args.candidates,
        "pointer_drift": float(args.pointer_drift),
        "pointer_bounds": str(args.pointer_bounds),
        "anti_lock_weight": float(args.anti_lock_weight),
        "anti_lock_sigma": float(args.anti_lock_sigma),
        "revisit_weight": float(args.revisit_weight),
        "revisit_window": int(args.revisit_window),
        "lookahead_steps": int(args.lookahead_steps),
        "lookahead_discount": float(args.lookahead_discount),
        "pref_repeat_weight": float(args.pref_repeat_weight),
        "cover_weight": float(args.cover_weight),
        "cover_sigma": float(args.cover_sigma),
        "cover_window": int(args.cover_window),
        "cover_warmup_frac": float(args.cover_warmup_frac),
        "epi_weight": float(args.epi_weight),
        "epi_normalize": int(args.epi_normalize),
        "epi_mode": str(args.epi_mode),
        "epi_obs_eps": float(args.epi_obs_eps),
        "epi_pred_floor": float(args.epi_pred_floor),
        "e0_steps": int(args.e0_steps),
        "episodes_per_class": int(args.episodes_per_class),
        "online_steps": int(args.online_steps),
        "offline_steps": int(args.offline_steps),
        "replay_mode": args.replay_mode,
        "run_e3": int(args.run_e3),
        "e3_task": str(args.e3_task),
        "e3_pairs": int(args.e3_pairs),
        "e3_seq_len": int(args.e3_seq_len),
        "e3_clue_len": int(args.e3_clue_len),
        "e3_key_len": int(args.e3_key_len),
        "e3_clue_margin": int(args.e3_clue_margin),
        "e3_policy": str(args.e3_policy),
        "e3_fixate_steps": int(args.e3_fixate_steps),
        "e3_fixate_mad_mult": float(args.e3_fixate_mad_mult),
        "e3_fixate_mode": str(args.e3_fixate_mode),
        "e3_fixate_scan_radius": int(args.e3_fixate_scan_radius),
        "e3_query_steps": int(args.e3_query_steps),
        "e3_query_key": str(args.e3_query_key),
        "e3_val_align": int(args.e3_val_align),
        "e3_speak": int(args.e3_speak),
        "e3_speak_vocab": int(args.e3_speak_vocab),
        "e3_speak_loss_weight": float(args.e3_speak_loss_weight),
        "e3_speak_use": int(args.e3_speak_use),
        "e3_speak_back_threshold": float(args.e3_speak_back_threshold),
        "e3_boundary_window": int(args.e3_boundary_window),
    }
    with open(os.path.join(save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Load texts and build vocab
    text_a = read_text_file(args.text_a, max_chars=int(args.max_chars))
    text_b = read_text_file(args.text_b, max_chars=int(args.max_chars))
    vocab = build_char_vocab([text_a, text_b], vocab_size=int(args.vocab_size))

    ids_a = torch.tensor(vocab.encode(text_a), dtype=torch.long, device=device)
    ids_b = torch.tensor(vocab.encode(text_b), dtype=torch.long, device=device)

    # Frozen embedding defines the sensory space y_t (dim_u == dim_q).
    emb = nn.Embedding(len(vocab.id_to_token), int(args.dim_q), padding_idx=vocab.pad_id).to(device)
    for p in emb.parameters():
        p.requires_grad_(False)
    nn.init.normal_(emb.weight, mean=0.0, std=0.5)
    with torch.no_grad():
        emb.weight[vocab.pad_id].zero_()

    port_a = GaussianTruncRenormReadPort(token_ids=ids_a, embedding=emb, sigma=float(args.sigma)).to(device)
    port_b = GaussianTruncRenormReadPort(token_ids=ids_b, embedding=emb, sigma=float(args.sigma)).to(device)

    # Model
    entity = create_soul_entity(
        {
            "dim_q": int(args.dim_q),
            "dim_u": int(args.dim_q),
            "num_charts": int(args.num_charts),
            "beta_kl": float(args.beta_kl),
            "gamma_pred": float(args.gamma_pred),
            "stiffness": float(args.stiffness),
            "contact_stiffness": float(args.contact_stiffness),
        }
    ).to(device)
    decoder = ObsDecoder(int(args.dim_q), int(args.dim_q), hidden=0).to(device)

    e3_task_init = str(args.e3_task or "marker_patch").strip().lower()
    if int(args.run_e3) == 1 and int(args.e3_speak) != 0 and e3_task_init != "kv_swap":
        raise ValueError("--e3_speak is currently only supported for --e3_task kv_swap")
    if int(args.run_e3) == 1 and e3_task_init == "kv_swap" and int(args.e3_speak) != 0:
        speak_vocab = int(args.e3_speak_vocab)
        if speak_vocab < 2:
            raise ValueError(f"--e3_speak_vocab must be >=2 (got {speak_vocab})")
        if "speak" not in entity.get_available_interfaces():
            entity.add_interface("speak", int(args.dim_q))
        entity.e3_speak_head = nn.Linear(int(args.dim_q), speak_vocab).to(device)
        entity.e3_speak_emb = nn.Embedding(speak_vocab, int(args.dim_q)).to(device)
        with torch.no_grad():
            entity.e3_speak_emb.weight.zero_()
            entity.e3_speak_head.weight.zero_()
            if entity.e3_speak_head.bias is not None:
                entity.e3_speak_head.bias.zero_()

    init_entity = {k: v.detach().clone() for k, v in entity.state_dict().items()}
    init_decoder = {k: v.detach().clone() for k, v in decoder.state_dict().items()}

    candidates = [float(x.strip()) for x in str(args.candidates).split(",") if x.strip()]
    log_path = os.path.join(save_dir, "log.jsonl")
    summary: Dict[str, object] = {}

    with open(log_path, "w", encoding="utf-8") as log_f:
        if int(args.run_e0) == 1:
            # E0: passive vs active, same init
            e0 = {}
            for mode in ("passive", "active"):
                entity.load_state_dict(init_entity, strict=True)
                decoder.load_state_dict(init_decoder, strict=True)
                entity.reset(batch_size=int(args.batch_size), device=device)
                opt = make_optimizer(
                    entity=entity,
                    decoder=decoder,
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    train_v=(int(args.train_v) == 1),
                    lr_v=(None if args.lr_v is None else float(args.lr_v)),
                )
                curves = run_online_steps(
                    entity=entity,
                    decoder=decoder,
                    read_port=port_a,
                    optimizer=opt,
                    device=device,
                    batch_size=int(args.batch_size),
                    steps=int(args.e0_steps),
                    dt=float(args.dt),
                    mode=mode,
                    step_size=float(args.step_size),
                    candidates=candidates,
                    beta_kl=float(args.beta_kl),
                    gamma_pred=float(args.gamma_pred),
                    cost_weight=float(args.cost_weight),
                    cost_power=float(args.cost_power),
                    cost_offset=float(args.cost_offset),
                    noise_std=float(args.noise_std),
                    pointer_drift=float(args.pointer_drift),
                    pointer_bounds=str(args.pointer_bounds),
                    anti_lock_weight=float(args.anti_lock_weight),
                    anti_lock_sigma=float(args.anti_lock_sigma),
                    revisit_weight=float(args.revisit_weight),
                    revisit_window=int(args.revisit_window),
                    lookahead_steps=int(args.lookahead_steps),
                    lookahead_discount=float(args.lookahead_discount),
                    pref_repeat_weight=float(args.pref_repeat_weight),
                    cover_weight=float(args.cover_weight),
                    cover_sigma=float(args.cover_sigma),
                    cover_window=int(args.cover_window),
                    cover_warmup_frac=float(args.cover_warmup_frac),
                    epi_weight=float(args.epi_weight),
                    epi_normalize=int(args.epi_normalize),
                    epi_mode=str(args.epi_mode),
                    epi_obs_eps=float(args.epi_obs_eps),
                    epi_pred_floor=float(args.epi_pred_floor),
                    log_f=log_f,
                    tag=f"e0_{mode}",
                    log_every=int(args.log_every),
                )
                e0[mode] = {
                    "pred_err_final": float(curves["pred_err"][-1]) if curves["pred_err"] else float("nan"),
                    "F_final": float(curves["F"][-1]) if curves["F"] else float("nan"),
                    "delta_mean_final": float(curves.get("final", {}).get("delta_mean", float("nan"))),
                    "s_std_final": float(curves.get("final", {}).get("s_std", float("nan"))),
                    "H_chart_final": float(curves.get("final", {}).get("H_chart", float("nan"))),
                    "V_term_final": float(curves.get("final", {}).get("V_term", float("nan"))),
                    "E_pred_final": float(curves.get("final", {}).get("E_pred", float("nan"))),
                    "cost_final": float(curves.get("final", {}).get("cost", float("nan"))),
                    "anti_lock_final": float(curves.get("final", {}).get("anti_lock", float("nan"))),
                    "revisit_final": float(curves.get("final", {}).get("revisit", float("nan"))),
                    "pref_final": float(curves.get("final", {}).get("pref", float("nan"))),
                    "cover_final": float(curves.get("final", {}).get("cover", float("nan"))),
                    "epi_final": float(curves.get("final", {}).get("epi", float("nan"))),
                    "epi_proxy_final": float(curves.get("final", {}).get("epi_proxy", float("nan"))),
                }
                with open(os.path.join(save_dir, f"e0_{mode}_curves.json"), "w", encoding="utf-8") as f:
                    json.dump(curves, f, ensure_ascii=False)
            summary["E0"] = e0

        if int(args.run_e1e2) == 1:
            # E1/E2: fixed rhythm episodes on A and B, active mode only.
            entity.load_state_dict(init_entity, strict=True)
            decoder.load_state_dict(init_decoder, strict=True)
            opt = make_optimizer(
                entity=entity,
                decoder=decoder,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                train_v=(int(args.train_v) == 1),
                lr_v=(None if args.lr_v is None else float(args.lr_v)),
            )

            feats_a, labs_a, e2_a = run_fixed_rhythm_episodes(
                entity=entity,
                decoder=decoder,
                read_port=port_a,
                optimizer=opt,
                device=device,
                batch_size=int(args.batch_size),
                episodes=int(args.episodes_per_class),
                online_steps=int(args.online_steps),
                offline_steps=int(args.offline_steps),
                dt=float(args.dt),
                step_size=float(args.step_size),
                candidates=candidates,
                beta_kl=float(args.beta_kl),
                gamma_pred=float(args.gamma_pred),
                cost_weight=float(args.cost_weight),
                cost_power=float(args.cost_power),
                cost_offset=float(args.cost_offset),
                noise_std=float(args.noise_std),
                pointer_drift=float(args.pointer_drift),
                pointer_bounds=str(args.pointer_bounds),
                anti_lock_weight=float(args.anti_lock_weight),
                anti_lock_sigma=float(args.anti_lock_sigma),
                revisit_weight=float(args.revisit_weight),
                revisit_window=int(args.revisit_window),
                lookahead_steps=int(args.lookahead_steps),
                lookahead_discount=float(args.lookahead_discount),
                pref_repeat_weight=float(args.pref_repeat_weight),
                cover_weight=float(args.cover_weight),
                cover_sigma=float(args.cover_sigma),
                cover_window=int(args.cover_window),
                cover_warmup_frac=float(args.cover_warmup_frac),
                epi_weight=float(args.epi_weight),
                epi_normalize=int(args.epi_normalize),
                epi_mode=str(args.epi_mode),
                epi_obs_eps=float(args.epi_obs_eps),
                epi_pred_floor=float(args.epi_pred_floor),
                replay_mode=str(args.replay_mode),
                log_f=log_f,
                label=0,
                tag="e1e2_A",
                log_every=int(args.log_every),
            )
            feats_b, labs_b, e2_b = run_fixed_rhythm_episodes(
                entity=entity,
                decoder=decoder,
                read_port=port_b,
                optimizer=opt,
                device=device,
                batch_size=int(args.batch_size),
                episodes=int(args.episodes_per_class),
                online_steps=int(args.online_steps),
                offline_steps=int(args.offline_steps),
                dt=float(args.dt),
                step_size=float(args.step_size),
                candidates=candidates,
                beta_kl=float(args.beta_kl),
                gamma_pred=float(args.gamma_pred),
                cost_weight=float(args.cost_weight),
                cost_power=float(args.cost_power),
                cost_offset=float(args.cost_offset),
                noise_std=float(args.noise_std),
                pointer_drift=float(args.pointer_drift),
                pointer_bounds=str(args.pointer_bounds),
                anti_lock_weight=float(args.anti_lock_weight),
                anti_lock_sigma=float(args.anti_lock_sigma),
                revisit_weight=float(args.revisit_weight),
                revisit_window=int(args.revisit_window),
                lookahead_steps=int(args.lookahead_steps),
                lookahead_discount=float(args.lookahead_discount),
                pref_repeat_weight=float(args.pref_repeat_weight),
                cover_weight=float(args.cover_weight),
                cover_sigma=float(args.cover_sigma),
                cover_window=int(args.cover_window),
                cover_warmup_frac=float(args.cover_warmup_frac),
                epi_weight=float(args.epi_weight),
                epi_normalize=int(args.epi_normalize),
                epi_mode=str(args.epi_mode),
                epi_obs_eps=float(args.epi_obs_eps),
                epi_pred_floor=float(args.epi_pred_floor),
                replay_mode=str(args.replay_mode),
                log_f=log_f,
                label=1,
                tag="e1e2_B",
                log_every=int(args.log_every),
            )

            feats = torch.cat([feats_a, feats_b], dim=0)
            labs = torch.cat([labs_a, labs_b], dim=0)
            probe = train_linear_probe(feats, labs, seed=int(args.seed))
            # Control: shuffled labels should collapse toward chance if separability is truly experience-conditioned.
            labs_shuf = labs[torch.randperm(int(labs.numel()))]
            probe_shuf = train_linear_probe(feats, labs_shuf, seed=int(args.seed) + 1)

            summary["E1"] = {
                "probe_acc": float(probe["acc"]),
                "probe_auc": float(probe["auc"]),
                "probe_acc_shuffled": float(probe_shuf["acc"]),
                "probe_auc_shuffled": float(probe_shuf["auc"]),
                "n_samples": int(labs.numel()),
                "feature": "mean_offline_q",
            }
            summary["E2"] = {
                "A": e2_a,
                "B": e2_b,
            }

        if int(args.run_e3) == 1:
            # E3: minimal reading closed-loop (localized evidence); from fresh init weights.
            entity.load_state_dict(init_entity, strict=True)
            decoder.load_state_dict(init_decoder, strict=True)
            opt = make_optimizer(
                entity=entity,
                decoder=decoder,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                train_v=(int(args.train_v) == 1),
                lr_v=(None if args.lr_v is None else float(args.lr_v)),
            )
            e3_task = str(args.e3_task or "marker_patch").strip().lower()
            if e3_task == "marker_patch":
                e3 = run_e3_marker_reading(
                    entity=entity,
                    decoder=decoder,
                    embedding=emb,
                    vocab=vocab,
                    optimizer=opt,
                    device=device,
                    batch_size=int(args.batch_size),
                    pairs=int(args.e3_pairs),
                    seq_len=int(args.e3_seq_len),
                    clue_len=int(args.e3_clue_len),
                    clue_margin=int(args.e3_clue_margin),
                    policy=str(args.e3_policy),
                    fixate_steps=int(args.e3_fixate_steps),
                    fixate_mad_mult=float(args.e3_fixate_mad_mult),
                    boundary_window=int(args.e3_boundary_window),
                    read_sigma=float(args.sigma),
                    online_steps=int(args.online_steps),
                    offline_steps=int(args.offline_steps),
                    dt=float(args.dt),
                    step_size=float(args.step_size),
                    candidates=candidates,
                    beta_kl=float(args.beta_kl),
                    gamma_pred=float(args.gamma_pred),
                    cost_weight=float(args.cost_weight),
                    cost_power=float(args.cost_power),
                    cost_offset=float(args.cost_offset),
                    noise_std=float(args.noise_std),
                    pointer_drift=float(args.pointer_drift),
                    pointer_bounds=str(args.pointer_bounds),
                    anti_lock_weight=float(args.anti_lock_weight),
                    anti_lock_sigma=float(args.anti_lock_sigma),
                    revisit_weight=float(args.revisit_weight),
                    revisit_window=int(args.revisit_window),
                    lookahead_steps=int(args.lookahead_steps),
                    lookahead_discount=float(args.lookahead_discount),
                    pref_repeat_weight=float(args.pref_repeat_weight),
                    cover_weight=float(args.cover_weight),
                    cover_sigma=float(args.cover_sigma),
                    cover_window=int(args.cover_window),
                    cover_warmup_frac=float(args.cover_warmup_frac),
                    epi_weight=float(args.epi_weight),
                    epi_normalize=int(args.epi_normalize),
                    epi_mode=str(args.epi_mode),
                    epi_obs_eps=float(args.epi_obs_eps),
                    epi_pred_floor=float(args.epi_pred_floor),
                    replay_mode=str(args.replay_mode),
                    seed=int(args.seed),
                    log_f=log_f,
                    tag="e3",
                    log_every=int(args.log_every),
                )
            elif e3_task == "kv_swap":
                e3 = run_e3_kv_swap(
                    entity=entity,
                    decoder=decoder,
                    embedding=emb,
                    vocab=vocab,
                    optimizer=opt,
                    device=device,
                    batch_size=int(args.batch_size),
                    pairs=int(args.e3_pairs),
                    seq_len=int(args.e3_seq_len),
                    key_len=int(args.e3_key_len),
                    value_len=int(args.e3_clue_len),
                    clue_margin=int(args.e3_clue_margin),
                    policy=str(args.e3_policy),
                    fixate_steps=int(args.e3_fixate_steps),
                    fixate_mad_mult=float(args.e3_fixate_mad_mult),
                    fixate_mode=str(args.e3_fixate_mode),
                    fixate_scan_radius=int(args.e3_fixate_scan_radius),
                    boundary_window=int(args.e3_boundary_window),
                    read_sigma=float(args.sigma),
                    online_steps=int(args.online_steps),
                    offline_steps=int(args.offline_steps),
                    dt=float(args.dt),
                    step_size=float(args.step_size),
                    candidates=candidates,
                    beta_kl=float(args.beta_kl),
                    gamma_pred=float(args.gamma_pred),
                    cost_weight=float(args.cost_weight),
                    cost_power=float(args.cost_power),
                    cost_offset=float(args.cost_offset),
                    noise_std=float(args.noise_std),
                    pointer_drift=float(args.pointer_drift),
                    pointer_bounds=str(args.pointer_bounds),
                    anti_lock_weight=float(args.anti_lock_weight),
                    anti_lock_sigma=float(args.anti_lock_sigma),
                    revisit_weight=float(args.revisit_weight),
                    revisit_window=int(args.revisit_window),
                    lookahead_steps=int(args.lookahead_steps),
                    lookahead_discount=float(args.lookahead_discount),
                    pref_repeat_weight=float(args.pref_repeat_weight),
                    cover_weight=float(args.cover_weight),
                    cover_sigma=float(args.cover_sigma),
                    cover_window=int(args.cover_window),
                    cover_warmup_frac=float(args.cover_warmup_frac),
                    epi_weight=float(args.epi_weight),
                    epi_normalize=int(args.epi_normalize),
                    epi_mode=str(args.epi_mode),
                    epi_obs_eps=float(args.epi_obs_eps),
                    epi_pred_floor=float(args.epi_pred_floor),
                    replay_mode=str(args.replay_mode),
                    query_steps=int(args.e3_query_steps),
                    query_key=str(args.e3_query_key),
                    val_align=int(args.e3_val_align),
                    speak=int(args.e3_speak),
                    speak_vocab=int(args.e3_speak_vocab),
                    speak_loss_weight=float(args.e3_speak_loss_weight),
                    speak_use=int(args.e3_speak_use),
                    speak_back_threshold=float(args.e3_speak_back_threshold),
                    seed=int(args.seed),
                    log_f=log_f,
                    tag="e3kv",
                    log_every=int(args.log_every),
                )
            else:
                raise ValueError(f"unknown e3_task: {args.e3_task}")
            summary["E3"] = e3

    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Run Card (markdown) for human-facing tracking
    cmd = " ".join(shlex.quote(x) for x in sys.argv)
    run_md = os.path.join(save_dir, "run.md")
    with open(run_md, "w", encoding="utf-8") as f:
        f.write(f"# Run Card: {run_id}\n\n")
        f.write(f"- Command: `{cmd}`\n")
        f.write(f"- Device: `{device}`\n")
        f.write(f"- Seed: `{int(args.seed)}`\n")
        f.write(f"- Defaults: `{V0Defaults()}`\n")
        f.write(f"- Text A: `{args.text_a}`\n")
        f.write(f"- Text B: `{args.text_b}`\n")
        f.write(f"- Vocab size: `{int(args.vocab_size)}`\n")
        f.write(f"- dim_q/dim_u: `{int(args.dim_q)}`\n")
        f.write(f"- Kernel: `{cfg['kernel'] if 'kernel' in cfg else 'gaussian_trunc_renorm'}` (σ={float(args.sigma)})\n")
        f.write(
            f"- v0.1 options: drift={float(args.pointer_drift)} bounds={str(args.pointer_bounds)} "
            f"anti_lock={float(args.anti_lock_weight)} (σ={float(args.anti_lock_sigma)}) "
            f"revisit={float(args.revisit_weight)} (win={int(args.revisit_window)}) "
            f"pref_repeat={float(args.pref_repeat_weight)}\n"
        )
        f.write(
            f"- v0.3 cover: weight={float(args.cover_weight)} sigma={float(args.cover_sigma)} "
            f"window={int(args.cover_window)} warmup_frac={float(args.cover_warmup_frac)}\n"
        )
        f.write(
            f"- v0.4 epistemic: weight={float(args.epi_weight)} normalize={int(args.epi_normalize)} "
            f"mode={str(args.epi_mode)} obs_eps={float(args.epi_obs_eps)} pred_floor={float(args.epi_pred_floor)}\n"
        )
        f.write(
            f"- Cost: weight={float(args.cost_weight)} power={float(args.cost_power)} offset={float(args.cost_offset)}\n"
        )
        f.write(
            f"- Control: lookahead_steps={int(args.lookahead_steps)} discount={float(args.lookahead_discount)}\n"
        )
        f.write(f"- Online/Offline: `{int(args.online_steps)}` / `{int(args.offline_steps)}` (fixed rhythm)\n")
        f.write("\n## Outputs\n")
        f.write(f"- `run_config.json`\n- `log.jsonl`\n- `summary.json`\n\n")
        f.write("## Summary\n")
        f.write("```json\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n```\n")

    print(f"[L7v2-v0] Saved run to: {save_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
