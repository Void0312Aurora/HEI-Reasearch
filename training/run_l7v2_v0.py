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
    parser.add_argument("--pointer_bounds", type=str, default="clamp", choices=["clamp", "reflect"])
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
