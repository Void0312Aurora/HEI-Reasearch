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
    anti_lock_weight: float = 0.0,
    anti_lock_sigma: float = 1.0,
    revisit_penalty_per_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    V_ps = compute_v_term_per_sample(entity, state)  # [B]
    V_term = V_ps.mean()
    KL = 0.5 * (entity.z ** 2).sum() * float(beta_kl)
    E_pred = float(gamma_pred) * pred_err_per_sample.mean()

    cost = V_term.new_zeros(())
    if action_delta is not None:
        cost = float(cost_weight) * (action_delta.to(dtype=V_term.dtype) ** 2).mean()

    anti_lock = V_term.new_zeros(())
    if action_delta is not None and float(anti_lock_weight) > 0.0:
        sigma = max(float(anti_lock_sigma), 1e-6)
        d = action_delta.to(dtype=V_term.dtype)
        anti_lock = float(anti_lock_weight) * torch.exp(-0.5 * (d / sigma) ** 2).mean()

    revisit = V_term.new_zeros(())
    if revisit_penalty_per_sample is not None:
        revisit = revisit_penalty_per_sample.to(dtype=V_term.dtype).mean()

    F = V_term + KL + E_pred + cost + anti_lock + revisit
    # Important: return tensors to avoid per-step GPU sync via `.item()`.
    return F, {
        "V_term": V_term.detach(),
        "KL": KL.detach(),
        "E_pred": E_pred.detach(),
        "cost": cost.detach(),
        "anti_lock": anti_lock.detach(),
        "revisit": revisit.detach(),
        "F": F.detach(),
    }


def _parts_to_float(parts: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().item()) for k, v in parts.items()}


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
    noise_std: float,
    pointer_drift: float = 0.0,
    anti_lock_weight: float = 0.0,
    anti_lock_sigma: float = 1.0,
    revisit_weight: float = 0.0,
    recent_s: Optional[torch.Tensor] = None,  # [K,B] long (quantized pointer history)
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

    # Vectorized candidate evaluation (reduces Python overhead & improves GPU utilization).
    L = float(read_port.length - 1)
    cand = candidates.to(device=device, dtype=torch.float32)  # [C]
    s0 = s.to(dtype=torch.float32)  # [B]
    s_new = (s0.unsqueeze(0) + float(pointer_drift) + cand.unsqueeze(1)).clamp(0.0, L)  # [C,B]

    y_flat = read_port.read(s_new.reshape(-1))  # [C*B,D]
    D = int(y_flat.shape[1])
    y = y_flat.view(C, B, D)

    y_in = y
    if noise_std > 0:
        y_in = y + float(noise_std) * torch.randn_like(y)

    state_flat = entity.state.flat  # [B,S]
    state_rep = state_flat.unsqueeze(0).expand(C, B, -1).reshape(C * B, -1)
    prev_w = getattr(entity, "_prev_chart_weights", None)
    prev_rep = None
    if prev_w is not None:
        prev_rep = prev_w.unsqueeze(0).expand(C, B, -1).reshape(C * B, -1)

    out = entity.forward_tensor(
        state_flat=state_rep,
        u_dict={"default": y_in.reshape(C * B, D)},
        dt=float(dt),
        prev_chart_weights=prev_rep,
        prediction_error=None,
        detach_next_prev_weights=True,
        compute_action=False,
        skip_free_energy=True,
    )
    next_state = ContactState(entity.dim_q, C * B, entity.state.device, out["next_state_flat"])
    y_hat = decoder(next_state.q).view(C, B, D)
    pred_err = (y_hat - y).pow(2).mean(dim=2)  # [C,B]

    V_ps = compute_v_term_per_sample(entity, next_state).view(C, B)  # [C,B]
    score = V_ps + float(gamma_pred) * pred_err
    score = score + float(cost_weight) * (cand.pow(2).unsqueeze(1))

    if float(anti_lock_weight) > 0.0:
        sigma = max(float(anti_lock_sigma), 1e-6)
        score = score + float(anti_lock_weight) * torch.exp(-0.5 * (cand / sigma).pow(2)).unsqueeze(1)

    if float(revisit_weight) > 0.0 and recent_s is not None and int(recent_s.numel()) > 0:
        s_new_round = s_new.round().to(dtype=torch.long)  # [C,B]
        hit = (recent_s.unsqueeze(0) == s_new_round.unsqueeze(1)).any(dim=1).to(dtype=score.dtype)  # [C,B]
        score = score + float(revisit_weight) * hit

    best_idx = score.argmin(dim=0)  # [B]
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
    noise_std: float,
    pointer_drift: float,
    anti_lock_weight: float,
    anti_lock_sigma: float,
    revisit_weight: float,
    revisit_window: int,
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

    pred_curve = torch.empty(int(steps), device=device, dtype=torch.float32)
    F_curve = torch.empty(int(steps), device=device, dtype=torch.float32)
    delta_curve = torch.empty(int(steps), device=device, dtype=torch.float32)
    last_parts: Dict[str, torch.Tensor] = {}
    last_s_mean = torch.tensor(float("nan"), device=device)
    last_s_std = torch.tensor(float("nan"), device=device)
    last_pred_err = torch.tensor(float("nan"), device=device)
    last_delta = torch.tensor(float("nan"), device=device)

    for t in range(int(steps)):
        if mode == "passive":
            delta = torch.full_like(s, float(step_size))
        elif mode == "active":
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
                noise_std=noise_std,
                pointer_drift=float(pointer_drift),
                anti_lock_weight=float(anti_lock_weight),
                anti_lock_sigma=float(anti_lock_sigma),
                revisit_weight=float(revisit_weight),
                recent_s=(recent_s[:recent_len] if recent_s is not None and recent_len > 0 else None),
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        s_new = (s + float(pointer_drift) + delta).clamp(0.0, float(read_port.length - 1))
        revisit_penalty_ps: Optional[torch.Tensor] = None
        if float(revisit_weight) > 0.0 and recent_s is not None and recent_len > 0:
            s_new_round = s_new.detach().round().to(dtype=torch.long)
            hit = (recent_s[:recent_len] == s_new_round.unsqueeze(0)).any(dim=0).to(dtype=s_new.dtype)
            revisit_penalty_ps = float(revisit_weight) * hit
            # push new pointer into ring buffer
            recent_s[recent_ptr] = s_new_round
            recent_ptr = (recent_ptr + 1) % int(recent_s.shape[0])
            recent_len = min(recent_len + 1, int(recent_s.shape[0]))
        s = s_new
        y = read_port.read(s)
        y_in = y
        if noise_std > 0:
            y_in = y + float(noise_std) * torch.randn_like(y)

        _entity_forward_step_fast(entity=entity, u_dict={"default": y_in}, dt=float(dt))

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
            anti_lock_weight=float(anti_lock_weight),
            anti_lock_sigma=float(anti_lock_sigma),
            revisit_penalty_per_sample=revisit_penalty_ps,
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
                "pointer_drift": float(pointer_drift),
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
    noise_std: float,
    pointer_drift: float,
    anti_lock_weight: float,
    anti_lock_sigma: float,
    revisit_weight: float,
    revisit_window: int,
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

    cand = torch.tensor(list(candidates), device=device, dtype=torch.float32) * float(step_size)
    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    # E2: pointer cycle diagnostics (quantize to int)
    cycle_hits = 0
    cycle_total = 0
    max_period = 8
    min_repeats = 3
    tail_window = 32

    for ep in range(int(episodes)):
        entity.reset(batch_size=batch_size, device=device)
        reset_experience(entity)
        # Fresh pointer per episode, away from edges
        s = torch.rand(batch_size, device=device) * float(read_port.length - 1)
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

        # Online
        entity.enter_online()
        for t in range(int(online_steps)):
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
                noise_std=noise_std,
                pointer_drift=float(pointer_drift),
                anti_lock_weight=float(anti_lock_weight),
                anti_lock_sigma=float(anti_lock_sigma),
                revisit_weight=float(revisit_weight),
                recent_s=(recent_s[:recent_len] if recent_s is not None and recent_len > 0 else None),
            )
            s_new = (s + float(pointer_drift) + delta).clamp(0.0, float(read_port.length - 1))
            revisit_penalty_ps: Optional[torch.Tensor] = None
            if float(revisit_weight) > 0.0 and recent_s is not None and recent_len > 0:
                s_new_round = s_new.detach().round().to(dtype=torch.long)
                hit = (recent_s[:recent_len] == s_new_round.unsqueeze(0)).any(dim=0).to(dtype=s_new.dtype)
                revisit_penalty_ps = float(revisit_weight) * hit
                recent_s[recent_ptr] = s_new_round
                recent_ptr = (recent_ptr + 1) % int(recent_s.shape[0])
                recent_len = min(recent_len + 1, int(recent_s.shape[0]))
            s = s_new
            s_tail[int(t) % int(tail_window)] = s.detach().round().to(dtype=torch.long)

            y = read_port.read(s)
            y_in = y
            if noise_std > 0:
                y_in = y + float(noise_std) * torch.randn_like(y)

            _entity_forward_step_fast(entity=entity, u_dict={"default": y_in}, dt=float(dt))
            # Fill experience buffer for offline replay conditioning (reward is a dummy scalar in this verifier).
            entity.experience.push(y_in, None, entity.state.flat, 0.0)

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
                anti_lock_weight=float(anti_lock_weight),
                anti_lock_sigma=float(anti_lock_sigma),
                revisit_penalty_per_sample=revisit_penalty_ps,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(entity.parameters()) + list(decoder.parameters()), max_norm=1.0)
            optimizer.step()
            entity.state = ContactState(entity.dim_q, batch_size, device, entity.state.flat.detach())

            if log_f is not None and (int(log_every) > 0) and (t % int(log_every) == 0 or t == online_steps - 1):
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
                    "pointer_drift": float(pointer_drift),
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
                if _detect_tail_cycle(tokens, max_period=max_period, min_repeats=min_repeats) is not None:
                    cycle_hits += 1

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

    e2 = {
        "ptr_cycle_rate": float(cycle_hits / max(1, cycle_total)),
        "ptr_cycle_hits": int(cycle_hits),
        "ptr_cycle_total": int(cycle_total),
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
    parser.add_argument("--beta_kl", type=float, default=0.01)
    parser.add_argument("--gamma_pred", type=float, default=1.0)
    parser.add_argument("--cost_weight", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=0.05)

    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--step_size", type=float, default=1.0)
    parser.add_argument("--candidates", type=str, default="-2,-1,0,1,2")

    # v0.1: anti dark-room (still within a single scalar F; optional, default off for v0 reproducibility)
    parser.add_argument("--pointer_drift", type=float, default=0.0)
    parser.add_argument("--anti_lock_weight", type=float, default=0.0)
    parser.add_argument("--anti_lock_sigma", type=float, default=1.0)
    parser.add_argument("--revisit_weight", type=float, default=0.0)
    parser.add_argument("--revisit_window", type=int, default=0)

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
        "beta_kl": float(args.beta_kl),
        "gamma_pred": float(args.gamma_pred),
        "cost_weight": float(args.cost_weight),
        "noise_std": float(args.noise_std),
        "sigma": float(args.sigma),
        "step_size": float(args.step_size),
        "candidates": args.candidates,
        "pointer_drift": float(args.pointer_drift),
        "anti_lock_weight": float(args.anti_lock_weight),
        "anti_lock_sigma": float(args.anti_lock_sigma),
        "revisit_weight": float(args.revisit_weight),
        "revisit_window": int(args.revisit_window),
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
                opt = torch.optim.AdamW(
                    list(entity.parameters()) + list(decoder.parameters()),
                    lr=float(args.lr),
                    weight_decay=1e-4,
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
                    noise_std=float(args.noise_std),
                    pointer_drift=float(args.pointer_drift),
                    anti_lock_weight=float(args.anti_lock_weight),
                    anti_lock_sigma=float(args.anti_lock_sigma),
                    revisit_weight=float(args.revisit_weight),
                    revisit_window=int(args.revisit_window),
                    log_f=log_f,
                    tag=f"e0_{mode}",
                    log_every=int(args.log_every),
                )
                e0[mode] = {
                    "pred_err_final": float(curves["pred_err"][-1]) if curves["pred_err"] else float("nan"),
                    "F_final": float(curves["F"][-1]) if curves["F"] else float("nan"),
                    "delta_mean_final": float(curves.get("final", {}).get("delta_mean", float("nan"))),
                    "s_std_final": float(curves.get("final", {}).get("s_std", float("nan"))),
                    "V_term_final": float(curves.get("final", {}).get("V_term", float("nan"))),
                    "E_pred_final": float(curves.get("final", {}).get("E_pred", float("nan"))),
                    "cost_final": float(curves.get("final", {}).get("cost", float("nan"))),
                    "anti_lock_final": float(curves.get("final", {}).get("anti_lock", float("nan"))),
                    "revisit_final": float(curves.get("final", {}).get("revisit", float("nan"))),
                }
                with open(os.path.join(save_dir, f"e0_{mode}_curves.json"), "w", encoding="utf-8") as f:
                    json.dump(curves, f, ensure_ascii=False)
            summary["E0"] = e0

        if int(args.run_e1e2) == 1:
            # E1/E2: fixed rhythm episodes on A and B, active mode only.
            entity.load_state_dict(init_entity, strict=True)
            decoder.load_state_dict(init_decoder, strict=True)
            opt = torch.optim.AdamW(
                list(entity.parameters()) + list(decoder.parameters()),
                lr=float(args.lr),
                weight_decay=1e-4,
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
                noise_std=float(args.noise_std),
                pointer_drift=float(args.pointer_drift),
                anti_lock_weight=float(args.anti_lock_weight),
                anti_lock_sigma=float(args.anti_lock_sigma),
                revisit_weight=float(args.revisit_weight),
                revisit_window=int(args.revisit_window),
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
                noise_std=float(args.noise_std),
                pointer_drift=float(args.pointer_drift),
                anti_lock_weight=float(args.anti_lock_weight),
                anti_lock_sigma=float(args.anti_lock_sigma),
                revisit_weight=float(args.revisit_weight),
                revisit_window=int(args.revisit_window),
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
            f"- v0.1 options: drift={float(args.pointer_drift)} anti_lock={float(args.anti_lock_weight)} "
            f"(σ={float(args.anti_lock_sigma)}) revisit={float(args.revisit_weight)} (win={int(args.revisit_window)})\n"
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
