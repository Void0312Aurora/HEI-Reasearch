"""
Stage 6: Plural Interfaces (Level 6) — A5-T1 Cross-interface Latent Alignment

Reference:
  - HEI/docs/plan/EXP/理论基础-7/VS.md  (A5-T1)

Goal:
  Starting from a trained L5 (language) core, add a new interface (vision) and
  verify that the internal representation can be aligned across interfaces
  without changing the core (ports-only adaptation).

Minimal synthetic "shared latent world":
  - latent: 2D grid position (x,y) in a small grid (default 4x4 -> 16 states)
  - language observation: a short ASCII string like "pos(2,3)"
  - vision observation: a 28x28 one-hot block image for the same grid cell

Training protocol (A5-T1):
  1) Load an L5 checkpoint and freeze it (entity + language port).
  2) Compute language-induced q prototypes for each latent state.
  3) Train ONLY the new vision port (port coupling + vision encoder) to match
     the language prototypes (MSE in q-space).
  4) Gate: nearest-prototype accuracy (clean + noisy vision).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure HEI is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from he_core.language_interface import ByteTokenizer
from he_core.vision import SimpleVisionEncoder

from curriculum.stage5_language import Stage5Config, Stage5Trainer, _normalize_device


@dataclass
class Stage6Config:
    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    # Init from Stage5
    stage5_ckpt: str = "checkpoints/curriculum/stage5_best.pt"
    stage5_text_file: str = "HEI/tests/dummy_corpus.txt"

    # Shared-latent world
    grid_size: int = 4  # 4x4 -> 16 states
    seq_len: int = 16
    image_size: int = 28  # must be divisible by grid_size

    # Dynamics rollout
    dt: float = 0.15
    evolution_steps: int = 2
    vision_steps: int = 8  # number of repeated vision "frames" (each runs evolution_steps)

    # Training (vision alignment only)
    batch_size: int = 512
    steps: int = 800
    lr: float = 1e-2
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 50
    # Performance
    use_cuda_graph: bool = False
    cuda_graph_warmup: int = 3

    # Initialization / adaptation policy
    init_vision_from_language: bool = True
    train_vision_port: bool = True

    # Alignment loss
    # - mse: regress q to prototypes (can collapse to mean)
    # - ce:  nearest-prototype classification via softmax over distances
    # - ce+mse: ce + mse_weight*mse (stabilizes distances)
    loss_type: str = "ce+mse"
    loss_tau: float = 1.0
    loss_mse_weight: float = 0.05

    # Robust eval
    eval_repeats: int = 16  # repeated noisy samples per latent
    noise_std: float = 0.25

    # Gate
    gate_profile: str = "base"  # base | robust
    gate_acc_clean: float = 0.90
    gate_acc_noisy: float = 0.85

    # Saving
    save_dir: str = "checkpoints/curriculum"
    # Save full Stage6Trainer state (large) or only the Stage6 adapter (fast, recommended).
    save_full_state: bool = False
    # If >0, only save "best" checkpoint every N steps (reduces IO stalls).
    save_best_every: int = 0

    def __str__(self) -> str:
        return (
            "Stage6Config("
            f"device={self.device}, grid={self.grid_size}x{self.grid_size}, "
            f"seq_len={self.seq_len}, img={self.image_size}, dt={self.dt}, evo={self.evolution_steps}, "
            f"vision_steps={self.vision_steps}, steps={self.steps}, batch={self.batch_size}, lr={self.lr}, "
            f"profile={self.gate_profile}, loss={self.loss_type}, cuda_graph={int(self.use_cuda_graph)}"
            ")"
        )


def _apply_env_overrides(cfg: Stage6Config) -> Stage6Config:
    def _get_bool(name: str, default: bool) -> bool:
        v = os.getenv(name)
        if v is None or v == "":
            return bool(default)
        v = str(v).strip().lower()
        return v in ("1", "true", "yes", "y", "on")

    def _get_int(name: str, default: int) -> int:
        v = os.getenv(name)
        return default if v is None or v == "" else int(v)

    def _get_float(name: str, default: float) -> float:
        v = os.getenv(name)
        return default if v is None or v == "" else float(v)

    def _get_str(name: str, default: str) -> str:
        v = os.getenv(name)
        return default if v is None or v == "" else str(v)

    cfg.device = _get_str("HEI_DEVICE", cfg.device)
    cfg.seed = _get_int("HEI_SEED", cfg.seed)

    cfg.stage5_ckpt = _get_str("HEI_STAGE5_CKPT", cfg.stage5_ckpt)
    cfg.stage5_text_file = _get_str("HEI_STAGE6_TEXT_FILE", cfg.stage5_text_file)

    cfg.grid_size = _get_int("HEI_STAGE6_GRID", cfg.grid_size)
    cfg.seq_len = _get_int("HEI_STAGE6_SEQ_LEN", cfg.seq_len)
    cfg.image_size = _get_int("HEI_STAGE6_IMAGE_SIZE", cfg.image_size)

    cfg.dt = _get_float("HEI_STAGE6_DT", cfg.dt)
    cfg.evolution_steps = _get_int("HEI_STAGE6_EVO", cfg.evolution_steps)
    cfg.vision_steps = _get_int("HEI_STAGE6_VISION_STEPS", cfg.vision_steps)

    cfg.batch_size = _get_int("HEI_BATCH_SIZE", cfg.batch_size)
    cfg.steps = _get_int("HEI_STAGE6_STEPS", cfg.steps)
    cfg.lr = _get_float("HEI_LR", cfg.lr)
    cfg.weight_decay = _get_float("HEI_WEIGHT_DECAY", cfg.weight_decay)
    cfg.grad_clip = _get_float("HEI_GRAD_CLIP", cfg.grad_clip)
    cfg.log_every = _get_int("HEI_LOG_EVERY", cfg.log_every)
    cfg.use_cuda_graph = _get_bool("HEI_CUDA_GRAPH", cfg.use_cuda_graph)
    cfg.cuda_graph_warmup = _get_int("HEI_CUDA_GRAPH_WARMUP", cfg.cuda_graph_warmup)

    cfg.init_vision_from_language = _get_bool("HEI_STAGE6_INIT_FROM_LANGUAGE", cfg.init_vision_from_language)
    cfg.train_vision_port = _get_bool("HEI_STAGE6_TRAIN_VISION_PORT", cfg.train_vision_port)
    cfg.loss_type = _get_str("HEI_STAGE6_LOSS", cfg.loss_type).strip().lower()
    cfg.loss_tau = _get_float("HEI_STAGE6_LOSS_TAU", cfg.loss_tau)
    cfg.loss_mse_weight = _get_float("HEI_STAGE6_LOSS_MSE_WEIGHT", cfg.loss_mse_weight)

    cfg.eval_repeats = _get_int("HEI_STAGE6_EVAL_REPEATS", cfg.eval_repeats)
    cfg.noise_std = _get_float("HEI_STAGE6_NOISE_STD", cfg.noise_std)

    cfg.gate_profile = _get_str("HEI_L6_GATE_PROFILE", cfg.gate_profile).lower()
    cfg.gate_acc_clean = _get_float("HEI_L6_GATE_ACC", cfg.gate_acc_clean)
    cfg.gate_acc_noisy = _get_float("HEI_L6_GATE_NOISY_ACC", cfg.gate_acc_noisy)

    cfg.save_dir = _get_str("HEI_SAVE_DIR", cfg.save_dir)
    cfg.save_full_state = _get_bool("HEI_STAGE6_SAVE_FULL", cfg.save_full_state)
    cfg.save_best_every = _get_int("HEI_STAGE6_BEST_EVERY", cfg.save_best_every)
    return cfg


def _infer_stage5_shapes(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int, bool]:
    """
    Infer (dim_q, dim_z, num_charts, decode_state) from a Stage5 checkpoint.
    """
    dim_z = 16
    if "entity.z" in sd and hasattr(sd["entity.z"], "shape"):
        dim_z = int(sd["entity.z"].shape[1])

    # Prefer port.readout_weight: [K, V, D_in]
    if "port.readout_weight" in sd:
        w = sd["port.readout_weight"]
        num_charts = int(w.shape[0])
        d_in = int(w.shape[2])
        # decode_state => d_in = 2*dq+1
        dim_q = (d_in - 1) // 2
        decode_state = (2 * dim_q + 1) == d_in
        return dim_q, dim_z, num_charts, decode_state

    # Fallback: language port coupling: [K, Du, Dq]
    for k in ("entity.generator.ports.language.W_stack", "entity.generator.ports.default.W_stack"):
        if k in sd:
            w = sd[k]
            num_charts = int(w.shape[0])
            dim_q = int(w.shape[2])
            return dim_q, dim_z, num_charts, True

    raise RuntimeError("Unable to infer Stage5 shapes from checkpoint (missing known keys).")


def _make_grid_language_table(tokenizer: ByteTokenizer, *, grid: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    pad = int(tokenizer.pad_id)
    rows = []
    masks = []
    for y in range(int(grid)):
        for x in range(int(grid)):
            s = f"pos({x},{y})"
            ids = tokenizer.encode(s, add_special=True)
            if len(ids) > int(seq_len):
                ids = ids[: int(seq_len)]
                if int(tokenizer.eos_id) not in ids:
                    ids[-1] = int(tokenizer.eos_id)
            m = [1] * len(ids)
            if len(ids) < int(seq_len):
                n = int(seq_len) - len(ids)
                ids = ids + [pad] * n
                m = m + [0] * n
            rows.append(torch.tensor(ids, dtype=torch.long))
            masks.append(torch.tensor(m, dtype=torch.long))
    return torch.stack(rows, dim=0), torch.stack(masks, dim=0)


def _make_grid_images(*, grid: int, image_size: int) -> torch.Tensor:
    if int(image_size) % int(grid) != 0:
        raise ValueError(f"image_size must be divisible by grid_size (got image_size={image_size}, grid={grid})")
    cell = int(image_size) // int(grid)
    imgs = []
    for y in range(int(grid)):
        for x in range(int(grid)):
            im = torch.zeros((1, int(image_size), int(image_size)), dtype=torch.float32)
            y0 = y * cell
            x0 = x * cell
            im[:, y0 : y0 + cell, x0 : x0 + cell] = 1.0
            imgs.append(im)
    return torch.stack(imgs, dim=0)  # [S,1,H,W]


class Stage6Trainer(nn.Module):
    def __init__(self, config: Stage6Config):
        super().__init__()
        self.config = config

        # Seed early so module initializations (vision encoder / new port) are reproducible.
        torch.manual_seed(int(self.config.seed))
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(int(self.config.seed))
            except Exception:
                pass

        if not os.path.exists(self.config.stage5_ckpt):
            raise FileNotFoundError(f"Stage5 checkpoint not found: {self.config.stage5_ckpt}")
        if not os.path.exists(self.config.stage5_text_file):
            raise FileNotFoundError(f"Stage5 text_file not found: {self.config.stage5_text_file}")

        sd = torch.load(self.config.stage5_ckpt, map_location="cpu")
        dim_q, dim_z, num_charts, decode_state = _infer_stage5_shapes(sd)

        # Build a minimal Stage5 trainer for feature extraction.
        cfg5 = Stage5Config(
            device=str(self.config.device),
            text_file=str(self.config.stage5_text_file),
            max_lines=50,
            eval_lines=5,
            sequence_len=max(8, int(self.config.seq_len)),
            dim_q=int(dim_q),
            dim_z=int(dim_z),
            num_charts=int(num_charts),
            dt=float(self.config.dt),
            evolution_steps=int(self.config.evolution_steps),
            port_decode_state=bool(decode_state),
            # Keep the rest as defaults; we won't train L5 here.
            steps=1,
            batch_size=1,
        )
        _normalize_device(cfg5)
        self.l5 = Stage5Trainer(cfg5)
        self.l5.load_state_dict(torch.load(self.config.stage5_ckpt, map_location=cfg5.device), strict=False)
        self.l5.eval()

        # Add a new vision interface to the SAME core.
        self.l5.entity.add_interface("vision", int(cfg5.dim_q))
        self._maybe_init_vision_port_from_language()

        # Vision encoder (trainable) -> u (dim_q)
        self.vision_encoder = SimpleVisionEncoder(in_channels=1, dim_out=int(cfg5.dim_q)).to(cfg5.device)

        # Precompute paired observations.
        self.tokenizer = ByteTokenizer()
        tok, mask = _make_grid_language_table(self.tokenizer, grid=int(self.config.grid_size), seq_len=int(self.config.seq_len))
        imgs = _make_grid_images(grid=int(self.config.grid_size), image_size=int(self.config.image_size))
        self.lang_tokens = tok.to(device=cfg5.device, non_blocking=True)
        self.lang_mask = mask.to(device=cfg5.device, non_blocking=True)
        self.vision_images = imgs.to(device=cfg5.device, non_blocking=True)

        # Freeze the L5 core + language port completely; train only new vision interface.
        for p in self.l5.parameters():
            p.requires_grad_(False)

        for p in self.vision_encoder.parameters():
            p.requires_grad_(True)
        for p in self.l5.entity.generator.ports["vision"].parameters():
            p.requires_grad_(bool(self.config.train_vision_port))

        self._dim_q = int(cfg5.dim_q)
        self._state_dim = 2 * self._dim_q + 1

        # Train params (used for grad-clip in CUDA-graph mode).
        self._train_params = [p for p in self.vision_encoder.parameters() if p.requires_grad]
        if bool(self.config.train_vision_port):
            self._train_params.extend([p for p in self.l5.entity.generator.ports["vision"].parameters() if p.requires_grad])

        # Cached language prototypes for each latent state.
        self._lang_proto_q = None

        # Performance knobs (match Stage5 style).
        torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def _maybe_init_vision_port_from_language(self) -> None:
        if not bool(getattr(self.config, "init_vision_from_language", True)):
            return
        ports = getattr(getattr(self.l5, "entity", None), "generator", None)
        ports = getattr(ports, "ports", None)
        if ports is None:
            return
        if ("language" not in ports) or ("vision" not in ports):
            return
        src = ports["language"]
        dst = ports["vision"]
        with torch.no_grad():
            for name in ("W_stack", "W_action_q", "W_action_p"):
                if not (hasattr(src, name) and hasattr(dst, name)):
                    continue
                a = getattr(src, name)
                b = getattr(dst, name)
                if hasattr(a, "shape") and hasattr(b, "shape") and tuple(a.shape) == tuple(b.shape):
                    b.copy_(a)

    def _adapter_payload(self) -> Dict[str, Any]:
        # NOTE: keep this payload small to avoid IO stalls.
        return {
            "type": "stage6_adapter_v1",
            "stage5_ckpt": str(self.config.stage5_ckpt),
            "config": asdict(self.config),
            "vision_encoder": self.vision_encoder.state_dict(),
            "vision_port": self.l5.entity.generator.ports["vision"].state_dict(),
        }

    def save_checkpoint(self, path: str) -> None:
        if bool(getattr(self.config, "save_full_state", False)):
            torch.save(self.state_dict(), path)
        else:
            torch.save(self._adapter_payload(), path)

    def load_checkpoint(self, path: str) -> None:
        obj = torch.load(path, map_location=str(self.config.device))
        if isinstance(obj, dict) and obj.get("type", "") == "stage6_adapter_v1":
            self.vision_encoder.load_state_dict(obj["vision_encoder"], strict=True)
            self.l5.entity.generator.ports["vision"].load_state_dict(obj["vision_port"], strict=True)
            return
        if isinstance(obj, dict) and ("vision_encoder" in obj) and ("vision_port" in obj):
            self.vision_encoder.load_state_dict(obj["vision_encoder"], strict=True)
            self.l5.entity.generator.ports["vision"].load_state_dict(obj["vision_port"], strict=True)
            return
        self.load_state_dict(obj, strict=False)

    @torch.no_grad()
    def _compute_language_prototypes(self) -> torch.Tensor:
        if self._lang_proto_q is not None:
            return self._lang_proto_q

        self.l5.eval()
        tokens = self.lang_tokens
        mask = self.lang_mask
        B, L = tokens.shape
        state = torch.zeros((B, self._state_dim), device=tokens.device, dtype=torch.float32)

        evo = int(self.config.evolution_steps)
        dt = float(self.config.dt)
        for t in range(int(L)):
            u = self.l5.port.encode(tokens[:, t])
            u = u * mask[:, t].to(dtype=u.dtype).unsqueeze(1)
            for _ in range(evo):
                out = self.l5.entity.forward_tensor(
                    state_flat=state,
                    u_dict={"language": u},
                    dt=dt,
                    prev_chart_weights=None,
                    prediction_error=None,
                    detach_next_prev_weights=True,
                    compute_action=False,
                    router_context=None,
                    skip_free_energy=True,
                )
                state = out["next_state_flat"]

        self._lang_proto_q = state[:, : self._dim_q].detach()
        return self._lang_proto_q

    def _rollout_vision_q(self, images: torch.Tensor) -> torch.Tensor:
        B = int(images.shape[0])
        device = images.device
        state = torch.zeros((B, self._state_dim), device=device, dtype=torch.float32)

        u = self.vision_encoder(images)
        u = torch.tanh(u)

        total = int(self.config.vision_steps) * int(self.config.evolution_steps)
        dt = float(self.config.dt)
        for _ in range(max(1, total)):
            out = self.l5.entity.forward_tensor(
                state_flat=state,
                u_dict={"vision": u},
                dt=dt,
                prev_chart_weights=None,
                prediction_error=None,
                detach_next_prev_weights=True,
                compute_action=False,
                router_context=None,
                skip_free_energy=True,
            )
            state = out["next_state_flat"]

        return state[:, : self._dim_q]

    @torch.no_grad()
    def evaluate_alignment(self, *, noisy: bool, repeats: int) -> Dict[str, float]:
        proto = self._compute_language_prototypes()  # [S,D]
        S = int(proto.shape[0])
        r = max(1, int(repeats))
        images = self.vision_images

        # Repeat each latent r times to estimate noisy accuracy stably.
        idx = torch.arange(S, device=images.device).repeat_interleave(r)  # [S*r]
        img = images.index_select(0, idx)
        if noisy:
            std = float(self.config.noise_std)
            if std > 0:
                img = (img + std * torch.randn_like(img)).clamp(0.0, 1.0)

        qv = self._rollout_vision_q(img)  # [S*r,D]
        dist = torch.cdist(qv, proto)  # [S*r,S]
        pred = dist.argmin(dim=1)
        acc = (pred == idx).to(dtype=torch.float32).mean()

        target = proto.index_select(0, idx)
        mse = F.mse_loss(qv, target)
        cos = F.cosine_similarity(qv, target, dim=1).mean()

        return {
            "acc": float(acc.item()),
            "mse": float(mse.item()),
            "cos": float(cos.item()),
            "n": int(idx.numel()),
        }

    @torch.no_grad()
    def evaluate_L6_gate(self) -> Dict[str, Any]:
        profile = str(self.config.gate_profile).lower()
        thr = {
            "gate_acc_clean": float(self.config.gate_acc_clean),
            "gate_acc_noisy": float(self.config.gate_acc_noisy),
        }

        clean = self.evaluate_alignment(noisy=False, repeats=1)
        noisy = self.evaluate_alignment(noisy=True, repeats=int(self.config.eval_repeats))

        passed_components = {
            "clean": bool(clean["acc"] >= thr["gate_acc_clean"]),
            "noisy": bool(noisy["acc"] >= thr["gate_acc_noisy"]) if profile == "robust" else True,
        }
        passed = bool(all(passed_components.values()))

        return {
            "profile": profile,
            "clean": clean,
            "noisy": noisy,
            "thresholds": thr,
            "passed_components": passed_components,
            "passed": passed,
        }

    def train_loop(self) -> Dict[str, Any]:
        use_cudagraph = (
            bool(getattr(self.config, "use_cuda_graph", False))
            and torch.cuda.is_available()
            and str(self.config.device).startswith("cuda")
        )
        if use_cudagraph:
            return self._train_loop_cuda_graph()

        print(f"Starting Stage 6 Training on {self.config.device}...")
        print(f"Config: {self.config}")
        print("L6: plural interfaces (vision aligned to language prototypes)")

        os.makedirs(self.config.save_dir, exist_ok=True)
        best_path = os.path.join(self.config.save_dir, "stage6_best.pt")
        final_path = os.path.join(self.config.save_dir, "stage6_final.pt")

        params = list(self._train_params)
        opt = torch.optim.AdamW(params, lr=float(self.config.lr), weight_decay=float(self.config.weight_decay))

        proto = self._compute_language_prototypes()  # cache
        S = int(proto.shape[0])
        images = self.vision_images

        best_loss = float("inf")
        save_best_every = int(getattr(self.config, "save_best_every", 0) or 0)
        if save_best_every <= 0:
            save_best_every = int(self.config.log_every)

        loss_type = str(getattr(self.config, "loss_type", "mse") or "mse").strip().lower()
        tau = float(getattr(self.config, "loss_tau", 1.0) or 1.0)
        mse_w = float(getattr(self.config, "loss_mse_weight", 0.0) or 0.0)
        if tau <= 0.0:
            tau = 1.0

        start = time.time()
        for step in range(1, int(self.config.steps) + 1):
            opt.zero_grad(set_to_none=True)

            idx = torch.randint(0, S, (int(self.config.batch_size),), device=images.device)
            img = images.index_select(0, idx)
            target = proto.index_select(0, idx)

            qv = self._rollout_vision_q(img)
            if loss_type == "mse":
                loss = F.mse_loss(qv, target)
            elif loss_type in ("ce", "ce+mse", "cemse", "ce_mse"):
                dist = torch.cdist(qv, proto)
                logits = -dist / float(tau)
                loss = F.cross_entropy(logits, idx)
                if "mse" in loss_type and mse_w > 0.0:
                    loss = loss + float(mse_w) * F.mse_loss(qv, target)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
            loss.backward()

            clip = float(self.config.grad_clip)
            if clip > 0:
                nn.utils.clip_grad_norm_(params, max_norm=clip)

            opt.step()

            if step % int(self.config.log_every) == 0 or step == 1:
                with torch.no_grad():
                    dist = torch.cdist(qv, proto)
                    pred = dist.argmin(dim=1)
                    acc = (pred == idx).to(dtype=torch.float32).mean().item()
                elapsed = time.time() - start
                print(f"Step {step}: loss={loss.item():.6f} acc={acc:.3f} time={elapsed:.1f}s", flush=True)
                if float(loss.item()) < best_loss and (step % save_best_every == 0):
                    best_loss = float(loss.item())
                    self.save_checkpoint(best_path)

        # Save final and print gate.
        try:
            if os.path.exists(best_path):
                self.load_checkpoint(best_path)
        except Exception:
            pass

        self.save_checkpoint(final_path)
        gate = self.evaluate_L6_gate()
        print("=== L6 Gate ===")
        print("Profile:", gate["profile"])
        print("Clean:", gate["clean"])
        print("Noisy:", gate["noisy"])
        print("Thresholds:", gate["thresholds"])
        print("Passed:", gate["passed"], gate["passed_components"])
        return gate

    def _clip_grad_norm_capturable_(self, max_norm: float = 1.0, eps: float = 1e-6) -> None:
        max_norm = float(max_norm)
        if max_norm <= 0.0:
            return
        grads = [p.grad for p in self._train_params if (p.grad is not None)]
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

    def _train_loop_cuda_graph(self) -> Dict[str, Any]:
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
            print(f"Starting Stage 6 Training (CUDA Graph) on {device}...")
            print(f"Config: {self.config}")
            print("L6: plural interfaces (vision aligned to language prototypes)")

            os.makedirs(self.config.save_dir, exist_ok=True)
            best_path = os.path.join(self.config.save_dir, "stage6_best.pt")
            final_path = os.path.join(self.config.save_dir, "stage6_final.pt")

            params = list(self._train_params)
            opt = torch.optim.AdamW(params, lr=float(self.config.lr), weight_decay=float(self.config.weight_decay), foreach=True)

            proto = self._compute_language_prototypes()
            images = self.vision_images
            S = int(proto.shape[0])

            B = int(self.config.batch_size)
            H = int(self.config.image_size)
            W = int(self.config.image_size)
            Dq = int(self._dim_q)

            img_buf = torch.empty((B, 1, H, W), device=device, dtype=torch.float32)
            tgt_buf = torch.empty((B, Dq), device=device, dtype=torch.float32)
            idx_buf = torch.empty((B,), device=device, dtype=torch.long)

            loss_buf = torch.zeros((), device=device)
            acc_buf = torch.zeros((), device=device)

            loss_type = str(getattr(self.config, "loss_type", "mse") or "mse").strip().lower()
            tau = float(getattr(self.config, "loss_tau", 1.0) or 1.0)
            mse_w = float(getattr(self.config, "loss_mse_weight", 0.0) or 0.0)
            if tau <= 0.0:
                tau = 1.0
            use_ce = loss_type in ("ce", "ce+mse", "cemse", "ce_mse")
            use_mse_aux = ("mse" in loss_type) and (loss_type != "mse") and (mse_w > 0.0)

            def refill_buffers() -> None:
                idx = torch.randint(0, S, (B,), device=device)
                idx_buf.copy_(idx)
                img_buf.copy_(images.index_select(0, idx))
                tgt_buf.copy_(proto.index_select(0, idx))

            def fwd_bwd_step() -> None:
                opt.zero_grad(set_to_none=False)
                qv = self._rollout_vision_q(img_buf)
                dist_for_acc = None
                if use_ce:
                    dist = torch.cdist(qv, proto)
                    logits = -dist / float(tau)
                    loss = F.cross_entropy(logits, idx_buf)
                    if use_mse_aux:
                        loss = loss + float(mse_w) * F.mse_loss(qv, tgt_buf)
                    dist_for_acc = dist.detach()
                else:
                    loss = F.mse_loss(qv, tgt_buf)
                loss_buf.copy_(loss.detach())
                # Lightweight training accuracy proxy (nearest prototype in q-space).
                if dist_for_acc is None:
                    dist_for_acc = torch.cdist(qv.detach(), proto)
                pred = dist_for_acc.argmin(dim=1)
                acc = (pred == idx_buf).to(dtype=torch.float32).mean()
                acc_buf.copy_(acc.detach())
                loss.backward()
                clip = float(getattr(self.config, "grad_clip", 0.0) or 0.0)
                if clip > 0.0:
                    self._clip_grad_norm_capturable_(clip)

            warmup = int(getattr(self.config, "cuda_graph_warmup", 3) or 3)
            warmup = max(1, warmup)
            for _ in range(warmup):
                refill_buffers()
                fwd_bwd_step()
                opt.step()

            torch.cuda.synchronize()
            pool = torch.cuda.graphs.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                fwd_bwd_step()

            best_loss = float("inf")
            save_best_every = int(getattr(self.config, "save_best_every", 0) or 0)
            if save_best_every <= 0:
                save_best_every = int(self.config.log_every)

            start = time.time()
            for step in range(1, int(self.config.steps) + 1):
                refill_buffers()
                graph.replay()
                opt.step()

                if step % int(self.config.log_every) == 0 or step == 1:
                    torch.cuda.synchronize()
                    elapsed = time.time() - start
                    lv = float(loss_buf.item())
                    av = float(acc_buf.item())
                    print(f"Step {step}: loss={lv:.6f} acc={av:.3f} time={elapsed:.1f}s", flush=True)
                    if (step % save_best_every == 0) and (lv < best_loss):
                        best_loss = lv
                        self.save_checkpoint(best_path)

            # Load best (if any), save final adapter, then evaluate gate.
            try:
                if os.path.exists(best_path):
                    self.load_checkpoint(best_path)
            except Exception:
                pass

            self.save_checkpoint(final_path)
            gate = self.evaluate_L6_gate()
            print("=== L6 Gate ===")
            print("Profile:", gate["profile"])
            print("Clean:", gate["clean"])
            print("Noisy:", gate["noisy"])
            print("Thresholds:", gate["thresholds"])
            print("Passed:", gate["passed"], gate["passed_components"])
            return gate
        finally:
            if prev_autograd_mt is not None:
                try:
                    torch.autograd.set_multithreading_enabled(prev_autograd_mt)
                except Exception:
                    pass


if __name__ == "__main__":
    cfg = _apply_env_overrides(Stage6Config())
    trainer = Stage6Trainer(cfg)
    trainer.train_loop()
