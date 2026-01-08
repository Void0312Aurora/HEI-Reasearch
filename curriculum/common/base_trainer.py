import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass

from he_core.soul_entity import SoulEntity, create_soul_entity
from he_core.state import ContactState

@dataclass
class CurriculumConfig:
    """Base configuration for all curriculum stages"""
    # Entity dims
    dim_q: int = 64
    dim_z: int = 16
    num_charts: int = 1

    # Numerics (curriculum default: no projection, fail-fast)
    integrator_method: str = "semi"
    damping: float = 0.1
    integrator_substeps: int = 1
    integrator_gamma_clip: float = 0.0
    sanitize_nonfinite: bool = False
    strict_nonfinite: bool = True
    q_clip_norm: float = 0.0
    p_clip_norm: float = 0.0
    s_clip_abs: float = 0.0
    transport_threshold: float = 0.0
    router_tau: float = 0.0
    router_context_dim: int = 0
    # Port coupling locality (atlas top-k): keep only top-k charts per sample in B(q).
    # 0 means dense mixture (all charts).
    port_top_k: int = 0
    port_topk_impl: str = "grouped"

    # L1 dissipativity bounds (AdaptiveDissipativeGenerator α(q) ∈ [alpha_min, alpha_max])
    alpha_min: float = 0.2
    alpha_max: float = 1.0
    
    # Training
    lr: float = 1e-4
    batch_size: int = 32
    steps: int = 1000
    dt: float = 0.1
    
    # Saving
    save_dir: str = "checkpoints/curriculum"
    log_every: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class BaseCurriculumTrainer(nn.Module):
    """Base trainer for curriculum stages"""
    def __init__(self, config: CurriculumConfig):
        super().__init__()
        self.config = config
        
        # Initialize Entity
        entity_config = {
            'dim_q': config.dim_q,
            'dim_u': config.dim_q, # Default u dim matches q
            'dim_z': config.dim_z,
            'num_charts': config.num_charts,
            # Default physics params
            'stiffness': getattr(config, 'stiffness', 0.1),
            'contact_stiffness': getattr(config, 'contact_stiffness', 0.1),
            'hyperbolic_c': getattr(config, 'hyperbolic_c', 0.1),
            'alpha_min': float(getattr(config, 'alpha_min', 0.2)),
            'alpha_max': float(getattr(config, 'alpha_max', 1.0)),
            'integrator_method': getattr(config, 'integrator_method', 'semi'),
            'damping': float(getattr(config, 'damping', 0.1)),
            'integrator_substeps': int(getattr(config, 'integrator_substeps', 1) or 1),
            'integrator_gamma_clip': float(getattr(config, 'integrator_gamma_clip', 0.0) or 0.0),
            # Numerical stability policy (optional)
            'sanitize_nonfinite': bool(getattr(config, 'sanitize_nonfinite', False)),
            'strict_nonfinite': bool(getattr(config, 'strict_nonfinite', True)),
            'q_clip_norm': getattr(config, 'q_clip_norm', 0.0),
            'p_clip_norm': getattr(config, 'p_clip_norm', 0.0),
            's_clip_abs': getattr(config, 's_clip_abs', 0.0),
            # L2/L1 carrier controls
            'transport_threshold': float(getattr(config, 'transport_threshold', 0.0) or 0.0),
            'router_tau': float(getattr(config, 'router_tau', 0.0) or 0.0),
            'router_context_dim': int(getattr(config, 'router_context_dim', 0) or 0),
            # Atlas locality for port coupling (optional)
            'port_top_k': int(getattr(config, 'port_top_k', 0) or 0),
            'port_topk_impl': str(getattr(config, 'port_topk_impl', 'grouped') or 'grouped'),
        }
        self.entity = create_soul_entity(entity_config)
        self.entity.to(config.device)
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.config.device))
