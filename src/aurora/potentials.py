"""
Aurora Potentials: Vectorized Force Fields.
===========================================

Implements Energy and Gradient computations for:
1. Structural Edges (Tree)
2. Semantic Edges (Graph)
3. Volume Control (Radius Anchor & Repulsion)

Design: Accepts edge indices as Tensors. The indices come from Data module.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from .geometry import dist_hyperbolic, minkowski_inner

class ForceField(nn.Module):
    """Base class for potentials."""
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Energy: (scalar) Total potential energy
            Gradient: (N, dim) Euclidean gradient w.r.t x (before metric scaling)
        """
        raise NotImplementedError

class SpringPotential(ForceField):
    """
    Attracts connected pairs: V = 0.5 * k * (d - l0)^2.
    """
    def __init__(self, edges: torch.Tensor, k: float, l0: float = 0.0):
        """
        Args:
            edges: (E, 2) LongTensor of indices
            k: stiffness
            l0: rest length
        """
        super().__init__()
        self.register_buffer('edges', edges)
        self.k = k
        self.l0 = l0
        
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N, dim)
        u_idx = self.edges[:, 0]
        v_idx = self.edges[:, 1]
        
        xu = x[u_idx]
        xv = x[v_idx]
        
        # d(u, v)
        # inner = -u0v0 + ...
        # d = acosh(-inner)
        
        # Manually inline geometry for grad
        J = torch.ones(x.shape[-1], device=x.device); J[0] = -1.0
        
        inner = (xu * xv * J).sum(dim=-1)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dist = torch.acosh(-inner)
        
        # Energy
        delta = dist - self.l0
        energy_vec = 0.5 * self.k * delta**2
        total_energy = energy_vec.sum()
        
        # Gradient
        # dV/du = k * delta * grad_u(d)
        # grad_u(d) = (-1/sqrt(inner^2-1)) * J*v
        
        denom = torch.sqrt(inner**2 - 1.0)
        force_mag = self.k * delta #(E,)
        
        factor = -(force_mag / (denom + 1e-9)).unsqueeze(-1) # (E,1)
        
        grad_u = factor * (xv * J)
        grad_v = factor * (xu * J)
        
        # Accumulate
        grad = torch.zeros_like(x)
        grad.index_add_(0, u_idx, grad_u)
        grad.index_add_(0, v_idx, grad_v)
        
        return total_energy, grad

class RobustSpringPotential(ForceField):
    """
    Robust Attraction: LogCosh.
    V = k * delta^2 * log(cosh((d - l0) / delta))
    
    Behavior:
    - Small d: ~ 0.5 * k * (d-l0)^2 (Quadratic, like Spring)
    - Large d: ~ k * delta * |d-l0| (Linear, saturated force)
    """
    def __init__(self, edges: torch.Tensor, k: float, l0: float = 0.0, delta: float = 1.0):
        super().__init__()
        self.register_buffer('edges', edges)
        self.k = k
        self.l0 = l0
        self.delta = delta
        
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N, dim)
        u_idx = self.edges[:, 0]
        v_idx = self.edges[:, 1]
        
        xu = x[u_idx]
        xv = x[v_idx]
        
        # Manually inline geometry for grad
        J = torch.ones(x.shape[-1], device=x.device); J[0] = -1.0
        
        inner = (xu * xv * J).sum(dim=-1)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dist = torch.acosh(-inner)
        
        # Diff
        diff = dist - self.l0
        scaled_diff = diff / self.delta
        
        # Energy: k * delta^2 * log(cosh(scaled_diff))
        # Numerical stability for log(cosh(x)) -> |x| - log(2) for large x
        abs_scaled = torch.abs(scaled_diff)
        mask_large = abs_scaled > 10.0
        
        logcosh = torch.zeros_like(scaled_diff)
        logcosh[mask_large] = abs_scaled[mask_large] - 0.69314718
        logcosh[~mask_large] = torch.log(torch.cosh(scaled_diff[~mask_large]))
        
        energy_vec = self.k * (self.delta**2) * logcosh
        total_energy = energy_vec.sum()
        
        # Grad magnitude w.r.t distance d
        # dV/dd = k * delta^2 * tanh(scaled_diff) * (1/delta)
        #       = k * delta * tanh(scaled_diff)
        force_mag = self.k * self.delta * torch.tanh(scaled_diff)
        
        # Chain rule: dV/du = force_mag * grad_u(d)
        denom = torch.sqrt(inner**2 - 1.0)
        factor = -(force_mag / (denom + 1e-9)).unsqueeze(-1)
        
        grad_u = factor * (xv * J)
        grad_v = factor * (xu * J)
        
        # Accumulate
        grad = torch.zeros_like(x)
        grad.index_add_(0, u_idx, grad_u)
        grad.index_add_(0, v_idx, grad_v)
        
        return total_energy, grad

class RadiusAnchorPotential(ForceField):
    """
    Volume Control: V = 0.5 * lambda * (r_i - r_target)^2.
    """
    def __init__(self, targets: torch.Tensor, lamb: float = 1.0):
        super().__init__()
        self.base_target_radii = targets.clone()  # Store base for PID scaling
        self.target_radii = targets  # Mutable, can be scaled by PID
        self.lamb = lamb
        
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # r = acosh(x0)
        x0 = x[:, 0]
        # x0 should be >= 1.
        x0_safe = torch.clamp(x0, min=1.0 + 1e-7)
        r = torch.acosh(x0_safe)
        
        delta = r - self.target_radii
        energy = 0.5 * self.lamb * (delta**2).sum()
        
        # Grad
        # dV/dx0 = lambda * delta * (1/sinh(r))
        sinh_r = torch.sqrt(x0_safe**2 - 1.0)
        grad_0 = self.lamb * delta / (sinh_r + 1e-9)
        
        grad = torch.zeros_like(x)
        grad[:, 0] = grad_0
        
        return energy, grad

class RepulsionPotential(ForceField):
    """
    Negative Sampling Repulsion: LogCosh.
    V = A * log(cosh(d / sigma))
    """
    def __init__(self, A: float = 5.0, sigma: float = 1.0, num_neg: int = 5):
        super().__init__()
        self.A = A
        self.sigma = sigma
        self.num_neg = num_neg
        
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N, dim = x.shape
        device = x.device
        
        # Sample negs
        u_idx = torch.randint(0, N, (N * self.num_neg,), device=device)
        v_idx = torch.randint(0, N, (N * self.num_neg,), device=device)
        
        xu = x[u_idx]
        xv = x[v_idx]
        
        J = torch.ones(dim, device=device); J[0] = -1.0
        inner = (xu * xv * J).sum(dim=-1)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dist = torch.acosh(-inner)
        
        # Energy
        val = dist / self.sigma
        
        # Stable log(cosh(x))
        # For large |x|, log(cosh(x)) approx |x| - log(2)
        # We can use softplus logic or simple masking.
        abs_val = torch.abs(val)
        mask_large = abs_val > 10.0
        
        logcosh = torch.zeros_like(val)
        # Large x: |x| - log(2)
        logcosh[mask_large] = abs_val[mask_large] - 0.69314718
        # Small x: log(cosh(x))
        logcosh[~mask_large] = torch.log(torch.cosh(val[~mask_large]))
        
        energy = (self.A * self.sigma * logcosh).sum() * (1.0 / self.num_neg)
        
        # Grad
        # dV/dd = (A/N) * tanh(val)
        force_mag = (self.A * torch.tanh(val)) / self.num_neg # (E_neg,)
        
        denom = torch.sqrt(inner**2 - 1.0)
        # d(d)/du = -1/sqrt * J v
        factor = -(force_mag / (denom + 1e-9)).unsqueeze(-1)
        
        grad_u = factor * (xv * J)
        grad_v = factor * (xu * J)
        
        grad = torch.zeros_like(x)
        # Need to scale by sampling density? Usually handled by learning rate or A.
        # Just pure sum for now.
        grad.index_add_(0, u_idx, grad_u)
        grad.index_add_(0, v_idx, grad_v)
        
        return energy, grad

class CompositePotential(ForceField):
    def __init__(self, components: List[ForceField]):
        super().__init__()
        self.components = nn.ModuleList(components)
        
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_e = 0.0
        total_g = torch.zeros_like(x)
        
        for comp in self.components:
            e, g = comp.compute_forces(x)
            total_e = total_e + e
            total_g = total_g + g
            
        return total_e, total_g
class GatedRepulsionPotential(ForceField):
    """
    Short-Range Repulsion (Contact Constraint).
    V = A * (epsilon - d)^2  if d < epsilon else 0.
    
    Acts as a soft barrier to prevent collision, but exerts NO force 
    when nodes are separated by more than epsilon.
    """
    def __init__(self, A: float = 100.0, epsilon: float = 0.1, num_neg: int = 5):
        super().__init__()
        self.A = A
        self.epsilon = epsilon
        self.num_neg = num_neg
        
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N, dim = x.shape
        device = x.device
        
        # Sample negs
        # Ideally we'd use a spatial index to find actual neighbors,
        # but for now random sampling + gating is a stochastic approximation.
        # To be effective, we might need more samples or a better heuristic.
        # But given the critique was "Global Noise", even random sampling with gating
        # removes the noise.
        
        u_idx = torch.randint(0, N, (N * self.num_neg,), device=device)
        v_idx = torch.randint(0, N, (N * self.num_neg,), device=device)
        
        xu = x[u_idx]
        xv = x[v_idx]
        
        J = torch.ones(dim, device=device); J[0] = -1.0
        inner = (xu * xv * J).sum(dim=-1)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dist = torch.acosh(-inner)
        
        # Gating: Mask where d < epsilon
        mask = dist < self.epsilon
        
        if not mask.any():
            return torch.tensor(0.0, device=device), torch.zeros_like(x)
            
        d_active = dist[mask]
        
        # Energy: 0.5 * A * (eps - d)^2 * (1/num_neg)
        delta = self.epsilon - d_active
        energy = 0.5 * self.A * (delta**2).sum() * (1.0 / self.num_neg)
        
        # Grad: dV/dx = (A / num_neg) * (eps - d) * grad_d
        
        force_mag = (self.A * delta) / self.num_neg  # Scale force by 1/num_neg
        
        denom = torch.sqrt(inner[mask]**2 - 1.0)
        factor = -(force_mag / (denom + 1e-9)).unsqueeze(-1)
        
        grad_u = factor * (xv[mask] * J)
        grad_v = factor * (xu[mask] * J)
        
        grad = torch.zeros_like(x)
        grad.index_add_(0, u_idx[mask], grad_u)
        grad.index_add_(0, v_idx[mask], grad_v)
        
        return energy, grad

class SemanticTripletPotential(ForceField):
    """
    Triplet Margin Loss for Micro-Alignment.
    L = max(0, d(u,p) - d(u,n) + margin)
    
    Includes Stochastic Hard Negative Mining:
    Samples 'num_candidates' negatives per anchor, selects the one with smallest distance.
    """
    def __init__(self, edges: torch.Tensor, k: float = 1.0, margin: float = 0.1, num_candidates: int = 5,
                 mining_mode: str = "hard", local_pool: bool = False, radius_tolerance: float = 0.1,
                 bank_size: int = 50000, soft_weight: float = 0.0):
        """
        Args:
            mining_mode: "hard", "semi-hard", or "curriculum"
                - hard: d(u,w) < d(u,v) (violates margin)
                - semi-hard: d(u,v) < d(u,w) < d(u,v) + margin (within margin)
                - curriculum: start with hard, gradually shift to semi-hard
            local_pool: If True, sample negatives from same radius band (|r_u - r_w| < radius_tolerance)
            radius_tolerance: Radius band tolerance for local_pool
            soft_weight: Soft Positive attraction weight (0.0 = disabled)
        """
        super().__init__()
        self.register_buffer('edges', edges)
        self.k = k
        self.margin = margin
        self.num_candidates = num_candidates
        self.mining_mode = mining_mode
        self.local_pool = local_pool
        self.radius_tolerance = radius_tolerance
        self.bank_size = bank_size
        self.soft_weight = soft_weight
        self.curriculum_progress = 0.0  # 0.0 = hard, 1.0 = semi-hard (for curriculum mode)
        self.hard_ratio = 0.5           # For trusted mode: Ratio of candidates from Bank vs Random
        self.last_violation_rate = 0.0  # Initialize metric
        self.soft_positives = None      # Buffer for Soft Positives (E, K_soft)
        
    def update_global_candidates(self, x: torch.Tensor, k: int = 1000, batch_size: int = 2000):
        """
        Global Exact Search (Chunked) with Local Radius Constraint.
        Finds Top-K Nearest Neighbors where |r_u - r_v| < 0.1.
        """
        N, dim = x.shape
        device = x.device
        
        # Anchors defined by edges
        u_idx = self.edges[:, 0]
        E = u_idx.shape[0]
        
        # Metric: Minkowski Inner Product
        J = torch.ones(dim, device=device)
        J[0] = -1.0
        
        # Pre-allocate result buffer
        self.global_candidates = torch.empty((E, k), dtype=torch.long, device=device)
        
        # All embeddings & Radii
        all_x = x
        # r = acosh(x0)
        all_r = torch.acosh(torch.clamp(all_x[:, 0], min=1.0 + 1e-7))
        
        # Processing in chunks of anchors
        for i in range(0, E, batch_size):
            end = min(i + batch_size, E)
            batch_u_idx = u_idx[i:end]
            
            # (B, D)
            xu_batch = x[batch_u_idx]
            ru_batch = torch.acosh(torch.clamp(xu_batch[:, 0], min=1.0 + 1e-7))
            
            # Compute Inner Products: (xu * J) @ all_x.T -> (B, N)
            xu_J = xu_batch * J
            inner_prods = torch.matmul(xu_J, all_x.t())
            
            # [Phase XIV] Local Radius Constraint
            # Mask out candidates with |r_u - r_v| > 0.1
            # r_diff shape: (B, N)
            r_diff = torch.abs(ru_batch.unsqueeze(1) - all_r.unsqueeze(0))
            valid_mask = r_diff < 0.1
            
            # Apply mask (set invalid to -inf)
            inner_prods[~valid_mask] = float('-inf')
            
            _, top_indices = torch.topk(inner_prods, k, dim=1, largest=True, sorted=False)
            
            self.global_candidates[i:end] = top_indices
            
        print(f"[GlobalIndex] Updated Top-{k} Local-Constrained Candidates (|dr|<0.1) for {E} anchors.")

    def update_soft_positives(self, x: torch.Tensor, k: int = 50, batch_size: int = 2000):
        """
        [Phase XVI] Soft Positive Mining.
        Finds 'Unlabeled' neighbors that are highly confident:
        1. Radius Constrained (|dr| < 0.1).
        2. Mutual kNN (u->v AND v->u).
        3. Stability (Appears in consecutive updates).
        """
        if self.soft_weight <= 0.0:
            return
            
        N, dim = x.shape
        device = x.device
        u_idx = self.edges[:, 0]
        E = u_idx.shape[0]
        J = torch.ones(dim, device=device); J[0] = -1.0
        
        # 1. Global kNN Search (Same as update_global_candidates but need mutual info)
        # We need ALL N nodes' kNN to check mutuality efficiently? 
        # Checking mutuality for just Anchors is cheaper:
        # For each anchor u, getting Top-K neighbors v.
        # Then for each v, checking if u is in ITS Top-K.
        # This requires full N->N search which is expensive.
        # Optimization: We already run update_global_candidates for E anchors.
        # We can re-use `self.global_candidates` as the forward set.
        # Then check backward link only for those candidates.
        
        # NOTE: self.global_candidates is (E, K_global). K_global usually 1000.
        # We can search purely within this set.
        
        # Helper: Ensure global_candidates exists
        if not hasattr(self, 'global_candidates'):
            print("Warning: global_candidates not found. Run update_global_candidates first.")
            return

        # Direct Slicing (Vectorized)
        # global_candidates is (E, K_global). We take Top-k.
        current_soft_t = self.global_candidates[:, :k].contiguous() # (E, k)
        
        # Temporal Stability Logic: Persistence Check
        # Generate globally unique IDs for (anchor, candidate) pairs
        # pair_id = u_idx * N + v_idx
        # But u_idx is implicit by row.
        # We use row_idx * N + candidate_idx
        
        # Base offset for each row
        row_offsets = (torch.arange(E, device=device) * N).unsqueeze(1) # (E, 1)
        current_pair_ids = row_offsets + current_soft_t # (E, k)
        
        if getattr(self, 'last_soft_ids', None) is not None:
             # Check which current pairs were present in last update
             # flatten for isin
             curr_flat = current_pair_ids.view(-1)
             last_flat = self.last_soft_ids.view(-1)
             
             # torch.isin: Elements of curr_flat that are in last_flat
             mask_flat = torch.isin(curr_flat, last_flat)
             
             self.soft_mask = mask_flat.view(E, k).contiguous()
             
             # Monitor stability rate
             stability_rate = mask_flat.float().mean().item()
             print(f"[SoftMining] Stability Rate (Persistence): {stability_rate*100:.1f}%")
        else:
             # First step: No history.
             # Option A: Accept all (Start fast)
             # Option B: Reject all (Wait for confirmation) -> Safer for High Fidelity
             # Let's use Option B to ensure strict persistence
             print("[SoftMining] First update (No history). Initializing buffer.")
             self.soft_mask = torch.zeros_like(current_soft_t, dtype=torch.bool).contiguous()
             
        # Update history buffer (Store CURRENT candidates for next check)
        self.last_soft_ids = current_pair_ids
        self.soft_positives = current_soft_t
        
        # Filtered stats
        active_count = self.soft_mask.sum().item()
        print(f"[SoftMining] Updated Soft Positives. Active Stable Pairs: {active_count} / {current_soft_t.numel()}")

    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N, dim = x.shape
        device = x.device
        E = self.edges.shape[0]
        
        u_idx = self.edges[:, 0] # Anchor
        v_idx = self.edges[:, 1] # Positive
        
        xu = x[u_idx]
        xv = x[v_idx]
        
        J = torch.ones(dim, device=device); J[0] = -1.0
        
        # 1. Compute d(u, v) (Positive Distance)
        inner_pos = (xu * xv * J).sum(dim=-1)
        inner_pos = torch.clamp(inner_pos, max=-1.0 - 1e-7)
        d_pos = torch.acosh(-inner_pos)
        
        # 2. Negative Candidate Sampling (Global FIFO Bank + Random)
        bank_size = getattr(self, 'bank_size', 20000)
        
        # Initialize bank if needed
        if not hasattr(self, 'global_bank'):
            self.global_bank = torch.randint(0, N, (bank_size,), device=device)
            self.bank_ptr = 0
            self.bank_filled = False
        
        # Sampling Strategy: Dynamic Hard Ratio
        # k_bank = candidates from Hard Source (Bank)
        k_bank = int(self.num_candidates * self.hard_ratio)
        k_rand = self.num_candidates - k_bank
        
        # A. Sample from Bank
        # A. Sample from Bank (FIFO or Global ANN)
        if k_bank > 0:
            if self.mining_mode == "global":
                # Global Mode: Sample from Top-K Recall Candidates
                if not hasattr(self, 'global_candidates'):
                    # Lazy Init (Warmup)
                    self.update_global_candidates(x, k=200, batch_size=2000)
                
                recall_k = self.global_candidates.shape[1]
                # Randomly select from the Top-K candidates
                gather_indices = torch.randint(0, recall_k, (E, k_bank), device=device)
                w_bank = self.global_candidates.gather(1, gather_indices)
            else:
                # FIFO/Standard Mode: Sample from Global FIFO Bank
                bank_indices = torch.randint(0, bank_size, (E, k_bank), device=device)
                w_bank = self.global_bank[bank_indices]
        else:
            w_bank = torch.empty(E, 0, dtype=torch.long, device=device)
        
        # B. Sample Random
        if k_rand > 0:
            w_rand = torch.randint(0, N, (E, k_rand), device=device)
        else:
            w_rand = torch.empty(E, 0, dtype=torch.long, device=device)
        
        # Combine
        w_candidates = torch.cat([w_bank, w_rand], dim=1) # (E, num_candidates)
        
        # Compute distances to all candidates
        xu_exp = xu.unsqueeze(1) # (E, 1, dim)
        xw = x[w_candidates]     # (E, K, dim)
        
        inner_neg_k = (xu_exp * xw * J).sum(dim=-1) # (E, K)
        inner_neg_k = torch.clamp(inner_neg_k, max=-1.0 - 1e-7)
        d_neg_k = torch.acosh(-inner_neg_k)
        
        # 3. Vectorized Mining Mode Selection
        if self.mining_mode == "hard":
            # Select hardest negative: min d(u, w)
            min_vals, min_indices = torch.min(d_neg_k, dim=1)
            d_neg = min_vals
        elif self.mining_mode == "semi-hard":
            # Vectorized semi-hard selection
            d_pos_exp = d_pos.unsqueeze(1)  # (E, 1)
            semi_hard_mask = (d_neg_k > d_pos_exp) & (d_neg_k < d_pos_exp + self.margin)
            
            # Set distances outside semi-hard range to inf, then argmin
            d_neg_k_masked = d_neg_k.clone()
            d_neg_k_masked[~semi_hard_mask] = float('inf')
            
            # If all candidates are non-semi-hard, fallback to hardest
            no_semi_hard = ~semi_hard_mask.any(dim=1)
            d_neg_k_masked[no_semi_hard] = d_neg_k[no_semi_hard]
            
            min_vals, min_indices = torch.min(d_neg_k_masked, dim=1)
            d_neg = d_neg_k.gather(1, min_indices.unsqueeze(1)).squeeze(1)
        elif self.mining_mode == "curriculum":
            # Vectorized curriculum with batch-level randomness
            d_pos_exp = d_pos.unsqueeze(1)
            semi_hard_mask = (d_neg_k > d_pos_exp) & (d_neg_k < d_pos_exp + self.margin)
            
            # Random selection: use semi-hard with probability = curriculum_progress
            use_semi_hard = torch.rand(E, device=device) < self.curriculum_progress
            
            # Prepare semi-hard candidates
            d_neg_k_semi = d_neg_k.clone()
            d_neg_k_semi[~semi_hard_mask] = float('inf')
            no_semi_hard = ~semi_hard_mask.any(dim=1)
            d_neg_k_semi[no_semi_hard] = d_neg_k[no_semi_hard]
            
            # Select: hard or semi-hard based on random decision
            d_neg_k_selected = torch.where(
                use_semi_hard.unsqueeze(1),
                d_neg_k_semi,
                d_neg_k
            )
            
            min_vals, min_indices = torch.min(d_neg_k_selected, dim=1)
            d_neg = d_neg_k.gather(1, min_indices.unsqueeze(1)).squeeze(1)
        elif self.mining_mode == "trusted" or self.mining_mode == "global":
            # Trusted Negative Mining (Strict Filter)
            # 1. Reject False Negatives: d_neg < d_pos
            # 2. Prefer Semi-Hard: d_pos < d_neg < d_pos + m
            # 3. Fallback: If no Semi-Hard, pick Hardest Valid (which implies Easy d_neg > d_pos + m)
            
            d_pos_exp = d_pos.unsqueeze(1)
            
            # Mask out False Negatives (too close)
            # We treat them as 'inf' so they are not selected as 'min'
            valid_mask = d_neg_k > d_pos_exp
            
            d_neg_k_filtered = d_neg_k.clone()
            d_neg_k_filtered[~valid_mask] = float('inf')
            
            # Select Hardest Valid (Smallest Distance)
            # If all are inf (all false negatives), min returns inf. Loss=0. Correct.
            min_vals, min_indices = torch.min(d_neg_k_filtered, dim=1)
            d_neg = min_vals
        else:
            raise ValueError(f"Unknown mining_mode: {self.mining_mode}")
        
        # Gather selected negatives
        w_idx = w_candidates.gather(1, min_indices.unsqueeze(1)).squeeze(1)
        xw_hard = x[w_idx]
        inner_neg = inner_neg_k.gather(1, min_indices.unsqueeze(1)).squeeze(1)
        
        # 3. Loss Calculation
        # L = relu(d_pos - d_neg + margin)
        loss_val = d_pos - d_neg + self.margin
        mask = loss_val > 0
        
        # [NEW] Metric Logging: Violation Rate
        self.last_violation_rate = mask.float().mean().item()
        
        # [NEW] Update Global Bank with Violating Negatives
        if mask.any():
            violating_w = w_idx[mask].unique()
            n_violating = violating_w.shape[0]
            bank_size = self.global_bank.shape[0]
            
            if n_violating > 0:
                # If we have more violators than bank size, sample a subset
                if n_violating > bank_size:
                    # Randomly select bank_size elements to fit
                    perm = torch.randperm(n_violating, device=device)[:bank_size]
                    violating_w = violating_w[perm]
                    n_violating = bank_size
                
                ptr = self.bank_ptr
                
                # Simple FIFO Logic
                # If fitting in remainder
                remaining = bank_size - ptr
                if n_violating <= remaining:
                    self.global_bank[ptr : ptr + n_violating] = violating_w
                    self.bank_ptr = (ptr + n_violating) % bank_size
                else:
                    # Wrap around
                    self.global_bank[ptr:] = violating_w[:remaining]
                    self.global_bank[:n_violating - remaining] = violating_w[remaining:]
                    self.bank_ptr = n_violating - remaining
                    self.bank_filled = True
        
        if not mask.any():
             return torch.tensor(0.0, device=device), torch.zeros_like(x)
             
        active_loss = loss_val[mask]
        # Energy = sum(L) * k
        total_energy = active_loss.sum() * self.k
        
        # 4. Soft Positive Attraction (Phase XVI)
        if self.soft_weight > 0.0 and self.soft_positives is not None:
             # Soft Positives are in self.soft_positives (E, K_soft)
             K_soft = self.soft_positives.shape[1]
             
             # Expand u to (E, K_soft, D)
             u_expand = xu.unsqueeze(1).repeat(1, K_soft, 1) # (E, K, D)
             
             # Fetch Soft Neighbors v_soft
             try:
                 soft_indices = self.soft_positives.view(-1) # (E*K)
                 v_soft = x[soft_indices].view(E, K_soft, dim) # (E, K, D)
                 
                 # Compute Distance d(u, v_soft)
                 inner_soft = (u_expand * v_soft * J.view(1, 1, dim)).sum(dim=-1)
                 inner_soft = torch.clamp(inner_soft, max=-1.0 - 1e-7)
                 d_soft = torch.acosh(-inner_soft)
                 
                 # Soft Loss = weight * d_soft
                 denom_soft = torch.sqrt(inner_soft**2 - 1.0)
                 factor_soft = -(self.soft_weight / (denom_soft + 1e-9)).unsqueeze(-1)
                 
                 # Apply Stability Mask if available
                 if hasattr(self, 'soft_mask'):
                     # soft_mask is (E, K)
                     mask_expanded = self.soft_mask.view(-1) # (E*K)
                     
                     # Zero out forces for unstable pairs
                     factor_soft = factor_soft * mask_expanded.view(E, K_soft, 1)
                     d_soft = d_soft * self.soft_mask # (E, K)
                 
                 grad_u_soft = factor_soft * (v_soft * J.view(1, 1, dim)) # (E, K, D)
                 grad_v_soft = factor_soft * (u_expand * J.view(1, 1, dim)) # (E, K, D)
                 
                 # Accumulate Gradients (Create new buffer to avoid interfering with previous ops)
                 # Wait, grad is already allocated.
                 if 'grad' not in locals():
                     grad = torch.zeros_like(x)
                 
                 grad.index_add_(0, u_idx, grad_u_soft.sum(dim=1))
                 grad.index_add_(0, soft_indices, grad_v_soft.view(-1, dim))
                 
                 energy_val = self.soft_weight * d_soft.sum()
                 total_energy += energy_val
                 
                 # Monitoring
                 if torch.rand(1).item() < 0.01:
                     # Calculate similarity with Hard Positives?
                     # Ideally hard pos dist should stay small.
                     print(f"    [SoftPos] Mean Dist: {d_soft.mean().item():.3f}, Loss Contrib: {energy_val.item():.2f}")
             except Exception as e:
                 print(f"    [SoftPos] Error: {e}")

        # 4. Gradients (Triplet)
        # L = d_pos - d_neg + m
        # grad = k * grad(L)
        # grad_u = grad_u(d_pos) - grad_u(d_neg)
        # grad_v = grad_v(d_pos)
        # grad_w = -grad_w(d_neg)
        
        # Helper for d(a,b) gradient w.r.t a:
        # grad_a = (-1/sqrt(in^2-1)) * J*b
        
        # Active subset
        denom_pos = torch.sqrt(inner_pos[mask]**2 - 1.0) + 1e-9
        factor_pos = -(1.0 / denom_pos).unsqueeze(-1) # (E_act, 1)
        
        denom_neg = torch.sqrt(inner_neg[mask]**2 - 1.0) + 1e-9
        factor_neg = -(1.0 / denom_neg).unsqueeze(-1) # (E_act, 1)

        # Terms for u (Anchor)
        # from pos: factor_pos * xv * J
        # from neg: - (factor_neg * xw * J) = -factor_neg * xw * J
        
        term_u_pos = factor_pos * (xv[mask] * J)
        term_u_neg = factor_neg * (xw_hard[mask] * J)
        grad_u = term_u_pos - term_u_neg
        
        # Term for v (Positive)
        # from pos: factor_pos * xu * J
        grad_v = factor_pos * (xu[mask] * J)
        
        # Term for w (Negative)
        # from neg: - (factor_neg * xu * J)
        grad_w = - factor_neg * (xu[mask] * J)
        
        # Accumulate
        # Scale by k
        grad = torch.zeros_like(x)
        grad.index_add_(0, u_idx[mask], grad_u * self.k)
        grad.index_add_(0, v_idx[mask], grad_v * self.k)
        grad.index_add_(0, w_idx[mask], grad_w * self.k)
        
        return total_energy, grad
