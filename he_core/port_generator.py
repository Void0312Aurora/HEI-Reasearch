import torch
import torch.nn as nn
from typing import Callable, Optional
from he_core.state import ContactState
from he_core.generator import BaseGenerator, DissipativeGenerator

class PortCoupling(nn.Module):
    """
    Defines the shape B(q) in the coupling term <u, B(q)>.
    Default: B(q) = -q (Linear Force Coupling).
    Learnable: B(q) = W q (Linear Map).
    Mixture: B(q) = sum w_k * W_k q (Gated Linear Map).
    """
    def __init__(
        self,
        dim_q: int,
        dim_u: int,
        learnable: bool = False,
        num_charts: int = 1,
        top_k: int = 0,
        topk_impl: str = "grouped",
    ):
        super().__init__()
        self.dim_q = dim_q
        self.dim_u = dim_u
        self.learnable = learnable
        self.num_charts = num_charts
        self.top_k = int(top_k or 0)
        self.topk_impl = str(topk_impl or "grouped")
        
        if learnable:
            # We need K matrices if num_charts > 1
            # Dictionary of matrices? Or tensor?
            # Tensor (K, dim_u, dim_q)
            self.W_stack = nn.Parameter(torch.Tensor(num_charts, dim_u, dim_q))
            
            # Init
            with torch.no_grad():
                 # Initialize all near Identity/Negative Identity
                 for k in range(num_charts):
                     if dim_q == dim_u:
                         self.W_stack[k].copy_(-torch.eye(dim_q) + torch.randn(dim_q, dim_q)*0.01)
                     else:
                         # Random initialization for non-square
                         nn.init.xavier_normal_(self.W_stack[k])
        
        if self.learnable:
            # We add an Action Readout Matrix W_out: (Dim_Q -> Dim_U)
            # Action a = W_out q
            # Using same shape as W_stack but separate param?
            # Let's keep it simple: A separate Parameter.
            # self.W_out = nn.Parameter(torch.Tensor(num_charts, dim_u, dim_q))
            # Actually, let's just reuse W_stack transpose if we want symmetry?
            # No, Active Inference usually implies a separate "Policy" or "Reflex".
            # Action a = W_action_q q + W_action_p p
            self.W_action_q = nn.Parameter(torch.Tensor(num_charts, dim_u, dim_q))
            self.W_action_p = nn.Parameter(torch.Tensor(num_charts, dim_u, dim_q))
            with torch.no_grad():
                nn.init.normal_(self.W_action_q, std=0.01)
                nn.init.normal_(self.W_action_p, std=0.01)

    def forward(self, q: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns B(q).
        """
        if not self.learnable:
            return -q # Assumes dim_q == dim_u

        weights_provided = weights is not None
        if weights is None:
            weights = torch.ones(q.shape[0], self.num_charts, device=q.device, dtype=q.dtype) / self.num_charts

        # Sparse mixture over charts (atlas locality): only keep top-k charts per sample.
        if weights_provided and self.top_k > 0 and self.top_k < self.num_charts:
            if self.topk_impl == "dense":
                if self.top_k == 1:
                    # Fast path: top-1 gating via argmax + gather (avoids per-step topk/scatter overhead).
                    idx = weights.argmax(dim=1)  # [B]
                    Y_all = torch.einsum("kuq,bq->bku", self.W_stack, q)  # [B,K,Du]
                    return Y_all.gather(1, idx.view(-1, 1, 1).expand(-1, 1, self.dim_u)).squeeze(1)
                w_vals, idx = torch.topk(weights, k=self.top_k, dim=1)  # [B,k]
                w_norm = w_vals / w_vals.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B,k]
                w_masked = torch.zeros_like(weights)
                w_masked.scatter_(1, idx, w_norm)

                # Dense compute (no Python loops) + masked top-k weights.
                Y_all = torch.einsum("kuq,bq->bku", self.W_stack, q)  # [B,K,Du]
                return (Y_all * w_masked.unsqueeze(2)).sum(dim=1)

            # NOTE: avoid materializing W_stack[idx] as [B,k,Du,Dq] (can OOM for large dims).
            # We group by selected chart id and accumulate with index_add_.
            w_vals, idx = torch.topk(weights, k=self.top_k, dim=1)  # [B,k]
            w_norm = w_vals / w_vals.sum(dim=1, keepdim=True).clamp(min=1e-8)

            batch_size = q.shape[0]
            k = int(self.top_k)
            idx_flat = idx.reshape(-1)  # [B*k]
            w_flat = w_norm.reshape(-1)  # [B*k]
            b_idx = torch.arange(batch_size, device=q.device).repeat_interleave(k)  # [B*k]

            out = torch.zeros(batch_size, self.dim_u, device=q.device, dtype=q.dtype)
            # Important: avoid `.tolist()` / `.any()` on CUDA tensors, which forces a device sync.
            for chart in range(self.num_charts):
                mask = idx_flat == chart
                rows = b_idx[mask]
                if rows.numel() == 0:
                    continue
                q_sub = q.index_select(0, rows)  # [n,Dq]
                y_sub = torch.matmul(q_sub, self.W_stack[chart].transpose(0, 1))  # [n,Du]
                y_sub = y_sub * w_flat[mask].unsqueeze(1)
                out.index_add_(0, rows, y_sub)
            return out

        # Dense mixture (all charts)
        Y_all = torch.einsum('kuq,bq->bku', self.W_stack, q)  # [B, K, Du]
        return (Y_all * weights.unsqueeze(2)).sum(dim=1)

    def get_action(self, q: torch.Tensor, p: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns Action a(q, p).
        """
        if not self.learnable:
            return torch.zeros(q.shape[0], self.dim_u, device=q.device)

        weights_provided = weights is not None
        if weights is None:
            weights = torch.ones(q.shape[0], self.num_charts, device=q.device, dtype=q.dtype) / self.num_charts

        # Sparse mixture over charts (atlas locality)
        if weights_provided and self.top_k > 0 and self.top_k < self.num_charts:
            if self.topk_impl == "dense":
                if self.top_k == 1:
                    idx = weights.argmax(dim=1)  # [B]
                    Aq_all = torch.einsum("kuq,bq->bku", self.W_action_q, q)
                    if p is not None:
                        Ap_all = torch.einsum("kuq,bq->bku", self.W_action_p, p)
                        A_all = Aq_all + Ap_all
                    else:
                        A_all = Aq_all
                    action_raw = A_all.gather(1, idx.view(-1, 1, 1).expand(-1, 1, self.dim_u)).squeeze(1)
                    return 5.0 * torch.tanh(action_raw)
                w_vals, idx = torch.topk(weights, k=self.top_k, dim=1)  # [B,k]
                w_norm = w_vals / w_vals.sum(dim=1, keepdim=True).clamp(min=1e-8)
                w_masked = torch.zeros_like(weights)
                w_masked.scatter_(1, idx, w_norm)

                Aq_all = torch.einsum("kuq,bq->bku", self.W_action_q, q)
                if p is not None:
                    Ap_all = torch.einsum("kuq,bq->bku", self.W_action_p, p)
                    A_all = Aq_all + Ap_all
                else:
                    A_all = Aq_all
                action_raw = (A_all * w_masked.unsqueeze(2)).sum(dim=1)
                return 5.0 * torch.tanh(action_raw)

            w_vals, idx = torch.topk(weights, k=self.top_k, dim=1)  # [B,k]
            w_norm = w_vals / w_vals.sum(dim=1, keepdim=True).clamp(min=1e-8)

            batch_size = q.shape[0]
            k = int(self.top_k)
            idx_flat = idx.reshape(-1)  # [B*k]
            w_flat = w_norm.reshape(-1)  # [B*k]
            b_idx = torch.arange(batch_size, device=q.device).repeat_interleave(k)  # [B*k]

            out = torch.zeros(batch_size, self.dim_u, device=q.device, dtype=q.dtype)
            # Important: avoid `.tolist()` / `.any()` on CUDA tensors, which forces a device sync.
            for chart in range(self.num_charts):
                mask = idx_flat == chart
                rows = b_idx[mask]
                if rows.numel() == 0:
                    continue
                q_sub = q.index_select(0, rows)
                a_sub = torch.matmul(q_sub, self.W_action_q[chart].transpose(0, 1))
                if p is not None:
                    p_sub = p.index_select(0, rows)
                    a_sub = a_sub + torch.matmul(p_sub, self.W_action_p[chart].transpose(0, 1))
                a_sub = a_sub * w_flat[mask].unsqueeze(1)
                out.index_add_(0, rows, a_sub)

            return 5.0 * torch.tanh(out)

        # Dense mixture
        Aq_all = torch.einsum('kuq,bq->bku', self.W_action_q, q)
        if p is not None:
            Ap_all = torch.einsum('kuq,bq->bku', self.W_action_p, p)
            A_all = Aq_all + Ap_all
        else:
            A_all = Aq_all

        Action_Raw = (A_all * weights.unsqueeze(2)).sum(dim=1)
        # Apply Tanh to bound action
        return 5.0 * torch.tanh(Action_Raw)

class PortCoupledGenerator(nn.Module):
    """
    Template 3: Hamiltonian with Multi-Port Coupling.
    H(x, u, t) = H_int(x) + sum_i <u_i, B_i(q)>
    """
    def __init__(
        self,
        internal_generator: BaseGenerator,
        dim_u: int,
        learnable_coupling: bool = False,
        num_charts: int = 1,
        top_k: int = 0,
        topk_impl: str = "grouped",
    ):
        super().__init__()
        self.internal = internal_generator
        self.dim_u = dim_u
        self.dim_q = internal_generator.dim_q
        self.num_charts = num_charts
        self.learnable = learnable_coupling
        self.top_k = int(top_k or 0)
        self.topk_impl = str(topk_impl or "grouped")
        
        # We store ports in a ModuleDict
        self.ports = nn.ModuleDict({
            'default': PortCoupling(
                self.dim_q,
                dim_u,
                learnable=learnable_coupling,
                num_charts=num_charts,
                top_k=self.top_k,
                topk_impl=self.topk_impl,
            )
        })
        
    def add_port(self, name: str, dim_u: int):
        self.ports[name] = PortCoupling(
            self.dim_q,
            dim_u,
            learnable=self.learnable,
            num_charts=self.num_charts,
            top_k=self.top_k,
            topk_impl=self.topk_impl,
        )

    def get_h_port(self, state: ContactState, port_name: str, u_val: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes <u, B(q)> for a specific port.
        """
        if port_name not in self.ports:
            return torch.zeros(state.batch_size, 1, device=state.device)
            
        B_q = self.ports[port_name](state.q, weights=weights)
        return (u_val * B_q).sum(dim=1, keepdim=True)

    def get_action(self, s: 'ContactState', port_name: str = 'default', weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get action from specific port, using state s (q and p).
        """
        if port_name in self.ports:
            return self.ports[port_name].get_action(s.q, s.p, weights=weights) # Pass s.p
        else:
            return torch.zeros(s.q.shape[0], self.ports[list(self.ports.keys())[0]].dim_u, device=s.q.device)

    def forward(self, state: ContactState, u_ext: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Backward compatibility for single port usage.
        Assumes u_ext is for the 'default' port.
        """
        H_int = self.internal(state)
        if u_ext is None:
            return H_int
        
        H_port = self.get_h_port(state, 'default', u_ext, weights=weights)
        return H_int + H_port
