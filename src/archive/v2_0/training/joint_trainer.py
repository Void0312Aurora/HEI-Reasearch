"""
Aurora Joint Trainer.
=====================

Trains Semantics (J) and Language (U) jointly but with decoupled dynamics.

Dynamics:
1. J-Step: Minimize Energy on Structural Tree + Core Semantics.
   - Objective: J aligns with neighbors in the Tree.
   - Constraint: U is Identity (or fixed) for Structural Edges?
     Actually, Structural edges usually imply U=I.
   - Gradients flow to J.
   
2. U-Step: Minimize Energy on Language Edges (Flow/Phrase/Dep).
   - Objective: U_uv rotates J_u to J_v.
   - Constraint: J is Fixed (Detached). We fit U to J.
   - Loss: Contrastive (Pos Energy < Neg Energy).
   
This prevents "Language Flow" (e.g. "eat" -> "apple") from pulling "eat" and "apple" to the same Semantic point (J_eat ~ J_apple).
Instead, it forces U_eat_apple to bridge the gap.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from ..gauge import GaugeField

class JointTrainer:
    def __init__(self, gauge_field: GaugeField, x: torch.Tensor, J: torch.Tensor, 
                 lr_J: float = 0.001, lr_U: float = 0.001, device='cuda'):
        self.gauge_field = gauge_field
        self.x = x
        self.J = J
        self.device = device
        
        # J is a Parameter (if we want optimizer to handle it)
        # But J is usually a Tensor.
        # If J requires grad:
        if not self.J.requires_grad:
            self.J.requires_grad = True
            
        self.opt_J = optim.Adam([self.J], lr=lr_J)
        self.opt_U = optim.Adam(self.gauge_field.parameters(), lr=lr_U)
        
        # Loss params
        self.margin = 0.5
        
    def train_step(self, struct_edges: List[Tuple[int, int]], lang_batch: List[Tuple[int, int, int, int]]):
        """
        Perform one update step.
        """
        # --- 1. J-Update (Semantics) ---
        # Driven by Structural Skeleton
        # Usually we sample a batch of Structural Edges
        # Or use ALL if small.
        # Let's sample batch matching lang_batch size
        bs = len(lang_batch)
        indices = np.random.choice(len(struct_edges), min(bs, len(struct_edges)), replace=False)
        s_batch = [struct_edges[i] for i in indices]
        
        u_s = torch.tensor([e[0] for e in s_batch], device=self.device)
        v_s = torch.tensor([e[1] for e in s_batch], device=self.device)
        
        self.opt_J.zero_grad()
        
        # Energy: 1 - <J_u, J_v> (Assuming Structural U=I)
        # Or use GaugeField with U=I logic?
        # Simple dot product.
        J_u = self.J[u_s]
        J_v = self.J[v_s]
        align_s = torch.sum(J_u * J_v, dim=-1)
        loss_J = torch.mean(1.0 - align_s)
        
        loss_J.backward()
        self.opt_J.step()
        
        # Renormalize J (Constraint)
        with torch.no_grad():
            self.J.data /= torch.norm(self.J.data, dim=-1, keepdim=True)
            
        # --- 2. U-Update (Language) ---
        # Driven by Language Edges (Flow/Phrase/Dep)
        # J is DETACHED (Frozen for this step)
        
        u_l = torch.tensor([e[0] for e in lang_batch], device=self.device)
        v_l = torch.tensor([e[1] for e in lang_batch], device=self.device)
        r_l = torch.tensor([e[2] for e in lang_batch], device=self.device)
        lbl = torch.tensor([e[3] for e in lang_batch], dtype=torch.float, device=self.device)
        
        self.opt_U.zero_grad()
        J_frozen = self.J.detach()
        
        # Get U
        edges_t = torch.stack([u_l, v_l], dim=1)
        U = self.gauge_field.get_U(x=self.x, edges=edges_t, relation_ids=r_l)
        
        # Transport
        J_u_f = J_frozen[u_l].unsqueeze(-1)
        J_trans = torch.matmul(U, J_u_f).squeeze(-1)
        J_v_f = J_frozen[v_l]
        
        # Alignment
        align_l = torch.sum(J_v_f * J_trans, dim=-1) # Range [-1, 1] approximately
        
        # Contrastive Loss:
        # If Label=1 (Pos): Maximize Align -> Minimize 1-Align
        # If Label=0 (Neg): Minimize Align -> Maximize 1-Align (Result > Margin)
        # Margin Ranking Logic:
        # Loss = Label * (1 - Align) + (1 - Label) * max(0, Align - Margin)
        # Wait, usually for ranking: max(0, -Align_pos + Align_neg + Margin).
        # But here input is mixed.
        # Hinge Loss:
        # Pos: Loss = max(0, 0.9 - Align) -> Encourage Align > 0.9
        # Neg: Loss = max(0, Align - 0.1) -> Encourage Align < 0.1
        
        loss_pos = torch.mean(torch.relu(0.9 - align_l[lbl==1]))
        loss_neg = torch.mean(torch.relu(align_l[lbl==0] - 0.1))
        
        # Budget Penalty (Holonomy approximation or Curvature)
        # For simplicity: L2 of Omega (keep rotations minimal/efficient)
        # But Phase 19 says we want U to be active.
        # Maybe just Contrastive is enough?
        loss_U = loss_pos + loss_neg
        
        loss_U.backward()
        self.opt_U.step()
        
        return loss_J.item(), loss_U.item()
