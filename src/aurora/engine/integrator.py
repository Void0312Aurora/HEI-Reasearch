"""
Aurora Integrator (CCD v3.1)
============================

Implements Axiom 2.4: Lie Group Integrator.
Updates Dynamical State U(t) = (q, p, J, z) on the manifold.

Key Logic:
1. q_{t+1} = Exp_q(v * dt)
2. p_{t+1} = PT_{GM}(p_t) + F * dt  (Levi-Civita Transport)
3. J_{t+1} = PT_{Gauge}(J_t)        (Wong Equation / Gauge Transport)
"""

import torch
import torch.nn as nn
from ..physics import geometry

class LieIntegrator:
    def __init__(self, method='euler'):
        self.method = method
    
    def step(self, q, p, J, m, force_field, dt=0.01, ctx_q=None, ctx_m=None, ctx_J=None):
        """
        Single step integration with context interaction.
        Args:
            q: (B, D) Active particles
            p: (B, D) Tangent vector at q
            J: (B, D, D) Gauge charge
            m: (B, 1) Mass
            force_field: Module with .compute_forces()
            dt: Time step
            ctx_q: (C, D) Context particles (LTM + Active History)
            ctx_m: (C, 1)
            ctx_J: (C, D, D)
        Returns:
            q_next, p_next, J_next
        """
        # 1. Compute forces (Gradient of Potential)
        # We need forces ON q FROM (q + ctx)
        
        if ctx_q is not None:
            # Combine System: [Context, Active]
            # Forces computation usually expects full batch.
            # We want forces on Active indices (last B).
            q_all = torch.cat([ctx_q, q], dim=0)
            m_all = torch.cat([ctx_m, m], dim=0)
            J_all = torch.cat([ctx_J, J], dim=0)
            
            # Compute Full Forces
            F_total, _ = force_field.compute_forces(q_all, m_all, J_all)
            
            # Extract Forces for Active particles
            B = q.size(0)
            F_tan = F_total[-B:]
        else:
            # Local only
            F_tan, _ = force_field.compute_forces(q, m, J)
        
        # 2. Update Momentum (Symplectic-like: p += F * dt)
        gamma = 2.0 # Friction
        p_star = p + (F_tan - gamma * p) * dt
        
        # 3. Update Position (Exponential Map)
        v = p_star 
        q_next = geometry.exp_map(q, v * dt)
        q_next = geometry.check_boundary(q_next)
        
        # 4. Parallel Transport Momentum (Levi-Civita)
        p_next = geometry.parallel_transport(q, q_next, p_star)
        
        # 5. Parallel Transport Gauge Charge (Gauge Connection)
        diff = geometry.mobius_add(-q, q_next)
        pt_gauge = force_field.connection(diff)
        J_next = torch.matmul(pt_gauge, torch.matmul(J, pt_gauge.transpose(1, 2)))
        
        return q_next, p_next, J_next
