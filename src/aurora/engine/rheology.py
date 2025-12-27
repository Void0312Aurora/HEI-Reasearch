"""
Aurora Rheological Optimizer (CCD v3.1).
========================================

Implements Bingham Plastic Flow for Parameter Evolution.
Replaces SGD.

Ref: Axiom 4.2.2.
"""

import torch
import torch.nn as nn

class RheologicalOptimizer:
    def __init__(self, params, lr: float = 0.01, yield_stress: float = 0.1, elastic_k: float = 0.001):
        self.params = list(params)
        self.lr = lr # inverse viscosity eta
        self.yield_stress = yield_stress
        self.elastic_k = elastic_k
        
    def step(self):
        """
        Update parameters based on their .grad (Stress).
        """
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                    
                # Stress = - grad (Convention: Stress points to reduction of energy)
                # But Gradient descent moves against grad.
                # So Stress ~ -Grad.
                # If we use standard optim, we subtract Grad.
                # Here we explicitly model Bingham.
                
                grad = p.grad
                stress_mag = torch.norm(grad)
                
                # Bingham Yield Check
                if stress_mag > self.yield_stress:
                    # Excess Stress
                    excess = stress_mag - self.yield_stress
                    # Flow rate = eta * excess
                    # Update = - flow_rate * direction
                    # direction = grad / mag
                    
                    update = self.lr * excess * (grad / (stress_mag + 1e-9))
                    p.data.sub_(update)
                    
                # Elastic Recovery (Forgetting)
                # Decay towards 0 (or MaxEnt state)?
                # Simple weight decay
                p.data.mul_(1.0 - self.elastic_k)
