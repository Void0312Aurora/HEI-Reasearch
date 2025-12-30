import torch
import torch.nn as nn
from typing import Dict, Any, List
from he_core.state import ContactState
from he_core.contact_dynamics import ContactIntegrator
from he_core.generator import DissipativeGenerator, DrivenGenerator
from he_core.interfaces import BaseInterface, ActionInterface

class GeometricEntity(nn.Module):
    """
    Entity v0.3: Geometric-Generator Realization.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dim_q = config.get('dim_q', 2)
        self.dim_ext = config.get('dim_ext', 2)
        
        # 1. State
        self.state = ContactState(self.dim_q)
        
        # 2. Generator (H)
        alpha = config.get('damping', 0.1)
        # Use DrivenGenerator to allow input coupling
        self.generator = DrivenGenerator(self.dim_q, alpha=alpha)
        
        # 3. Integrator
        self.integrator = ContactIntegrator()
        
        # 4. Interface (Default to ActionInterface)
        self.interface = ActionInterface(self.dim_q, self.dim_ext)
        
    def set_interface(self, interface: BaseInterface):
        self.interface = interface
        
    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        One step map.
        1. Read Env -> Drive
        2. Integrate State
        3. Write Output
        """
        u_env = torch.tensor(obs.get('x_ext_proxy', [0]*self.dim_ext), dtype=torch.float32).view(1, -1)
        
        # 1. Read Input
        drive = self.interface.read(u_env) # (B, dim_q)
        
        # 2. Integrate
        # We need to pass 'drive' to generator.
        # But ContactIntegrator only accepts state -> H.
        # Solution: Lambda or Wrapper.
        
        def H_func(s):
            return self.generator.forward_with_drive(s, drive)
            
        dt = 0.1
        self.state = self.integrator.step(self.state, H_func, dt)
        
        # 3. Write Output
        u_self = self.interface.write(self.state)
        
        # Return Log Dict (Compliance with v0.3 Log Contract)
        return {
            'x_int': self.state.flat.detach().numpy(),
            'sensory': u_env.detach().numpy().flatten(),
            'active': u_self.detach().numpy().flatten(),
            'q_norm': self.state.q.norm().item(),
            'p_norm': self.state.p.norm().item(),
            's_val': self.state.s.item(),
            'step': 0 # TODO: manage step count
        }
