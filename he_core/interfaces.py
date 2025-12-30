import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from he_core.state import ContactState

class BaseInterface(nn.Module, ABC):
    def __init__(self, dim_q: int, dim_ext: int):
        super().__init__()
        self.dim_q = dim_q
        self.dim_ext = dim_ext
        
    @abstractmethod
    def read(self, u_env: torch.Tensor) -> torch.Tensor:
        """Map External Input -> Internal Force/Drive (B, dim_q)"""
        pass
        
    @abstractmethod
    def write(self, state: ContactState) -> torch.Tensor:
        """Map Internal State -> External Output (B, dim_ext)"""
        pass

class ActionInterface(BaseInterface):
    """
    Standard Motor Interface.
    Write: p -> Action (Force)
    Read: u_env -> Force (Direct Drive)
    """
    def __init__(self, dim_q: int, dim_ext: int):
        super().__init__(dim_q, dim_ext)
        self.proj = nn.Linear(dim_q, dim_ext, bias=False) # p -> action
        self.embed = nn.Linear(dim_ext, dim_q, bias=False) # env -> force
        
        # Init near Identity
        with torch.no_grad():
            if dim_q == dim_ext:
                self.proj.weight.copy_(torch.eye(dim_q))
                self.embed.weight.copy_(torch.eye(dim_q))
            else:
                self.proj.weight.normal_(0, 0.1)
                self.embed.weight.normal_(0, 0.1)
        
    def read(self, u_env: torch.Tensor) -> torch.Tensor:
        # u_env (B, ext) -> (B, q)
        return self.embed(u_env)
        
    def write(self, state: ContactState) -> torch.Tensor:
        # p (B, q) -> (B, ext)
        # Action reflects momentum/intent
        return self.proj(state.p)

class AuxInterface(BaseInterface):
    """
    Symbolic/Discrete Interface.
    Write: q -> Discrete Symbols (via Softmax/Argmax proxy) -> vector
    Read: Symbol -> Force (via Embedding)
    """
    def __init__(self, dim_q: int, dim_ext: int):
        super().__init__(dim_q, dim_ext)
        self.proj = nn.Linear(dim_q, dim_ext) # q -> logits
        self.embed = nn.Linear(dim_ext, dim_q)
        
    def read(self, u_env: torch.Tensor) -> torch.Tensor:
        # u_env (B, ext) is one-hot or embedding?
        # Assume vector.
        return self.embed(u_env)
        
    def write(self, state: ContactState) -> torch.Tensor:
        # q -> logits
        # For simplicity, just return logits or softmax?
        # Return projected vector.
        return torch.tanh(self.proj(state.q))


class PortContract(nn.Module):
    """
    Port Contract Layer (Phase 15.2).
    Normalizes/clamps port output to ensure bounded gain.
    
    Methods:
    - 'tanh': y' = max_gain * tanh(y)
    - 'soft_norm': y' = max_gain * y / (1 + ||y||)
    - 'saturation': y' = clamp(y, -max_gain, max_gain)
    """
    def __init__(self, method: str = 'tanh', max_gain: float = 1.0):
        super().__init__()
        self.method = method
        self.max_gain = max_gain
        
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.method == 'tanh':
            return self.max_gain * torch.tanh(y)
        elif self.method == 'soft_norm':
            norm = y.norm(dim=-1, keepdim=True) + 1e-6
            return self.max_gain * y / (1.0 + norm)
        elif self.method == 'saturation':
            return torch.clamp(y, -self.max_gain, self.max_gain)
        else:
            return y  # Identity


class PortInterface(BaseInterface):
    """
    Energy-Semantic Port Interface (Phase 15.1).
    
    Defines conjugate effort/flow port variables:
    - Input u (effort): External input mapped to internal force.
    - Output y (flow): Conjugate output derived from state, with energy semantics.
    
    Power: P = u · y (has physical meaning).
    
    Key difference from ActionInterface:
    - y is NOT just a projection of p; it's constructed to be conjugate to u.
    - For port-Hamiltonian systems: y = ∂H_port/∂u.
    - For simplicity, we use y = B(q)^T p (collocated output).
    
    Includes optional PortContract for gain bounding.
    """
    def __init__(self, dim_q: int, dim_u: int, use_contract: bool = True, contract_method: str = 'tanh', max_gain: float = 1.0):
        super().__init__(dim_q, dim_u)
        self.dim_u = dim_u
        
        # Input Embedding: u -> internal force
        self.G_u = nn.Linear(dim_u, dim_q, bias=False)  # Input gain matrix
        
        # Output Projection: state -> y (conjugate flow)
        # y = G_u^T @ p gives collocated output with passivity structure
        # But for flexibility, we use a separate learnable projection
        self.G_y = nn.Linear(dim_q, dim_u, bias=False)  # Output projection
        
        # Initialize near Identity for passivity
        with torch.no_grad():
            if dim_q == dim_u:
                self.G_u.weight.copy_(torch.eye(dim_q))
                self.G_y.weight.copy_(torch.eye(dim_q))
            else:
                # Small random init
                self.G_u.weight.normal_(0, 0.1)
                self.G_y.weight.normal_(0, 0.1)
                
        # Port Contract (gain bounding)
        self.use_contract = use_contract
        if use_contract:
            self.contract = PortContract(method=contract_method, max_gain=max_gain)
        else:
            self.contract = None
            
    def read(self, u_ext: torch.Tensor) -> torch.Tensor:
        """Map external effort u to internal force."""
        # Force = G_u @ u
        return self.G_u(u_ext)
        
    def write(self, state: ContactState) -> torch.Tensor:
        """Compute conjugate flow output y from state."""
        # Collocated output: y = G_y @ p
        # This gives passivity when G_y = G_u^T (which we approximate)
        y = self.G_y(state.p)
        
        # Apply contract if enabled
        if self.use_contract and self.contract is not None:
            y = self.contract(y)
            
        return y
        
    def read_u(self, u_ext: torch.Tensor) -> torch.Tensor:
        """Alias for read (explicit naming for port semantics)."""
        return self.read(u_ext)
        
    def write_y(self, state: ContactState) -> torch.Tensor:
        """Alias for write (explicit naming for port semantics)."""
        return self.write(state)
