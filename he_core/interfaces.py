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
        self.proj = nn.Linear(dim_q, dim_ext) # p -> action
        self.embed = nn.Linear(dim_ext, dim_q) # env -> force
        
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
