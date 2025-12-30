from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

class Env(ABC):
    """
    Abstract Base Class for Environments.
    Follows a simplified Gym interface.
    """
    @abstractmethod
    def reset(self, seed: int = None) -> Dict[str, Any]:
        """
        Resets the environment.
        Returns:
            obs: Observation dictionary.
        """
        pass
        
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Steps the environment.
        Args:
            action: Action array.
        Returns:
            obs: Observation dictionary.
            reward: Scalar reward (optional).
            done: Termination flag.
            info: Auxiliary info.
        """
        pass

class ObsAdapter(ABC):
    """
    Adapts Environment Observation to Entity Input format.
    Target format: Dict with keys ['x_ext_proxy'] or 'u_env' ready for Entity.
    """
    @abstractmethod
    def adapt(self, env_obs: Dict[str, Any]) -> Dict[str, Any]:
        pass

class ActAdapter(ABC):
    """
    Adapts Entity Action to Environment Action.
    Entity Action is usually unbounded internal force/readout.
    Env Action might need clipping, scaling, or discretization.
    """
    @abstractmethod
    def adapt(self, entity_action: np.ndarray) -> np.ndarray:
        pass
