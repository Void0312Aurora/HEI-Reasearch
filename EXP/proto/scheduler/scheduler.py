from typing import Tuple, Dict
import torch
import numpy as np

class Scheduler:
    """
    Manages the Online/Offline phase switching and input source control.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.phase = "online" # or "offline"
        self.step_count = 0
        self.online_steps = config.get("online_steps", 100)
        self.offline_steps = config.get("offline_steps", 100)
        
    def step(self) -> Dict[str, str]:
        self.step_count += 1
        
        # Simple cycle: Online -> Offline -> Stop (or loops)
        total_cycle = self.online_steps + self.offline_steps
        cycle_idx = self.step_count % total_cycle
        
        if cycle_idx < self.online_steps:
            self.phase = "online"
            u_source = "env"
        else:
            self.phase = "offline"
            # offline source can be "replay" or "internal"
            # For MVS v0, we might just cut off external input.
            u_source = self.config.get("offline_u_source", "internal")
            
        return {
            "phase": self.phase,
            "u_source": u_source,
            "step": self.step_count
        }

    def process_input(self, u_env: torch.Tensor, u_self: torch.Tensor, meta: Dict) -> torch.Tensor:
        """
        Combines u_env and u_self based on the current phase.
        """
        if meta["phase"] == "online":
            # Online: u = u_env + w * u_self
            # For MVS v0, let's assume simple addition or just u_env for pure online
            # But "Port Loop" implies u_self is always there.
            return u_env + u_self
        else:
            # Offline: u_env is blocked or replaced.
            if meta["u_source"] == "internal":
                # Pure internal: u = u_self (env blocked)
                return u_self
            elif meta["u_source"] == "replay":
                # Replay: u = stored_buffer (env blocked)
                # For v0 simplicity, let's treat replay as a special u_self logic outside, 
                # here we just block u_env.
                return u_self
            else:
                return u_self
