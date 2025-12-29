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
        Includes mandatory port clipping (Passivity v0).
        """
        # 1. Combine
        if meta["phase"] == "online":
            # Online: u = u_env + w * u_self
            u_combined = u_env + u_self
        else:
            # Offline: u_env is blocked or replaced.
            if meta["u_source"] == "internal":
                u_combined = u_self
            elif meta["u_source"] == "replay":
                # Replay: u = stored_buffer + u_self
                u_combined = u_env + u_self
            else:
                u_combined = u_self
                
        # 2. Clip (Stabilization)
        # Limit magnitude to avoid energy injection explosion
        clip_val = self.config.get("u_clip", 1.0)
        u_combined = torch.clamp(u_combined, -clip_val, clip_val)
        
        return u_combined
