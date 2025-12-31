
import torch
import torch.nn as nn
from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    dim_q = 128
    config = {'dim_q': dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(dim_q, 256), nn.Tanh(), nn.Linear(256, 1))
    adaptive = AdaptiveDissipativeGenerator(dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, 1, True, 1)
    entity.internal_gen = adaptive
    entity.generator.add_port('context', dim_u=2)
    entity.to(DEVICE)
    
    print("Entity Init Done.")
    
    curr = torch.zeros(1, 2*dim_q + 1, device=DEVICE)
    u_val = torch.zeros(1, 1, device=DEVICE)
    ctx = torch.zeros(1, 2, device=DEVICE)
    dt = 0.1
    
    print("Step 1 Start...")
    out = entity.forward_tensor(curr, {'default': u_val, 'context': ctx}, dt)
    curr = out['next_state_flat'].detach()
    print("Step 1 Done.")

if __name__ == "__main__":
    main()
