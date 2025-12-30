
import torch
import torch.nn as nn
from he_core.entity_v4 import UnifiedGeometricEntity

def run_entity_test():
    print("DEBUG: Starting Entity Refactor Test")
    
    # Config matching train_mnist_adaptive
    config = {
        'dim_q': 784,
        'dim_u': 784,
        'num_charts': 1,
        'damping': 0.0, # Unused by Adaptive
        'learnable_coupling': True, # To match complex graph
        'use_port_interface': False
    }
    
    # 1. Init Entity
    entity = UnifiedGeometricEntity(config)
    
    # Inject Adaptive Generator manually if needed, 
    # but Entity v4 init uses PortCoupled(Dissipative).
    # Dissipative is constant alpha.
    # Adaptive requires replacing internal_gen.
    
    from he_core.adaptive_generator import AdaptiveDissipativeGenerator
    adaptive_gen = AdaptiveDissipativeGenerator(784, net_V=nn.Linear(784, 1))
    
    # Replace internal generator in PortCoupled
    # entity.generator is PortCoupledGenerator
    entity.generator.internal = adaptive_gen
    
    # 2. Loop
    batch_size = 32
    steps = 10
    dt = 0.1
    
    # State
    x = torch.randn(batch_size, 2*784+1, requires_grad=True)
    u_ext = torch.randn(batch_size, 784)
    
    current_x = x
    
    for t in range(steps):
        # Forward Tensor
        # This calls: Checkpoint(run_step) or run_step direct
        # Inspect entity_v4 source to see what is enabled.
        
        out = entity.forward_tensor(current_x, u_ext, dt)
        current_x = out['next_state_flat']
        
        if t % 2 == 0:
            print(f"Step {t}: x_norm={current_x.norm().item()}")
            
    # 3. Backward
    loss = current_x.sum()
    print("DEBUG: Calling Backward...") 
    loss.backward()
    
    # Check Gradients on Alpha
    grad_norm = adaptive_gen.net_Alpha[0].weight.grad.norm().item()
    print(f"PASS: Alpha Grad Norm = {grad_norm}")

if __name__ == "__main__":
    try:
        run_entity_test()
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()
