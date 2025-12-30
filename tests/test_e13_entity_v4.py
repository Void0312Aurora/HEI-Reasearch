import torch
import unittest
from he_core.entity_v4 import UnifiedGeometricEntity

class TestE13EntityV4(unittest.TestCase):
    def test_initialization_and_flow(self):
        print("\n--- Task 13.0: Entity v0.4 Flow Test ---")
        config = {
            'dim_q': 2,
            'learnable_coupling': True,
            'num_charts': 3,
            'damping': 0.1
        }
        entity = UnifiedGeometricEntity(config)
        entity.reset()
        
        # Input
        obs = {'x_ext': torch.randn(1, 2)}
        
        # Step
        out = entity(obs, dt=0.1)
        
        print(f"Output Keys: {out.keys()}")
        print(f"Chart Weights: {out['chart_weights']}")
        
        self.assertIn('chart_weights', out)
        self.assertEqual(len(out['chart_weights']), 3)
        self.assertIn('action', out)
        
    def test_trainability(self):
        print("\n--- Task 13.0: Entity v0.4 Trainability Test ---")
        config = {
            'dim_q': 2,
            'learnable_coupling': True,
            'num_charts': 2
        }
        entity = UnifiedGeometricEntity(config)
        entity.train() # Set training mode
        entity.reset()
        
        # Forward Tensor
        u_ext = torch.randn(1, 2)
        state_in = entity.state.flat.clone().detach().requires_grad_(True)
        
        out = entity.forward_tensor(state_in, u_ext)
        
        print(f"Tensor Out Keys: {out.keys()}")
        
        # Check Backprop
        # Loss = action sum
        loss = out['action'].sum()
        loss.backward()
        
        # Now Router is used in Dynamics (via H_func -> generator -> weights)
        # So Router should get gradient from 'action' loss (via next_state -> H -> weights)
        
        router_grad = list(entity.atlas.router.parameters())[0].grad
        print(f"Router Grad from Action Loss: {router_grad.norm().item() if router_grad is not None else 'None'}")
        
        # It MUST be non-None now
        self.assertIsNotNone(router_grad)
        self.assertGreater(router_grad.norm().item(), 0.0)
        
        # Check W grad too
        W_grad = entity.generator.coupling.W_stack.grad
        print(f"Coupling W_stack Grad: {W_grad.norm().item() if W_grad is not None else 'None'}")
        self.assertIsNotNone(W_grad)

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
