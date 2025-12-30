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
        
        # Check Gradients
        # Port Coupling weights
        W_grad = entity.generator.coupling.W.weight.grad
        print(f"Coupling W Grad: {W_grad.norm().item() if W_grad is not None else 'None'}")
        
        self.assertIsNotNone(W_grad)
        self.assertGreater(W_grad.norm().item(), 0.0)
        
        # Router weights (Router depends on q, q depends on state_in)
        # But we are testing graph from input to output. 
        # Router depends on 'state_in'. state_in is input.
        # But Router params?
        # Router: q -> weights.
        # Loss involves 'action'. Action depends on 'next_state'.
        # next_state depends on H.
        # H depends on PortCoupling.
        # Router output is NOT in 'action' path in v0.4 yet (it's monitoring only for now, unless we route dynamics).
        # In UnifiedEntity v0.4 init:
        # self.generator is ONE generator.
        # self.atlas.router is called but 'chart_weights' not used in dynamics?
        
        # Wait, if router is unused in dynamics, it won't get grad from action loss.
        # We need to test Router learning separately or link it?
        # Protocol: "Train Router to maximize Coverage".
        # So we direct loss on 'chart_weights'.
        
        loss_router = out['chart_weights'].sum()
        loss_router.backward()
        
        router_grad = list(entity.atlas.router.parameters())[0].grad
        print(f"Router Grad: {router_grad.norm().item() if router_grad is not None else 'None'}")
        self.assertIsNotNone(router_grad)

if __name__ == '__main__':
    unittest.main()
