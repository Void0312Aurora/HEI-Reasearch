import torch
import torch.nn as nn
import torch.optim as optim
import unittest
from he_core.atlas import Atlas

class TestAtlasRouter(unittest.TestCase):
    def test_router_learnability(self):
        print("\n--- Protocol 4: Atlas Router Learning ---")
        dim_q = 2
        num_charts = 2
        atlas = Atlas(num_charts, dim_q)
        
        optimizer = optim.SGD(atlas.router.parameters(), lr=0.1)
        criterion = nn.MSELoss()
        
        # Data: q1 -> [1, 0], q2 -> [0, 1]
        q1 = torch.ones(1, dim_q)
        t1 = torch.tensor([[1.0, 0.0]])
        
        q2 = -torch.ones(1, dim_q)
        t2 = torch.tensor([[0.0, 1.0]])
        
        print("Training Router...")
        for i in range(50):
            optimizer.zero_grad()
            
            w1 = atlas.router(q1)
            loss1 = criterion(w1, t1)
            
            w2 = atlas.router(q2)
            loss2 = criterion(w2, t2)
            
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Step {i}, Loss: {loss.item():.4f}")
                
        # Verify Separation
        w1_final = atlas.router(q1)
        w2_final = atlas.router(q2)
        
        print(f"Final Weights q1: {w1_final.detach()}")
        print(f"Final Weights q2: {w2_final.detach()}")
        
        self.assertGreater(w1_final[0,0], 0.8, "Failed to learn routing for q1")
        self.assertGreater(w2_final[0,1], 0.8, "Failed to learn routing for q2")

if __name__ == '__main__':
    unittest.main()
