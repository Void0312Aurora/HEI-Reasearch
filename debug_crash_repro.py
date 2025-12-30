
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Import core modules
from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.adapters import ImagePortAdapter
from he_core.losses import SelfSupervisedLosses

# Define Model (Same as train_mnist_adaptive.py)
class CognitiveAgent(nn.Module):
    def __init__(self, dim_q: int = 64):
        super().__init__()
        self.dim_q = dim_q
        self.dim_u = 32 # Drive dimension
        
        self.adapter = ImagePortAdapter(in_channels=1, dim_out=self.dim_u)
        
        net_V = nn.Sequential(
            nn.Linear(dim_q, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1) 
        )
        
        self.gen = AdaptiveDissipativeGenerator(dim_q, net_V=net_V)
        
        config = {
            'dim_q': dim_q,
            'dim_u': self.dim_u,
            'learnable_coupling': True,
            'num_charts': 1,
            'damping': 0.5,
            'use_port_interface': True
        }
        
        self.entity = UnifiedGeometricEntity(config)
        self.entity.internal_gen = self.gen
        
        self.classifier = nn.Linear(dim_q, 10)
        self.drive_scale = 1.0
        
    def forward(self, img):
        drive_scale = getattr(self, 'drive_scale', 1.0)
        u_ext = self.adapter.get_drive(img) * drive_scale 
        
        batch_size = img.shape[0]
        
        from he_core.state import ContactState
        if self.entity.state.batch_size != batch_size:
            self.entity.state = ContactState(self.dim_q, batch_size, device=img.device)
            
        self.entity.state.q = torch.zeros(batch_size, self.dim_q, device=img.device)
        self.entity.state.p = torch.zeros(batch_size, self.dim_q, device=img.device)
        self.entity.state.s = torch.zeros(batch_size, 1, device=img.device)
        
        # Test Long Horizon
        steps = 10 
        dt = 0.1
        energies = []
        
        for _ in range(steps):
            out = self.entity.forward_tensor(self.entity.state.flat, u_ext, dt)
            self.entity.state.flat = out['next_state_flat']
            
            K = 0.5 * (self.entity.state.p**2).sum(dim=1, keepdim=True)
            energies.append(K)
            
        final_q = self.entity.state.q
        logits = self.classifier(final_q)
        
        return logits, torch.cat(energies, dim=1)

def run_synthetic_test():
    print("DEBUG: Starting Synthetic Test")
    
    device = torch.device("cpu") # Test CPU first
    model = CognitiveAgent(dim_q=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Synthetic Data Loop
    batch_size = 64
    steps = 100 # Simulate 100 batches
    
    for i in range(steps):
        data = torch.randn(batch_size, 1, 28, 28).to(device)
        target = torch.randint(0, 10, (batch_size,)).to(device)
        
        optimizer.zero_grad()
        logits, energy_seq = model(data)
        
        loss_diss = SelfSupervisedLosses.l1_dissipative_loss(energy_seq.T)
        loss_cls = nn.functional.cross_entropy(logits, target)
        loss = loss_cls + 0.1 * loss_diss
        
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i}: Loss={loss.item():.4f}")
            sys.stdout.flush()
            
    print("DEBUG: Synthetic Test Passed")

if __name__ == "__main__":
    try:
        run_synthetic_test()
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()
