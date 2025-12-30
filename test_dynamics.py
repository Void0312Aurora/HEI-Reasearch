print("DYN TEST START")
import torch
import torch.nn as nn
from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adapters import ImagePortAdapter
from he_core.generator import DeepDissipativeGenerator
print("Imports OK")

# Config
dim_q = 64
dim_u = 32
net_V = nn.Sequential(nn.Linear(dim_q, 1))
gen = DeepDissipativeGenerator(dim_q, alpha=0.5, net_V=net_V)
adapter = ImagePortAdapter(in_channels=1, dim_out=dim_u)

config = {
    'dim_q': dim_q,
    'dim_u': dim_u,
    'learnable_coupling': True,
    'num_charts': 1,
    'damping': 0.5,
    'use_port_interface': True
}

entity = UnifiedGeometricEntity(config)
entity.internal_gen = gen
print("Entity Init OK")

# Forward
batch_size = 64
state_flat = torch.zeros(batch_size, dim_q * 2 + 1)
# u_ext = torch.randn(batch_size, dim_u)
# Use Adapter
img = torch.randn(batch_size, 1, 28, 28)
u_ext = adapter.get_drive(img)

dt = 0.1

print(f"Running Forward Loop (B={batch_size})...")
try:
    energies = []
    # Loop
    for i in range(10):
        # We need to manually update flat because UnifiedGeometricEntity.forward_tensor 
        # uses the passed state_flat but returns next_state_flat.
        # But UnifiedGeometricEntity internally wraps state_flat into ContactState.
        
        out = entity.forward_tensor(state_flat, u_ext, dt)
        state_flat = out['next_state_flat']
        
        # Access p through temp state (as in CognitiveAgent.forward logic)
        # CognitiveAgent does: self.entity.state.flat = out['next_state_flat']
        # Then accesses self.entity.state.p
        
        # Here we mimic access:
        # We need a ContactState wrapper to interpret state_flat
        # UnifiedGeometricEntity doesn't expose a persistent state in forward_tensor, 
        # but CognitiveAgent DOES maintain self.entity.state 
        
        # Let's verify if access crashes
        # To access p, we need to wrap it
        from he_core.state import ContactState
        temp_state = ContactState(dim_q, batch_size, flat_tensor=state_flat)
        p = temp_state.p
        K = 0.5 * (p**2).sum(dim=1, keepdim=True)
        energies.append(K)
        
    print("Loop OK")
    
    # Loss
    from he_core.losses import SelfSupervisedLosses
    energy_seq = torch.cat(energies, dim=1) # (B, Steps)
    energies_t = energy_seq.T
    loss_diss = SelfSupervisedLosses.l1_dissipative_loss(energies_t)
    print("Loss Diss OK:", loss_diss.item())

    # Classifier
    classifier = nn.Linear(dim_q, 10)
    # Extract q from the FINAL state_flat
    final_state_wrapper = ContactState(dim_q, batch_size, flat_tensor=state_flat)
    final_q = final_state_wrapper.q
    
    logits = classifier(final_q)
    target = torch.randint(0, 10, (batch_size,)).long()
    loss_cls = nn.functional.cross_entropy(logits, target)
    print("Loss Cls OK:", loss_cls.item())
    
    # Backward (Total)
    loss = loss_cls + 0.0 * loss_diss
    print("Running Backward...")
    loss.backward()
    print("Backward OK")

except Exception as e:
    print("CRASH:", e)
    import traceback
    traceback.print_exc()

print("DYN TEST END")
