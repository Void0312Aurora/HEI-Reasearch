
import torch
import torch.nn as nn
import torch.nn.functional as F

def run_exact_repro():
    print("DEBUG: Starting Exact Logic Repro")
    
    # 7-dim State (dim_q = 3) -> 2*3+1 = 7
    dim_q = 3
    dim = 2 * dim_q + 1
    B = 2
    steps = 10
    
    # Init x
    x = torch.randn(B, dim, requires_grad=True)
    
    # Networks (Theta)
    net_V = nn.Linear(dim_q, 1)
    net_Alpha = nn.Linear(dim_q, 1)
    
    # Loop
    for t in range(steps):
        # 1. Slice
        q = x[:, :dim_q]             # (B, 3)
        p = x[:, dim_q:2*dim_q]      # (B, 3)
        s = x[:, 2*dim_q:]           # (B, 1)
        
        # 2. Compute H
        # K(p) = 0.5 * |p|^2
        K = 0.5 * (p**2).sum(dim=1, keepdim=True)
        
        # V(q) = Net(q)
        V = net_V(q)
        
        # Alpha(q) = Softplus(Net(q))
        alpha = F.softplus(net_Alpha(q))
        
        # H = K + V + alpha * s
        H_val = K + V + alpha * s
        H_sum = H_val.sum()
        
        # 3. Grads
        # create_graph=True, retain_graph=False (Simulating proposed fix)
        grads = torch.autograd.grad(H_sum, x, create_graph=True)[0]
        
        # 4. Update
        # Simulate Contact Dynamics Update where H is used in dot_s
        # dot_s = p*dH/dp - H
        # We use H_val in the update rule
        
        # Note: In real code, dot_s = term1 - H
        # So x_new depends on -H.
        
        x = x + grads * 0.1 - H_val * 0.01 
        
        if t % 2 == 0:
             print(f"Step {t}: x_norm={x.norm().item()}")
             
    # Loss & Backward
    loss = x.sum()
    print("DEBUG: Calling Backward...")
    loss.backward()
    
    # Check Gradients
    nn_grad = net_V.weight.grad
    if nn_grad is not None:
        print(f"PASS: Net Grad Norm = {nn_grad.norm().item()}")
    else:
        print("FAIL: No grad on Net")

if __name__ == "__main__":
    try:
        run_exact_repro()
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()
