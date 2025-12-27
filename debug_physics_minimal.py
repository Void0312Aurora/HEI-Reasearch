import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from aurora.model.injector import ContinuousInjector
from aurora.engine.forces import ForceField
from aurora.physics import geometry

def debug():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Injector
    print("Init Injector...")
    injector = ContinuousInjector(dim=16).to(device)
    
    x = torch.rand(32, 1, device=device) # Batch 32
    r = torch.rand(32, 1, device=device)
    
    print("Running Injector Forward...")
    m, q, J, p = injector(x, r)
    print(f"Injector Output: q={q.shape}, J={J.shape}")
    
    if torch.isnan(q).any(): print("FATAL: q contains NaNs!")
    if torch.isnan(m).any(): print("FATAL: m contains NaNs!")
    if torch.isnan(J).any(): print("FATAL: J contains NaNs!")
    if torch.isnan(p).any(): print("FATAL: p contains NaNs!")
    
    # 2. Geometry
    print("Testing Geometry...")
    d = geometry.dist(q, q)
    print(f"Dist shape: {d.shape}")
    
    # 3. Forces
    print("Init ForceField...")
    ff = ForceField(dim=16).to(device)
    
    print("Computing Forces Components...")
    # Explicitly call each
    V_geo = ff.potential_geometry(q)
    print(f"V_geo: {V_geo.item()}")
    
    V_mass = ff.potential_mass(q, m)
    print(f"V_mass: {V_mass.item()}")
    
    V_gauge = ff.potential_gauge(q, J)
    print(f"V_gauge: {V_gauge.item()}")
    
    V_chem = ff.potential_chemical(m)
    print(f"V_chem: {V_chem.item()}")
    
    F, E = ff.compute_forces(q, m, J)
    print(f"Total Forces Shape: {F.shape}, Energy: {E.item()}")
    
    print("SUCCESS")

if __name__ == "__main__":
    debug()
