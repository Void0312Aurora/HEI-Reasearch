"""
Network STL Monitors (Phase 14.3).

Extends Signal Temporal Logic monitoring to multi-entity networks.

Specifications:
- A1 (Network Boundedness): All nodes remain within bounds.
- Cooperative Liveness: Signal propagates from one node to another.
"""
import torch
from typing import Dict, List, Callable, Any
from EXP.diag.monitors import robustness_always, robustness_eventually


def network_always_bounded(trajectories: Dict[str, List[Dict]], bound: float = 10.0) -> Dict[str, Any]:
    """
    Check A1 (Boundedness) for all entities in the network.
    
    Spec: G[0,T](||action_A|| < bound AND ||action_B|| < bound)
    
    Returns robustness values per entity and overall.
    """
    results = {}
    min_robustness = float('inf')
    
    for entity_id, traj in trajectories.items():
        # Extract action norms
        action_norms = torch.tensor([step['action'].flatten() for step in traj])
        norms = torch.norm(action_norms, dim=1)
        
        # Predicate: ||action|| < bound
        pred_values = bound - norms  # Positive if bounded
        
        # Always robustness
        rob = pred_values.min().item()
        results[entity_id] = rob
        min_robustness = min(min_robustness, rob)
        
    results['overall'] = min_robustness
    results['satisfied'] = min_robustness > 0
    
    return results


def network_eventually_response(trajectories: Dict[str, List[Dict]], source_id: str, target_id: str, 
                                 response_threshold: float = 0.01, window: int = 10) -> Dict[str, Any]:
    """
    Check Cooperative Liveness: If source acts, target eventually responds.
    
    Spec: G[0,T-window]( ||action_source|| > 0 => F[0,window](||action_target|| > threshold) )
    
    Simplified: Check if target ever exceeds threshold after source exceeds 0.
    """
    source_traj = trajectories[source_id]
    target_traj = trajectories[target_id]
    
    T = len(source_traj)
    
    # Find first time source action is non-trivial
    source_active_t = None
    for t, step in enumerate(source_traj):
        if torch.tensor(step['action']).abs().max() > 0.001:
            source_active_t = t
            break
            
    if source_active_t is None:
        return {'source_active': False, 'target_responded': False, 'robustness': 0.0}
        
    # Check if target responds within window after source active
    response_window = target_traj[source_active_t:min(source_active_t + window, T)]
    
    max_target_response = 0.0
    for step in response_window:
        target_norm = torch.tensor(step['action']).norm().item()
        max_target_response = max(max_target_response, target_norm)
        
    robustness = max_target_response - response_threshold
    
    return {
        'source_active': True,
        'source_active_t': source_active_t,
        'target_responded': robustness > 0,
        'max_target_response': max_target_response,
        'robustness': robustness
    }


class NetworkSTLMonitor:
    """
    Composite monitor for network-level STL specifications.
    """
    def __init__(self, bound: float = 10.0, response_threshold: float = 0.01):
        self.bound = bound
        self.response_threshold = response_threshold
        
    def check_all(self, trajectories: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Run all network STL checks."""
        report = {
            'A1_boundedness': network_always_bounded(trajectories, self.bound),
            'A_to_B_liveness': network_eventually_response(trajectories, 'A', 'B', self.response_threshold),
            'B_to_A_liveness': network_eventually_response(trajectories, 'B', 'A', self.response_threshold)
        }
        
        report['all_satisfied'] = (
            report['A1_boundedness']['satisfied'] and
            report['A_to_B_liveness']['target_responded'] and
            report['B_to_A_liveness']['target_responded']
        )
        
        return report


if __name__ == "__main__":
    # Example usage
    from he_core.wiring import Edge, WiringDiagram, TwoEntityNetwork
    
    config = {'dim_q': 2, 'learnable_coupling': True, 'num_charts': 2, 'damping': 2.0}
    edges = [
        Edge(source_id='A', target_id='B', gain=0.3),
        Edge(source_id='B', target_id='A', gain=0.3)
    ]
    wiring = WiringDiagram(edges)
    
    network = TwoEntityNetwork(config, config, wiring)
    trajectories = network.rollout(steps=50)
    
    monitor = NetworkSTLMonitor(bound=5.0)
    report = monitor.check_all(trajectories)
    
    print("--- Network STL Monitor ---")
    print(f"  A1 Boundedness: {report['A1_boundedness']['satisfied']}")
    print(f"  A->B Liveness: {report['A_to_B_liveness']['target_responded']}")
    print(f"  B->A Liveness: {report['B_to_A_liveness']['target_responded']}")
    print(f"  All Satisfied: {report['all_satisfied']}")
