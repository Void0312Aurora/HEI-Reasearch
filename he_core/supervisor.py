"""
Phase 17.3: Hard Gating Integration (Supervisor).

The Supervisor monitors training loops and enforces "Hard Gates" to prevention
catastrophic forgetting or instability.

Gates:
1. Small-Gain Gate: Loop Gain < 1.0 (Critical Safety).
2. Robustness Trend Gate: Slope >= 0 (Prevent degrading performance).
3. Boundedness Gate: A1 Robustness > 0.
"""
from typing import Dict, Any
import torch

class TrainingSupervisor:
    def __init__(self, use_small_gain: bool = True, use_trend: bool = True):
        self.use_small_gain = use_small_gain
        self.use_trend = use_trend
        
    def check_gates(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Check all enabled gates.
        metrics: {
            'loop_gain': float,
            'robustness_slope': float,
            'min_robustness': float
        }
        Returns: {'gate_name': passed (bool)}
        """
        results = {}
        all_passed = True
        
        # 1. Small-Gain
        if self.use_small_gain:
            if 'loop_gain' in metrics:
                passed = metrics['loop_gain'] < 1.0
                results['small_gain'] = passed
                if not passed: all_passed = False
            else:
                results['small_gain'] = False # Validation failed due to missing metric/data
                
        # 2. Trend (Allow small fluctuation -0.01)
        if self.use_trend:
            if 'robustness_slope' in metrics:
                passed = metrics['robustness_slope'] > -0.01
                results['trend'] = passed
                if not passed: all_passed = False
                
        # 3. Boundedness (A1)
        if 'min_robustness' in metrics:
            passed = metrics['min_robustness'] > 0
            results['boundedness'] = passed
            if not passed: all_passed = False
            
        return {'all_passed': all_passed, 'details': results}
        
    def decide_action(self, gate_results: Dict[str, Any]) -> str:
        """
        Decide training action based on gate results.
        Returns: 'CONTINUE', 'ROLLBACK', 'STOP'
        """
        details = gate_results.get('details', {})
        
        # Critical Failure -> ROLLBACK
        if not details.get('small_gain', True):
            return 'ROLLBACK'
            
        if not details.get('boundedness', True):
            return 'ROLLBACK'
            
        # Warning Failure -> STOP/ANNEAL (Simulated here as STOP)
        if not details.get('trend', True):
            return 'STOP' # Degrading trend
            
        return 'CONTINUE'
