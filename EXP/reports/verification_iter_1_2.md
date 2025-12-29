# Iteration 1.2 Robustness Verification (N=10)
## Hypothesis: Ordered Replay should have significantly higher Resonator Energy than Mismatched.

```
              Condition ResEnergy (Mean) ResEnergy (Std) ResEnergy (CI95) D1 Var (Mean) p-value (vs Replay)
       Replay (Ordered)            1.743           0.972           ±0.603         1.143                   -
     Mismatch (Reverse)            1.639           1.035           ±0.642         1.128             0.82873
Mismatch (BlockShuffle)            1.236           0.339           ±0.210         0.951             0.16713
```

## Raw Data Summary
### Replay (Ordered)
- Energies: [np.float32(1.5976557), np.float32(3.127492), np.float32(3.5490723), np.float32(1.0966821), np.float32(1.0330329), np.float32(2.8902285), np.float32(1.0331116), np.float32(1.0334259), np.float32(1.0330329), np.float32(1.0330329)]
- Variances: [0.5683956146240234, 0.2566765546798706, 4.391202926635742, 0.309431254863739, 0.600429356098175, 2.9359636306762695, 0.6315822005271912, 0.4596315026283264, 0.9310410022735596, 0.34370186924934387]
### Mismatch (Reverse)
- Energies: [np.float32(1.3600694), np.float32(1.5120715), np.float32(3.5263724), np.float32(0.99212855), np.float32(1.0330329), np.float32(3.8321857), np.float32(1.0330167), np.float32(1.032495), np.float32(1.0330329), np.float32(1.0330329)]
- Variances: [1.0491384267807007, 0.6160744428634644, 2.4959611892700195, 0.3508315086364746, 0.600429356098175, 3.8088765144348145, 0.6289705634117126, 0.4586111009120941, 0.9310410022735596, 0.34370186924934387]
### Mismatch (BlockShuffle)
- Energies: [np.float32(1.168896), np.float32(1.3814045), np.float32(1.9431778), np.float32(0.9144451), np.float32(1.0330329), np.float32(1.7905068), np.float32(1.0246301), np.float32(1.0330111), np.float32(1.0330329), np.float32(1.0330329)]
- Variances: [0.658143162727356, 0.5848791599273682, 2.1236624717712402, 0.35219240188598633, 0.600429356098175, 2.818537712097168, 0.6327381134033203, 0.4600772261619568, 0.9310410022735596, 0.34370186924934387]
