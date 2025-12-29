# Iteration 1.5 Ablation & Consistency Audit
Env: LimitCycle (Pure Decay), Seeds: 10

## 1. Discrimination Power (Replay vs Shuffle)
- Plastic Metric: Weight Norm
- Resonant Metric: Resonator Energy

```
  Kernel  Replay (Mean)  Shuffle (Mean)  p-value  DistCheck (Replay u^2)  DistCheck (Shuffle u^2)  Shuffle Energy Ratio
 plastic       0.285709        0.119256 0.004097                0.341955                 0.340544              0.995873
resonant       3.079354        1.593778 0.014445                0.341955                 0.340544              0.995873
```

## 2. Distribution Consistency Check
If 'Shuffle Energy Ratio' is close to 1.0, the shuffle is chemically pure (energy preserving).
