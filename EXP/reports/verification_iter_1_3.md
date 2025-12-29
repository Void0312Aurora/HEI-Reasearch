# Iteration 1.3 Plasticity Verification (N=10)
## Hypothesis: Causal Replay -> Higher Learned Weight Norm

```
              Condition WeightNorm (Mean) WeightNorm (Std)    CI95 p-value
       Replay (Ordered)            0.3145           0.0859 ±0.0533       -
     Mismatch (Reverse)            0.3332           0.0691 ±0.0428 0.61785
Mismatch (BlockShuffle)            0.2386           0.0816 ±0.0506 0.07044
```
