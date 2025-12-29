# Iteration 0.1 Verification Results

```
           Condition  d2_max_eig  d2_gap  d2_ratio  d3_correlation  d3_gain  d3_amplification  d1_variance  d1_speed  d1_fixed_point  d1_periodicity  d1_lag5_corr  d1_max_excursion
            Baseline         1.0     0.0       1.0       -0.966292 2.005755               NaN     2.545837  1.868068             0.0        0.854370      0.624959          4.613805
   Ablation (Zero U)         1.0     0.0       1.0             NaN      NaN               0.0     3.523499  2.215040             0.0        0.820057      0.509420          4.638992
Mismatch (Shuffle U)         NaN     NaN       NaN        0.017010 2.002848               NaN     2.545837  1.868068             0.0        0.854370      0.624959          4.613805
    FastSlow (e=1.0)         1.0     0.0       1.0       -0.966292 2.005755               NaN     2.545837  1.868068             0.0        0.854370      0.624959          4.613805
    FastSlow (e=0.1)         1.0     0.0       1.0       -0.812228 1.433774               NaN     1.400907  2.087132             0.0        0.947044      0.788088          3.684538
   FastSlow (e=0.01)         1.0     0.0       1.0       -0.756792 1.395132               NaN     1.264182  2.087362             0.0        0.951192      0.807705          3.121424
```
