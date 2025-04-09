# Optimization Algorithm Benchmark Results

This document presents benchmark results for various optimization algorithms tested across different environments in PyRosetta.

## Overview

We evaluated several reinforcement learning and optimization algorithms:
- **OCF**: Optimization Control Flow (zero-order and first-order variants)
- **TRPO**: Trust Region Policy Optimization (with discount factors: 0.99, 0.9, 0.8, 0.6)
- **PPO**: Proximal Policy Optimization (with discount factors: 0.99, 0.9, 0.8, 0.6)
- **SAC**: Soft Actor-Critic (with discount factors: 0.99, 0.9, 0.8, 0.6)

Each algorithm was tested on the following environments:
- **shape_boundary**: Boundary optimization for shape-based problems
- **molecule**: Molecular structure optimization
- **shape**: General shape optimization
- **naive** variants of each environment (simplified versions)

## Results Summary

### Shape Boundary Environment

| Algorithm | Score (lower is better) |
|-----------|-------------------------|
| SAC_0_9 | **5.04** |
| SAC_0_6 | 5.88 |
| OCF_first_order | 5.90 |
| TRPO_0_99 | 6.09 |
| TRPO_0_9 | 6.32 |
| OCF_zero_order | 6.76 |
| TRPO_0_8 | 7.47 |
| SAC_0_8 | 11.27 |
| TRPO_0_6 | 12.07 |
| SAC_0_99 | 15.61 |
| PPO_0_99 | 18.94 |
| PPO_0_9 | 19.18 |
| PPO_0_6 | 19.90 |
| PPO_0_8 | 20.07 |

### Naive Shape Boundary Environment

| Algorithm | Score (lower is better) |
|-----------|-------------------------|
| TRPO_0_99 | **6.32** |
| SAC_0_99 | 11.30 |
| PPO_0_99 | 20.39 |

### Molecule Environment

| Algorithm | Score (lower is better) |
|-----------|-------------------------|
| SAC_0_8 | **649.26** |
| SAC_0_99 | 662.40 |
| SAC_0_6 | 680.71 |
| PPO_0_8 | 694.41 |
| SAC_0_9 | 697.64 |
| PPO_0_6 | 700.79 |
| OCF_first_order | 702.92 |
| TRPO_0_99 | 709.48 |
| PPO_0_9 | 712.76 |
| TRPO_0_8 | 715.30 |
| TRPO_0_6 | 715.51 |
| TRPO_0_9 | 720.70 |
| OCF_zero_order | 723.32 |
| PPO_0_99 | 730.40 |

### Naive Molecule Environment

| Algorithm | Score (lower is better) |
|-----------|-------------------------|
| SAC_0_99 | **564.08** |
| PPO_0_99 | 702.24 |
| TRPO_0_99 | 723.17 |

### Shape Environment

| Algorithm | Score (lower is better) |
|-----------|-------------------------|
| TRPO_0_9 | **5.59** |
| OCF_zero_order | 6.08 |
| TRPO_0_8 | 6.56 |
| TRPO_0_99 | 7.04 |
| PPO_0_99 | 7.13 |
| PPO_0_9 | 7.17 |
| OCF_first_order | 7.20 |
| TRPO_0_6 | 7.20 |
| PPO_0_8 | 7.31 |
| SAC_0_99 | 7.34 |
| PPO_0_6 | 7.39 |
| SAC_0_9 | 7.58 |
| SAC_0_8 | 8.14 |
| SAC_0_6 | 8.31 |

### Naive Shape Environment

| Algorithm | Score (lower is better) |
|-----------|-------------------------|
| TRPO_0_99 | **6.93** |
| PPO_0_99 | 7.33 |
| SAC_0_99 | 9.62 |

## Key Findings

1. **Best Performers By Environment:**
   
    | **Environment** | **Top Score** | **Second Best** |
    |-----------------|---------------|-----------------|
    | Shape Boundary | 	SAC γ = 0.9 (5.04) | SAC γ = 0.6 (5.88) |
    | Naive Shape Boundary | TRPO γ = 0.99 (6.32) | — |
    | Molecule | 9.62 | SAC γ = 0.8 (649.26) | SAC γ = 0.99 (662.40) |
    | Naive Molecule | SAC γ = 0.99 (564.08) | — |
    | Shape | TRPO γ = 0.9 (5.59) | OCF_zero_order (6.08) |
    | Naive Shape| TRPO γ = 0.99 (6.93) | — |

2. **Algorithm Performance Patterns:**
   - **SAC** performs exceptionally well on molecular optimization and certain shape boundary tasks.
   - **OCF** is competitive on shape‑based environments (2nd place on shape) but is mid‑pack elsewhere.
   - **TRPO** offers balanced performance: it wins both naïve shape tasks and is always mid‑table or better, but it trails SAC on naïve_molecule.
   - **PPO** underperforms relative to the other methods in every environment (never places in the top three).

3. **Discount Factor Impact:**
   - On shape‑based tasks, γ = 0.9 yields the best TRPO score (shape) and the best SAC score (shape_boundary).
   - For molecule optimisation, γ = 0.99 (SAC) is the second‑best result and the clear winner in naïve_molecule.
   - Lower γ values (0.8 or 0.6) occasionally help (e.g., SAC 0.8 on molecule), but gains are environment‑specific.

4. **Standard vs. Naive Environments:**
   - SAC improves markedly when the molecule task is simplified (564 → 649).
   - TRPO 0.99 dominates both naïve shape variants but not naïve_molecule.
   - Only three algorithms were benchmarked on naïve tasks, so cross‑algorithm comparisons there are limited.

## Conclusions

- **No Single Winner**: No algorithm dominates across all environments, suggesting task-specific selection is important
- **SAC Recommendation**: For molecular optimization, SAC with appropriate discount factors shows superior performance
- **OCF Efficiency**: OCF_zero_order provides strong shape performance with a simpler optimisation routine.
- **Discount Factor Tuning**: Careful tuning of discount factors can significantly improve algorithm performance
- **Naive vs. Standard**: The significant performance improvement of SAC in naive environments suggests that simplification of certain problem spaces may benefit particular algorithms