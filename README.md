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
| OCF_first_order | **5.90** |
| SAC_0_9 | **5.04** |
| TRPO_0_99 | 6.09 |
| TRPO_0_9 | 6.32 |
| OCF_zero_order | 6.76 |
| TRPO_0_8 | 7.47 |
| SAC_0_6 | 5.88 |
| SAC_0_8 | 11.27 |
| TRPO_0_6 | 12.07 |
| SAC_0_99 | 15.61 |
| PPO_0_99 | 18.94 |
| PPO_0_9 | 19.18 |
| PPO_0_6 | 19.90 |
| PPO_0_8 | 20.07 |

**Naive Shape Boundary Environment Results:**
- TRPO_0_99: 6.32
- SAC_0_99: 11.30
- PPO_0_99: 20.39

### Molecule Environment

| Algorithm | Score (lower is better) |
|-----------|-------------------------|
| SAC_0_99 | **662.40** |
| SAC_0_8 | **649.26** |
| SAC_0_6 | 680.71 |
| SAC_0_9 | 697.64 |
| PPO_0_8 | 694.41 |
| PPO_0_6 | 700.79 |
| OCF_first_order | 702.92 |
| PPO_0_9 | 712.76 |
| TRPO_0_8 | 715.30 |
| TRPO_0_6 | 715.51 |
| TRPO_0_9 | 720.70 |
| OCF_zero_order | 723.32 |
| TRPO_0_99 | 709.48 |
| PPO_0_99 | 730.40 |

**Naive Molecule Environment Results:**
- SAC_0_99: **564.08**
- PPO_0_99: 702.24
- TRPO_0_99: 723.17

### Shape Environment

| Algorithm | Score (lower is better) |
|-----------|-------------------------|
| TRPO_0_9 | **5.59** |
| OCF_zero_order | **6.08** |
| TRPO_0_8 | 6.56 |
| TRPO_0_99 | 7.04 |
| OCF_first_order | 7.20 |
| TRPO_0_6 | 7.20 |
| PPO_0_99 | 7.13 |
| PPO_0_9 | 7.17 |
| PPO_0_8 | 7.31 |
| SAC_0_99 | 7.34 |
| PPO_0_6 | 7.39 |
| SAC_0_9 | 7.58 |
| SAC_0_8 | 8.14 |
| SAC_0_6 | 8.31 |

**Naive Shape Environment Results:**
- TRPO_0_99: 6.93
- PPO_0_99: 7.33
- SAC_0_99: 9.62

## Key Findings

1. **Best Performers By Environment:**
   - **Shape Boundary**: SAC with γ=0.9 (5.04) and OCF_first_order (5.90)
   - **Molecule**: SAC with γ=0.8 (649.26) and SAC with γ=0.99 (662.40)
   - **Shape**: TRPO with γ=0.9 (5.59) and OCF_zero_order (6.08)

2. **Algorithm Performance Patterns:**
   - **SAC** performs exceptionally well on molecular optimization
   - **OCF** methods show strong and consistent performance on shape-related tasks
   - **TRPO** demonstrates good balance across all environments
   - **PPO** generally underperforms on shape_boundary but shows competitive results on other environments

3. **Discount Factor Impact:**
   - Different discount factors significantly affect performance
   - γ=0.9 often produces better results than γ=0.99, especially for TRPO and SAC
   - Lower discount factors (γ=0.8, γ=0.6) occasionally outperform higher values in specific scenarios

4. **Naive vs. Standard Environments:**
   - SAC shows dramatic improvement in the naive_molecule environment (564.08)
   - Most algorithms perform similarly or slightly worse on naive variants
   - The performance gap between algorithms is generally maintained across variants

## Conclusions

- **No Single Winner**: No algorithm dominates across all environments, suggesting task-specific selection is important
- **SAC Recommendation**: For molecular optimization, SAC with appropriate discount factors shows superior performance
- **OCF Efficiency**: OCF methods provide excellent performance for shape optimization with potentially lower computational costs
- **Discount Factor Tuning**: Careful tuning of discount factors can significantly improve algorithm performance