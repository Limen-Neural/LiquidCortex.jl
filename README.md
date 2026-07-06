<p align="center">
  <img src="docs/logo.png" width="220" alt="LiquidCortex">
</p>

<h1 align="center">LiquidCortex.jl</h1>
<p align="center">GPU-accelerated sparse liquid state machine for neuromorphic computing</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Julia-9558B2" alt="Julia">
  <img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue" alt="MIT/Apache-2.0">
</p>

---

Production-grade CUDA-accelerated sparse Liquid State Machine (LSM) with
OU-SDE membrane dynamics, multi-lobe ensemble architecture, cuSPARSE mat-vec,
and STDP covariance learning.

## Features

- `SparseBrain` — configurable reservoir: N neurons, connectivity probability, Float16 sparse weights on GPU
- Configurable input/output dimensions (`n_in`, `n_out`)
- OU-SDE dynamics: `dV = ((V_rest - V)/τ + I_rec + I_ext)dt + σ dW`
- cuSPARSE Float16 sparse mat-vec on GPU (fits 65k neurons in 16 GB VRAM)
- STDP covariance learning with eligibility traces
- 1000-tick rolling spike history buffer (circular, on-GPU)
- `EnsembleBrain` — multi-lobe: multiple reservoirs with different time constants
- Generic inhibition interface — caller provides a stress signal

## Installation

```julia
using Pkg
Pkg.add("LiquidCortex")
```

## Quick Start

```julia
using LiquidCortex

# Create a 65,536-neuron sparse LSM lobe
brain = SparseBrain(20.0f0)  # τ_m = 20ms, default n_in=14, n_out=16

# Or with custom dimensions
brain = SparseBrain(20.0f0; n_in=8, n_out=4)

# Or create the full 4-lobe ensemble (262,144 neurons)
ensemble = EnsembleBrain()

# Step the reservoir with an input vector
u = CUDA.zeros(Float32, 14)
step!(brain, u; inhibition=0.3f0)

# Read the output
output = get_output(brain)
```

## Public API

| Type / Function | Description |
|-----------------|-------------|
| `SparseBrain(tau_m; n_in, n_out)` | Create a 65,536-neuron sparse reservoir lobe |
| `EnsembleBrain(; n_in, n_out)` | Create 4-lobe ensemble (262,144 neurons) |
| `step!(brain, u; inhibition, reflex_eta)` | Execute one simulation timestep |
| `ensemble_step!(eb, u; inhibition, reflex_eta, reflex_signal)` | Step all lobes and aggregate |
| `get_output(brain)` | Copy readout from GPU to CPU |
| `get_ensemble_output(eb)` | Copy aggregated readout |
| `compute_reservoir_covariance!(brain)` | Compute subsampled covariance matrix |
| `diagnostics(brain)` | Return diagnostic string |
| `ensemble_diagnostics(eb)` | Per-lobe diagnostic summary |

## OU-SDE Membrane Dynamics

```
dV = ((V_rest - V)/τ  +  W_rec·s(t)  +  W_in·x(t)) dt  +  σ dW
```

Discretized as Euler-Maruyama. Spike when `V ≥ θ`; reset to `V_rest`.

*Ornstein & Uhlenbeck (1930); Maass, Natschläger & Markram (2002)*

## STDP Covariance Learning

```
ΔW_ij = η (⟨s_i s_j⟩ - ⟨s_i⟩⟨s_j⟩)
```

Computed on a subsampled 8192-neuron window to avoid O(N²) blow-up.

*Bi & Poo (1998); Hebb (1949)*

## Provenance

Extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander), a private
neuromorphic GPU supervisor. The LSM core has been fully decoupled from
domain-specific logic so it works with any time-series application.

## License

Licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
