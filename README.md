<p align="center">
  <img src="docs/logo.png" width="220" alt="LiquidCortex">
</p>

<h1 align="center">LiquidCortex.jl</h1>
<p align="center">GPU-accelerated sparse liquid state machine for neuromorphic computing</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Julia-9558B2" alt="Julia">
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

Production-grade CUDA-accelerated sparse Liquid State Machine (LSM) with
OU-SDE membrane dynamics, multi-lobe ensemble architecture, cuSPARSE mat-vec,
and STDP covariance learning.

## Features

- `SparseBrain` — configurable reservoir: N neurons, connectivity probability, Float16 sparse weights on GPU
- OU-SDE dynamics: `dV = ((V_rest - V)/τ + I_rec + I_ext)dt + σ dW`
- cuSPARSE Float16 sparse mat-vec on GPU (fits 65k neurons in 16 GB VRAM)
- STDP covariance learning with eligibility traces
- 1000-tick rolling spike history buffer (circular, on-GPU)
- `EnsembleBrain` — multi-lobe: multiple reservoirs with different time constants
- Hardware proprioception: global inhibition driven by external stress signals
- `MarketPulse` — optional domain-specific input decoding for financial time-series

## Installation

```julia
using Pkg
Pkg.add("LiquidCortex")
```

## Quick Start

```julia
using LiquidCortex

# Create a 65,536-neuron sparse LSM lobe
brain = SparseBrain(20.0f0)  # τ_m = 20ms

# Or create the full 4-lobe ensemble (262,144 neurons)
ensemble = EnsembleBrain()

# Step the reservoir with a 14-element input vector
u = CUDA.zeros(Float32, 14)  # or use pulse_to_input(market_pulse)
step!(brain, u, 45.0f0, 0.3f0)  # (input, gpu_temp, buffer_load)

# Read the 16-element output
output = get_output(brain)
```

## Public API

| Type / Function | Description |
|-----------------|-------------|
| `SparseBrain(tau_m)` | Create a 65,536-neuron sparse reservoir lobe |
| `EnsembleBrain()` | Create 4-lobe ensemble (262,144 neurons) |
| `step!(brain, u, gpu_temp, basys_load; ...)` | Execute one simulation timestep |
| `get_output(brain)` | Copy 16-element readout from GPU to CPU |
| `get_ensemble_output(eb)` | Copy aggregated 16-element readout |
| `compute_reservoir_covariance!(brain)` | Compute subsampled covariance matrix |
| `diagnostics(brain)` | Return diagnostic string |
| `ensemble_diagnostics(eb)` | Per-lobe diagnostic summary |
| `MarketPulse` | Market data packed struct (120 bytes) |
| `decode_market_pulse(buf)` | Zero-copy decode from byte buffer |
| `pulse_to_input(pulse)` | Convert MarketPulse → 14-element input vector |

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
domain-specific ingestion and execution so it works with any time-series
application.

## License

GPL-3.0-or-later