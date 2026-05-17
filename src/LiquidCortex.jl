"""
    LiquidCortex

GPU-accelerated sparse Liquid State Machine for neuromorphic computing.

Provides two LSM implementations:
- **EnsembleBrain** (`sparse_brain.jl`) — 4-lobe, 65,536-neuron/lobe sparse CUDA LSM
  with OU-SDE dynamics, STDP covariance learning, and rolling 1,000-tick spike history.
  Requires RTX-class GPU with ≥14 GB VRAM.

- **Reference LSM** (`market_lsm.jl`) — 2,048-neuron dense CUDA reservoir with
  16-channel input/output for rapid prototyping.

Optional market integration (`market_pulse.jl`) provides domain-specific input
decoding for time-series applications.

Both GPU implementations are conditionally loaded when a CUDA GPU is available.

## Minimal Usage

```julia
using LiquidCortex

brain = SparseBrain(20.0f0)       # 65,536-neuron sparse reservoir
ensemble = EnsembleBrain()        # 4-lobe, 262,144 neurons
```
"""
module LiquidCortex

using CUDA

function __init__()
    if !CUDA.functional()
        @warn "No CUDA-capable GPU found. LiquidCortex GPU kernels are unavailable. " *
              "Core types will load, but step! and GPU operations require a CUDA device."
    end
end

# ── Core reservoir (always loaded — structs and constructors are CPU-safe) ──
include("sparse_brain.jl")
include("market_lsm.jl")

# ── Optional market integration (always available, user-facing) ──
include("market_pulse.jl")

# ── Public API ───────────────────────────────────────────────────────────────

export SparseBrain, EnsembleBrain
export step!, ensemble_step!, get_output, get_ensemble_output
export compute_reservoir_covariance!, diagnostics, ensemble_diagnostics
export MarketPulse, decode_market_pulse, pulse_to_input

end # module LiquidCortex