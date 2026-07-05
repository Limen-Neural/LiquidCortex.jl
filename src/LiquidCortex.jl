# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    LiquidCortex

GPU-accelerated sparse Liquid State Machine for neuromorphic computing.

Provides two LSM implementations:
- **EnsembleBrain** (`sparse_brain.jl`) — 4-lobe, 65,536-neuron/lobe sparse CUDA LSM
  with OU-SDE dynamics, STDP covariance learning, and rolling 1,000-tick spike history.
  Configurable input/output dimensions. Requires RTX-class GPU with ≥14 GB VRAM.

- **Reference LSM** (`reference_lsm.jl`) — 2,048-neuron dense CUDA reservoir for
  rapid prototyping.

On CPU-only systems, the module loads cleanly — types and API functions are defined
but GPU allocations are deferred until a CUDA device is available at runtime.
Check `LiquidCortex._cuda_available[]` to test CUDA availability.
"""
module LiquidCortex

using CUDA

# ── CUDA availability flag ────────────────────────────────────────────────
# Checked at __init__ time. All GPU allocations are deferred until this is true.
const _cuda_available = Ref{Bool}(false)

function __init__()
    if CUDA.functional()
        _cuda_available[] = true
        @info "LiquidCortex: CUDA functional — GPU kernels available on $(CUDA.name(CUDA.device()))."
        # Eagerly initialize the reference LSM reservoir on the GPU
        _init_ref_lsm!()
    else
        @warn "LiquidCortex: No CUDA-capable GPU found. " *
              "Core types will load, but step! and GPU operations require a CUDA device."
    end
end

# ── GPU source files (structs defined at load; GPU allocations deferred to
#    constructors/runtime, guarded by _cuda_available[]) ─────────────────────
include("sparse_brain.jl")
include("reference_lsm.jl")

# ── Public API ───────────────────────────────────────────────────────────────

export SparseBrain, EnsembleBrain
export step!, ensemble_step!, get_output, get_ensemble_output
export compute_reservoir_covariance!, diagnostics, ensemble_diagnostics

end # module LiquidCortex
