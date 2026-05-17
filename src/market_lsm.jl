# market_lsm.jl - Liquid State Machine (LSM) Reservoir (2,048 Neurons)
#
# Reference LSM: A simpler 2,048-neuron dense CUDA reservoir for rapid prototyping.
# 16-channel input/output, tanh activation, hardware proprioception inhibition.
#
# GPU state is lazily initialized in __init__ to allow CPU-only imports.

using CUDA
using Statistics
using LinearAlgebra

# Reservoir parameters
const REF_N = 2048
const REF_IN = 16
const REF_OUT = 16

# ── Lazy-initialized GPU state ─────────────────────────────────────────────────
# These globals start as `nothing` and are allocated only when CUDA is functional.
# This allows `using LiquidCortex` to succeed on CPU-only machines.

const _ref_W = Ref{Union{Nothing, CuMatrix{Float32}}}(nothing)
const _ref_Win = Ref{Union{Nothing, CuMatrix{Float32}}}(nothing)
const _ref_Wout = Ref{Union{Nothing, CuMatrix{Float32}}}(nothing)
const _ref_x = Ref{Union{Nothing, CuVector{Float32}}}(nothing)

"""
    _init_ref_lsm!()

Initialize the 2,048-neuron reference LSM reservoir on GPU.
Called from `LiquidCortex.__init__()` only when `CUDA.functional()` is true.
"""
function _init_ref_lsm!()
    _ref_W[] = CUDA.randn(Float32, REF_N, REF_N) .* 0.02f0
    _ref_Win[] = CUDA.randn(Float32, REF_N, REF_IN) .* 0.5f0
    _ref_Wout[] = CUDA.randn(Float32, REF_OUT, REF_N) .* 0.1f0
    _ref_x[] = CUDA.zeros(Float32, REF_N)
    return nothing
end

"""
    run_lsm_step(inputs_vec, inhibit_val)

`inputs_vec`: 16-element Float32 vector (receptor stimulus)
`inhibit_val`: Hardware thermal stress [0.0, 1.0]
"""
function run_lsm_step(inputs_vec::Vector{Float32}, inhibit_val::Float32)
    W = _ref_W[]
    Win = _ref_Win[]
    Wout = _ref_Wout[]
    x_local = _ref_x[]

    if W === nothing || Win === nothing || Wout === nothing || x_local === nothing
        error("Reference LSM not initialized — no CUDA GPU available. " *
              "Call LiquidCortex on a machine with a functional CUDA device.")
    end

    # Move inputs to GPU
    u = cu(inputs_vec)

    # Reservoir dynamics: x(t+1) = tanh(W*x(t) + Win*u(t))
    # Hardware Sync: Proprioception (Temp/Watts) acts as "Inhibitory Signal"
    # High inhibit_val reduces the gain of the recurrent connections.
    gain = 1.0f0 - (inhibit_val * 0.4f0)

    # Step the reservoir
    x_local .= tanh.(gain .* (W * x_local) .+ (Win * u))
    _ref_x[] = x_local

    # Readout Layer
    y = Wout * x_local

    return Array(y) # Return 16-element vector
end

"""
    run_lsm_step_str(inputs_vec, inhibit_val)

Returns result as a comma-separated string for easier integration.
"""
function run_lsm_step_str(inputs_vec::Vector{Float32}, inhibit_val::Float32)
    y = run_lsm_step(inputs_vec, inhibit_val)
    return join(y, ",")
end