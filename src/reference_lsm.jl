# SPDX-License-Identifier: MIT OR Apache-2.0
#
# reference_lsm.jl — Reference Liquid State Machine (2,048 Neurons)
#
# A simple 2,048-neuron dense CUDA reservoir for rapid prototyping.
# Configurable input/output dimensions, tanh activation, generic inhibition.
#
# GPU state is lazily initialized in __init__ to allow CPU-only imports.

using CUDA
using Statistics
using LinearAlgebra

# Reservoir parameters
const REF_N = 2048
const REF_IN_DEFAULT = 16
const REF_OUT_DEFAULT = 16

# ── Lazy-initialized GPU state ─────────────────────────────────────────────────
# These globals start as `nothing` and are allocated only when CUDA is functional.
# This allows `using LiquidCortex` to succeed on CPU-only machines.

const _ref_W = Ref{Union{Nothing, CuMatrix{Float32}}}(nothing)
const _ref_Win = Ref{Union{Nothing, CuMatrix{Float32}}}(nothing)
const _ref_Wout = Ref{Union{Nothing, CuMatrix{Float32}}}(nothing)
const _ref_x = Ref{Union{Nothing, CuVector{Float32}}}(nothing)
const _ref_n_in = Ref{Int}(REF_IN_DEFAULT)
const _ref_n_out = Ref{Int}(REF_OUT_DEFAULT)

"""
    _init_ref_lsm!(; n_in=16, n_out=16)

Initialize the 2,048-neuron reference LSM reservoir on GPU.
Called from `LiquidCortex.__init__()` only when `CUDA.functional()` is true.
"""
function _init_ref_lsm!(; n_in::Int=REF_IN_DEFAULT, n_out::Int=REF_OUT_DEFAULT)
    n_in > 0 || throw(ArgumentError("n_in must be positive, got $n_in"))
    n_out > 0 || throw(ArgumentError("n_out must be positive, got $n_out"))
    _ref_W[] = CUDA.randn(Float32, REF_N, REF_N) .* 0.02f0
    _ref_Win[] = CUDA.randn(Float32, REF_N, n_in) .* 0.5f0
    _ref_Wout[] = CUDA.randn(Float32, n_out, REF_N) .* 0.1f0
    _ref_x[] = CUDA.zeros(Float32, REF_N)
    _ref_n_in[] = n_in
    _ref_n_out[] = n_out
    return nothing
end

"""
    run_lsm_step(inputs_vec, inhibit_val)

`inputs_vec`: Float32 vector matching the configured input dimension.
`inhibit_val`: Inhibition signal [0.0, 1.0]
"""
function run_lsm_step(inputs_vec::Vector{Float32}, inhibit_val::Float32)
    # Lazy initialization on first call
    if _ref_W[] === nothing || _ref_Win[] === nothing || _ref_Wout[] === nothing || _ref_x[] === nothing
        LiquidCortex._cuda_available[] || error(
            "Reference LSM requires a CUDA GPU. No CUDA device available.")
        n = length(inputs_vec)
        _init_ref_lsm!(; n_in=n, n_out=REF_OUT_DEFAULT)

    end

    W = _ref_W[]
    Win = _ref_Win[]
    Wout = _ref_Wout[]
    x_local = _ref_x[]

    length(inputs_vec) == _ref_n_in[] || throw(DimensionMismatch(
        "input has length $(length(inputs_vec)), expected $(_ref_n_in[])"))

    # Move inputs to GPU
    u = cu(inputs_vec)

    # Reservoir dynamics: x(t+1) = tanh(W*x(t) + Win*u(t))
    # High inhibit_val reduces the gain of the recurrent connections.
    gain = 1.0f0 - (inhibit_val * 0.4f0)

    # Step the reservoir
    x_local .= tanh.(gain .* (W * x_local) .+ (Win * u))
    _ref_x[] = x_local

    # Readout Layer
    y = Wout * x_local

    return Array(y)
end

"""
    run_lsm_step_str(inputs_vec, inhibit_val)

Returns result as a comma-separated string for easier integration.
"""
function run_lsm_step_str(inputs_vec::Vector{Float32}, inhibit_val::Float32)
    y = run_lsm_step(inputs_vec, inhibit_val)
    return join(y, ",")
end
