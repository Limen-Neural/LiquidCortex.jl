# market_lsm.jl - Liquid State Machine (LSM) Reservoir (2,048 Neurons)
#
# Reference LSM: A simpler 2,048-neuron dense CUDA reservoir for rapid prototyping.
# 16-channel input/output, tanh activation, hardware proprioception inhibition.

using CUDA
using Statistics
using LinearAlgebra

# Reservoir parameters
const REF_N = 2048
const REF_IN = 16
const REF_OUT = 16

# Lazy GPU state — initialized on first `run_lsm_step` when CUDA is functional.
const _market_lsm_lock = ReentrantLock()
_market_lsm_initialized = false

function _ensure_market_reservoir_locked!()
    global W, Win, Wout, x, _market_lsm_initialized
    _market_lsm_initialized && return nothing
    CUDA.functional() ||
        error("run_lsm_step requires a CUDA-capable GPU (CUDA.functional() == false)")
    W = CUDA.randn(Float32, REF_N, REF_N) .* 0.02f0
    Win = CUDA.randn(Float32, REF_N, REF_IN) .* 0.5f0
    Wout = CUDA.randn(Float32, REF_OUT, REF_N) .* 0.1f0
    x = CUDA.zeros(Float32, REF_N)
    _market_lsm_initialized = true
    return nothing
end

function _ensure_market_reservoir!()
    lock(_market_lsm_lock) do
        _ensure_market_reservoir_locked!()
    end
    return nothing
end

"""
    run_lsm_step(inputs_vec, inhibit_val)

`inputs_vec`: 16-element Float32 vector (receptor stimulus)
`inhibit_val`: Hardware thermal stress [0.0, 1.0]
"""
function run_lsm_step(inputs_vec::Vector{Float32}, inhibit_val::Float32)
    _ensure_market_reservoir!()
    global x, W, Win, Wout

    # Move inputs to GPU
    u = cu(inputs_vec)

    # Reservoir dynamics: x(t+1) = tanh(W*x(t) + Win*u(t))
    # Hardware Sync: Proprioception (Temp/Watts) acts as "Inhibitory Signal"
    # High inhibit_val reduces the gain of the recurrent connections.
    gain = 1.0f0 - (inhibit_val * 0.4f0)

    # Step the reservoir
    # x = (1.0 - leak)*x + leak * tanh(...)
    # For a liquid state machine, we can use a simpler version:
    x = tanh.(gain .* (W * x) .+ (Win * u))

    # Readout Layer
    y = Wout * x

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
