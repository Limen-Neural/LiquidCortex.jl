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

# Persistent state (stored in module scope)
if !isdefined(Main, :reservoir_initialized)
    # Use float32 for performance on RTX 5080
    global W = CUDA.randn(Float32, REF_N, REF_N) .* 0.02f0
    # Spectral radius check is slow; using empirical initialization for dense matrix.

    global Win = CUDA.randn(Float32, REF_N, REF_IN) .* 0.5f0
    global Wout = CUDA.randn(Float32, REF_OUT, REF_N) .* 0.1f0

    global x = CUDA.zeros(Float32, REF_N)

    global reservoir_initialized = true
end

"""
    run_lsm_step(inputs_vec, inhibit_val)

`inputs_vec`: 16-element Float32 vector (receptor stimulus)
`inhibit_val`: Hardware thermal stress [0.0, 1.0]
"""
function run_lsm_step(inputs_vec::Vector{Float32}, inhibit_val::Float32)
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