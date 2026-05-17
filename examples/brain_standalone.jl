# Standalone brain test — requires CUDA GPU
# Run: julia --project -e 'include("examples/brain_standalone.jl")'

using CUDA
using SparseArrays
using Statistics
using LinearAlgebra
using Printf

open("/tmp/brain_test.log", "w") do io
    println(io, "Starting Julia Brain test...")
end

# Try loading packages
using ZMQ

open("/tmp/brain_test.log", "a") do io
    println(io, "All packages loaded successfully!")
    println(io, "CUDA device: ", CUDA.name(CUDA.device()))
end

include(joinpath(@__DIR__, "..", "src", "sparse_brain.jl"))

open("/tmp/brain_test.log", "a") do io
    println(io, "sparse_brain.jl included")
end