using Test
using LiquidCortex
using CUDA

@testset "LiquidCortex" begin

    @testset "Package loads" begin
        @test @isdefined(LiquidCortex)
        @test LiquidCortex isa Module
    end

    @testset "Public API is exported" begin
        # Core types
        @test isdefined(LiquidCortex, :SparseBrain)
        @test isdefined(LiquidCortex, :EnsembleBrain)

        # Core methods
        @test isdefined(LiquidCortex, :step!)
        @test isdefined(LiquidCortex, :get_output)
        @test isdefined(LiquidCortex, :get_ensemble_output)
        @test isdefined(LiquidCortex, :compute_reservoir_covariance!)
        @test isdefined(LiquidCortex, :diagnostics)
        @test isdefined(LiquidCortex, :ensemble_diagnostics)

        # Market integration
        @test isdefined(LiquidCortex, :MarketPulse)
        @test isdefined(LiquidCortex, :decode_market_pulse)
        @test isdefined(LiquidCortex, :pulse_to_input)
    end

    if CUDA.functional()
        @testset "GPU: SparseBrain constructor" begin
            brain = SparseBrain(20.0f0; name="test")
            @test brain isa SparseBrain
            @test brain.tau_m == 20.0f0
            @test brain.tick_count == 0
        end

        @testset "GPU: MarketPulse decode" begin
            # Build a minimal 120-byte buffer (zeros)
            buf = zeros(UInt8, 120)
            # Set timestamp at offset 0 (UInt64)
            buf[1:8] = reinterpret(UInt8, [UInt64(42)])
            # Set first float at offset 8 (dnx_price = 1.0f32)
            buf[9:12] = reinterpret(UInt8, [Float32(1.0)])

            pulse = decode_market_pulse(buf)
            @test pulse.timestamp_ns == 42
            @test pulse.dnx_price ≈ 1.0f0

            # pulse_to_input returns 14-element vector
            input_vec = pulse_to_input(pulse)
            @test length(input_vec) == 14
        end
    else
        @info "Skipping GPU tests — no CUDA device available"
        @test_skip "GPU tests skipped (no CUDA)"
    end

end