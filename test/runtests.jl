using Test
using LiquidCortex

@testset "LiquidCortex" begin

    @testset "Package loads" begin
        @test @isdefined(LiquidCortex)
        @test LiquidCortex isa Module
    end

    @testset "Public API is exported" begin
        # Verify each name is both defined AND exported (not just defined)
        exports = names(LiquidCortex; all=false)
        for sym in [:SparseBrain, :EnsembleBrain,
                    :step!, :ensemble_step!, :get_output, :get_ensemble_output,
                    :compute_reservoir_covariance!, :diagnostics, :ensemble_diagnostics,
                    :MarketPulse, :decode_market_pulse, :pulse_to_input]
            @test sym in exports
        end
    end

    # ── CPU-only tests (always run, no GPU needed) ──────────────────────────

    @testset "CPU: MarketPulse decode" begin
        # Build a 120-byte buffer (zeros)
        buf = zeros(UInt8, 120)
        # Set timestamp at offset 0 (UInt64)
        buf[1:8] = reinterpret(UInt8, [UInt64(42)])
        # Set first float at offset 8 (dnx_price = 1.0f32)
        buf[9:12] = reinterpret(UInt8, [Float32(1.0)])
        # Set Qubic fields at offsets 108-120
        buf[109:112] = reinterpret(UInt8, [Float32(0.5)])   # qubic_tick_trace
        buf[113:116] = reinterpret(UInt8, [Float32(100.0)]) # qubic_tick_rate
        buf[117:120] = reinterpret(UInt8, [Float32(0.75)])  # qubic_epoch_progress

        pulse = decode_market_pulse(buf)
        @test pulse.timestamp_ns == 42
        @test pulse.dnx_price ≈ 1.0f0
        @test pulse.qubic_tick_trace ≈ 0.5f0
        @test pulse.qubic_tick_rate ≈ 100.0f0
        @test pulse.qubic_epoch_progress ≈ 0.75f0
    end

    @testset "CPU: pulse_to_input vector" begin
        pulse = MarketPulse(
            UInt64(0),
            1.0f0, 0.1f0,   # dnx
            2.0f0, 0.2f0,   # quai
            3.0f0, 0.3f0,   # qubic
            4.0f0, 0.4f0,   # kaspa
            5.0f0, 0.5f0,   # monero
            6.0f0, 0.6f0,   # ocean
            7.0f0, 0.7f0,   # verus
            0.0f0,          # confidence
            0.0f0, 0.0f0, 0.0f0, 0.0f0,  # institutional
            45.0f0, 200.0f0, 95.0f0, 0.3f0,  # hardware
            0.0f0, 0.0f0,   # dYdX
            0.0f0, 0.0f0, 0.0f0  # Qubic
        )
        input_vec = pulse_to_input(pulse)
        @test length(input_vec) == 14
        @test input_vec[1] ≈ 1.0f0   # dnx_price
        @test input_vec[2] ≈ 0.1f0   # dnx_vol
    end

    @testset "CPU: decode_market_pulse rejects bad input" begin
        @test_throws ArgumentError decode_market_pulse(zeros(UInt8, 100))
        @test_throws ArgumentError decode_market_pulse(zeros(UInt8, 200))
    end

    # ── GPU tests (only run when CUDA is available) ──────────────────────────

    if LiquidCortex._cuda_available[]
        @testset "GPU: SparseBrain constructor" begin
            brain = SparseBrain(20.0f0; name="test")
            @test brain isa SparseBrain
            @test brain.tau_m == 20.0f0
            @test brain.tick_count == 0
        end

        @testset "GPU: EnsembleBrain constructor" begin
            ensemble = EnsembleBrain()
            @test ensemble isa EnsembleBrain
            @test length(ensemble.lobes) == 4
        end

        @testset "GPU: pulse_to_input returns CuVector" begin
            using CUDA
            buf = zeros(UInt8, 120)
            pulse = decode_market_pulse(buf)
            input_vec = pulse_to_input(pulse)
            @test input_vec isa CuVector{Float32}
            @test length(input_vec) == 14
        end
    else
        @info "Skipping GPU tests — no CUDA device available"
        @test_skip "GPU tests skipped (no CUDA)"
    end

end