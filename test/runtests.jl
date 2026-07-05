using Test
using LiquidCortex
using CUDA

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
                    :compute_reservoir_covariance!, :diagnostics, :ensemble_diagnostics]
            @test sym in exports
        end
    end

    @testset "Removed domain symbols are NOT exported" begin
        exports = names(LiquidCortex; all=false)
        for sym in [:MarketPulse, :decode_market_pulse, :pulse_to_input]
            @test !(sym in exports)
        end
    end

    # ── GPU tests (only run when CUDA is available) ──────────────────────────

    if LiquidCortex._cuda_available[]
        @testset "GPU: SparseBrain default dims" begin
            brain = SparseBrain(20.0f0; name="test")
            @test brain isa SparseBrain
            @test brain.tau_m == 20.0f0
            @test brain.tick_count == 0
            @test brain.n_in == 14
            @test brain.n_out == 16
        end

        @testset "GPU: SparseBrain custom dims" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="custom")
            @test brain isa SparseBrain
            @test brain.n_in == 8
            @test brain.n_out == 4
            @test length(brain.output) == 4
            @test size(brain.W_in, 2) == 8
        end

        @testset "GPU: EnsembleBrain default dims" begin
            ensemble = EnsembleBrain()
            @test ensemble isa EnsembleBrain
            @test length(ensemble.lobes) == 4
            @test ensemble.lobes[1].n_in == 14
            @test ensemble.lobes[1].n_out == 16
        end

        @testset "GPU: EnsembleBrain custom dims" begin
            ensemble = EnsembleBrain(n_in=8, n_out=4)
            @test ensemble isa EnsembleBrain
            @test length(ensemble.lobes) == 4
            @test ensemble.lobes[1].n_in == 8
            @test ensemble.lobes[1].n_out == 4
        end

        @testset "GPU: step! with generic inhibition" begin
            brain = SparseBrain(20.0f0; n_in=8, n_out=4, name="step-test")
            u = CUDA.zeros(Float32, 8)
            step!(brain, u; inhibition=0.5f0)
            @test brain.tick_count == 1
            @test brain.v_thresh_dynamic > LiquidCortex.V_THRESH
        end

        @testset "GPU: ensemble_step! with generic inhibition" begin
            ensemble = EnsembleBrain(n_in=8, n_out=4)
            u = CUDA.zeros(Float32, 8)
            ensemble_step!(ensemble, u; inhibition=0.3f0, reflex_signal=0.2f0)
            output = get_ensemble_output(ensemble)
            @test length(output) == 4
        end
    else
        @info "Skipping GPU tests — no CUDA device available"
        @test_skip "GPU tests skipped (no CUDA)"
    end

end
