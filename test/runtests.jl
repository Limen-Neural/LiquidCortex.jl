using Test
using LiquidCortex
using CUDA

function snapshot_reference_lsm_state()
    # Copy reservoir state: run_lsm_step mutates _ref_x[] in-place.
    x_snap = LiquidCortex._ref_x[]
    return (
        W=LiquidCortex._ref_W[],
        Win=LiquidCortex._ref_Win[],
        Wout=LiquidCortex._ref_Wout[],
        x=isnothing(x_snap) ? nothing : copy(x_snap),
        n_in=LiquidCortex._ref_n_in[],
        n_out=LiquidCortex._ref_n_out[],
        initialized=LiquidCortex._ref_initialized[],
    )
end

function clear_reference_lsm_state!()
    LiquidCortex._ref_W[] = nothing
    LiquidCortex._ref_Win[] = nothing
    LiquidCortex._ref_Wout[] = nothing
    LiquidCortex._ref_x[] = nothing
    LiquidCortex._ref_n_in[] = LiquidCortex.REF_IN_DEFAULT
    LiquidCortex._ref_n_out[] = LiquidCortex.REF_OUT_DEFAULT
    LiquidCortex._ref_initialized[] = false
    return nothing
end

function restore_reference_lsm_state!(state)
    LiquidCortex._ref_W[] = state.W
    LiquidCortex._ref_Win[] = state.Win
    LiquidCortex._ref_Wout[] = state.Wout
    LiquidCortex._ref_x[] = state.x
    LiquidCortex._ref_n_in[] = state.n_in
    LiquidCortex._ref_n_out[] = state.n_out
    LiquidCortex._ref_initialized[] = state.initialized
    return nothing
end

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
        @testset "GPU: Reference LSM lazy initialization" begin
            original_state = snapshot_reference_lsm_state()
            try
                clear_reference_lsm_state!()
                @test !LiquidCortex._ref_is_initialized()

                output = LiquidCortex.run_lsm_step(
                    zeros(Float32, LiquidCortex.REF_IN_DEFAULT),
                    0.5f0,
                )

                @test LiquidCortex._ref_is_initialized()
                @test length(output) == LiquidCortex.REF_OUT_DEFAULT
                @test size(LiquidCortex._ref_Win[]) == (
                    LiquidCortex.REF_N,
                    LiquidCortex.REF_IN_DEFAULT,
                )
                @test size(LiquidCortex._ref_Wout[]) == (
                    LiquidCortex.REF_OUT_DEFAULT,
                    LiquidCortex.REF_N,
                )
            finally
                restore_reference_lsm_state!(original_state)
            end
        end

        @testset "GPU: Reference LSM custom dimensions" begin
            original_state = snapshot_reference_lsm_state()
            try
                clear_reference_lsm_state!()
                LiquidCortex._init_ref_lsm!(; n_in=8, n_out=4)

                @test LiquidCortex._ref_is_initialized()
                @test LiquidCortex._ref_n_in[] == 8
                @test LiquidCortex._ref_n_out[] == 4
                @test size(LiquidCortex._ref_Win[]) == (LiquidCortex.REF_N, 8)
                @test size(LiquidCortex._ref_Wout[]) == (4, LiquidCortex.REF_N)

                output = LiquidCortex.run_lsm_step(zeros(Float32, 8), 0.0f0)
                @test length(output) == 4
            finally
                restore_reference_lsm_state!(original_state)
            end
        end

        @testset "GPU: Reference LSM dimension validation" begin
            original_state = snapshot_reference_lsm_state()
            try
                clear_reference_lsm_state!()
                @test_throws ArgumentError LiquidCortex._init_ref_lsm!(; n_in=0, n_out=4)
                @test_throws ArgumentError LiquidCortex._init_ref_lsm!(; n_in=4, n_out=0)

                # Hardcode n_in so the mismatch vs input length 8 is guaranteed
                # regardless of REF_IN_DEFAULT.
                LiquidCortex._init_ref_lsm!(; n_in=6, n_out=4)
                @test_throws DimensionMismatch LiquidCortex.run_lsm_step(
                    zeros(Float32, 8),
                    0.0f0,
                )
            finally
                restore_reference_lsm_state!(original_state)
            end
        end

        @testset "GPU: Reference LSM string output" begin
            original_state = snapshot_reference_lsm_state()
            try
                clear_reference_lsm_state!()
                output = LiquidCortex.run_lsm_step_str(zeros(Float32, 6), 0.0f0; n_out=3)
                parts = split(output, ",")

                @test output isa String
                @test length(parts) == 3
                @test all(part -> !isempty(part), parts)
            finally
                restore_reference_lsm_state!(original_state)
            end
        end

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
