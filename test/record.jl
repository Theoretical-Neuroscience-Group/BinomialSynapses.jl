CUDA.functional() && @testset "record.jl" begin
    @info "Testing record.jl"
    N = 10
    p = 0.85
    q = 1.0
    σ = 0.2
    τ = 0.2

    sim = NestedFilterSimulation(
            N, p, q, σ, τ,
            1:20,
            LinRange(0.00, 1.00, 100),
            LinRange(0.00, 2.00, 100),
            LinRange(0.05, 2.00, 100),
            LinRange(0.05, 2.00, 100),
            2048, 512, 12,
            timestep = FixedTimestep(0.1)
          )

    # Recording meant to record hstate and not save it
    f1(sim, _) = sim.hstate
    f2 = nothing
    rec = Recording(f1, f2, sim)

    @test length(rec.data) == 1
    @test eltype(rec.data) == typeof(sim.hstate)

    times, epsps = run!(sim, T = 10, recording = rec)
    @test length(rec.data) == 11
    @test last(rec.data) == sim.hstate

    # Remove first element after recording
    f2(data) = popfirst!(data)
    rec = Recording(f1, f2, sim)
    
    @test length(rec.data) == 1

    times, epsps = run!(sim, T = 10, recording = rec)
    @test length(rec.data) == 10
    @test last(rec.data) == sim.hstate
end
