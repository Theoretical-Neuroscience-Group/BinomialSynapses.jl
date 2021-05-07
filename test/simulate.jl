@testset "simulate.jl" begin
    println("             > simulate.jl")
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
    
    for i in 1:10
        obs = propagate!(sim)
        @test obs.dt ≈ 0.1
    end

    times, epsps = run!(sim, T = 10)
    for i in eachindex(times)
        @test times[i] ≈ 0.1*i
    end
end
