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
    
    @test length(sim.times) == 0
    @test length(sim.epsps) == 0
    
    initialize!(sim)
    @test sim.times == [0.]
    
    for i in 1:10
        propagate!(sim)
        @test sim.times[end] ≈ 0.1*i
    end

    @test length(sim.times) == 11
    @test length(sim.epsps) == 11

    times, epsps = run!(sim, T = 10)
    @test length(sim.times) == 21
    @test length(sim.epsps) == 21

    for i in eachindex(times)
        @test times[i] ≈ 0.1*(i-1)
    end
end
