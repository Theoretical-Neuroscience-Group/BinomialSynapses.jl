@testset "simulate.jl" begin
    @info "Testing simulate.jl"
    @testset "Device = $device" for device in DEVICES
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
                timestep = FixedTimestep(0.1),
                device = device
            )
        
        @test length(sim.times) == 0
        @test length(sim.epsps) == 0

        @test m_out(sim) == 2048
        
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

        # Test `DeterministicTrain`
        sim = NestedFilterSimulation(
                N, p, q, σ, τ,
                1:20,
                LinRange(0.00, 1.00, 100),
                LinRange(0.00, 2.00, 100),
                LinRange(0.05, 2.00, 100),
                LinRange(0.05, 2.00, 100),
                2048, 512, 12,
                timestep = DeterministicTrain([1., 2., 3.]),
                device = device
            )
        @test length(sim.times) == 0
        @test length(sim.epsps) == 0

        initialize!(sim)
        @test sim.times == [0.]

        propagate!(sim)
        @test sim.times[2] == 1.

        propagate!(sim)
        @test sim.times[3] == 3.

        propagate!(sim)
        @test sim.times[4] == 6.

        @test isnothing(propagate!(sim))

        sim = NestedFilterSimulation(
                N, p, q, σ, τ,
                1:20,
                LinRange(0.00, 1.00, 100),
                LinRange(0.00, 2.00, 100),
                LinRange(0.05, 2.00, 100),
                LinRange(0.05, 2.00, 100),
                2048, 512, 12,
                timestep = DeterministicTrain([1., 2., 3.]),
                device = device
            )

        run!(sim, T = 10)
        @test length(sim.times) == 4
        @test length(sim.epsps) == 4
    end
end
