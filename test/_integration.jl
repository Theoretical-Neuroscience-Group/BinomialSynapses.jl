CUDA.functional() && @testset "integration tests" begin
    println("             > INTEGRATION TESTS")

    function benchmark(timestep::Timestep)
        N = 10
        p = 0.85
        q = 1.0
        σ = 0.2
        τ = 0.2
        sim = NestedFilterSimulation(
                N, p, q, σ, τ,
                1:20,
                LinRange(0.00, 1.00, 45),
                LinRange(0.00, 2.00, 45),
                LinRange(0.05, 2.00, 45),
                LinRange(0.05, 2.00, 45),
                2048, 512, 12,
                timestep = timestep
              )
        initialize!(sim)
        display(@benchmark CUDA.@sync propagate!($sim))
    end

    @testset "benchmark of filter" begin
        println("")
        println("Benchmarking single iteration with exponential random timestep")
        benchmark(RandomTimestep(Exponential(0.121)))
        println("")
        println("")
    end

    # OED benchmarks
    @testset "benchmark of OED" begin
        candidates = LinRange(0.005, 2, 8)
        @testset "OEDPolicy: Uniform" begin
            println("")
            println("Benchmarking single iteration with OEDPolicy: Uniform")
            benchmark(Uniform(candidates))
            println("")
            println("")
        end
        @testset "OEDPolicy: Myopic" begin
            println("")
            println("Benchmarking single iteration with OEDPolicy: Myopic")
            benchmark(Myopic(candidates))
            println("")
            println("")
        end
        @testset "OEDPolicy: MyopicFast" begin
            println("")
            println("Benchmarking single iteration with OEDPolicy: MyopicFast")
            benchmark(MyopicFast(candidates))
            println("")
            println("")
        end
    end

    function test_convergence(timestep::Timestep, T::Int)
        N = 10
        p = 0.85
        q = 1.0
        σ = 0.2
        τ = 0.2
        sim = NestedFilterSimulation(
                N, p, q, σ, τ,
                1:20,
                LinRange(0.00, 1.00, 45),
                LinRange(0.00, 2.00, 45),
                LinRange(0.05, 2.00, 45),
                LinRange(0.05, 2.00, 45),
                2048, 512, 12,
                timestep = timestep
              )
        times, epsps = run!(sim, T = T)
        mapmodel = MAP(sim)

        @test abs(mapmodel.N - N) <= 6
        @test abs(mapmodel.p - p) <= 0.15
        @test abs(mapmodel.q - q) <= 0.15
        @test abs(mapmodel.σ - σ) <= 0.3
        @test abs(mapmodel.τ - τ) <= 0.3
    end

    @testset "convergence of filter" begin
        test_convergence(RandomTimestep(Exponential(0.121)), 1000)
    end

    # OED tests
    @testset "convergence of OED" begin
        candidates = LinRange(0.005, 2, 8)
        T = 1000
        @testset "OEDPolicy: Uniform" begin
            test_convergence(Uniform(candidates), T)
        end
        @testset "OEDPolicy: Myopic" begin
            test_convergence(Myopic(candidates), T)
        end
        @testset "OEDPolicy: MyopicFast" begin
            test_convergence(MyopicFast(candidates), T)
        end
    end
end
