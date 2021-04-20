@testset "integration tests" begin
    println("             > INTEGRATION TESTS")
    @testset "convergence" begin
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
                2048, 512, 12
              )
        times, epsps = run!(sim, T = 1000)
        map = MAP(sim)

        @test abs(map[:N] - N) <= 6
        @test abs(map[:p] - p) <= 0.1
        @test abs(map[:q] - q) <= 0.1
        @test abs(map[:σ] - σ) <= 0.3
        @test abs(map[:τ] - τ) <= 0.3
    end
end
