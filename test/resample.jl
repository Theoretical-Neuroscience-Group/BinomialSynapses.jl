@testset "outer_resample!" begin
    @testset "benchmark" begin
        m_out = 1024
        m_in  = 1024
        state = BinomialState(128, m_out, m_in)
        model = BinomialGridModel(
            m_out,
            1:5,
            LinRange(0.05,0.95,5),
            LinRange(0.1,2,5),
            LinRange(0.05,2,5),
            LinRange(0.05,2,5)
        )

        u = CUDA.rand(m_out)
        println("")
        println("Benchmarking function outer_resample!: should take about 900μs")
        display(@benchmark CUDA.@sync outer_resample!($state, $model, $u))
        println("")
    end
    @testset "correctness of values" begin
        using BinomialSynapses: outer_indices!
        M_out = 4
        u = cu([1f3, 1f0, 1f0, 1f3])
        outer_indices!(u)
        @test u == cu([1f3, 1.001f3, 1.002f3, 2.002f3])

        count = 0
        for i in 1:100
            u = cu([1f3, 1f0, 1f0, 1f3])
            outer_indices!(u)
            uu = Array(u)
            for j in 1:4
                if uu[j] == 2 || uu[j] == 3
                    count += 1
                end
            end
        end
        @test count / 400 < 0.01
    end
end
