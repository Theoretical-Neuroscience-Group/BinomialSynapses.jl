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
        println("Benchmarking function outer_resample!: should take about 300μs")
        display(@benchmark outer_resample!($state, $model, $u))
        println("")
    end

    @testset "correctness of values" begin
        using BinomialSynapses: outer_indices!

        for M_out in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
            u = rand(M_out)
            v = cumsum(u)
            outer_indices!(u)
            @test u[2:M_out] ≈ v[1:M_out-1]
        end

        count = 0
        for i in 1:100
            u = cu([1f3, 1f0, 1f0, 1f3])
            idx = Array(outer_indices!(u))
            for j in 1:4
                if idx[j] == 2 || idx[j] == 3
                    count += 1
                end
            end
        end
        @test count / 400 < 0.05
    end

    @testset "sortedness of idx" begin
        using BinomialSynapses: outer_indices!

        for M_out in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
            u = rand(M_out)
            idx = outer_indices!(u)
            @test issorted(idx)
        end
    end
end
