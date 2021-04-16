@testset "outer_resample!" begin
    @testset "internals" begin
        using BinomialSynapses: cu_alias_sample
        N, K = 10, 10

        wv = Array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        a = CuArray{Int64}(1:N)
        x = CuArray{Int64}(zeros(Int64, K));

        cu_alias_sample!(a, wv, x)

        # Resampling should only pick the second index (the only one with a non-
        # zero likelihood)
        @test all(x .== 2)
    end

    @testset "user API" begin
        m_out = 1024
        m_in = 1024
        ns = CuArray(rand(1:128, m_out, m_in))
        ks = CUDA.zeros(Int, m_out, m_in);

        state = BinomialState(ns, ks)
        u = CUDA.rand(m_out)
        println("")
        println("Benchmarking function outer_resample!: should take about 4ms")
        display(@benchmark CUDA.@sync outer_resample!($state, $u))
        println("")
    end
end
