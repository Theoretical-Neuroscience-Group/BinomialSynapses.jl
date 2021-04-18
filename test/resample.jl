@testset "outer_resample!" begin
    @testset "user API" begin
        m_out = 1024
        m_in = 1024
        ns = CuArray(rand(1:128, m_out, m_in))
        ks = CUDA.zeros(Int, m_out, m_in);

        state = BinomialState(ns, ks)
        u = CUDA.rand(m_out)
        println("")
        println("Benchmarking function outer_resample!: should take about 900μs")
        display(@benchmark CUDA.@sync outer_resample!($state, $u))
        println("")
    end
end
