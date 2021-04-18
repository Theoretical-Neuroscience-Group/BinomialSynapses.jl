@testset "outer_resample!" begin
    @testset "user API" begin
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
end
