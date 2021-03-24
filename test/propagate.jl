@testset "propagate!" begin
    m_out  = 1024
    m_in   = 1024
    Ns     = CuArray(rand(1:128, m_out))
    ps     = CUDA.rand(m_out)
    qs     = CUDA.rand(m_out)
    sigmas = CUDA.rand(m_out)
    taus   = CUDA.rand(m_out)
    model  = BinomialModel(Ns, ps, qs, sigmas, taus);

    ns = CuArray(rand(1:128, m_out, m_in))
    ks = CUDA.zeros(Int, m_out, m_in);

    for i in 1:100
        propagate!(ns, ks, model, 0.1f0)
        @test minimum(ns) >= 0
        @test minimum(ks) >= 0
        @test minimum(ns .- ks) >= 0
        @test minimum(Ns .- ns) >= 0
    end

    println("")
    println("Benchmarking function propagate!: should take about 4ms")
    display(@benchmark CUDA.@sync propagate!($ns, $ks, $model, 0.1f0))
    println("")
end
