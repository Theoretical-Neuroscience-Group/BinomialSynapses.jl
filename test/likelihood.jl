@testset "likelihood" begin
    @testset "consistency" begin
        using BinomialSynapses: likelihood_indices

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

        ks .= ns .% 64

        u1      = likelihood(ks, model, 0.3f0)
        u2, idx = likelihood_indices(ks, model, 0.3f0)

        @test u1 ≈ u2
        @test maximum(idx) <= m_in
        @test minimum(idx) >= 1

        println("")
        println("Benchmarking function likelihood!: should take about 300μs")
        display(@benchmark CUDA.@sync likelihood($ks, $model, 0.3f0))
        println("")
        println("Benchmarking function likelihood_indices: should take about 4ms")
        display(@benchmark CUDA.@sync likelihood_indices($ks, $model, 0.3f0))
        println("")
    end
    @testset "correctness of values" begin
        observation = 3.0f0
        M_out  = 3
        M_in   = 3
        Ns     = 5 .* CUDA.ones(Int, M_out)
        ps     = CUDA.rand(M_out)
        qs     = CUDA.ones(M_out)
        println("")
        println("Scalar operations warning expected here:")
        qs[1]  = 1f0
        qs[2]  = 1f0
        qs[3]  = 3f0
        sigmas = 0.1f0 .* CUDA.ones(M_out)
        taus   = CUDA.rand(M_out)
        model  = BinomialModel(Ns, ps, qs, sigmas, taus);

        ks     = 3 .* CUDA.ones(Int, M_out, M_in)
        u      = likelihood(ks, model, observation)

        @test u[1] > u[3]
        @test u[2] > u[3]
        println("Scalar operations over")
        println("-------------------------------------------------------------------------")
    end
end
