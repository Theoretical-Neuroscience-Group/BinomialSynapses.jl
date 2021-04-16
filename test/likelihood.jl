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
        println("")
        println("Benchmarking function likelihood_indices: should take about 4ms")
        display(@benchmark CUDA.@sync likelihood_indices($ks, $model, 0.3f0))
        println("")
    end
    @testset "correctness of values" begin
        observation = 3.0
        M_out = 6
        M_in = 4
        Ns = 10 .* CUDA.ones(Int, M_out)
        ps = CUDA.rand(M_out)
        qs = CUDA.ones(M_out)
        println("")
        println("Scalar operations warning expected here:")
        qs[3] = 3.0
        qs[4] = 3.0
        qs[6] = 3.0
        sigmas = Float32(0.1) .* CUDA.ones(M_out)
        taus = CUDA.rand(M_out)
        model = BinomialModel(Ns, ps, qs, sigmas, taus);

        ks = CUDA.ones(Int, M_out, M_in)
        ks[1,:] = [3,3,3,3]
        ks[4,:] = [3,3,3,3]
        ks[5,1] = 3
        ks[6,2:4] = [3,3,3]

        u,idx = likelihood_indices(ks, model, observation)

        # Combinations of q and k that correspond to the observation should have
        # a high likelihood
        @test u[1] > 3.9894
        @test u[3] > 3.9894

        # Combinations of q and k that do not match the observation should have
        # a low likelihood
        @test u[2] < 1.5f-10
        @test u[4] < 1.5f-10

        # Resampling should pick the particles that match q and the observation
        @test all(idx[5:6,:].==1)
        println("Scalar operations over")
        println("-------------------------------------------------------------------------")
    end
end
