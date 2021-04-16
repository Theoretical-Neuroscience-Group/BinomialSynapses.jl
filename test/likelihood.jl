@testset "likelihood" begin
    observation = 3.0
    M_out = 6
    M_in = 4

    Ns     = 10 .* CUDA.ones(Int, M_out)
    ps     = CUDA.rand(M_out)
    qs     = CUDA.ones(M_out)
    qs[3] = 3.0
    qs[4] = 3.0
    qs[6] = 3.0
    sigmas = Float32(0.1) .* CUDA.ones(M_out)
    taus   = CUDA.rand(M_out)
    model  = BinomialModel(Ns, ps, qs, sigmas, taus);

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
end
