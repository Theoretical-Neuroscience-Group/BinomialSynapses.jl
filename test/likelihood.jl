@testset "likelihood.jl" begin
    @info "Testing likelihood.jl"
    @testset "consistency on device = :cpu" begin
        using BinomialSynapses: likelihood_indices

        m_out = 1024
        m_in  = 1024
        Ns = rand(1:128, m_out)
        ps = rand(m_out)
        qs = rand(m_out)
        σs = rand(m_out)
        τs = rand(m_out)
        model = BinomialModel(Ns, ps, qs, σs, τs);

        ns = rand(1:128, m_out, m_in)
        ks = zeros(Int, m_out, m_in);

        ks .= ns .% 64

        u1      = likelihood(ks, model, 0.3)
        u2, idx = likelihood_indices(ks, model, 0.3)

        @test u1 ≈ u2
        @test maximum(idx) <= m_in
        @test minimum(idx) >= 1

        for r in eachrow(idx)
            @test issorted(r)
        end
    end

    CUDA.functional() && @testset "consistency on device = :gpu" begin
        using BinomialSynapses: likelihood_indices

        m_out = 1024
        m_in  = 1024
        Ns = CuArray(rand(1:128, m_out))
        ps = CUDA.rand(m_out)
        qs = CUDA.rand(m_out)
        σs = CUDA.rand(m_out)
        τs = CUDA.rand(m_out)
        model = BinomialModel(Ns, ps, qs, σs, τs);

        ns = CuArray(rand(1:128, m_out, m_in))
        ks = CUDA.zeros(Int, m_out, m_in);

        ks .= ns .% 64

        u1      = likelihood(ks, model, 0.3f0)
        u2, idx = likelihood_indices(ks, model, 0.3f0)

        @test u1 ≈ u2
        @test maximum(idx) <= m_in
        @test minimum(idx) >= 1

        idx = Array(idx)
        for r in eachrow(idx)
            @test issorted(r)
        end

        if RUN_BENCHMARKS
            println("")
            println("Benchmarking function likelihood!: should take about 300μs")
            display(@benchmark CUDA.@sync likelihood($ks, $model, 0.3f0))
            println("")
            println("")
            println("Benchmarking function likelihood_resample!: should take about 4ms")
            state = BinomialState(ns, ks)
            obs   = BinomialObservation(0.3f0, 0.1f0)
            display(@benchmark CUDA.@sync likelihood_resample!($state, $model, $obs))
            println("")
            println("")
        end
    end

    @testset "correctness of values on device = :cpu" begin
        observation = 3.0
        M_out = 6
        M_in = 4
        Ns = 10 .* ones(Int, M_out)
        ps = rand(M_out)
        qs = [1., 1., 3., 3., 1., 3.]
        σs = 0.1 .* ones(M_out)
        τs = rand(M_out)
        model = BinomialModel(Ns, ps, qs, σs, τs);

        ks = [3 3 3 3; 1 1 1 1; 1 1 1 1; 3 3 3 3; 3 1 1 1; 1 3 3 3]

        u, idx = likelihood_indices(ks, model, observation)

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

    CUDA.functional() && @testset "correctness of values on device = :gpu" begin
        observation = 3.0
        M_out = 6
        M_in = 4
        Ns = 10 .* CUDA.ones(Int, M_out)
        ps = CUDA.rand(M_out)
        qs = cu([1f0, 1f0, 3f0, 3f0, 1f0, 3f0])
        σs = Float32(0.1) .* CUDA.ones(M_out)
        τs = CUDA.rand(M_out)
        model = BinomialModel(Ns, ps, qs, σs, τs);

        ks = cu([3 3 3 3; 1 1 1 1; 1 1 1 1; 3 3 3 3; 3 1 1 1; 1 3 3 3])

        u, idx = likelihood_indices(ks, model, observation)
        u = Array(u)
        idx = Array(idx)

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

    @testset "three dimensions on device = :cpu" begin
        using BinomialSynapses: likelihood_indices
        M_dt = 2
        M_out = 3
        M_in = 4
        observation = repeat([1.0, 3.0], 1, M_out)
        Ns = 10 .* ones(Int, M_dt, M_out)
        ps = rand(M_dt, M_out)
        qs = [1. 1. 3.; 3. 1. 3.]
        σs = 0.1 .* ones(M_dt, M_out)
        τs = rand(M_dt, M_out)
        model = BinomialModel(Ns, ps, qs, σs, τs);

        ks = permutedims(
            cat(
                [3 3 3 1; 1 1 1 1; 1 1 1 3], 
                [1 1 1 1; 1 1 3 1; 1 3 3 3], 
                dims = 3
            ), 
            (3,1,2)
        )

        u, idx = likelihood_indices(ks, model, observation)

        @test size(u) == (2, 3)
        @test size(idx) == (2, 3, 4)

        # Combinations of q and k that correspond to the observation should have
        # a high likelihood
        @test u[1, 2] > 3.9894
        @test u[2, 1] > 3.9894

        # Combinations of q and k that do not match the observation should have
        # a low likelihood
        @test u[1, 3] < 1.0
        @test u[2, 3] < 1.0

        # Resampling should pick the particles that match q and the observation
        @test all(idx[1, 1, :] .== 4)
        @test all(idx[2, 2, :] .== 3)
        @test all(idx[2, 3, :] .== 1)
    end
    
    CUDA.functional() && @testset "three dimensions on device = :gpu" begin
        using BinomialSynapses: likelihood_indices
        M_dt = 2
        M_out = 3
        M_in = 4
        observation = cu(repeat([1.0f0, 3.0f0], 1, M_out))
        Ns = 10 .* CUDA.ones(Int, M_dt, M_out)
        ps = CUDA.rand(M_dt, M_out)
        qs = cu([1f0 1f0 3f0; 3f0 1f0 3f0])
        σs = Float32(0.1) .* CUDA.ones(M_dt, M_out)
        τs = CUDA.rand(M_dt, M_out)
        model = BinomialModel(Ns, ps, qs, σs, τs);

        ks = cu(permutedims(
            cat(
                [3 3 3 1; 1 1 1 1; 1 1 1 3], 
                [1 1 1 1; 1 1 3 1; 1 3 3 3], 
                dims = 3
            ), 
            (3,1,2)
        ))

        u, idx = likelihood_indices(ks, model, observation)

        @test size(u) == (2, 3)
        @test size(idx) == (2, 3, 4)

        u = Array(u)
        idx = Array(idx)

        # Combinations of q and k that correspond to the observation should have
        # a high likelihood
        @test u[1, 2] > 3.9894
        @test u[2, 1] > 3.9894

        # Combinations of q and k that do not match the observation should have
        # a low likelihood
        @test u[1, 3] < 1.0
        @test u[2, 3] < 1.0

        # Resampling should pick the particles that match q and the observation
        @test all(idx[1, 1, :] .== 4)
        @test all(idx[2, 2, :] .== 3)
        @test all(idx[2, 3, :] .== 1)
    end
end
