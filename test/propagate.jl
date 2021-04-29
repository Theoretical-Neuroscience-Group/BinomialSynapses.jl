@testset "propagate.jl" begin
    println("             > propagate.jl")
    @testset "GPU" begin
        m_out = 1024
        m_in  = 1024

        state = BinomialState(128, m_out, m_in)
        model = BinomialModel(128, m_out)

        for i in 1:100
            propagate!(state, model, 0.1f0)
            @test minimum(state.n) >= 0
            @test minimum(state.k) >= 0
            @test minimum(state.n .- state.k) >= 0
            @test minimum(state.n .- state.n) >= 0
        end

        if RUN_BENCHMARKS
            println("")
            println("Benchmarking function propagate!: should take about 4ms")
            display(@benchmark CUDA.@sync propagate!($state, $model, 0.1f0))
            println("")
        end

        dt    = 0.1f0
        m_out = 4
        m_in  = 4
        N     = 10 .* CUDA.ones(Int, m_out)
        p     = cu([0f0, 0f0, 1f0, 1f0])
        q     = CUDA.rand(m_out)
        σ     = CUDA.rand(m_out)
        τ     = cu([1f3, 1f-4, 1f-4, 1f3])
        model = BinomialModel(N, p, q, σ, τ);
        n     = 5 .* CUDA.ones(Int, m_out, m_in)
        k     = 4 .* CUDA.ones(Int, m_out, m_in)

        state = BinomialState(n, k)

        propagate!(state, model, dt)
        n = Array(state.n)
        k = Array(state.k)

        # When τ is much larger than dt, no vesicle is refilled
        @test all(n[1,:] .== 1)
        @test all(n[4,:] .== 1)
        # When τ is much shorter than dt, all vesicles are refilled
        @test all(n[2,:] .== 10)
        @test all(n[3,:] .== 10)

        # When the release probability is 0, no vesicle is released
        @test all(k[1,:] .== 0)
        @test all(k[2,:] .== 0)
        # When the release probability is 1, all vesicles are released
        @test all(k[3,:] .== 10)
        @test all(k[4,:] .== 1)
    end#GPU

    @testset "CPU" begin
        m_out = 16
        m_in  = 16

        state = BinomialState(128, m_out, m_in, :cpu)
        model = BinomialModel(128, m_out, :cpu)

        for i in 1:10
            propagate!(state, model, 0.1)
            @test minimum(state.n) >= 0
            @test minimum(state.k) >= 0
            @test minimum(state.n .- state.k) >= 0
            @test minimum(state.n .- state.n) >= 0
        end

        dt    = 0.1
        m_out = 4
        m_in  = 4
        N     = 10 .* ones(Int, m_out)
        p     = [0., 0., 1., 1.]
        q     = rand(m_out)
        σ     = rand(m_out)
        τ     = [1e3, 1e-4, 1e-4, 1e3]
        model = BinomialModel(N, p, q, σ, τ);
        n     = 5 .* ones(Int, m_out, m_in)
        k     = 4 .* ones(Int, m_out, m_in)

        state = BinomialState(n, k)

        propagate!(state, model, dt)
        n = Array(state.n)
        k = Array(state.k)

        # When τ is much larger than dt, no vesicle is refilled
        @test all(n[1,:] .== 1)
        @test all(n[4,:] .== 1)
        # When τ is much shorter than dt, all vesicles are refilled
        @test all(n[2,:] .== 10)
        @test all(n[3,:] .== 10)

        # When the release probability is 0, no vesicle is released
        @test all(k[1,:] .== 0)
        @test all(k[2,:] .== 0)
        # When the release probability is 1, all vesicles are released
        @test all(k[3,:] .== 10)
        @test all(k[4,:] .== 1)
    end#CPU
end
