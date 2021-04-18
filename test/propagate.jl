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

    dt     = 0.1
    m_out  = 4
    m_in   = 4
    Ns     = 10 .* CUDA.ones(Int, m_out)
    ps     = cu([0f0, 0f0, 1f0, 1f0])
    qs     = CUDA.rand(m_out)
    sigmas = CUDA.rand(m_out)
    taus   = cu([1f3, 1f-4, 1f-4, 1f3])
    model  = BinomialModel(Ns, ps, qs, sigmas, taus);
    ns     = 5 .* CUDA.ones(Int, m_out, m_in)
    ks     = 4 .* CUDA.ones(Int, m_out, m_in)

    propagate!(ns, ks, model, dt)
    ns = Array(ns)
    ks = Array(ks)

    # When tau is much larger than dt, no vesicle is refilled
    @test all(ns[1,:] .== 1)
    @test all(ns[4,:] .== 1)
    # When tau is much shorter than dt, all vesicles are refilled
    @test all(ns[2,:] .== 10)
    @test all(ns[3,:] .== 10)

    # When the release probability is 0, no vesicle is released
    @test all(ks[1,:] .== 0)
    @test all(ks[2,:] .== 0)
    # When the release probability is 1, all vesicles are released
    @test all(ks[3,:] .== 10)
    @test all(ks[4,:] .== 1)
end
