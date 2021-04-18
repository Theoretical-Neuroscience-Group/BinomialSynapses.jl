@testset "high-level filtering logic" begin
    m_out  = 1024
    m_in   = 1024

    ns = CuArray(rand(1:128, m_out, m_in))
    ks = CUDA.zeros(Int, m_out, m_in);

    Nrng     = CuArray(1:5)
    prng     = CuArray(Float32.(LinRange(0.05,0.95,5)))
    qrng     = CuArray(Float32.(LinRange(0.1,2,5)))
    sigmarng = CuArray(Float32.(LinRange(0.05,2,5)))
    taurng   = CuArray(Float32.(LinRange(0.05,2,5)))
    Nind     = cu(rand(1:length(Nrng), m_out))
    pind     = cu(rand(1:length(prng), m_out))
    qind     = cu(rand(1:length(qrng), m_out))
    sigmaind = cu(rand(1:length(sigmarng), m_out))
    tauind   = cu(rand(1:length(taurng), m_out))

    state = BinomialState(ns, ks)
    model = BinomialGridModel(
        Nind, pind, qind, sigmaind, tauind,
        Nrng, prng, qrng, sigmarng, taurng
    )

    fstate = NestedParticleState(state, model)
    filter = NestedParticleFilter(12)
    obs    = BinomialObservation(0.3f0, 0.1f0)

    println("")
    println("Benchmarking one filter step: should take under 10ms")
    display(@benchmark CUDA.@sync update!($fstate, $obs, $filter))
    println("")
end
