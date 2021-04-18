abstract type AbstractBinomialModel end

struct BinomialModel{T1,T2} <: AbstractBinomialModel
    N::T1
    p::T2
    q::T2
    sigma::T2
    tau::T2
end

struct BinomialGridModel{T1, T2, T3, T4, T5} <: AbstractBinomialModel
    Nind::T1
    pind::T1
    qind::T1
    sigmaind::T1
    tauind::T1
    Nrng::T2
    prng::T3
    qrng::T3
    sigmarng::T3
    taurng::T3
    N::T4
    p::T5
    q::T5
    sigma::T5
    tau::T5
end

function BinomialGridModel(
    Nind, pind, qind, sigmaind, tauind,
    Nrng, prng, qrng, sigmarng, taurng
)
    N     = Nrng[Nind]
    p     = prng[pind]
    q     = qrng[qind]
    sigma = sigmarng[sigmaind]
    tau   = taurng[tauind]

    return BinomialGridModel(
        Nind, pind, qind, sigmaind, tauind,
        Nrng, prng, qrng, sigmarng, taurng,
        N,    p,    q,    sigma,    tau
    )
end

function BinomialGridModel(m_out::Int, my_Nrng, my_prng, my_qrng, my_sigmarng, my_taurng)
    Nrng     = CuArray(Int.(my_Nrng))
    prng     = CuArray(Float32.(my_prng))
    qrng     = CuArray(Float32.(my_qrng))
    sigmarng = CuArray(Float32.(my_sigmarng))
    taurng   = CuArray(Float32.(my_taurng))

    Nind     = cu(rand(1:length(Nrng),     m_out))
    pind     = cu(rand(1:length(prng),     m_out))
    qind     = cu(rand(1:length(qrng),     m_out))
    sigmaind = cu(rand(1:length(sigmarng), m_out))
    tauind   = cu(rand(1:length(taurng),   m_out))

    return BinomialGridModel(
                Nind, pind, qind, sigmaind, tauind,
                Nrng, prng, qrng, sigmarng, taurng
            )
end

function refresh!(model::BinomialGridModel)
    model.N     .= model.Nrng[model.Nind]
    model.p     .= model.prng[model.pind]
    model.q     .= model.qrng[model.qind]
    model.sigma .= model.sigmarng[model.sigmaind]
    model.tau   .= model.taurng[model.tauind]
    return model
end

# special outer constructor to convert a BinomialGridModel into a BinomialModel
function BinomialModel(model::BinomialGridModel)
   N     = model.Nrng[model.Nind]
   p     = model.prng[model.pind]
   q     = model.qrng[model.qind]
   sigma = model.sigmarng[model.sigmaind]
   tau   = model.taurng[model.tauind]
   return BinomialModel(N, p, q, sigma, tau)
end

struct BinomialState{T}
    n::T
    k::T
end

function BinomialState(nmax::Int, m_out::Int, m_in::Int)
    n = CuArray(rand(1:nmax, m_out, m_in))
    k = CUDA.zeros(Int, m_out, m_in)
    return BinomialState(n, k)
end

struct BinomialObservation{T1, T2}
    EPSP::T1
    dt::T2
end
