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

struct BinomialObservation{T1, T2}
    EPSP::T1
    dt::T2
end
