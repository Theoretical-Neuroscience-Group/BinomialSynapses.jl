struct BinomialModel{T1,T2}
    N::T1
    p::T2
    q::T2
    sigma::T2
    tau::T2
end

struct BinomialGridModel{T1, T2, T3}
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
end
# special outer constructor to convert a BinomialModelGridGPU into a BinomialModel
function BinomialModel(gridmodel::BinomialGridModel)
   N     = gridmodel.Nrng[gridmodel.Nind]
   p     = gridmodel.Nrng[gridmodel.pind]
   q     = gridmodel.Nrng[gridmodel.qind]
   sigma = gridmodel.Nrng[gridmodel.sigmaind]
   tau   = gridmodel.Nrng[gridmodel.tauind]
   return BinomialModel(N, p, q, sigma, tau)
end

struct BinomialState{T}
    n::T
    k::T
end
