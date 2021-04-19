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
   return BinomialModel(model.N, model.p, model.q, model.sigma, model.tau)
end

function BinomialModel(nmax::Int, m_out::Int, device = :gpu)
    if device == :gpu
        N     = CuArray(rand(1:nmax, m_out))
        p     = CUDA.rand(m_out)
        q     = CUDA.rand(m_out)
        sigma = CUDA.rand(m_out)
        tau   = CUDA.rand(m_out)
    elseif device == :cpu
        N     = rand(1:nmax, m_out)
        p     = rand(m_out)
        q     = rand(m_out)
        sigma = rand(m_out)
        tau   = rand(m_out)
    end
    return BinomialModel(N, p, q, sigma, tau)
end

function BinomialModel(m_out::Int, my_Nrng, my_prng, my_qrng, my_sigmarng, my_taurng)
    gridmodel = BinomialGridModel(m_out, my_Nrng, my_prng, my_qrng, my_sigmarng, my_taurng)
    return BinomialModel(gridmodel)
end

function ScalarBinomialModel(nmax, device = :cpu)
    return BinomialModel(nmax, 1, device)
end

function ScalarBinomialModel(N::Int, p, q, sigma, tau, device = :cpu)
    if device == :cpu
        Ns     = N .* ones(Int, 1)
        ps     = p .* ones(1)
        qs     = q .* ones(1)
        sigmas = sigma .* ones(1)
        taus   = tau .* ones(1)
    elseif device == :gpu
        Ns     = N .* CUDA.ones(Int, 1)
        ps     = Float32(p) .* CUDA.ones(1)
        qs     = Float32(q) .* CUDA.ones(1)
        sigmas = Float32(sigma) .* CUDA.ones(1)
        taus   = Float32(tau) .* CUDA.ones(1)
    end
    return BinomialModel(Ns, ps, qs, sigmas, taus)
end

struct BinomialState{T}
    n::T
    k::T
end

function BinomialState(nmax::Int, m_out::Int, m_in::Int, device = :gpu)
    if device == :gpu
        n = CuArray(rand(1:nmax, m_out, m_in))
        k = CUDA.zeros(Int, m_out, m_in)
    elseif device == :cpu
        n = rand(1:nmax, m_out, m_in)
        k = zeros(Int, m_out, m_in)
    end
    return BinomialState(n, k)
end

function ScalarBinomialState(nmax::Int, device = :cpu)
    return BinomialState(nmax, 1, 1, device)
end

function ScalarBinomialState(n::Int, k::Int, device = :cpu)
    if device == :cpu
        ns     = n .* ones(Int, 1, 1)
        ks     = k .* ones(Int, 1, 1)
    elseif device == :gpu
        ns     = n .* CUDA.ones(Int, 1, 1)
        ks     = k .* CUDA.ones(Int, 1, 1)
    end
    return BinomialState(ns, ks)
end

struct BinomialObservation{T1, T2}
    EPSP::T1
    dt::T2
end
