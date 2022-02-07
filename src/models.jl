abstract type AbstractBinomialModel end
abstract type AbstractBinomialState end

struct BinomialModel{T1,T2} <: AbstractBinomialModel
    N::T1
    p::T2
    q::T2
    σ::T2
    τ::T2
end

struct BinomialGridModel{T1, T2, T3, T4, T5} <: AbstractBinomialModel
    Nind::T1
    pind::T1
    qind::T1
    σind::T1
    τind::T1
    Nrng::T2
    prng::T3
    qrng::T3
    σrng::T3
    τrng::T3
    N::T4
    p::T5
    q::T5
    σ::T5
    τ::T5
end

function BinomialGridModel(
    Nind, pind, qind, σind, τind,
    Nrng, prng, qrng, σrng, τrng
)
    N = Nrng[Nind]
    p = prng[pind]
    q = qrng[qind]
    σ = σrng[σind]
    τ = τrng[τind]

    return BinomialGridModel(
        Nind, pind, qind, σind, τind,
        Nrng, prng, qrng, σrng, τrng,
        N,    p,    q,    σ,    τ
    )
end

function BinomialGridModel(m_out::Int, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)
    Nrng = CuArray(Int.(my_Nrng))
    prng = CuArray(Float32.(my_prng))
    qrng = CuArray(Float32.(my_qrng))
    σrng = CuArray(Float32.(my_σrng))
    τrng = CuArray(Float32.(my_τrng))

    Nind = cu(rand(1:length(Nrng), m_out))
    pind = cu(rand(1:length(prng), m_out))
    qind = cu(rand(1:length(qrng), m_out))
    σind = cu(rand(1:length(σrng), m_out))
    τind = cu(rand(1:length(τrng), m_out))

    return BinomialGridModel(
                Nind, pind, qind, σind, τind,
                Nrng, prng, qrng, σrng, τrng
            )
end

function refresh!(model::BinomialGridModel)
    model.N .= model.Nrng[model.Nind]
    model.p .= model.prng[model.pind]
    model.q .= model.qrng[model.qind]
    model.σ .= model.σrng[model.σind]
    model.τ .= model.τrng[model.τind]
    return model
end

# special outer constructor to convert a BinomialGridModel into a BinomialModel
function BinomialModel(model::BinomialGridModel)
    refresh!(model)
    return BinomialModel(model.N, model.p, model.q, model.σ, model.τ)
end

function BinomialModel(nmax::Int, m_out::Int, device = :gpu)
    if device == :gpu
        N = CuArray(rand(1:nmax, m_out))
        p = CUDA.rand(m_out)
        q = CUDA.rand(m_out)
        σ = CUDA.rand(m_out)
        τ = CUDA.rand(m_out)
    elseif device == :cpu
        N = rand(1:nmax, m_out)
        p = rand(m_out)
        q = rand(m_out)
        σ = rand(m_out)
        τ = rand(m_out)
    end
    return BinomialModel(N, p, q, σ, τ)
end

function BinomialModel(m_out::Int, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)
    gridmodel = BinomialGridModel(m_out, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)
    return BinomialModel(gridmodel)
end

function ScalarBinomialModel(nmax, device = :cpu)
    return BinomialModel(nmax, 1, device)
end

function ScalarBinomialModel(N::Int, p, q, σ, τ, device = :cpu)
    if device == :cpu
        Ns = N .* ones(Int, 1)
        ps = p .* ones(1)
        qs = q .* ones(1)
        σs = σ .* ones(1)
        τs = τ .* ones(1)
    elseif device == :gpu
        Ns = N .* CUDA.ones(Int, 1)
        ps = Float32(p) .* CUDA.ones(1)
        qs = Float32(q) .* CUDA.ones(1)
        σs = Float32(σ) .* CUDA.ones(1)
        τs = Float32(τ) .* CUDA.ones(1)
    end
    return BinomialModel(Ns, ps, qs, σs, τs)
end

struct BinomialState{T} <: AbstractBinomialState
    n::T
    k::T
end

function BinomialState(nmax::Int, m_out::Int, m_in::Int, device::Symbol = :gpu)
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
        ns = n .* ones(Int, 1, 1)
        ks = k .* ones(Int, 1, 1)
    elseif device == :gpu
        ns = n .* CUDA.ones(Int, 1, 1)
        ks = k .* CUDA.ones(Int, 1, 1)
    end
    return BinomialState(ns, ks)
end

struct BinomialObservation{T1, T2}
    EPSP::T1
    dt::T2
end

function Base.show(io::IO, ::MIME"text/plain", ::AbstractBinomialModel)
    print(io, "Binomial release model")
end

function Base.show(io::IO, ::AbstractBinomialModel)
    print(io, "Binomial release model")
end

function Base.show(io::IO, ::MIME"text/plain", ::AbstractBinomialState)
    print(io, "n =, k =")
end

function Base.show(io::IO, ::AbstractBinomialState)
    print(io, "n =, k =")
end
