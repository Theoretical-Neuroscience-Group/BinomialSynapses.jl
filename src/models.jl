"""
    AbstractBinomialModel

An abstract type for binomial synaptic models.
A binomial model always has the following parameters:
- `N`: number of release sites
- `p`: probability of release
- `q`: quantum of release
- `σ`: observation noise
- `τ`: refilling time constant
"""
abstract type AbstractBinomialModel end

"""
    AbstractBinomialState

An abstract type for binomial model states.
A binomial model state has to have the following variables:
- `n`: number of readily releasable vesicles
- `k`: number of released vesicles
"""
abstract type AbstractBinomialState end

"""
    BinomialModel(N, p, q, σ, τ)

The standard structure for a binomial model or model ensemble.
"""
struct BinomialModel{T1,T2} <: AbstractBinomialModel
    N::T1
    p::T2
    q::T2
    σ::T2
    τ::T2
end

"""
    BinomialGridModel(
        Nind, pind, qind, σind, τind,
        Nrng, prng, qrng, σrng, τrng,
        N,    p,    q,    σ,    τ
       )

A binomial model ensemble whose parameters are constrained to live on a grid
defined by `Nrng`, `prng`, etc.
"""
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

"""
    BinomialGridModel(
        Nind, pind, qind, σind, τind,
        Nrng, prng, qrng, σrng, τrng
    )

Construct a binomial model ensemble with parameters on a grid, e.g. `Nrng`, 
based on choosing indices, e.g. `Nind`.
"""
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

"""
    BinomialGridModel(m_out, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng, device = :gpu)

Randomly initialize a binomial model ensemble with parameters uniformly sampled from
the specified grid (on the GPU).
"""
function BinomialGridModel(
    m_out::Integer, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng;
    device::Symbol = :gpu
)
    Nrng = my_Nrng
    prng = my_prng
    qrng = my_qrng
    σrng = my_σrng
    τrng = my_τrng

    Nind = rand(1:length(Nrng), m_out)
    pind = rand(1:length(prng), m_out)
    qind = rand(1:length(qrng), m_out)
    σind = rand(1:length(σrng), m_out)
    τind = rand(1:length(τrng), m_out)

    if device === :cpu
        return BinomialGridModel(
            Nind, pind, qind, σind, τind,
            Nrng, prng, qrng, σrng, τrng
        )
    elseif device === :gpu
        Nrng = CuArray(Int.(Nrng))
        prng = CuArray(Float32.(prng))
        qrng = CuArray(Float32.(qrng))
        σrng = CuArray(Float32.(σrng))
        τrng = CuArray(Float32.(τrng))

        Nind = cu(Nind)
        pind = cu(pind)
        qind = cu(qind)
        σind = cu(σind)
        τind = cu(τind)

        return BinomialGridModel(
            Nind, pind, qind, σind, τind,
            Nrng, prng, qrng, σrng, τrng
        )
    end
    throw(ArgumentError("Device must be either :cpu or :gpu."))
end

"""
    refresh!(model::BinomialGridModel)

Set the parameters of the model ensemble according to its current set of
indices.
"""
function refresh!(model::BinomialGridModel)
    model.N .= model.Nrng[model.Nind]
    model.p .= model.prng[model.pind]
    model.q .= model.qrng[model.qind]
    model.σ .= model.σrng[model.σind]
    model.τ .= model.τrng[model.τind]
    return model
end

"""
    BinomialModel(model::BinomialGridModel)

Convert a `BinomialGridModel` into a `BinomialModel`.
"""
function BinomialModel(model::BinomialGridModel)
    refresh!(model)
    return BinomialModel(model.N, model.p, model.q, model.σ, model.τ)
end

"""
    BinomialModel(Nmax, m_out, device = :gpu)

Randomly initialize a binomial model ensemble of size `m_out`,
with maximum value for `N` of `Nmax` on the GPU or CPU.
"""
function BinomialModel(Nmax::Integer, m_out::Integer; device::Symbol = :gpu)
    if device === :gpu
        N = CuArray(rand(1:Nmax, m_out))
        p = CUDA.rand(m_out)
        q = CUDA.rand(m_out)
        σ = CUDA.rand(m_out)
        τ = CUDA.rand(m_out)
    elseif device == :cpu
        N = rand(1:Nmax, m_out)
        p = rand(m_out)
        q = rand(m_out)
        σ = rand(m_out)
        τ = rand(m_out)
    end
    return BinomialModel(N, p, q, σ, τ)
end

"""
    BinomialModel(m_out, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)

Randomly initialize a binomial model ensemble on a grid, but throw away the grid info
and just keep the parameters.
"""
function BinomialModel(
    m_out::Integer, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng;
    device::Symbol = :gpu
)
    gridmodel = BinomialGridModel(
        m_out, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng, 
        device = device
    )
    return BinomialModel(gridmodel)
end

"""
    ScalarBinomialModel(Nmax, device = :cpu)

Randomly initialize a model ensemble of size 1, which corresponds to a scalar model (used for the hidden state).
"""
function ScalarBinomialModel(Nmax, device::Symbol = :cpu)
    return BinomialModel(Nmax, 1, device)
end

"""
    ScalarBinomialModel(N, p, q, σ, τ, device = :cpu)

Initialize a scalar binomial model with the given parameters.
"""
function ScalarBinomialModel(N::Integer, p, q, σ, τ; device::Symbol = :cpu)
    if device == :cpu
        Ns = N .* ones(Int, 1)
        ps = p .* ones(1)
        qs = q .* ones(1)
        σs = σ .* ones(1)
        τs = τ .* ones(1)
    elseif device === :gpu
        Ns = N .* CUDA.ones(Int, 1)
        ps = Float32(p) .* CUDA.ones(1)
        qs = Float32(q) .* CUDA.ones(1)
        σs = Float32(σ) .* CUDA.ones(1)
        τs = Float32(τ) .* CUDA.ones(1)
    end
    return BinomialModel(Ns, ps, qs, σs, τs)
end

"""
    BinomialState(n, k)

An ensemble of states of the binomial model.
"""
struct BinomialState{T} <: AbstractBinomialState
    n::T
    k::T
end

"""
    BinomialState(Nmax, m_out, m_in, device = :gpu)

Randomly initialize a state ensemble of size `m_out` x `m_in` and maximum value of `N` 
equal to `Nmax` on the specified device `:gpu` or `:cpu`.
"""
function BinomialState(Nmax::Integer, m_out::Integer, m_in::Integer; device::Symbol = :gpu)
    if device === :gpu
        n = CuArray(rand(1:Nmax, m_out, m_in))
        k = CUDA.zeros(Int, m_out, m_in)
    elseif device == :cpu
        n = rand(1:Nmax, m_out, m_in)
        k = zeros(Int, m_out, m_in)
    end
    return BinomialState(n, k)
end

"""
    BinomialState(N, m_in)

Initialize a state ensemble with `n` equal to `N` across inner particles and `k` equal to zero. The same device is used for `n` and `k` as for `N`.
"""
function BinomialState(N::AbstractArray{<:Integer}, m_in::Integer)
    m_out = length(N)
    n = repeat(N, 1, m_in)
    k = zeros(Int, m_out, m_in)
    return BinomialState(n, k)
end

function BinomialState(N::AnyCuArray{<:Integer}, m_in::Integer)
    m_out = length(N)
    n = repeat(N, 1, m_in)
    k = CUDA.zeros(Int, m_out, m_in)
    return BinomialState(n, k)
end

"""
    ScalarBinomialState(Nmax, device = :cpu)

Randomly initialize a state ensemble of size 1, which corresponds to a scalar model (used for the hidden state).
"""
function ScalarBinomialState(Nmax::Integer; device = :cpu)
    return BinomialState(Nmax, 1, 1, device = device)
end

"""
    ScalarBinomialState(n, k, device = :cpu)

Initialize a scalar state with the specified values of the variables.
"""
function ScalarBinomialState(n::Integer, k::Integer; device::Symbol = :cpu)
    if device == :cpu
        ns = n .* ones(Int, 1, 1)
        ks = k .* ones(Int, 1, 1)
    elseif device === :gpu
        ns = n .* CUDA.ones(Int, 1, 1)
        ks = k .* CUDA.ones(Int, 1, 1)
    end
    return BinomialState(ns, ks)
end

"""
    BinomialObservation

A structure for an observation consisting of an EPSP and a time step.
"""
struct BinomialObservation{T1, T2}
    EPSP::T1
    dt::T2
end
