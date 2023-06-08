"""
    MyopicPolicy <: OEDPolicy
A myopic OEDPolicy, i.e. one in which only the effect of the next time step on the parameter estimation, e.g. the entropy of a parameter, is taken into account.
"""
abstract type MyopicPolicy <: OEDPolicy end

"""
    Myopic(dts, target)
A parallel implementation of a myopic policy with candidate time steps `dts` and optimization target `target`, in which multiple copies of the particles 
are propagated in parallel.
Implemented settings of `target`: choose time step such that it
- `_entropy`: minimizes the joint entropy of the posterior distribution over parameters
- `_tauentropy`: minimizes the marginal entropy of `τ`
"""
struct Myopic{T1, T2, T3} <: MyopicPolicy
    dts::T1
    target::T2
    penalty::T3
end

"""
    MyopicFast(dts, target)
MyopicFast` is the same as `Myopic`, except that instead of expanding states and parameters along another dimension, and propagating each parameter with each dt, `dts` are randomly assigned to members of the parameter ensemble.
Implemented settings of `target`: choose time step such that it
- `_entropy`: minimizes the joint entropy of the posterior distribution over parameters
- `_tauentropy`: minimizes the marginal entropy of `τ`
"""
struct MyopicFast{T1, T2, T3} <: MyopicPolicy
    dts::T1
    target::T2
    penalty::T3    
end

# default target is miniminum entropy
"""
    Myopic(dts)
Minimize the joint entropy.
"""
Myopic(dts,penalty) = Myopic(dts, _entropy,penalty)

"""
    MyopicFast(dts)
Minimize the joint entropy.
"""
MyopicFast(dts,penalty) = MyopicFast(dts, _entropy,penalty)

"""
    Myopic_tau(dts)
Minimize the entropy of τ.
"""
Myopic_tau(dts,penalty) = Myopic(dts, _tauentropy,penalty)

"""
    MyopicFast_tau(dts)
Minimize the entropy of τ.
"""
MyopicFast_tau(dts,penalty) = MyopicFast(dts, _tauentropy,penalty)

function (policy::MyopicPolicy)(sim::NestedFilterSimulation)
    obs = _synthetic_obs(sim, policy)
    temp_state = _temp_state(sim, policy)

    update!(temp_state, obs, sim.filter)

    return policy.target(temp_state.model, obs, policy) 
end

function _synthetic_obs(sim, policy)
    epsp_vector = _temp_epsps(sim)
    dt_vector = _temp_dts(sim, policy)
    return BinomialObservation(epsp_vector, dt_vector)
end


_temp_state(sim, ::Myopic) = _repeat(sim.fstate, length(sim.tsteps.dts))

function _repeat(fstate::NestedParticleState, m)
    # return a new NestedParticleState which repeats `fstate.state` along a third dimension,
    # `m` times, and `fstate.model` along a second dimension,
    # the first dimension of each of them represents the different entries of `dt_vector`
    state = _repeat(fstate.state, m)
    model = _repeat(fstate.model, m)

    return NestedParticleState(state, model)
end

function _repeat(state::BinomialState, m)
    n = _repeat(state.n, m)
    k = _repeat(state.k, m)
    return BinomialState(n, k)
end

function _repeat(model::BinomialModel, m)
    N = _repeat(model.N, m)
    p = _repeat(model.p, m)
    q = _repeat(model.q, m)
    σ = _repeat(model.σ, m)
    τ = _repeat(model.τ, m)
    return BinomialModel(N, p, q, σ, τ)
end

function _repeat(model::BinomialGridModel, m)
    Nind = _repeat(model.Nind, m)
    pind = _repeat(model.pind, m)
    qind = _repeat(model.qind, m)
    σind = _repeat(model.σind, m)
    τind = _repeat(model.τind, m)
    return BinomialGridModel(
        Nind,       pind,       qind,       σind,       τind, 
        model.Nrng, model.prng, model.qrng, model.σrng, model.τrng
    )
end

function _repeat(A::AbstractArray, m)
    B = similar(A, size(A)..., m)
    B .= A
    return PermutedDimsArray(B, circshift(1:ndims(B), 1))
end


_temp_state(sim, ::MyopicFast) = deepcopy(sim.fstate)

_temp_dts(sim, policy) = _temp_dts(sim.tsteps.dts, sim.fstate.model.N, policy, m_out(sim)) 

_temp_dts(dts, ::AbstractArray, ::Myopic, ::Integer) = collect(dts)
_temp_dts(dts, ::AbstractArray, ::MyopicFast, m::Integer) = repeat(dts, m ÷ length(dts))
_temp_dts(dts, ::AnyCuArray, ::Myopic, ::Integer) = cu(collect(dts))
_temp_dts(dts, ::AnyCuArray, ::MyopicFast, m::Integer) = cu(repeat(dts, m ÷ length(dts)))

function _temp_epsps(sim)
    dts = sim.tsteps.dts
    map = MAP(sim.fstate.model)
    times = sim.times

    N_star = map.N
    p_star = map.p
    q_star = map.q
    τ_star = map.τ

    e_temp = _temp_epsps(times, N_star, p_star, q_star, τ_star, dts, sim.fstate.model.N)

    return return _shape_epsps(e_temp, sim, sim.tsteps)
end

function _temp_epsps(times, N_star, p_star, q_star, τ_star, dts, ::AnyCuArray)
    return cu(_temp_epsps(times, N_star, p_star, q_star, τ_star, dts))
end

# CPU fallback:
function _temp_epsps(times, N_star, p_star, q_star, τ_star, dts, ::AbstractArray)
    return _temp_epsps(times, N_star, p_star, q_star, τ_star, dts)
end

function _temp_epsps(times, N_star, p_star, q_star, τ_star, dts)
    x = 1.
    L = length(times)
    if L > 1
        for ii in 2:L
            x = 1-(1-(1-p_star)*x)*exp(-(times[ii]-times[ii-1])/τ_star)
        end
    end

    e_temp = zeros(length(dts))
    for kk in 1:length(e_temp)
        x_temp = 1-(1-(1-p_star)*x)*exp(-dts[kk]/τ_star)
        e_temp[kk] = x_temp*N_star*p_star*q_star
    end
    return e_temp
end

_shape_epsps(e_temp, sim, ::Myopic) = e_temp
_shape_epsps(e_temp, sim, ::MyopicFast) = repeat(e_temp, m_out(sim)÷length(sim.tsteps.dts))

function _entropy(model::BinomialGridModel, obs::BinomialObservation, policy::Myopic)
    # CPU algorithm: move index arrays to CPU
    Nind = Array(model.Nind)
    pind = Array(model.pind)
    qind = Array(model.qind)
    σind = Array(model.σind)
    τind = Array(model.τind)

    dts = Array(obs.dt)

    η = policy.penalty

    minent = Inf
    imin = 0
    @inbounds for i in 1:size(Nind, 1)
        dict = Dict{NTuple{5, Int64}, Int}()
        @inbounds for j in 1:size(Nind, 2)
            iN = Nind[i, j]
            ip = pind[i, j]
            iq = qind[i, j]
            iσ = σind[i, j]
            iτ = τind[i, j]
            key = (iN, ip, iq, iσ, iτ)
            dict[key] = get!(dict, key, 0) + 1
        end
        ent = 0.
        for value in values(dict)
            p = value/size(Nind, 2)
            ent -= p * log(p)
        end
        if ent + η*dts[i] < minent
            minent = ent + η*dts[i]
            imin = i 
        end
    end
    return dts[imin]
end

function _entropy(model::BinomialGridModel, obs::BinomialObservation, policy::MyopicFast) 
    # CPU algorithm: move index arrays to CPU
    Nind = Array(model.Nind)
    pind = Array(model.pind)
    qind = Array(model.qind)
    σind = Array(model.σind)
    τind = Array(model.τind)

    dts = Array(obs.dt)
    
    η = policy.penalty

    counts = Dict{Tuple{Float64, Int, Int, Int, Int, Int}, Int}()
    totals = Dict{Float64, Int}() # total counts per dt
    entropies = Dict{Float64, Float64}()
    @inbounds for i in 1:length(Nind)
        iN = Nind[i]
        ip = pind[i]
        iq = qind[i]
        iσ = σind[i]
        iτ = τind[i]
        dt = dts[i]
        key = (dt, iN, ip, iq, iσ, iτ)
        counts[key] = get!(counts, key, 0) + 1
        totals[dt] = get!(totals, dt, 0.) + 1
        entropies[dt] = η*dt
    end

    @inbounds for (key, count) in counts
        dt = key[1]
        p = count/totals[dt]
        entropies[dt] = get!(entropies, dt, 0.) - p * log(p)
    end
    
    return argmin(entropies)
end

function _tauentropy(model::BinomialGridModel, obs::BinomialObservation, policy::Myopic)
    # CPU algorithm: move index arrays to CPU
    τind = Array(model.τind)

    dts = Array(obs.dt)
    
    η = policy.penalty

    minent = Inf
    imin = 0
    @inbounds for i in 1:size(τind, 1)
        dict = Dict{NTuple{1, Int64}, Int}()
        @inbounds for j in 1:size(τind, 2)
            iτ = τind[i, j]
            key = (iτ,)
            dict[key] = get!(dict, key, 0) + 1
        end
        ent = 0.
        for value in values(dict)
            p = value/size(τind, 2)
            ent -= p * log(p)
        end
        if ent + η*dts[i] < minent
            minent = ent + η*dts[i]
            imin = i 
        end
    end
    return dts[imin]
end

function _tauentropy(model::BinomialGridModel, obs::BinomialObservation, policy::MyopicFast) 
    # CPU algorithm: move index arrays to CPU
    τind = Array(model.τind)

    dts = Array(obs.dt)
    
    η = policy.penalty

    counts = Dict{Tuple{Float64, Int}, Int}()
    totals = Dict{Float64, Int}() # total counts per dt
    entropies = Dict{Float64, Float64}()
    @inbounds for i in 1:length(τind)
        iτ = τind[i]
        dt = dts[i]
        key = (dt, iτ)
        counts[key] = get!(counts, key, 0) + 1
        totals[dt] = get!(totals, dt, 0.) + 1
        entropies[dt] = η*dt
    end

    
    @inbounds for (key, count) in counts
        dt = key[1]
        p = count/totals[dt]
        entropies[dt] = get!(entropies, dt, 0.) - p * log(p)
    end
    
    return argmin(entropies)
end

function Base.show(io::IO, ::MIME"text/plain", ::Myopic)
    print(io, "Myopic OED policy")
end

function Base.show(io::IO, ::Myopic)
    print(io, "Myopic OED policy")
end

function Base.show(io::IO, ::MIME"text/plain", ::MyopicFast)
    print(io, "Fast myopic OED policy")
end

function Base.show(io::IO, ::MyopicFast)
    print(io, "Fast myopic OED policy")
end
