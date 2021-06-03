abstract type MyopicPolicy <: OEDPolicy end

# parallel implementation of the myopic policy, in which multiple copies of the particles 
# are propagated in parallel
struct Myopic{T1, T2} <: MyopicPolicy
    dts::T1
    costfun::T2
end

# `MyopicFast` is the same as `Myopic`, except that instead of expanding states and
# parameters along another dimension, and propagating each parameter with each dt,
# `dts` are randomly assigned to parameters
struct MyopicFast{T1, T2} <: MyopicPolicy
    dts::T1
    costfun::T2
end

Myopic(dts) = Myopic(dts, _entropy)
MyopicFast(dts) = MyopicFast(dts, _entropy)

function _oed!(sim, ::MyopicPolicy)
    policy = sim.tsteps

    obs = _synthetic_obs(sim, policy)
    temp_state = _temp_state(sim, policy)

    # `propagate!(temp_state.state, temp_state.model, obs.dt)` should work out of the box
    # the likelihood kernel will need to be rewritten to take BinomialObservation with 
    # vectors
    update!(temp_state, obs, sim.filter)

    costs = policy.costfun(temp_state.model, policy) 
    return policy.dts[argmin(costs)]
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
# this can be implemented using `repeat`
    state = fstate.state
    model = fstate.model

    n = _repeat(state.n, m)
    k = _repeat(state.k, m)

    state = BinomialState(n, k)

    N = _repeat(model.N, m)
    p = _repeat(model.p, m)
    q = _repeat(model.q, m)
    σ = _repeat(model.σ, m)
    τ = _repeat(model.τ, m)

    model = BinomialModel(N, p, q, σ, τ)

    return NestedParticleState(state, model)
end


function _repeat(A::AbstractArray, m)
    B = similar(A, size(A)..., m)
    B .= A
    return PermutedDimsArray(B, circshift(1:ndims(B), 1))
end


_temp_state(sim, ::MyopicFast) = deepcopy(sim.fstate)

_temp_dts(sim, ::Myopic) = collect(sim.tsteps.dts)
_temp_dts(sim, ::MyopicFast) = rand(sim.tsteps.dts, m_out(sim))

function _temp_epsps(sim)
    dts = sim.tsteps.dts
    map = MAP(sim.fstate.model)
    times = sim.times

    N_star = map.N
    p_star = map.p
    q_star = map.q
    τ_star = map.τ

    # TODO: this can be made more efficient by storing the previous value of `x`
    # as part of the policy data structure
    x = 1.
    L = length(times)
    if L > 1
        for ii in 2:L
            x = 1-(1-(1-p_star)*x)*exp(-(times[ii]-times[ii-1])/τ_star)
        end
    end

    # TODO: this can be made more efficient by not reallocating `e_temp`, but
    # allocating as part of the policy data structure
    e_temp = zeros(length(dts))
    for kk in 1:length(e_temp)
        x_temp = 1-(1-(1-p_star)*x)*exp(-dts[kk]/τ_star)
        e_temp[kk] = x_temp*N_star*p_star*q_star
    end
    return e_temp
end
