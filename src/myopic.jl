abstract type MyopicPolicy <: OEDPolicy end

struct Myopic{T} <: MyopicPolicy
    dts::T
end

# `MyopicFast` is the same as `Myopic`, except that instead of expanding states and
# parameters along another dimension, and propagating each parameter with each dt,
# `dts` are randomly assigned to parameters
struct MyopicFast{T} <: MyopicPolicy
    dts::T
end

function _oed!(sim, ::MyopicPolicy)
    policy = sim.tsteps

    obs = _synthetic_obs(sim, policy)
    temp_state = _temp_state(sim, policy)

    # `propagate!(temp_state.state, temp_state.model, obs.dt)` should work out of the box
    # the likelihood kernel will need to be rewritten to take BinomialObservation with 
    # vectors
    update!(temp_state, obs, sim.filter)

    entropies = _entropy(temp_state.model, policy) 
    return policy.dts[argmax(entropies)]
end

function _synthetic_obs(sim, policy)
    epsp_vector = _temp_epsps(sim)
    dt_vector = _temp_dts(sim, policy)
    return BinomialObservation(epsp_vector, dt_vector)
end


_temp_state(sim, ::Myopic) = _repeat(sim.fstate, length(sim.tsteps.dts))

function _repeat(fstate, m)
# return a new NestedParticleState which repeats `fstate.state` along a third dimension,
# `m` times, and `fstate.model` along a second dimension,
# the first dimension of each of them represents the different entries of `dt_vector`
# this can be implemented using `repeat`
    state = fstate.state
    model = fstate.model

    n = permutedims(repeat(state.n, 1, 1, m), [3, 1, 2])
    k = permutedims(repeat(state.k, 1, 1, m), [3, 1, 2])

    state = BinomialState(n, k)

    N = repeat(model.N, 1, m)'
    p = repeat(model.p, 1, m)'
    q = repeat(model.q, 1, m)'
    σ = repeat(model.σ, 1, m)'
    τ = repeat(model.τ, 1, m)'

    model = BinomialModel(N, p, q, σ, τ)

    return NestedParticleState(state, model)
end


_temp_state(sim, ::MyopicFast) = copy(sim.fstate)


function _entropy(model, ::Myopic) 
    # TODO: simply compute entropy for each i in the first dimension of the parameter arrays
end

function _entropy(model, policy::MyopicFast) 
    # TODO: aggregate entropies of each dt by looping over the parameter arrays
    # this may be faster and easier to implement on the CPU than on the GPU
end

_temp_dts(sim, ::Myopic) = collect(sim.tsteps.dts)
_temp_dts(sim, ::MyopicFast) = cu(rand(sim.tsteps.dts), m_out(sim))

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
    x = 1
    if i>1
        for ii in 2:i
            x = 1-(1-(1-p_star)*x)*exp(-(times[ii]-times[ii-1])/τ_star)
        end
    end

    # TODO: this can be made more efficient by not reallocating `e_temp`, but
    # allocating as part of the policy data structure
    e_temp = zeros(length(dts))
    for kk in 1:length(e_temp)
        x_temp = 1-(1-(1-p_star)*x)*exp(-policy.dts[kk]/τ_star)
        e_temp[kk] = x_temp*N_star*p_star*q_star
    end
    return e_temp
end
