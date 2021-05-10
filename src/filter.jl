struct NestedParticleFilter
    jittering_width::Int
end

struct NestedParticleState{T1, T2}
    state::T1
    model::T2
end

function NestedParticleState(
    m_out::Int, m_in::Int,
    my_Nrng, my_prng, my_qrng, my_σrng, my_τrng
)
    model = BinomialGridModel(m_out, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)
    n     = repeat(model.N, 1, m_in)
    k     = CUDA.zeros(Int, m_out, m_in)
    state = BinomialState(n, k)
    propagate!(state, model, 0f0)
    return NestedParticleState(state, model)
end

function update!(
    filterstate::NestedParticleState,
    observation::BinomialObservation,
    filter::NestedParticleFilter
)
    state = filterstate.state
    model = filterstate.model

    jitter!(model, filter.jittering_width)
    propagate!(state, model, observation.dt)
    u = likelihood_resample!(state, model, observation)
    N_max = model.N[argmax(u)]
    p_max = model.p[argmax(u)]
    q_max = model.q[argmax(u)]
    σ_max = model.σ[argmax(u)]
    τ_max = model.τ[argmax(u)]
    outer_resample!(state, model, u)
    return filterstate, N_max, p_max, q_max, σ_max, τ_max
end
