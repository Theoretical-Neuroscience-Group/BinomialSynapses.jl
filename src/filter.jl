struct NestedParticleFilter
    jittering_width::Int
end

struct NestedParticleState{T1, T2}
    state::T1
    model::T2
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
    outer_resample!(state, u)
    return filterstate
end
