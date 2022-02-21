"""
    NestedParticleFilter(jittering_width)

Construct a nested particle filter with a given jittering kernel width parameter.
"""
struct NestedParticleFilter
    jittering_width::Int
end

"""
    NestedParticleState(state, model)

Construct a particle system (ensemble) consisting of a state ensemble (inner and outer particles) and a model ensemble (outer particles).
"""
struct NestedParticleState{T1, T2}
    state::T1
    model::T2
end

"""
    NestedParticleState(m_out, m_in, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)

Construct a randomly initialized particle system with a given number of outer (`m_out`) and inner (`m_in`) particles and specified grids for the parameters.
"""
function NestedParticleState(
    m_out::Integer, m_in::Integer,
    my_Nrng, my_prng, my_qrng, my_σrng, my_τrng
)
    model = BinomialGridModel(m_out, my_Nrng, my_prng, my_qrng, my_σrng, my_τrng)
    n     = repeat(model.N, 1, m_in)
    k     = CUDA.zeros(Int, m_out, m_in)
    state = BinomialState(n, k)
    return NestedParticleState(state, model)
end

m_out(fstate::NestedParticleState) = size(fstate.state.n)[end-1]
m_in(fstate::NestedParticleState) = size(fstate.state.n)[end]


"""
    update!(filterstate, obs, filter)

Update the particle filter state based on a given observation and filter.
"""
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
    outer_resample!(state, model, u)
    return filterstate
end

function Base.show(io::IO, ::MIME"text/plain", filter::NestedParticleFilter)
    print(io, "Nested particle filter with jittering width = ", filter.jittering_width)
end

function Base.show(io::IO, filter::NestedParticleFilter)
    print(io, "Nested particle filter with jittering width = ", filter.jittering_width)
end

function Base.show(io::IO, ::MIME"text/plain", state::NestedParticleState)
    print(io, "Nested particle state with
model: $(state.model)
state: $(state.state)")
end

function Base.show(io::IO, state::NestedParticleState)
    print(io, "Nested particle filter with jittering width = ", filter.jittering_width)
end
