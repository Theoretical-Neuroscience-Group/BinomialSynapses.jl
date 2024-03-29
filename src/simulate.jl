"""
    NestedFilterSimulation(hmodel, filter, hstate, fstate, tsteps, times, epsps)

This object stores static (`hmodel`, `filter`, `tsteps`) and dynamic (`hstate`, `fstate`, `tsteps`, `times`, `epsps`) information about the simulation.
"""
struct NestedFilterSimulation{T1, T2, T3, T4, T5, T6, T7}
    hmodel::T1
    filter::T2
    hstate::T3
    fstate::T4
    tsteps::T5
    times::T6
    epsps::T7
end

"""
    NestedFilterSimulation(
        N, p, q, σ, τ,
        Nrng, prng, qrng, σrng, τrng,
        m_out, m_in, width;
        timestep::Timestep = RandomTimestep(Exponential(0.121)),
        device = :gpu
    )

This is the main way simulations are supposed to be constructed by the user, i.e.
by specifying 
- ground truth parameters `N`, `p`, `q`, `σ`, `τ`.
- parameter grids for the model ensemble (outer particles) `Nrng`, `prng`,...
- size of the ensemble, i.e. number of outer_particles `m_out` and inner particles `m_in`
- the width of the jittering kernel `width`
- a method for choosing time steps `timestep`
"""
function NestedFilterSimulation(
    N, p, q, σ, τ,
    Nrng, prng, qrng, σrng, τrng,
    m_out, m_in, width;
    timestep::Timestep = RandomTimestep(Exponential(0.121)),
    device::Symbol = :gpu
)
    hmodel = ScalarBinomialModel(N, p, q, σ, τ)
    filter = NestedParticleFilter(width)
    hstate = ScalarBinomialState(N, 0)
    fstate = NestedParticleState(
                m_out, m_in,
                Nrng, prng, qrng, σrng, τrng,
                device = device
             )
    times = zeros(0)
    epsps = zeros(0)
    return NestedFilterSimulation(hmodel, filter, hstate, fstate, timestep, times, epsps)
end

"""
    m_out(sim)

Return the number of outer particles of the simulation `sim`.
"""
m_out(sim::NestedFilterSimulation) = m_out(sim.fstate)


"""
    propagate_hidden!(sim, dt)

Propagate the hidden state for a time step `dt`.
"""
function propagate_hidden!(sim, dt)
    return propagate!(sim.hstate, sim.hmodel, dt)
end

"""
    emit(sim, dt)

Emit an EPSP after time step `dt`.
"""
function emit(sim::NestedFilterSimulation, dt)
    return emit(sim.hstate, sim.hmodel, dt)
end

"""
    filter_update!(sim, obs)

Update the filter state based on observation `obs`.
"""
function filter_update!(sim::NestedFilterSimulation, obs)
    return update!(sim.fstate, obs, sim.filter)
end

"""
    initialize!(sim)

Initialize the simulation.
"""
function initialize!(sim::NestedFilterSimulation)
    dt = 0.
    propagate_hidden!(sim, dt)
    obs = emit(sim, dt)
    filter_update!(sim, obs)
    push!(sim.times, dt)
    push!(sim.epsps, obs.EPSP)
    return sim
end

(ts::Timestep)(::NestedFilterSimulation) = ts()


"""
    propagate!(sim)

Propagate the simulation, i.e. choose a time step and then propagate the simulation by it.
"""
function propagate!(sim::NestedFilterSimulation)
    dt = sim.tsteps(sim)
    propagate!(sim, dt)
end

"""
    propagate!(sim, dt)

Propagate the simulation by time step `dt`.
"""
function propagate!(sim::NestedFilterSimulation, dt)
    propagate_hidden!(sim, dt)
    obs = emit(sim, dt)
    filter_update!(sim, obs)
    push!(sim.times, sim.times[end] + dt)
    push!(sim.epsps, obs.EPSP)
    return sim
end

propagate!(::NestedFilterSimulation, ::Nothing) = nothing

"""
    run!(
        sim; 
        T, 
        plot_each_timestep = false, 
        recording = NoRecording
    )

Run a simulation for `T` time steps.
Set `plot_each_timestep = true` to get a live update of the simulation (this will reduce performance) and set `recording` to collect data while running the simulation (see `Recording`).
"""
function run!(
    sim::NestedFilterSimulation; 
    T::Integer, 
    plot_each_timestep::Bool = false, 
    recording::Recording = NoRecording
)
    if length(sim.times) == 0
        initialize!(sim)
    end
    for i in 1:T
        time = @timed begin
            r = propagate!(sim)
            if isnothing(r)
                @warn "Simulation ended prematurely due to `get_step` returning `nothing`."
                break
            end
        end
        if plot_each_timestep
            posterior_plot(sim)
        end
        update!(recording, sim, time) 
    end
    save(recording)
    return sim.times, sim.epsps
end

MAP(sim::NestedFilterSimulation; kwargs...) = MAP(sim.fstate.model; kwargs...)

"""
    Recording(f1, f2, sim)

Initialize a `Recording` for the specified functions `f1` and `f2` and already existing simulation `sim`.
"""
function Recording(f1, f2, sim::NestedFilterSimulation)
    begin
        time = @timed nothing
    end
    res = f1(sim, time)
    data = [res]
    return Recording(f1, f2, data)
end

function Base.show(io::IO, ::MIME"text/plain", sim::NestedFilterSimulation)
    # status = isempty(sim.epsps) ? "Uninitialized" : "Initialized"
    print(io, "Nested particle filter simulation 
    Filter: ", sim.filter, "
    time step counter: ", length(sim.epsps), "
    # of outer particles: ", m_out(sim.fstate), "
    # of inner particles: ", m_in(sim.fstate), "
    True model: ", sim.hmodel, "
    Initial hidden state: ", sim.hstate)
end

function Base.show(io::IO, ::NestedFilterSimulation)
    print(io, "Nested particle filter simulation")
end
