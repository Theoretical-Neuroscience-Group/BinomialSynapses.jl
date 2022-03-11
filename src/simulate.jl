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
        timestep::Timestep = RandomTimestep(Exponential(0.121))
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
    timestep::Timestep = RandomTimestep(Exponential(0.121))
)
    hmodel = ScalarBinomialModel(N, p, q, σ, τ)
    filter = NestedParticleFilter(width)
    hstate = ScalarBinomialState(N, 0)
    fstate = NestedParticleState(
                m_out, m_in,
                Nrng, prng, qrng, σrng, τrng
             )
    times = zeros(0)
    epsps = zeros(0)
    return NestedFilterSimulation(hmodel, filter, hstate, fstate, timestep, times, epsps)
end

"""
    m_out(sim)

Return the number of outer particles of the simulation `sim`.
"""
function m_out(sim::NestedFilterSimulation)
    return size(sim.fstate.state.n, 1)
end

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

get_step(sim::NestedFilterSimulation) = get_step(sim.tsteps)
function get_step(sim::NestedFilterSimulation{T1, T2, T3, T4, T5, T6, T7}) where 
{T1, T2, T3, T4, T5 <: DeterministicTrain, T6, T7}
    i = length(sim.times)
    return sim.tsteps.train[i]
end

"""
    propagate!(sim)

Propagate the simulation, i.e. choose a time step and then propagate the simulation by it.
"""
function propagate!(sim::NestedFilterSimulation)
    time1 = @timed get_step(sim)
    dt = time1.value
    time2 = propagate!(sim, dt)
    return time1.time + time2
end

"""
    propagate!(sim, dt)

Propagate the simulation by time step `dt`.
"""
function propagate!(sim::NestedFilterSimulation, dt)
    time1 = @timed propagate_hidden!(sim, dt)
    obs = emit(sim, dt)
    time2 = @timed filter_update!(sim, obs)
    push!(sim.times, sim.times[end] + dt)
    push!(sim.epsps, obs.EPSP)
    return time1.time + time2.time
end

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
        begin
            time = propagate!(sim)
        end
        if plot_each_timestep
            posterior_plot(sim,i)
        end
        update!(recording, sim, time) 
    end
    save(recording)
    return sim.times, sim.epsps
end

function runBatch!(
    sim::NestedFilterSimulation;
    T::Integer,
    plot_each_timestep::Bool = false,
    recording::Recording = NoRecording
)
    if length(sim.times) == 0
        initialize!(sim)
    end
    for i in 1:T
        begin
            entrop = zeros(length(keys(sim.tsteps.train)))
            for j in 1:length(keys(sim.tsteps.train))
                train = sim.tsteps.train[j]

                entropy_temp = []
                for l in 1:10
		    print(l)
                    sim_copy = copy(sim)
                    sim_copy.tsteps = DeterministicTrain(train)
                    for k in 1:length(train)
                        propagate!(sim_copy)
                    end
                    append!(entropy_temp,compute_entropy(sim_copy.fstate.model))
                end
                entrop[j] = mean(entropy_temp)
            end
	end
        train_opt = sim.tsteps.train[argmin(entrop)]
        sim.tsteps = DeterministicTrain(train_opt)
	for j in 1:length(train_opt)
	    begin
		time = propagate!(sim)
	    end
	    if plot_each_timestep
		posterior_plot(sim,j)
	    end
	    update!(recording, sim, time)
	end
    end
    save(recording)
    return sim.times, sim.epsps
end	

function compute_entropy(model)
    Nind = Array(model.Nind)
    pind = Array(model.pind)
    qind = Array(model.qind)
    σind = Array(model.σind)
    τind = Array(model.τind)

    Nrng = Array(model.Nrng)
    prng = Array(model.prng)
    qrng = Array(model.qrng)
    σrng = Array(model.σrng)
    τrng = Array(model.τrng)

    samples = [Nrng[Nind]';prng[pind]';qrng[qind]';σrng[σind]';τrng[τind]']
    Σ_est = cov(samples')
    determinant = det(2*pi*ℯ*Σ_est)
    ent = 0.5*log(determinant)

    return ent

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
    res = f1(sim, time.time)
    data = [res]
    return Recording(f1, f2, data)
end
