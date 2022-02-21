struct NestedFilterSimulation{T1, T2, T3, T4, T5, T6, T7}
    hmodel::T1
    filter::T2
    hstate::T3
    fstate::T4
    tsteps::T5
    times::T6
    epsps::T7
end

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

function m_out(sim::NestedFilterSimulation)
    return size(sim.fstate.state.n, 1)
end

function propagate_hidden!(sim, dt)
    return propagate!(sim.hstate, sim.hmodel, dt)
end

function emit(sim::NestedFilterSimulation, dt)
    return emit(sim.hstate, sim.hmodel, dt)
end

function filter_update!(sim::NestedFilterSimulation, obs)
    return update!(sim.fstate, obs, sim.filter)
end

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

function propagate!(sim::NestedFilterSimulation)
    dt = get_step(sim)
    propagate!(sim, dt)
end

function propagate!(sim::NestedFilterSimulation, dt)
    propagate_hidden!(sim, dt)
    obs = emit(sim, dt)
    filter_update!(sim, obs)
    push!(sim.times, sim.times[end] + dt)
    push!(sim.epsps, obs.EPSP)
    return sim
end

function run!(
    sim::NestedFilterSimulation; 
    T::Int, 
    plot_each_timestep::Bool = false, 
    recording::Recording = NoRecording
)
    if length(sim.times) == 0
        initialize!(sim)
    end
    for i in 1:T
        begin
            time = @timed propagate!(sim)
        end
        print(time)
        print('\n')
        if plot_each_timestep
            posterior_plot(sim)
        end
        update!(recording, sim, time) 
    end
    save(recording)
    return sim.times, sim.epsps
end

#function Recording(f1, f2, sim::NestedFilterSimulation)
#    begin
#        time = @timed nothing
#    end
#    res = f1(sim, time)
#    data = [res]
#    return Recording(f1, f2, data)
#end
