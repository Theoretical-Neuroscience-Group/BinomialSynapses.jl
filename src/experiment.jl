struct NestedFilterExperiment{T1, T2, T3}
    filter::T1
    fstate::T2
    tsteps::T3
end

function NestedFilterExperiment(
    Nrng, prng, qrng, σrng, τrng,
    m_out, m_in, width;
    timestep::Timestep = RandomTimestep(Exponential(0.121))
)
    filter = NestedParticleFilter(width)
    fstate = NestedParticleState(
                m_out, m_in,
                Nrng, prng, qrng, σrng, τrng
             )
    return NestedFilterExperiment(filter, fstate, timestep)
end

function propagate!(sim::NestedFilterExperiment, epsc, dt)
    obs = BinomialObservation(epsc, dt)
    filter_update!(sim, obs)
end

function filter_update!(sim::NestedFilterExperiment, obs)
    return update!(sim.fstate, obs, sim.filter)
end

function run_experiment!(
    sim::NestedFilterExperiment,
    epscs, dts; 
    T::Integer, 
    plot_each_timestep::Bool = false, 
    recording::Recording = NoRecording
)
    for i in 1:T
        propagate!(sim, epscs[i], dts[i])
        if plot_each_timestep
            posterior_plot(sim,i)
        end
    end
end
