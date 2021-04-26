struct NestedFilterSimulation{T1, T2, T3, T4}
    hmodel::T1
    filter::T2
    hstate::T3
    fstate::T4
end

function NestedFilterSimulation(
    N, p, q, σ, τ,
    Nrng, prng, qrng, σrng, τrng,
    m_out, m_in, width;
    λ = 0.121,
    dt = nothing
)
    hmodel = ScalarBinomialModel(N, p, q, σ, τ)
    filter = NestedParticleFilter(width)
    hstate = ScalarBinomialState(N, 0)
    fstate = NestedParticleState(
                m_out, m_in,
                Nrng, prng, qrng, σrng, τrng
             )
    return NestedFilterSimulation(hmodel, filter, hstate, fstate)
end

function propagate!(sim::NestedFilterSimulation)
    obs = propagate_emit!(sim.hstate, sim.hmodel)
    update!(sim.fstate, obs, sim.filter)
    return obs
end

function run!(sim::NestedFilterSimulation; T::Int, plot_each_timestep = false)
    times = zeros(0)
    epsps = zeros(0)
    time = 0.
    for i in 1:T
        @time begin
            obs = propagate!(sim)
        end
        push!(times, time += obs.dt)
        push!(epsps, obs.EPSP)
        if plot_each_timestep
            posterior_plot(sim.fstate, times, epsps, truemodel = sim.hmodel)
        end
        v = variance(sim)
        print(v[:p])
    end
    return times, epsps
end

MAP(sim::NestedFilterSimulation) = MAP(sim.fstate.model)
variance(sim::NestedFilterSimulation) = variance(sim.fstate.model)
