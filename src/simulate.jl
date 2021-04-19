struct NestedFilterSimulation{T1, T2, T3, T4}
    hmodel::T1
    filter::T2
    hstate::T3
    fstate::T4
end

function NestedFilterSimulation(
    N, p, q, sigma, tau,
    Nrng, prng, qrng, sigmarng, taurng,
    m_out, m_in, width;
    λ = 0.121,
    dt = nothing
)
    hmodel = ScalarBinomialModel(10, 0.85, 1.0, 0.2, 0.2)
    filter = NestedParticleFilter(width)
    hstate = ScalarBinomialState(N, 0)
    fstate = NestedParticleState(
                m_out, m_in,
                Nrng, prng, qrng, sigmarng, taurng
             )
    return NestedFilterSimulation(hmodel, filter, hstate, fstate)
end

function propagate!(sim::NestedFilterSimulation)
    obs = propagate_emit!(sim.hstate, sim.hmodel)
    update!(sim.fstate, obs, sim.filter)
    return obs
end

function run!(sim::NestedFilterSimulation; T::Int)
    times = zeros(T)
    epsps = zeros(T)
    time = 0.
    for i in 1:T
        obs = propagate!(sim)
        times[i] = time += obs.dt
        epsps[i] = obs.EPSP
    end
    return times, epsps
end
