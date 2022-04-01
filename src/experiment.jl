struct NestedFilterExperiment{T1, T2, T3}
    filter::T1
    fstate::T2
    tsteps::T3
end

function NestedFilterExperiment(
    Nrng, prng, qrng, σrng, τrng,
    m_out, m_in, width;
    timestep
)
    filter = NestedParticleFilter(width)
    fstate = NestedParticleState(
                m_out, m_in,
                Nrng, prng, qrng, σrng, τrng
             )
    return NestedFilterExperiment(filter, fstate, timestep)
end
