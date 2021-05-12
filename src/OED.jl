abstract type OEDPolicy <: Timestep end

policy(sim::NestedFilterSimulation) = sim.tsteps

function get_step(sim::NestedFilterSimulation{T1, T2, T3, T4, T5, T6, T7}) where 
{T1, T2, T3, T4, T5 <: OEDPolicy, T6, T7}
    policy = policy(sim)
    return _oed!(sim, policy)
end

_oed!(sim, policy::OEDPolicy) = _oed!(policy)



struct Uniform{T} <: OEDPolicy
    dts::T
end

_oed!(policy::Uniform) = rand(policy.dts)
