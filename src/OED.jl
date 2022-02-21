"""
    OEDPolicy <: Timestep

An abstract type for choosing time steps based on optimizing a given cost function.
This is provided in order to do active inference.
"""
abstract type OEDPolicy <: Timestep end

"""
    policy(sim)

Return the instance of OEDPolicy used in simulation `sim`.
"""
policy(sim::NestedFilterSimulation) = sim.tsteps

function get_step(sim::NestedFilterSimulation{T1, T2, T3, T4, T5, T6, T7}) where 
{T1, T2, T3, T4, T5 <: OEDPolicy, T6, T7}
    policy = sim.tsteps
    return _oed!(sim, policy)
end

_oed!(sim, policy::OEDPolicy) = _oed!(policy)


"""
    Uniform(dts)

Randomly sample from a discrete set of time steps `dts`.
This is equivalent to, but more convenient to use than `RandomTimestep(dist)` with `dist` a uniform distribution on `dts`. 
"""
struct Uniform{T} <: OEDPolicy
    dts::T
end

_oed!(policy::Uniform) = rand(policy.dts)
