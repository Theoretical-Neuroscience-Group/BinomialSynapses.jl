"""
    Timestep

An abstract type specifying the way a time step is chosen. 
The following basic types are supported:
- `FixedTimestep`: a constant, fixed number is used
- `RandomTimestep`: the time step is random from a fixed distribution
- `OEDPolicy`: an optimal time step is chosen based on some objective function
"""
abstract type Timestep end

"""
    get_step(::Timestep)

Returns a value for the time step, based on the chosen method for choosing a time step.
"""
function get_step(::Timestep) end

"""
    FixedTimestep(dt)

    Choose a fixed time step of size `dt`.
"""
struct FixedTimestep{T} <: Timestep
    dt::T
end

"""
    RandomTimestep(dist)

Choose a random time step from distribution `dist`.
There must be an implementation of `rand` for `typeof(dist)`.
"""
struct RandomTimestep{T} <: Timestep
    distribution::T
end

"""
    DeterministicTrain(train)

Follows a predefined fixed train of ISIs.
"""
struct DeterministicTrain{T} <: Timestep
    train::T
end

get_step(timestep::FixedTimestep) = timestep.dt
get_step(timestep::RandomTimestep) = rand(timestep.distribution)
