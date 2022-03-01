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
    function FixedTimestep(dt::Real)
        dt <= 0 &&
            throw(ErrorException("FixedTimestep must have strictly positive argument."))
        return new{typeof(dt)}(dt)
    end
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

Produces a predefined finite sequence of time steps.
The simulation terminates when the sequence is exhausted.
"""
struct DeterministicTrain{T} <: Timestep
    train::T
    function DeterministicTrain(v::AbstractVector{<:Real})
        isempty(v) && 
            throw(ErrorException("DeterminisicTrain must have nonempty argument."))
        any(v .<= 0) && 
            throw(ErrorException("DeterminisicTrain needs strictly positive arguments."))
        return new{typeof(v)}(reverse(v))
    end
end

get_step(timestep::FixedTimestep) = timestep.dt
get_step(timestep::RandomTimestep) = rand(timestep.distribution)

function get_step(timestep::DeterministicTrain)
    if isempty(timestep.train)
        return nothing
    else
        return pop!(timestep.train)
    end
end
