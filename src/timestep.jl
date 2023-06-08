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
    (ts::Timestep)()

Returns a value for the time step, based on the chosen method for choosing a time step.
"""
function (ts::Timestep)() end

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
    stack::T
    function DeterministicTrain(v::AbstractVector{<:Real})
        any(v .<= 0) && 
            throw(ErrorException("DeterminisicTrain needs strictly positive arguments."))
        return new{typeof(v)}(reverse(v))
    end
end

(timestep::FixedTimestep)() = timestep.dt
(timestep::RandomTimestep)() = rand(timestep.distribution)
(timestep::DeterministicTrain)() = isempty(timestep.stack) ? nothing : pop!(timestep.stack)
  

function Base.show(io::IO, ::MIME"text/plain", timestep::FixedTimestep)
    print(io, "Fixed time step dt = ", timestep.dt)
end

function Base.show(io::IO, timestep::FixedTimestep)
    print(io, "Fixed time step dt = ", timestep.dt)
end

function Base.show(io::IO, ::MIME"text/plain", timestep::RandomTimestep)
    print(io, "Random time step with distribution ", timestep.distribution)
end

function Base.show(io::IO, timestep::RandomTimestep)
    print(io, "Random time step with distribution ", timestep.distribution)
end


function Base.show(io::IO, ::MIME"text/plain", timestep::DeterministicTrain)
    print(io, "Deterministic sequence of timesteps")
end

function Base.show(io::IO, timestep::DeterministicTrain)
    print(io, "Deterministic sequence of timesteps")
end