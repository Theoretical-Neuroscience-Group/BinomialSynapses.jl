abstract type Timestep end

function get_step(::Timestep) end

struct FixedTimestep{T} <: Timestep
    dt::T
end

struct RandomTimestep{T} <: Timestep
    distribution::T
end

get_step(timestep::FixedTimestep) = timestep.dt
get_step(timestep::RandomTimestep) = rand(timestep.distribution)

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
