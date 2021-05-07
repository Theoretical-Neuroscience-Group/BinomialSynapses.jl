abstract type Timestep end

struct FixedTimestep{T} <: Timestep
    dt::T
end

struct RandomTimestep{T} <: Timestep
    distribution::T
end

get_step(timestep::FixedTimestep) = timestep.dt
get_step(timestep::RandomTimestep) = rand(timestep.distribution)